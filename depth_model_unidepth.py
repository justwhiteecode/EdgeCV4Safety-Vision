import os
import numpy as np
import cv2
import onnxruntime as ort

import re # For version extraction

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)06d [%(name)s] - %(message)s',
    datefmt='%H:%M:%S' # To add date: %Y-%m-%d
)

unidepth_logger = logging.getLogger('UNIDEPTH')

def get_tensorrt_version():
    """Try to import the TensorRT library and return its version as a tuple of integers."""
    try:
        import tensorrt as trt
        version_str = trt.__version__
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if match:
            return tuple(map(int, match.groups()))
    except (ImportError, AttributeError):
        # If the library is not installed or does not have the __version__ attribute
        pass
    return None

def is_tensorrt_compatible():
    """Check if the installed TensorRT version meets the minimum required version."""
    MIN_TRT_VERSION = (8, 6, 0) # To use TensorRT EP with UniDepth, at least version 8.6.0 seems to be required
    version = get_tensorrt_version()
    if version is None:
        return False
    return version >= MIN_TRT_VERSION

class DepthEstimatorUniDepth:
    """
    Depth estimation using UniDepth V2 ONNX model, optimized with the best
    available ONNX Runtime Execution Provider.
    """
    def __init__(self, model_size='small', provider=None):
        unidepth_logger.info("Initializing DepthEstimator (UniDepth) with ONNX Runtime.")

        model_map = {
            'small': './models/unidepthv2s.onnx',
            'base': './models/unidepthv2b.onnx',
            'large': './models/unidepthv2l.onnx'
        }
        model_path = model_map.get(model_size.lower())

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        unidepth_logger.info(f"Loading ONNX model from: {model_path}")

        # Logic to automatically select the best available provider for object detection
        available_providers = ort.get_available_providers()
        provider_options = None
        sess_options = ort.SessionOptions()

        if 'TensorrtExecutionProvider' in available_providers and is_tensorrt_compatible() and (provider is None or 'tensorrt' in provider.lower()):
            unidepth_logger.info("Using TensorRT Execution Provider.")
            provider = 'TensorrtExecutionProvider'
            cache_path = os.path.join(os.path.dirname(__file__), "trt_cache_unidepth")
            if not os.path.exists(cache_path): os.makedirs(cache_path)
            provider_options = [{
                'trt_engine_cache_enable': True, 
                'trt_engine_cache_path': cache_path,
                #'trt_fp16_enable': False,
            }]
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        elif 'CUDAExecutionProvider' in available_providers and (provider is None or 'cuda' in provider.lower()):
            provider = 'CUDAExecutionProvider'
            unidepth_logger.info("Using CUDA Execution Provider.")
        elif 'DmlExecutionProvider' in available_providers and (provider is None or 'dml' in provider.lower()):
            unidepth_logger.info("Using DirectML Execution Provider (for Windows AMD/Intel GPU).")
            provider = 'DmlExecutionProvider'
        else:
            provider = 'CPUExecutionProvider'
            unidepth_logger.info("Using CPU Execution Provider.")

        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=[provider],
                provider_options=provider_options
            )
            unidepth_logger.info(f"ONNX session created successfully using provider: {self.session.get_providers()[0]}")
        except Exception as e:
            unidepth_logger.info(f"Error loading ONNX model: {e}. Falling back to CPU.")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # Automatically read input dimensions from the ONNX model
        input_details = self.session.get_inputs()[0]
        self.input_name = input_details.name
        self.input_height = input_details.shape[2]
        self.input_width = input_details.shape[3]
        unidepth_logger.info(f"Model expects input of size: ({self.input_height}, {self.input_width})")

        self.output_names = [output.name for output in self.session.get_outputs()]
        unidepth_logger.info(f"Model provides outputs: {self.output_names}")

    # Create default camera intrinsics
    def _create_default_intrinsics(self, h, w):
        focal_length = w
        cx, cy = w / 2, h / 2
        intrinsics = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32)
        return np.expand_dims(intrinsics, axis=0)

    # Preprocess the input image
    def _preprocess(self, image):
        # Resize to fixed dimensions required by the model
        image_resized = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        image_normalized = image_rgb.astype('float32') / 255.0

        image_transposed = image_normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(image_transposed, axis=0).astype('float32')
        return input_tensor

    # Estimate depth for the input image
    def estimate_depth(self, image):
        original_h, original_w = image.shape[:2]
        
        if image is None or image.size == 0:
            logging.error("e ] estimate_depth received a NULL or empty image.")
            return np.zeros((original_h, original_w), dtype=np.float32)

        try:
            input_tensor = self._preprocess(image)
            
            if input_tensor is None or input_tensor.size == 0:
                logging.error("e ] _preprocess returned an empty or None tensor.")
                return np.zeros((original_h, original_w), dtype=np.float32)

            inputs = {self.input_name: input_tensor}
            outputs = self.session.run(self.output_names, inputs)

            pts_3d = outputs[0]

            if pts_3d is None or pts_3d.size == 0:
                logging.error("The ONNX model (UniDepth) returned a null or empty output for 'pts_3d'.")
                return np.zeros((original_h, original_w), dtype=np.float32)

            depth_map_low_res = pts_3d[0, 2, :, :]
            if depth_map_low_res.ndim != 2:
                logging.error(f"e ] Extracted depth map has incorrect dimensions. Shape: {depth_map_low_res.shape}")
                return np.zeros((original_h, original_w), dtype=np.float32)

            if np.isnan(depth_map_low_res).any() or np.isinf(depth_map_low_res).any():
                logging.warning("The calculated depth map contains NaN or Inf values. Cleaning up.")
                depth_map_low_res = np.nan_to_num(depth_map_low_res)

            depth_resized_to_original = cv2.resize(depth_map_low_res, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            return depth_resized_to_original
        except Exception as e:
            logging.error(f"e ] Unhandled exception occurred in estimate_depth: {e}", exc_info=True)
            return np.zeros((original_h, original_w), dtype=np.float32)

    # Colorize the depth map for visualization
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min: depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else: depth_normalized = np.zeros_like(depth_map, dtype=float)
        depth_map_uint8 = (depth_normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_map_uint8, cmap)

    # Get depth value at a specific point
    def get_depth_at_point(self, depth_map, x, y):
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]: return depth_map[int(y), int(x)]
        return 0.0
    
    # Get depth statistics in a bounding box region
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1,y1=max(0,x1),max(0,y1)
        x2,y2=min(depth_map.shape[1]-1,x2),min(depth_map.shape[0]-1,y2)
        if x1>=x2 or y1>=y2: return 0.0
        region=depth_map[y1:y2,x1:x2]
        if region.size==0: return 0.0
        if method=='mean': return float(np.mean(region))
        return float(np.median(region))