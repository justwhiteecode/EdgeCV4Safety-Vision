import os
import numpy as np
import cv2
import onnxruntime as ort

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(message)s',
    datefmt='%H:%M:%S' # To add date: %Y-%m-%d
)

class DepthEstimatorUniDepthONNX:
    """
    Depth estimation using UniDepth V2 ONNX model, optimized with the best
    available ONNX Runtime Execution Provider.
    """
    def __init__(self, model_path='unidepth-v2-vitl14.onnx'):
        logging.info("Initializing DepthEstimator (UniDepth) with ONNX Runtime.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        logging.info(f"Loading ONNX model from: {model_path}")

        # Logic to automatically select the best available provider for object detection
        available_providers = ort.get_available_providers()
        provider_options = None

        if 'TensorrtExecutionProvider' in available_providers:
            logging.info("Using TensorRT Execution Provider.")
            provider = 'TensorrtExecutionProvider'
            cache_path = os.path.join(os.path.dirname(__file__), "trt_cache_unidepth")
            if not os.path.exists(cache_path): os.makedirs(cache_path)
            provider_options = [{'trt_engine_cache_enable': True, 'trt_engine_cache_path': cache_path}]
        elif 'CUDAExecutionProvider' in available_providers:
            provider = 'CUDAExecutionProvider'
            logging.info("Using CUDA Execution Provider.")
        else:
            provider = 'CPUExecutionProvider'
            logging.info("Using CPU Execution Provider.")

        try:
            self.session = ort.InferenceSession(model_path, providers=[provider], provider_options=provider_options)
            logging.info(f"ONNX session created successfully using provider: {self.session.get_providers()[0]}")
        except Exception as e:
            logging.info(f"Error loading ONNX model: {e}. Falling back to CPU.")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # Automatically read input dimensions from the ONNX model
        input_details = self.session.get_inputs()[0]
        self.input_name = input_details.name
        self.input_height = input_details.shape[2]
        self.input_width = input_details.shape[3]
        self.intrinsics_name = self.session.get_inputs()[1].name
        logging.info(f"Model expects input of size: ({self.input_height}, {self.input_width})")

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
        image_transposed = image_rgb.transpose(2, 0, 1)
        input_tensor = np.expand_dims(image_transposed, axis=0).astype('float32')
        return input_tensor

    # Estimate depth for the input image
    def estimate_depth(self, image):
        original_h, original_w = image.shape[:2]
        
        input_tensor = self._preprocess(image)
        intrinsics_tensor = self._create_default_intrinsics(original_h, original_w)

        inputs = {self.input_name: input_tensor, self.intrinsics_name: intrinsics_tensor}
        result = self.session.run(None, inputs)
        
        depth_map_low_res = np.squeeze(result[0])

        # Resize the depth map to the original image size
        depth_resized_to_original = cv2.resize(depth_map_low_res, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return depth_resized_to_original

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