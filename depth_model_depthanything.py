import os
import numpy as np
import cv2
import onnxruntime as ort

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)06d [%(name)s] - %(message)s',
    datefmt='%H:%M:%S' # To add date: %Y-%m-%d
)

depthanything_logger = logging.getLogger('DEPTH ANYTHING')

class DepthEstimatorDepthAnything:
    """
    Depth estimation using Depth Anything v2 ONNX model, optimized with the best
    available ONNX Runtime Execution Provider for the current hardware.
    """
    def __init__(self, model_size='small', provider=None):
        depthanything_logger.info("Initializing DepthEstimator (DepthAnything) with ONNX Runtime.")

        model_map = {
            'small': './models/depth_anything_v2_metric_indoor_small.onnx',
            'base': './models/depth_anything_v2_metric_indoor_base.onnx',
            'large': './models/depth_anything_v2_metric_indoor_large.onnx'
        }
        onnx_model_path = model_map.get(model_size.lower())
        if not onnx_model_path or not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Model file '{onnx_model_path}' not found for model size '{model_size}'.")
        
        depthanything_logger.info(f"Loading ONNX model from: {onnx_model_path}")

        # Logic to automatically select the best available provider for depth estimation
        available_providers = ort.get_available_providers()
        provider_options = None
        sess_options = ort.SessionOptions()
        
        if 'TensorrtExecutionProvider' in available_providers and (provider is None or 'tensorrt' in provider.lower()):
            depthanything_logger.info("Using TensorRT Execution Provider.")
            provider = 'TensorrtExecutionProvider'
            cache_path = os.path.join(os.path.dirname(__file__), "trt_cache_depthanything")
            if not os.path.exists(cache_path): os.makedirs(cache_path)
            provider_options = [{
                'trt_engine_cache_enable': True, 
                'trt_engine_cache_path': cache_path,
                #'trt_fp16_enable': False,
            }]
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        elif 'CUDAExecutionProvider' in available_providers and (provider is None or 'cuda' in provider.lower()):
            provider = 'CUDAExecutionProvider'
            depthanything_logger.info("Using CUDA Execution Provider.")
        elif 'DmlExecutionProvider' in available_providers and (provider is None or 'dml' in provider.lower()):
            provider = 'DmlExecutionProvider'
            depthanything_logger.info("Using DirectML Execution Provider.")
        else:
            provider = 'CPUExecutionProvider'
            depthanything_logger.info("Using CPU Execution Provider.")

        try:
            self.session = ort.InferenceSession(
                onnx_model_path,
                sess_options=sess_options,
                providers=[provider], 
                provider_options=provider_options
            )
            depthanything_logger.info(f"ONNX session created successfully using provider: {self.session.get_providers()[0]}")
        except Exception as e:
            depthanything_logger.info(f"Error loading ONNX model: {e}. Falling back to CPU.")
            self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

        input_details = self.session.get_inputs()[0]
        self.input_name = input_details.name
        self.input_height = input_details.shape[2]
        self.input_width = input_details.shape[3]
        depthanything_logger.info(f"Model expects input of size: ({self.input_height}, {self.input_width})")

    # Preprocess the input image
    def _preprocess(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image_resized = cv2.resize(image_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_resized - mean) / std
        image_transposed = image_normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(image_transposed, axis=0).astype('float32')
        return input_tensor

    # Estimate depth for the input image
    def estimate_depth(self, image):
        original_h, original_w = image.shape[:2]
        input_tensor = self._preprocess(image)
        result = self.session.run(None, {self.input_name: input_tensor})
        depth_map = np.squeeze(result[0])
        depth_resized = cv2.resize(depth_map, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return depth_resized

    # Colorize the depth map for visualization
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_map, dtype=float)
        depth_map_uint8 = (depth_normalized * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth

    # Get depth value at a specific point
    def get_depth_at_point(self, depth_map, x, y):
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[int(y), int(x)]
        return 0.0

    # Get depth statistics in a bounding box region
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1] - 1, x2), min(depth_map.shape[0] - 1, y2)
        if x1 >= x2 or y1 >= y2: return 0.0
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0: return 0.0
        if method == 'mean': return float(np.mean(region))
        if method == 'min': return float(np.min(region))
        return float(np.median(region))