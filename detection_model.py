import os
import cv2
import numpy as np
import onnxruntime as ort

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(message)s',
    datefmt='%H:%M:%S' # To add date: %Y-%m-%d
)

class ObjectDetector:
    """
    Object detection using a YOLOv11 ONNX model, optimized with the best
    available ONNX Runtime Execution Provider for the current hardware.
    """
    def __init__(self, model_size='small', conf_thres=0.25, iou_thres=0.45, classes=None):
        logging.info("Initializing ObjectDetector with ONNX Runtime.")
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        model_map = {
            'nano': './models/yolo11n.onnx',
            'small': './models/yolo11s.onnx',
            'medium': './models/yolo11m.onnx',
            'large': './models/yolo11l.onnx',
            'extra': './models/yolo11x.onnx'
        }
        onnx_model_path = model_map.get(model_size.lower())
        if not onnx_model_path or not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Model file '{onnx_model_path}' not found for size '{model_size}'.")
        
        logging.info(f"Loading ONNX model from: {onnx_model_path}")
        
        # Logic to automatically select the best available provider for object detection
        available_providers = ort.get_available_providers()
        provider_options = None
        
        if 'TensorrtExecutionProvider' in available_providers:
            logging.info("Using TensorRT Execution Provider.")
            provider = 'TensorrtExecutionProvider'
            cache_path = os.path.join(os.path.dirname(__file__), "trt_cache_yolo")
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            provider_options = [{
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_path,
            }]
        elif 'CUDAExecutionProvider' in available_providers:
            logging.info("Using CUDA Execution Provider.")
            provider = 'CUDAExecutionProvider'
        elif 'DmlExecutionProvider' in available_providers:
            logging.info("Using DirectML Execution Provider (for Windows AMD/Intel GPU).")
            provider = 'DmlExecutionProvider'
        else:
            logging.info("No specialized GPU provider found. Using CPU Execution Provider.")
            provider = 'CPUExecutionProvider'

        try:
            self.session = ort.InferenceSession(
                onnx_model_path, 
                providers=[provider], 
                provider_options=provider_options
            )
            logging.info(f"ONNX session created successfully for YOLOv11 using provider: {self.session.get_providers()[0]}")
        except Exception as e:
            logging.info(f"Error loading YOLOv11 ONNX model: {e}. Falling back to CPU.")
            self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

        input_details = self.session.get_inputs()[0]
        self.input_name = input_details.name
        self.input_height = input_details.shape[2]
        self.input_width = input_details.shape[3]
        logging.info(f"YOLOv11 model expects input of size: ({self.input_height}, {self.input_width})")

        self.filter_classes = classes
        self.classes = self._load_coco_classes()

    # Load COCO class names
    def _load_coco_classes(self):
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # Preprocess the input image
    def _preprocess(self, image):
        img_h, img_w, _ = image.shape
        ratio = min(self.input_width / img_w, self.input_height / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded_img[(self.input_height - new_h) // 2:(self.input_height - new_h) // 2 + new_h, 
                   (self.input_width - new_w) // 2:(self.input_width - new_w) // 2 + new_w, :] = resized_img
        input_tensor = padded_img.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor, ratio, (self.input_height - new_h) // 2, (self.input_width - new_w) // 2

    # Postprocess the model output
    def _postprocess(self, output, ratio, pad_y, pad_x):
        predictions = np.squeeze(output).T
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        if len(predictions) == 0:
            return [], [], []
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        if self.filter_classes is not None:
            class_mask = np.isin(class_ids, self.filter_classes)
            predictions = predictions[class_mask]
            scores = scores[class_mask]
            class_ids = class_ids[class_mask]
            if len(predictions) == 0:
                return [], [], []
        boxes = predictions[:, :4]
        x1 = (boxes[:, 0] - boxes[:, 2] / 2)
        y1 = (boxes[:, 1] - boxes[:, 3] / 2)
        x2 = (boxes[:, 0] + boxes[:, 2] / 2)
        y2 = (boxes[:, 1] + boxes[:, 3] / 2)
        boxes_scaled = np.column_stack((x1, y1, x2, y2))
        boxes_scaled[:, 0] = (boxes_scaled[:, 0] - pad_x) / ratio
        boxes_scaled[:, 1] = (boxes_scaled[:, 1] - pad_y) / ratio
        boxes_scaled[:, 2] = (boxes_scaled[:, 2] - pad_x) / ratio
        boxes_scaled[:, 3] = (boxes_scaled[:, 3] - pad_y) / ratio
        indices = cv2.dnn.NMSBoxes(boxes_scaled.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        final_boxes, final_scores, final_class_ids = [], [], []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes_scaled[i])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])
        return final_boxes, final_scores, final_class_ids

    # Main detection method
    def detect(self, image):
        input_tensor, ratio, pad_y, pad_x = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes, scores, class_ids = self._postprocess(outputs[0], ratio, pad_y, pad_x)
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append([box, score, class_id, None])
        return image, detections

    def get_class_names(self):
        return self.classes