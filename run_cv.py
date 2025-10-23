#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
import socket
import struct
import logging
import math

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(message)s',
    datefmt='%H:%M:%S' # To add date: %Y-%m-%d
)

# Classes imports
from detection_model import ObjectDetector
from depth_model_depthanything import DepthEstimatorDepthAnything
from depth_model_unidepth import DepthEstimatorUniDepth
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

# Aravis (GigE Vision) https://github.com/AravisProject/aravis
import gi
gi.require_version('Aravis', '0.10')
from gi.repository import Aravis

def check_keypress():
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
        return True
    return False

def main():
    # ======================================================================================================
    """
    --------------------------------------------------------------------------------------------------------
    Configuration variables (modify these as needed)
    --------------------------------------------------------------------------------------------------------
    """

    # Models settings
    DEPTH_MODEL_CHOICE = "unidepth" # Depth estimation model: "unidepth", "depthanything"
    DEPTH_MODEL_SIZE = "small"  # Depth model size: "small", "base", "large"
    YOLO_MODEL_SIZE = "extra" # YOLO11 model size: "nano", "small", "medium", "large", "extra"

    # Detection settings
    CONF_THRESHOLD = 0.75  # Confidence threshold for object detection
    IOU_THRESHOLD = 0.6  # IoU threshold for NMS
    CLASSES = [0]  # Filter by class, e.g., [0, 1, 2] for specific CLASSES, None for all classes available

    # Feature toggles
    ENABLE_BEV = False  # Enable Bird's Eye View visualization
    ENABLE_PSEUDO_3D = True  # Enable pseudo-3D visualization

    # Preview enabled/disabled (strongly affects real-time performance)
    WINDOW_CAMERA_PREVIEW = False  # Show camera preview window
    WINDOW_RESULTS_PREVIEW = False  # Show results window

    # Camera settings
    CAMERA_IP = '192.168.37.150' # None for aravis auto-choice (first found)
    CAMERA_FRAME_RATE = 22 # Check max support (pixel format dependent) i.e. on https://www.baslerweb.com/en/tools/frame-rate-calculator/camera=a2A2448-23gcBAS
    CAMERA_PIXEL_FORMAT = Aravis.PIXEL_FORMAT_BAYER_RG_8
    CAMERA_GAIN = 30.0
    CAMERA_AUTO_EXPOSURE = True
    CAMERA_EXPOSURE_TIME = 8000
    CAMERA_BUFFER_TIMEOUT = 200000
    CAMERA_IMAGE_ROTATION_ANGLE = 0 # 0 to disable
    CAMERA_ROI_HEIGHT = 0 # 0 to disable
    CAMERA_ROI_WIDTH = 0 # 0 to disable

    # Camera infos (Set to 0 to use distances from camera)
    CAMERA_HEIGHT_FROM_GROUND = 1.7 # Camera height from ground in meters
    CAMERA_DISTANCE_FROM_FIXED_OBJECT = 2 # Distance from a known fixed object straight to camerain meters (used for punctual depth estimation from this object)

    # Output Target Node settings
    TARGET_NODE_IP = '192.168.37.50'
    TARGET_NODE_PORT = 13750

    # ======================================================================================================

    # Socket UDP for data sending to Target Node
    try:
        udp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logging.info(f"i ] UDP Socket created to sendo data to {TARGET_NODE_IP}:{TARGET_NODE_PORT}")
    except socket.error as e:
        logging.info(f"e ] Error in creating UDP Socket: {e}")
        udp_client_socket = None
    
    # Initialize models
    logging.info("Initializing models...")
    detector = ObjectDetector(
        model_size=YOLO_MODEL_SIZE,
        conf_thres=CONF_THRESHOLD, 
        iou_thres=IOU_THRESHOLD,   
        classes=CLASSES
    )
    try:
        if DEPTH_MODEL_CHOICE == "unidepth":
            logging.info("i ] Using UniDepthV2-ONNX model.")
            # Assicurati che il percorso corrisponda al nome del file ONNX che hai generato
            depth_estimator = DepthEstimatorUniDepth(model_size=DEPTH_MODEL_SIZE)
        elif DEPTH_MODEL_CHOICE == "depthanything":
            logging.info("i ] Using DepthAnythingV2-ONNX model.")
            depth_estimator = DepthEstimatorDepthAnything(model_size=DEPTH_MODEL_SIZE)
        else:
            raise ValueError(f"Unknown depth model choice: {DEPTH_MODEL_CHOICE}")    
    except Exception as e:
        logging.error(f"e ] CRITICAL: Error initializing depth estimator: {e}")
        exit(1)

    #-------------------------------------------------------------------------------------------------------
    # Windows setup
    if WINDOW_CAMERA_PREVIEW:
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 800, 800)

    if WINDOW_RESULTS_PREVIEW:
        cv2.namedWindow("3D Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("3D Object Detection", 800, 800)
    
    #-------------------------------------------------------------------------------------------------------
    # Initialize 3D bounding box estimator with default parameters
    # Simplified approach - focus on 2D detection with depth information
    if ENABLE_PSEUDO_3D:
    bbox3d_estimator = BBox3DEstimator()
    # Initialize Bird's Eye View if enabled
    if ENABLE_BEV:
        # Use a scale that works well for the 1-5 meter range
        bev = BirdEyeView(scale=60, size=(300, 300))  # Increased scale to spread objects out

    #-------------------------------------------------------------------------------------------------------
    # Camera creation & configuration
    aravis_camera = None
    aravis_stream = None
    try:
        aravis_camera = Aravis.Camera.new(CAMERA_IP)
    except TypeError:
        logging.info(f"e ] Error: Could not find Aravis camera at IP {CAMERA_IP}. Exiting.")
        exit(2)
    if not aravis_camera:
        logging.info("e ] Error: No Aravis-compatible camera found. Exiting.")
        exit(2)
    aravis_camera.set_frame_rate(CAMERA_FRAME_RATE)
    aravis_camera.set_pixel_format(CAMERA_PIXEL_FORMAT)
    aravis_camera.set_gain(CAMERA_GAIN)
    if CAMERA_AUTO_EXPOSURE:
        try:
            aravis_camera.set_exposure_mode(Aravis.ExposureMode.CONTINUOUS)
            logging.info("i ] Exposure set to continuous.")
        except AttributeError:
            try:
                aravis_camera.set_exposure_mode(Aravis.ExposureMode.AUTO)
                logging.info("i ] Attempting to set 'ExposureAuto' to 'Continuous'.")
            except Exception as e:
                logging.info(f"e ] Error setting ExposureAuto: {e}. Automatic exposure may not be configured.")
        except Exception as e:
            logging.info(f"e ] Error during setting of automatic exposure: {e}")
    else:
        try:
            aravis_camera.set_exposure_mode(Aravis.ExposureMode.MANUAL)
            aravis_camera.set_exposure_time(CAMERA_EXPOSURE_TIME)
            logging.info(f"i ] Exposure set to manual with exposure time: {CAMERA_EXPOSURE_TIME} µs")
        except AttributeError:
            try:
                aravis_camera.set_feature('ExposureMode', 'Off')
                aravis_camera.set_feature('ExposureTime', CAMERA_EXPOSURE_TIME)
                logging.info(f"i ] Attempting to set 'ExposureMode' to 'Off' and 'ExposureTime' to {CAMERA_EXPOSURE_TIME} µs.")
            except Exception as e:
                logging.info(f"e ] Error setting ExposureMode/ExposureTime: {e}. Manual exposure may not be configured.")
        except Exception as e:
            logging.info(f"e ] Error during setting of manual exposure: {e}")
    #-------------------------------------------------------------------------------------------------------
    # Get input video properties
    width, height = aravis_camera.get_sensor_size()
    aravis_camera.set_region(CAMERA_ROI_WIDTH, CAMERA_ROI_HEIGHT, width, height)
    payload = aravis_camera.get_payload()
    logging.info(f"i ] Opening video source: {aravis_camera.get_model_name()} by {aravis_camera.get_vendor_name()} [{width} x {height}]")
    aravis_stream = aravis_camera.create_stream(None, None)
    for _ in range(CAMERA_FRAME_RATE): 
        aravis_stream.push_buffer(Aravis.Buffer.new_allocate(payload))
    
    #-------------------------------------------------------------------------------------------------------
    logging.info("> ] Starting camera acquisition...")
    aravis_camera.start_acquisition()

    #-------------------------------------------------------------------------------------------------------
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    #-------------------------------------------------------------------------------------------------------
    logging.info("i ] Starting processing...")
    while True:
        try:
            # Step 0: Frame acquisition from camera    
            try:
                # Read frame (last arrived, discard previous for real-time performance)
                buffer = None
                while True:
                    b = aravis_stream.try_pop_buffer()
                    if b is None:
                        break
                    if buffer:
                        aravis_stream.push_buffer(buffer)
                    buffer = b

                if buffer is None:
                    continue
                # Data to image
                frame = np.ndarray(
                    buffer=buffer.get_data(),
                    shape=(height, width),
                    dtype=np.uint8
                )

                # Frame conversion to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)

                if WINDOW_CAMERA_PREVIEW: cv2.imshow("Camera", frame_rgb)

                # Make copies for different visualizations
                original_frame = frame_rgb.copy()
                detection_frame = frame_rgb.copy()
                depth_frame = frame_rgb.copy()
                result_frame = frame_rgb.copy()

            # Step 1: Object Detection (YOLO-ONNX)
            try:
                detection_frame, detections = detector.detect(detection_frame)
            except Exception as e:
                    logging.info(f"e ] Error during object detection: {e}")
                    detections = []
                    cv2.putText(detection_frame, "Detection Error", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Step 2: Depth Estimation (UnidpethV2-ONNX or DepthAnything-ONNX)
            depth_map = depth_estimator.estimate_depth(original_frame)
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                logging.info(f"Error during depth estimation: {e}")
                # Create a dummy depth map
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Step 3: 3D Bounding Box Estimation
            boxes_3d = []
            active_ids = []
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    
                    # Get class name
                    class_name = detector.get_class_names()[class_id]
                    
                    # Get depth in the region of the bounding box
                    # Try different methods for depth estimation
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        # For people and animals, use the center point depth
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        # For other objects, use the median depth in the region
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'

                    # Create a simplified 3D box representation
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': depth_method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score
                    }
                    boxes_3d.append(box_3d)
                    # Keep track of active IDs for tracker cleanup
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    logging.info(f"Error processing detection: {e}")
                    continue
            # Clean up trackers for objects that are no longer detected
            bbox3d_estimator.cleanup_trackers(active_ids)

            # Step 4: Visualization & Distance selection
            # Draw boxes on the result frame && data send
            min_depth_value = float('inf')
            for box_3d in boxes_3d:
                try:
                    act_distance = abs((math.sqrt(pow(box_3d['depth_value'], 2) - pow(CAMERA_HEIGHT_FROM_GROUND, 2))) - CAMERA_DISTANCE_FROM_FIXED_OBJECT) # distance calculated from terrain projection of the camera to object (Pitagora theorem) and subtracting camera distance from a known fixed point
                    logging.info(f"r ] Detected {box_3d['class_name']} ({box_3d['score']:.2f}) at depth {act_distance:.2f} m.")
                    # Taking the minimum depth value (the closest object detected)
                    if act_distance < min_depth_value:
                        min_depth_value = act_distance
                    
                    if WINDOW_RESULTS_PREVIEW:
                    # Determine color based on class
                        class_name = box_3d['class_name'].lower()
                        if 'car' in class_name or 'vehicle' in class_name:
                            color = (0, 0, 255)  # Red
                        elif 'person' in class_name:
                            color = (0, 255, 0)  # Green
                        elif 'bicycle' in class_name or 'motorcycle' in class_name:
                            color = (255, 0, 0)  # Blue
                        elif 'potted plant' in class_name or 'plant' in class_name:
                            color = (0, 255, 255)  # Yellow
                        else:
                            color = (255, 255, 255)  # White
                        
                        # Draw box with depth information
                        result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                except Exception as e:
                    logging.info(f"Error drawing box: {e}")
                    continue

            # Step 5: Send data via UDP if socket is available
            if udp_client_socket:
                try:
                    packed_distance = struct.pack('<f', float(min_depth_value)) # distance from arm
                    udp_client_socket.sendto(packed_distance, (TARGET_NODE_IP, TARGET_NODE_PORT))
                    logging.info(f"Minimum distance '{min_depth_value:.2f}' m send via UDP to {TARGET_NODE_IP}:{TARGET_NODE_PORT}")
                except socket.error as e:
                    logging.info(f"e ] Error while sending UDP packet: {e}")
            else:
                logging.info("e ] UDP Socket not initialized. Impossible to send data.")
            
            # Draw Bird's Eye View if enabled
            if ENABLE_BEV:
                try:
                    # Reset BEV and draw objects
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    bev_image = bev.get_image()
                    
                    # Resize BEV image to fit in the corner of the result frame
                    bev_height = height // 4  # Reduced from height/3 to height/4 for better fit
                    bev_width = bev_height
                    
                    # Ensure dimensions are valid
                    if bev_height > 0 and bev_width > 0:
                        # Resize BEV image
                        bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                        
                        # Create a region of interest in the result frame
                        roi = result_frame[height - bev_height:height, 0:bev_width]
                        
                        # Simple overlay - just copy the BEV image to the ROI
                        result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                        
                        # Add a border around the BEV visualization
                        cv2.rectangle(result_frame, 
                                     (0, height - bev_height), 
                                     (bev_width, height), 
                                     (255, 255, 255), 1)
                        
                        # Add a title to the BEV visualization
                        cv2.putText(result_frame, "Bird's Eye View", 
                                   (10, height - bev_height + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    logging.info(f"Error drawing BEV: {e}")
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"
                logging.info(f"i ] {fps_display}")
            # Add FPS and device info to the result frame
            if WINDOW_RESULTS_PREVIEW:
                cv2.putText(result_frame, f"{fps_display} | Device: {device}", (30, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                # Add depth map to the corner of the result frame
                try:
                    depth_height = height // 4
                    depth_width = depth_height * width // height
                    depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                    result_frame[0:depth_height, 0:depth_width] = depth_resized
                except Exception as e:
                    logging.info(f"Error adding depth map to result: {e}")
                
                cv2.imshow("3D Object Detection", result_frame)

            # Check for key press to exit
            if (WINDOW_RESULTS_PREVIEW or WINDOW_CAMERA_PREVIEW) and check_keypress():
                logging.info("i ] Exiting program...")
                break

        except Exception as e:
            logging.info(f"Error processing frame: {e}")
            # Also check for key press during exception handling
            if (WINDOW_RESULTS_PREVIEW or WINDOW_CAMERA_PREVIEW) and check_keypress():
                logging.info("i ] Exiting program...")
                break
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nProgram interrupted by user (Ctrl+C)")
    finally:
        # Cleanup
        logging.info("Cleaning up resources...")
        aravis_camera.stop_acquisition()
        cv2.destroyAllWindows()
        if buffer is not None:
            aravis_stream.push_buffer(buffer)