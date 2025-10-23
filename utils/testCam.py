#!/usr/bin/env python3
import gi
gi.require_version('Aravis', '0.10')
from gi.repository import Aravis, GLib
import cv2
import numpy as np

CAMERA_FRAME_RATE = 5
CAMERA_PIXEL_FORMAT = Aravis.PIXEL_FORMAT_BAYER_RG_8
CAMERA_GAIN = 30.0
CAMERA_AUTO_EXPOSURE = False
CAMERA_EXPOSURE_TIME = 500  # microseconds

# Initialize Aravis
camera = Aravis.Camera.new(None)
device = camera.get_device()

# Basic parameters
try:
    camera.set_frame_rate(CAMERA_FRAME_RATE)
except Exception as e:
    print(f"Warning: unable to set frame rate: {e}")

try:
    camera.set_pixel_format(CAMERA_PIXEL_FORMAT)
except Exception as e:
    print(f"Warning: unable to set pixel format: {e}")

try:
    camera.set_gain(CAMERA_GAIN)
except Exception as e:
    print(f"Warning: unable to set gain: {e}")

# Exposure
if CAMERA_AUTO_EXPOSURE:
    try:
        camera.set_exposure_mode(Aravis.ExposureMode.CONTINUOUS)
        print("Exposure set to automatic (Continuous).")
    except AttributeError:
        try:
            camera.set_feature('ExposureAuto', 'Continuous')
            print("Attempting to set 'ExposureAuto' to 'Continuous'.")
        except Exception as e:
            print(f"Error setting ExposureAuto: {e}. Automatic exposure may not be configured.")
    except Exception as e:
        print(f"Error during automatic exposure setting: {e}")
else:
    try:
        camera.set_exposure_mode(Aravis.ExposureMode.MANUAL)
        camera.set_exposure_time(CAMERA_EXPOSURE_TIME)
        print(f"Exposure set to manual with exposure time: {CAMERA_EXPOSURE_TIME} µs")
    except AttributeError:
        try:
            camera.set_feature('ExposureMode', 'Off')
            camera.set_feature('ExposureTime', CAMERA_EXPOSURE_TIME)
            print(f"Attempting to set 'ExposureMode' to 'Off' and 'ExposureTime' to {CAMERA_EXPOSURE_TIME} µs.")
        except Exception as e:
            print(f"Error setting ExposureMode/ExposureTime: {e}. Manual exposure may not be configured.")
    except Exception as e:
        print(f"Error during manual exposure setting: {e}")

# Stream setup
payload = camera.get_payload()
stream = camera.create_stream(None, None)
for i in range(5):
    stream.push_buffer(Aravis.Buffer.new_allocate(payload))

camera.start_acquisition()

# Setup window
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 800, 800)

print("Press 'q' to exit.")

while True:
    buffer = stream.pop_buffer()
    if buffer:
        data = buffer.get_data()
        height = camera.get_integer("Height")
        width = camera.get_integer("Width")

        # Convert to numpy array Bayer (uint8)
        frame = np.ndarray(
            buffer=data,
            shape=(height, width),
            dtype=np.uint8
        )

        # Convert BayerRG → RGB (CPU)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)

        # Show image
        cv2.imshow("Camera", rgb)

        stream.push_buffer(buffer)

    # Keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.stop_acquisition()
cv2.destroyAllWindows()
