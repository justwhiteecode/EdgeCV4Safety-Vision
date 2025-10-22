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
CAMERA_EXPOSURE_TIME = 500  # microsecondi

# Inizializza Aravis
camera = Aravis.Camera.new(None)
device = camera.get_device()

# Parametri baseq
try:
    camera.set_frame_rate(CAMERA_FRAME_RATE)
except Exception as e:
    print(f"Warning: impossibile impostare frame rate: {e}")

try:
    camera.set_pixel_format(CAMERA_PIXEL_FORMAT)
except Exception as e:
    print(f"Warning: impossibile impostare pixel format: {e}")

try:
    camera.set_gain(CAMERA_GAIN)
except Exception as e:
    print(f"Warning: impossibile impostare gain: {e}")

# Esposizione
if CAMERA_AUTO_EXPOSURE:
    try:
        camera.set_exposure_mode(Aravis.ExposureMode.CONTINUOUS)
        print("Esposizione impostata su automatica (Continuous).")
    except AttributeError:
        try:
            camera.set_feature('ExposureAuto', 'Continuous')
            print("Tentativo di impostare 'ExposureAuto' su 'Continuous'.")
        except Exception as e:
            print(f"Errore nel settare ExposureAuto: {e}. L'esposizione automatica potrebbe non essere configurata.")
    except Exception as e:
        print(f"Errore durante l'impostazione dell'esposizione automatica: {e}")
else:
    try:
        camera.set_exposure_mode(Aravis.ExposureMode.MANUAL)
        camera.set_exposure_time(CAMERA_EXPOSURE_TIME)
        print(f"Esposizione impostata su manuale con tempo di esposizione: {CAMERA_EXPOSURE_TIME} µs")
    except AttributeError:
        try:
            camera.set_feature('ExposureMode', 'Off')
            camera.set_feature('ExposureTime', CAMERA_EXPOSURE_TIME)
            print(f"Tentativo di impostare 'ExposureMode' su 'Off' e 'ExposureTime' su {CAMERA_EXPOSURE_TIME} µs.")
        except Exception as e:
            print(f"Errore nel settare ExposureMode/ExposureTime: {e}. L'esposizione manuale potrebbe non essere configurata.")
    except Exception as e:
        print(f"Errore durante l'impostazione dell'esposizione manuale: {e}")

# Stream setup
payload = camera.get_payload()
stream = camera.create_stream(None, None)
for i in range(5):
    stream.push_buffer(Aravis.Buffer.new_allocate(payload))

camera.start_acquisition()

# Setup finestra
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 800, 800)

print("Premi 'q' per uscire.")

while True:
    buffer = stream.pop_buffer()
    if buffer:
        data = buffer.get_data()
        height = camera.get_integer("Height")
        width = camera.get_integer("Width")

        # Converte in numpy array Bayer (uint8)
        frame = np.ndarray(
            buffer=data,
            shape=(height, width),
            dtype=np.uint8
        )

        # Conversione BayerRG → RGB (CPU)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)

        # Mostra immagine
        cv2.imshow("Camera", rgb)

        stream.push_buffer(buffer)

    # Tasti
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.stop_acquisition()
cv2.destroyAllWindows()
