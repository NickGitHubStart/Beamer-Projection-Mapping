"""
YOLO-Segmentierung + Homographie: wie 03_yolo_projection, aber mit Instanz-Masken
(Umriss pro Objekt) statt nur Bounding-Box-Brackets.

Modell: yolov8n-seg.pt (wird von Ultralytics bei Bedarf geladen).
"""
import numpy as np
import cv2
import os
import pickle
from ultralytics import YOLO


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CAMERA_INDEX = 1

relative_homographic_tranform_path = '../02_homogrphic_transform/homographic_tranform.pckl'

bool_fullscreen = False
YOLO_CONF = 0.25

# Segmentierungsmodell (nicht yolov8n.pt — das ist nur Detection)
model = YOLO("yolov8n-seg.pt")

COLORS = [
    (0, 255, 128),
    (0, 200, 255),
    (255, 80, 80),
    (255, 200, 0),
    (180, 0, 255),
]


# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------
def warp_image(image, H, output_width, output_height):
    return cv2.warpPerspective(image, H, (output_width, output_height))


# -------------------------------------------------
# Homographie
# -------------------------------------------------
if not os.path.exists(relative_homographic_tranform_path):
    print("Homographie fehlt. Erst 02_calc_pose_trans.py ausfuehren.")
    exit()

with open(relative_homographic_tranform_path, 'rb') as f:
    H, output_width, output_height = pickle.load(f)

# -------------------------------------------------
# Kamera + Fenster
# -------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise IOError(f"Kamera {CAMERA_INDEX} nicht verfuegbar")

cv2.namedWindow('Beamer_Window', cv2.WINDOW_NORMAL)
if bool_fullscreen:
    cv2.setWindowProperty('Beamer_Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Press ESC to exit — YOLO-Segmentierung auf gewarpter Ebene -> Beamer_Window")

# -------------------------------------------------
# Loop
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    dewarped_img = warp_image(frame, H, output_width, output_height)
    BeamerImage = np.zeros(dewarped_img.shape, np.uint8)

    results = model(dewarped_img, conf=YOLO_CONF, verbose=False)[0]

    if results.masks is not None and len(results.masks.xy) > 0:
        classes = results.boxes.cls.cpu().numpy().astype(int)
        for i, polygon_xy in enumerate(results.masks.xy):
            cls_id = int(classes[i])
            color = COLORS[cls_id % len(COLORS)]
            pts = np.array(polygon_xy, dtype=np.int32).reshape((-1, 1, 2))
            if pts.shape[0] >= 3:
                cv2.fillPoly(BeamerImage, [pts], color)

    cv2.imshow("Beamer_Window", BeamerImage)
    cv2.imshow("output", dewarped_img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
