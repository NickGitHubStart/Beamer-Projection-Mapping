"""
SAM (Segment Anything) + Homographie -> Beamer_Window.

Ultralytics-Pipeline: **YOLO erkennt Objekte (Boxen + Klasse)**, **SAM** bekommt diese
Boxen als **Prompt** und liefert **scharfe Masken** am Objektrand (statt YOLO-Seg direkt).

Voraussetzungen:
  pip install -U ultralytics
Erster Start laedt Modelle (yolov8n.pt, sam_b.pt — einige hundert MB).

GPU dringend empfohlen; auf CPU wird es langsam.
"""
import numpy as np
import cv2
import os
import pickle

from ultralytics import SAM, YOLO


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CAMERA_INDEX = 1

relative_homographic_tranform_path = '../02_homogrphic_transform/homographic_tranform.pckl'

bool_fullscreen = False
DET_CONF = 0.25

# Detection: klein/schnell — nur fuer Box-Prompts
detector = YOLO("yolov8n.pt")
# SAM: Box-Prompt -> Maske (sam_s kleiner/schneller, sam_b mittel)
segmenter = SAM("sam_b.pt")

COLORS = [
    (0, 255, 128),
    (0, 200, 255),
    (255, 80, 80),
    (255, 200, 0),
    (180, 0, 255),
]


def warp_image(image, H, output_width, output_height):
    return cv2.warpPerspective(image, H, (output_width, output_height))


if not os.path.exists(relative_homographic_tranform_path):
    print("Homographie fehlt. Erst 02_calc_pose_trans.py ausfuehren.")
    exit()

with open(relative_homographic_tranform_path, 'rb') as f:
    H, output_width, output_height = pickle.load(f)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise IOError(f"Kamera {CAMERA_INDEX} nicht verfuegbar")

cv2.namedWindow('Beamer_Window', cv2.WINDOW_NORMAL)
if bool_fullscreen:
    cv2.setWindowProperty('Beamer_Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("ESC = beenden — YOLO (Boxen) + SAM (Masken) auf gewarpter Ebene -> Beamer_Window")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dewarped_img = warp_image(frame, H, output_width, output_height)
    BeamerImage = np.zeros(dewarped_img.shape, np.uint8)

    det = detector(dewarped_img, conf=DET_CONF, verbose=False)[0]
    if det.boxes is None or len(det.boxes) == 0:
        cv2.imshow("Beamer_Window", BeamerImage)
        cv2.imshow("output", dewarped_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    xyxy = det.boxes.xyxy.cpu().numpy()
    cls = det.boxes.cls.cpu().numpy().astype(int)
    bboxes = xyxy.tolist()

    sam_out = segmenter.predict(
        source=dewarped_img,
        bboxes=bboxes,
        verbose=False,
    )
    r0 = sam_out[0]

    if r0.masks is not None and len(r0.masks.xy) > 0:
        for i, polygon_xy in enumerate(r0.masks.xy):
            cls_id = int(cls[i]) if i < len(cls) else 0
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
