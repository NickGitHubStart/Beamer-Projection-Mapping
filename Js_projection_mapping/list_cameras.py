"""
Probiert OpenCV-Videoindizes durch und zeigt, welche Geraete oeffnen und Frames liefern.
Ausfuehren:  python list_cameras.py   (aus Js_projection_mapping/)
Nach jedem OK kommt automatisch der naechste Index — es wird die ganze Liste durchlaufen.
"""
import sys
import time

import cv2

MAX_INDEX = 8  # testet Indizes 0 .. MAX_INDEX-1 (bei Bedarf erhoehen)

PREVIEW_SEC = 1.5


def main():
    print(
        "Kamera-Test (backend: default). Pro Kamera ca. "
        f"{PREVIEW_SEC:.1f} s Vorschau. Laueft immer alle Indizes durch — "
        "waehrend einer Vorschau **Q** = ganz abbrechen.\n"
    )
    try:
        for i in range(MAX_INDEX):
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                print(f"  [{i}] nicht verfuegbar (oeffnet nicht)")
                continue
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                print(f"  [{i}] geoeffnet, aber kein Frame (evt. belegt oder Fake-Device)")
                continue
            h, w = frame.shape[:2]
            print(f"  [{i}] OK — Aufloesung ca. {w}x{h} (diesen Index in den Skripten eintragen)")
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Camera index = {i}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            t_end = time.monotonic() + PREVIEW_SEC
            user_quit = False
            while time.monotonic() < t_end:
                cv2.imshow("list_cameras — aktueller Index", preview)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), ord("Q")):
                    user_quit = True
                    break
            cv2.destroyAllWindows()
            if user_quit:
                print("\n  (Abbruch durch Q)")
                break
    finally:
        cv2.destroyAllWindows()
    print("\nFertig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
