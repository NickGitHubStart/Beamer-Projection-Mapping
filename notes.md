# Notizen

**Frage:** Wo KI / Computer Vision an den Kamerastream haengen und wo der Beamer-Inhalt entsteht?

**Antwort:**
- **CV / Erkennung:** Im `while True`-Loop, **nach** `dewarped_img` (oder anderem gewaehltem Eingangsbild), **vor** `cv2.imshow("Beamer_Window", BeamerImage)`.
- **Beamer "bearbeiten":** Alles Sichtbare kommt aus **`BeamerImage`** (`np.zeros`, Zeichnungen, Overlays) - KI-Ergebnisse dort einzeichnen bzw. daraus steuern.

**Frage:** Stimmen die Pixel-Koordinaten von Erkennung und Beamer ueberein?

**Antwort:** Ja. YOLO (und jede andere Erkennung) laeuft auf `dewarped_img` — dem bereits gewarpten Bild in Beamer-Koordinaten. Die Koordinaten die YOLO zurueckgibt sind deshalb direkt identisch mit den Beamer-Pixeln. Pixel (300, 200) bei YOLO = Pixel (300, 200) auf dem Beamer. Kein Umrechnen noetig.

**Frage:** Wo und wie werden die ArUco-Marker erkannt — OpenCV oder manuell fuer die Speziellen?

**Antwort:** Erkennung laeuft ueber **OpenCV** (`cv2.aruco`): vorgegebenes Woerterbuch z.B. `DICT_5X5_50`, dann `detectMarkers` auf dem (gewarpten) Graustufenbild. Welche Marker *bedeuten* was (Ecken 0–3, Overlay bei ID 5 usw.), steht **im Code** als feste IDs / Logik — keine separate Trainings-KI, nur Dictionary + Parameter passend zu den gedruckten/generierten Markern halten.

**Frage:** Ist das Setup nicht nur **2D** (eine gewarpte Bildebene), und wenn ich auf ein **3D-Objekt** projizieren will, stimmt **Tiefe** nicht — der Beamer projiziert doch nur auf eine bestimmte Flaeche, nicht tiefenrichtig / angepasst? Oder reicht **Pixel** theoretisch auch fuer 3D?

**Antwort:** Richtig: **Homographie + warp** = Abbildung **Kameraebene → eine Referenz-Ebene** (deine „Wand“-Flaeche). Es gibt **kein** eingebautes **Tiefenmodell** der Szene: alles, was „drauf“ gezeichnet wird, liegt faktisch in dieser 2D-Projektion. Auf **krumme 3D-Geometrie** passt das nur **optisch da**, wo die reale Oberflaeche der gedachten Ebene entspricht (z.B. flache Wand). Auf **Kubus, Kopf, Statue** wirkt es **verzerrt / schief**, eben weil der **Lichtweg des Beamers** und die **Sicht der Kamera** nicht automatisch auf jede Tiefe / jeden Winkel abgestimmt sind. **Pixel allein** loesen das nicht: du kannst weiterhin 2D-Inhalt an **(u,v)** legen, aber **korrektes 3D-Mapping** braechte z.B. **Kalibrierung Kamera+Beamer**, ein **3D-Modell / Mesh** der Oberflaeche und **Perspektiven-/Projektions-Rechnung** (oder strukturiertes Licht / stereo) — deutlich mehr als der aktuelle Pipeline-Schritt.

**Stand (Erkennung + Projektion):** Aktuell u.a. **YOLO** (`03_yolo_projection.py`) und **ArUco** (`02_aruco_plus_transform_on_live_stream.py`): Es werden **Bounding-Boxes** bzw. **Marker-Positionen** genutzt; der Beamer legt Inhalt **in diese Boxen** oder **an die Marker** — jeweils so, wie es im Code eingestellt ist.

**Ausblick (genauere Objekt-Umrisse):** Ziel waere, Inhalte **nicht** als Rechteck um ein Objekt zu projizieren, sondern **am eigentlichen Umriss** — z.B. nur **Arm**, **Person**, **Flasche** ohne den Bereich ausserhalb der Box. Dafuer reicht ein reiner **Detektor + Box** nicht; man braechte z.B. **Instanz-Segmentierung** (Masken pro Objekt), **Pose/Keypoints** (nur Koerperteil) oder Modelle wie **SAM** — andere Architekturen / Post-Processing, um eine **Pixel-Maske** statt einer Box auf `dewarped_img` / `BeamerImage` zu legen.

---

# Pipeline - Schritt fuer Schritt nachmachen

**Kamera-Index** ist **nicht** in allen Dateien gleich — vor dem Start pruefen und anpassen:
- `03_aruco_tracking/` (`02_aruco_plus_transform_on_live_stream.py`, `03_yolo_projection.py`), `02_homogrphic_transform/` (`02_calc_pose_trans.py`, `03_apply_transform_on_live_stream.py`): typ. **`1`**
- `01_intrinsic_calibration/00_*` und `02_record_images.py`: dort z.T. **`2`**
- `src/run_model.py`: aktuell **`3`**
Bei falscher Quelle **0 / 1 / 2** (bzw. konstante im jeweiligen Skript) durchprobieren.
ChArUco-Blatt ggf. ausgedruckt; **Eck-Vorlage** kommt aus Schritt 2a (oder gedruckt statt Beamer).

**Aufbau vor Homographie (2a → 2b):** Laptop **erweiterte Anzeige** (zweiter Bildschirm = Beamer). Kamera so montieren, dass sie auf die **Projektionsflaeche / den Beamer-Bildschirm** schaut. Auf den Beamer legst du die **01-Marker-Vorlage** (`BeamerImage.png` aus `01_marker_representation.py`, Vollbild) — die **vier ArUco-Ecken** muessen im Kamerabild klar sichtbar sein. Erst dann **`02_calc_pose_trans.py`** (Kalibrierung der Homographie); ohne diese Vorlage auf der Beamerflaeche erkennt das Skript die Eck-Marker nicht zuverlaessig.

# segmenting models for cooler beamer projection
ah mach das noch in die notes. es gibt sogenannte segmentierungsmodelle, die ein segment quasi erkennen (statt einer box). beispeilsweise SAM, aber das braucht einen prompt nach das es suchen soll, quasi wie yolo objekt erkennen, aber statt box, macht es dann einen punkt davon und legt einemaske bis zu allen anderen übereinstimmenden darüber
bzw. rechteck wie yolo und sam verfeinert den rand. 



```bash
# --- 1. Kamera kalibrieren --- SKIP: bool_load_cam_calib=False gesetzt, fuer Top-Down nicht noetig ---

# --- 2a. Eck-Marker-Vorlage fuer den Beamer (4 ArUco, schwarz) ---
cd ../02_homogrphic_transform
# In 01_marker_representation.py Zeile bool_fullscreen:
#   False = normales Fenster (Vorschau / Fenster auf den Beamer-Monitor ziehen)
#   True  = sofort Vollbild (direkt komplette Beamerflaeche)
python 01_marker_representation.py
# -> schreibt BeamerImage.png hier; Q schliesst. Ohne Beamer: PNG z.B. in Viewer Vollbild auf Projektor.
# Alternative: dieselben Marker gedruckt kleben — ID0=oben-links, ID1=oben-rechts, ID2=unten-links, ID3=unten-rechts

# --- 2b. Homographie berechnen ---
# Vorher: Beamer mit BeamerImage.png (01) bespielt, Kamera darauf ausgerichtet — siehe Absatz "Aufbau vor Homographie".
python 02_calc_pose_trans.py           # Marker sichtbar (Beamer oder Druck), ESC wenn fertig

# --- 3. Projektion starten ---
cd ../03_aruco_tracking
# python 03_yolo_projection.py           # YOLO Detection -> Eck-Brackets auf Beamer_Window
# python 04_yolo_seg_projection.py       # YOLO-Seg -> Masken (trainiertes YOLO-Seg)
# python 05_sam_projection.py            # YOLO Boxen als Prompt + SAM -> scharfe Masken (Ultralytics)
python 02_aruco_plus_transform_on_live_stream.py
# Wichtig: nur "Beamer_Window" braucht man fuer die Ausgabe (auf Beamer / Vollbild).
# "output" = Kontrollansicht (gewarpter Kamerastream mit Markierungen) — kann man ignorieren oder wegstellen.
# Beamer_Window = die "Oberflaeche": was die Kamera in der Projektionsflaeche sieht, wird dort als Projektion abgebildet;
#   ausserhalb des Kamerablicks (wenn du nur den Laptop-Bildschirm anschaust) ist das im Wesentlichen Schwarz plus die Projektionen.
# ESC = beenden
```
