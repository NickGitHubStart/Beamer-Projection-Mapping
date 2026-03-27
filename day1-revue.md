# Day 1 - Revue

## Was heute gebaut wurde

### Pipeline-Ueberblick

**Schritt 1 - Marker-Repraesentation:**
Ein Skript generiert vier ArUco-Marker und zeigt sie an den Ecken des Beamer-Bildschirms an.

**Schritt 2 - Homographie berechnen:**
Ein zweites Skript (parallel in eigenem Terminal gestartet) sieht die vier Eck-Marker ueber die Kamera und berechnet daraus die Transformation zwischen Kamera-Perspektive und Beamer-Perspektive. Das Ergebnis wird in einer Pickle-Datei gespeichert und kann jederzeit neu geladen werden.

**Schritt 3 - ArUco Tracking + Projektion:**
Ein drittes Skript laedt die Pickle-Datei, nimmt das bereits gewarpte (umgewandelte) Kamerabild und erkennt darin ArUco-Marker per OpenCV. Die erkannten Pixel-Koordinaten werden direkt auf ein schwarzes Bild uebertragen - das schwarze Bild ist die Beamer-Flaeche. Schwarz = kein Licht, alles Gezeichnete wird projiziert. Da die Koordinaten aus dem bereits gewarpten Bild stammen, stimmen sie 1:1 mit den Beamer-Pixeln ueberein.

**Schritt 4 - YOLO Objekt-Erkennung:**
Gleiches Prinzip, aber statt ArUco-Markern laeuft ein YOLO-Modell auf dem gewarpten Kamerabild. YOLO gibt Bounding-Boxes zurueck, die direkt auf die Beamer-Flaeche gezeichnet werden - als AR-Brackets an den Ecken der erkannten Objekte.

**Schritt 5 - YOLO Segmentierung:**
Statt Bounding-Boxes liefert ein YOLO-Segmentierungsmodell pixelgenaue Masken der erkannten Objekte (z.B. Hand, Flasche). Diese Masken werden auf die Beamer-Flaeche uebertragen - praeziser als eine Box, direkt am Umriss des Objekts.

---

## Was gut funktioniert hat

- Alle vier Skripte laufen
- Nach Neukalibrierung funktioniert die Projektion sehr gut
- Koordinaten-Uebertragung von Kamera auf Beamer stimmt

## Limitierung: Tiefe (Depth)

Die aktuelle Pipeline arbeitet rein in 2D - eine Kamera-Ebene wird auf eine Beamer-Ebene gemappt. Je nach Kamerawinkel werden Objekte leicht verzerrt erkannt, weil keine Tiefeninformation vorhanden ist. Das fuehrt dazu, dass die Projektion nicht immer exakt auf dem physischen Objekt sitzt, sondern leicht versetzt wirkt.
