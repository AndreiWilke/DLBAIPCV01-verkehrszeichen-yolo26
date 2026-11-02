"""Erkennungsverlauf je Fahrt: alle 10 Bilder die Erkennungen eines Modells (Original vom 05.10.2026, im
Arbeitsverzeichnis ausgefuehrt). Ergebnis: analyse/<video>/det_<modell>.csv, Kopien in results/metrics/route/det_*.csv.
Aufruf: python det_verlauf.py IMG_9733 E11 runs/train/E11_s1280_robust_farbe/weights/best.pt
"""
import cv2, sys, csv, os
from ultralytics import YOLO
name, tag, wpath = sys.argv[1], sys.argv[2], sys.argv[3]
m = YOLO(wpath); c = cv2.VideoCapture(rf"videos\input\{name}.MOV"); i = 0
od = rf"analyse\{name}"; os.makedirs(od, exist_ok=True)
w = csv.writer(open(rf"{od}\det_{tag}.csv", "w", newline=""), delimiter=";"); w.writerow(["frame","sek","n","max_conf","klassen"])
while c.grab():
    if i % 10 == 0:
        r = m.predict(c.retrieve()[1], imgsz=1280, conf=0.25, device=0, verbose=False)[0]
        cf = r.boxes.conf.tolist(); cl = [int(x) for x in r.boxes.cls.tolist()]
        w.writerow([i, round(i/30,2), len(cf), round(max(cf),3) if cf else 0, " ".join(f"{a}:{b:.2f}" for a,b in zip(cl,cf))])
    i += 1
print(name, tag, "det fertig", i, flush=True)
