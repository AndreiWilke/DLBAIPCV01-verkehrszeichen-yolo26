"""Gleiche Streckenpunkte aus Tag-, Dämmerungs- und Nachtfahrt für die CVAT-Annotation.

Die Zuordnung Tag -> Dämmerung/Nacht kommt aus dem bildbasierten Streckenabgleich
(results/metrics/route/zeitabgleich_strecke1.csv). Ausgewählt werden
  a) alle Schildstellen der Tagreferenz (Bild mit der höchsten Konfidenz je Stelle) und
  b) zufällige Streckenpunkte ohne Schildstelle (mind. 3 s Abstand, Seed 42),
jeweils in allen drei Fahrten. So enthält die Testmenge an jedem Punkt dieselbe Szene
bei drei Lichtverhältnissen, und Frames ohne Erkennung decken übersehene Zeichen auf.

Vorannotation: Vereinigung der Vorschläge von E3, E9 und E11 (conf 0,25, NMS IoU 0,5), damit
keines der beiden verglichenen Modelle bevorzugt wird. Jede Box muss in CVAT geprüft werden;
übersehene Zeichen ergänzen.

Aufruf (im Ordner 02_Code):
  python src/extract_strecke.py --zufall 30
"""

from __future__ import annotations

import argparse
import csv
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import DATA_DIR, ROOT, SUPER_CLASSES, resolve_path

FAHRTEN = [("IMG_9733", "tag_s", "tag"), ("IMG_9726", "daemmerung_s", "daemmerung"), ("IMG_9730", "nacht_s", "nacht")]


def lade_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def zahl(s: str) -> float:
    return float(str(s).replace(",", "."))


def vereinige(boxes: list[tuple], iou: float = 0.5) -> list[tuple]:
    """boxes: (x1, y1, x2, y2, conf, cls); einfache NMS über beide Modelle hinweg."""
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        ok = True
        for k in keep:
            xx1, yy1 = max(b[0], k[0]), max(b[1], k[1])
            xx2, yy2 = min(b[2], k[2]), min(b[3], k[3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            uni = (b[2] - b[0]) * (b[3] - b[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if uni > 0 and inter / uni >= iou:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zuordnung", default=str(ROOT / "results/metrics/route/zeitabgleich_strecke1.csv"))
    ap.add_argument("--pruefliste", default=str(ROOT / "results/metrics/route/pruefliste_strecke1.csv"))
    ap.add_argument("--zufall", type=int, default=30, help="Anzahl zufälliger Streckenpunkte ohne Schildstelle")
    ap.add_argument("--gewichte", nargs="+", default=["runs/train/E3_s1280/weights/best.pt",
                                                      "runs/train/E9_s1280_robust/weights/best.pt",
                                                      "runs/train/E11_s1280_robust_farbe/weights/best.pt"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    zu = lade_csv(Path(args.zuordnung))
    tag = np.array([zahl(r["tag_s"]) for r in zu])
    spalten = {c: np.array([zahl(r[c]) for r in zu]) for _, c, _ in FAHRTEN}
    t0, t1 = tag.min(), tag.max()

    ref = sorted(zahl(r["sek"]) for r in lade_csv(Path(args.pruefliste)) if r["art"].startswith("ref"))
    rng = random.Random(42)
    zuf = []
    while len(zuf) < args.zufall:
        t = round(rng.uniform(t0 + 2, t1 - 2), 1)
        if all(abs(t - r) >= 3 for r in ref) and all(abs(t - z) >= 3 for z in zuf):
            zuf.append(t)
    punkte = [(f"S{i:02d}", t, "schildstelle") for i, t in enumerate(ref)] + \
             [(f"Z{i:02d}", t, "zufall") for i, t in enumerate(sorted(zuf))]

    out = resolve_path(args.out) if args.out else DATA_DIR / "eigene_annotation" / "strecke1"
    img_dir, lbl_dir = out / "images" / "test", out / "labels" / "test"
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    modelle = [YOLO(str(resolve_path(w))) for w in args.gewichte]

    manifest, dateien, labels = [], [], {}
    for video, spalte, name in FAHRTEN:
        cap = cv2.VideoCapture(str(resolve_path(f"videos/input/{video}.MOV")))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for pid, t_tag, art in punkte:
            t = float(np.interp(t_tag, tag, spalten[spalte]))
            fr = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                print("nicht lesbar:", video, fr)
                continue
            f = img_dir / f"{pid}_{name}_{video}_{fr:06d}.jpg"
            cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = img.shape[:2]
            boxes = []
            for m in modelle:
                r = m.predict(img, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
                for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    boxes.append((x1, y1, x2, y2, c, int(k)))
            lines = [f"{k} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                     for x1, y1, x2, y2, c, k in vereinige(boxes)]
            labels[f] = lines
            dateien.append(f)
            manifest.append([f.name, pid, art, name, video, round(t_tag, 1), round(t, 2), fr, len(lines)])
        cap.release()
        print(f"{name}: {len(punkte)} Punkte")

    for f in dateien:
        (lbl_dir / f"{f.stem}.txt").write_text("\n".join(labels[f]) + ("\n" if labels[f] else ""), encoding="utf-8")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh, delimiter=";")
        wr.writerow(["datei", "punkt", "art", "bedingung", "video", "tag_s", "video_s", "frame", "vorschlaege"])
        wr.writerows(manifest)
    y = {"path": str(out.resolve()), "train": "images/test", "val": "images/test", "test": "images/test",
         "names": {i: n for i, n in enumerate(SUPER_CLASSES)}}
    (out / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")

    with zipfile.ZipFile(out / "bilder.zip", "w", zipfile.ZIP_STORED) as z:
        for f in dateien:
            z.write(f, f.name)
    with zipfile.ZipFile(out / "vorannotation_yolo11.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(SUPER_CLASSES) + "\n")
        z.writestr("obj.data", f"classes = {len(SUPER_CLASSES)}\nnames = data/obj.names\ntrain = data/train.txt\n")
        z.writestr("train.txt", "\n".join(f"data/obj_train_data/{f.name}" for f in dateien) + "\n")
        for f in dateien:
            z.writestr(f"obj_train_data/{f.stem}.txt", "\n".join(labels[f]) + ("\n" if labels[f] else ""))
    n_box = sum(len(v) for v in labels.values())
    print(f"Fertig: {len(dateien)} Bilder, {n_box} Vorschläge -> {out}")


if __name__ == "__main__":
    main()
