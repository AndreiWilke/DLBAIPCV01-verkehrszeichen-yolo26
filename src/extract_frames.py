"""Extrahiert Einzelbilder aus einem eigenen Fahrvideo und bereitet die Annotation vor.

Die Aufgabenstellung verlangt die "Markierung und Annotation relevanter Frames".
Um den Aufwand zu senken, schlägt das trainierte Modell die Boxen vor (Vorannotation);
der Mensch prüft und korrigiert nur noch. Ablauf:
  1. Frames extrahieren und vorannotieren (dieses Skript, Option --vorannotieren).
  2. In CVAT eine Aufgabe mit den Labels prohibitory, danger, mandatory, other anlegen,
     bilder.zip hochladen, danach "Upload annotations" im Format "YOLO 1.1" mit vorannotation_yolo11.zip.
  3. Jede Box prüfen: falsche löschen, Klasse korrigieren, Rahmen anpassen und vor allem
     ÜBERSEHENE Zeichen ergänzen. Ohne diese Prüfung wäre die Testmenge wertlos, weil das
     Modell dann an seinen eigenen Vorschlägen gemessen würde.
  4. Export im Format "YOLO 1.1" und mit import_cvat.py zurückholen, dann evaluate_model.py.

Auswahl (--auswahl N): höchstens N Frames, davon etwa zwei Drittel mit Modellerkennung und ein Drittel
zufällig OHNE Erkennung (Seed 42). Die Frames ohne Erkennung sind nötig, um übersehene Zeichen zu finden;
sonst wäre die Stichprobe zugunsten des Modells verzerrt.

Aufruf:
  python src/extract_frames.py --video videos/input/fahrt_tag_stadt.mp4 --every-sec 1 ^
         --vorannotieren runs/train/E3_s1280/weights/best.pt --auswahl 150
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml

from common import DATA_DIR, SUPER_CLASSES, resolve_path


def extract(video: Path, img_dir: Path, every_sec: float, start_sec: float, end_sec: float | None) -> list[Path]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Video nicht lesbar: {video} (HEVC? dann als H.264 aufnehmen oder umwandeln)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_sec))
    start, end = int(start_sec * fps), (int(end_sec * fps) if end_sec else None)
    files, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (end is not None and i > end):
            break
        if i >= start and (i - start) % step == 0:
            f = img_dir / f"{video.stem}_{i:06d}.jpg"
            cv2.imwrite(str(f), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            files.append(f)
        i += 1
    cap.release()
    print(f"{len(files)} Frames extrahiert (alle {step} Frames bei {fps:.1f} FPS)")
    return files


def predict_labels(files: list[Path], weights: str, conf: float, imgsz: int, device: str | None) -> dict[Path, list[str]]:
    from ultralytics import YOLO

    model = YOLO(str(resolve_path(weights)))
    kw = dict(imgsz=imgsz, conf=conf, verbose=False)
    if device is not None:
        kw["device"] = device
    out = {}
    for f in files:
        r = model.predict(str(f), **kw)[0]
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for (x, y, w, h), c in zip(r.boxes.xywhn.cpu().tolist(), r.boxes.cls.cpu().tolist())]
        out[f] = lines
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--every-sec", type=float, default=1.0)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vorannotieren", default=None, help="Gewichte für Vorschläge, z. B. runs/train/E3_s1280/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--auswahl", type=int, default=None, help="höchstens N Frames behalten (2/3 mit, 1/3 ohne Erkennung)")
    args = ap.parse_args()

    video = resolve_path(args.video)
    out = resolve_path(args.out) if args.out else DATA_DIR / "eigene_annotation" / video.stem
    img_dir, lbl_dir = out / "images" / "test", out / "labels" / "test"
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
    files = extract(video, img_dir, args.every_sec, args.start_sec, args.end_sec)

    labels = {f: [] for f in files}
    if args.vorannotieren:
        labels = predict_labels(files, args.vorannotieren, args.conf, args.imgsz, args.device)
        mit = [f for f in files if labels[f]]
        print(f"Vorannotation: {len(mit)} von {len(files)} Frames mit mindestens einer Erkennung")

    if args.auswahl and len(files) > args.auswahl:
        rng = random.Random(42)
        mit = [f for f in files if labels[f]]
        ohne = [f for f in files if not labels[f]]
        n_ohne = min(len(ohne), args.auswahl - round(args.auswahl * 2 / 3))
        n_mit = min(len(mit), args.auswahl - n_ohne)
        n_ohne = min(len(ohne), args.auswahl - n_mit)  # falls zu wenige Frames mit Erkennung
        keep = set(rng.sample(mit, n_mit) + rng.sample(ohne, n_ohne))
        verworfen = out / "_nicht_ausgewaehlt"
        verworfen.mkdir(exist_ok=True)
        for f in files:
            if f not in keep:
                shutil.move(str(f), verworfen / f.name)
        files = sorted(keep)
        print(f"Auswahl: {n_mit} Frames mit und {n_ohne} ohne Erkennung (Rest in {verworfen.name})")

    for f in files:
        (lbl_dir / f"{f.stem}.txt").write_text("\n".join(labels[f]) + ("\n" if labels[f] else ""), encoding="utf-8")

    y = {"path": str(out.resolve()), "train": "images/test", "val": "images/test", "test": "images/test",
         "names": {i: n for i, n in enumerate(SUPER_CLASSES)}}
    (out / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")

    # Pakete für CVAT: Bilder für die Aufgabe, Vorschläge im Format YOLO 1.1
    with zipfile.ZipFile(out / "bilder.zip", "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, f.name)
    with zipfile.ZipFile(out / "vorannotation_yolo11.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(SUPER_CLASSES) + "\n")
        z.writestr("obj.data", f"classes = {len(SUPER_CLASSES)}\nnames = data/obj.names\ntrain = data/train.txt\n")
        z.writestr("train.txt", "\n".join(f"data/obj_train_data/{f.name}" for f in files) + "\n")
        for f in files:
            z.writestr(f"obj_train_data/{f.stem}.txt", "\n".join(labels[f]) + ("\n" if labels[f] else ""))
    print(f"Fertig: {out}")
    print("  bilder.zip               -> in CVAT beim Anlegen der Aufgabe hochladen")
    print("  vorannotation_yolo11.zip -> danach 'Upload annotations', Format YOLO 1.1")


if __name__ == "__main__":
    main()
