"""Konvertiert den German Traffic Sign Detection Benchmark (GTSDB) ins YOLO-Format.

Eingabe:  Ordner FullIJCNN2013 mit 00000.ppm ... 00899.ppm und gt.txt
          gt.txt-Zeile: Dateiname;x_min;y_min;x_max;y_max;class_id
Ausgabe:  data/processed/gtsdb_yolo_<modus>/
              images/{train,val,test}/*.jpg
              labels/{train,val,test}/*.txt   (class x_center y_center width height, normiert)
              data.yaml
          results/metrics/datensatz_<modus>_*.csv/json (Statistik für Kapitel 4)

Aufteilung:
  - Bilder 00000-00599 = offizieller Trainingsteil, daraus werden val-ratio (Standard 15 %)
    zufällig (Seed 42) als Validierung abgetrennt.
  - Bilder 00600-00899 = offizieller Evaluationsteil, wird nur als Testmenge verwendet.
  Bilder ohne Verkehrszeichen bleiben als Hintergrundbilder enthalten (leere Label-Datei).

Aufruf:
  python src/convert_gtsdb_to_yolo.py --src data/raw/FullIJCNN2013 --mode super
  python src/convert_gtsdb_to_yolo.py --src data/raw/FullIJCNN2013 --mode full
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import yaml

from common import DATA_DIR, METRICS_DIR, names_for_mode, resolve_path, save_json, to_super_class

OFFICIAL_TRAIN_LAST_INDEX = 599


def read_ground_truth(gt_file: Path) -> dict[str, list[tuple[int, int, int, int, int]]]:
    boxes: dict[str, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    with gt_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 6:
                raise ValueError(f"gt.txt Zeile {line_no} unerwartet: {line}")
            name, x1, y1, x2, y2, cid = parts
            boxes[name].append((int(x1), int(y1), int(x2), int(y2), int(cid)))
    return boxes


def to_yolo_line(box, w: int, h: int, mode: str) -> tuple[str, int, int, int]:
    x1, y1, x2, y2, cid = box
    # GTSDB-Koordinaten sind inklusiv, daher +1 bei Breite und Höhe
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    xc, yc = x1 + bw / 2, y1 + bh / 2
    cls = to_super_class(cid) if mode == "super" else cid
    return f"{cls} {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}", cls, bw, bh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Ordner mit *.ppm und gt.txt")
    ap.add_argument("--mode", choices=["super", "full"], default="super",
                    help="super = 4 Obergruppen, full = 43 Einzelklassen")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--quality", type=int, default=95, help="JPEG-Qualität")
    args = ap.parse_args()

    src: Path = resolve_path(args.src)
    gt_file = src / "gt.txt"
    if not gt_file.exists():
        raise SystemExit(f"gt.txt nicht gefunden in {src}")
    out: Path = resolve_path(args.out) if args.out else DATA_DIR / "processed" / f"gtsdb_yolo_{args.mode}"
    names = names_for_mode(args.mode)

    gt = read_ground_truth(gt_file)
    images = sorted(src.glob("*.ppm"))
    if not images:
        raise SystemExit(f"Keine .ppm-Bilder in {src}")

    train_pool = [p for p in images if int(p.stem) <= OFFICIAL_TRAIN_LAST_INDEX]
    test_set = [p for p in images if int(p.stem) > OFFICIAL_TRAIN_LAST_INDEX]
    rng = random.Random(args.seed)
    shuffled = train_pool[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * args.val_ratio)
    val_set = set(shuffled[:n_val])
    split_of = {p: ("val" if p in val_set else "train") for p in train_pool}
    split_of.update({p: "test" for p in test_set})

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_counts: dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}
    img_counts = Counter()
    bg_counts = Counter()
    sizes: dict[str, list[int]] = {s: [] for s in ("train", "val", "test")}
    split_rows = []
    imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if args.format == "jpg" else []

    for p in images:
        split = split_of[p]
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit(f"Bild nicht lesbar: {p}")
        h, w = img.shape[:2]
        target = out / "images" / split / f"{p.stem}.{args.format}"
        cv2.imwrite(str(target), img, imwrite_params)

        lines = []
        for box in gt.get(p.name, []):
            line, cls, bw, bh = to_yolo_line(box, w, h, args.mode)
            lines.append(line)
            class_counts[split][cls] += 1
            sizes[split].append(max(bw, bh))
        (out / "labels" / split / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                              encoding="utf-8")
        img_counts[split] += 1
        if not lines:
            bg_counts[split] += 1
        split_rows.append((p.stem, split, len(lines)))

    data_yaml = {
        "path": str(out.resolve()),  # absolut, damit Ultralytics nicht im globalen datasets_dir sucht
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Statistik für den Bericht
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / f"datensatz_{args.mode}_klassen.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["klasse_id", "klasse", "train", "val", "test"])
        for i, n in enumerate(names):
            wr.writerow([i, n, class_counts["train"][i], class_counts["val"][i], class_counts["test"][i]])
    with (METRICS_DIR / f"datensatz_{args.mode}_aufteilung.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["bild", "split", "anzahl_zeichen"])
        wr.writerows(split_rows)

    summary = {
        "modus": args.mode,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "bilder": dict(img_counts),
        "hintergrundbilder_ohne_zeichen": dict(bg_counts),
        "zeichen": {s: sum(c.values()) for s, c in class_counts.items()},
        "zeichengroesse_px_max_seite": {
            s: ({"min": min(v), "median": median(v), "max": max(v)} if v else None) for s, v in sizes.items()
        },
        "klassen_ohne_beispiel_im_train": [names[i] for i in range(len(names)) if class_counts["train"][i] == 0],
        "klassen_ohne_beispiel_im_test": [names[i] for i in range(len(names)) if class_counts["test"][i] == 0],
        "data_yaml": str((out / "data.yaml").resolve()),
    }
    save_json(summary, METRICS_DIR / f"datensatz_{args.mode}_zusammenfassung.json")

    print(f"Fertig: {out}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
