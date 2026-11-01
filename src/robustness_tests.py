"""Robustheitstests unter simulierten Licht- und Wetterbedingungen.

Die Testbilder werden mit OpenCV/NumPy künstlich verändert (Dunkelheit, Nacht,
Überbelichtung, Nebel, Regen, Bewegungsunschärfe, Rauschen, starke JPEG-Kompression).
Die Geometrie bleibt unverändert, daher gelten die Original-Labels weiter.
Für jede Bedingung wird das Modell auf der veränderten Testmenge ausgewertet.

Ergebnis:
  results/metrics/robustheit.csv          mAP je Bedingung und Differenz zum Original
  results/figures/robustheit_beispiele.jpg Beispielbild in allen Bedingungen

Aufruf:
  python src/robustness_tests.py --weights runs/train/E0_baseline_n1280/weights/best.pt ^
         --data data/processed/gtsdb_yolo_super/data.yaml --imgsz 1280
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import DATA_DIR, FIGURES_DIR, METRICS_DIR, resolve_path
from evaluate_model import evaluate

RNG_SEED = 42


def _gamma(img, g):
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def dunkel(img, rng):
    return _gamma(img, 2.2)


def nacht(img, rng):
    out = _gamma(img, 2.8).astype(np.float32)
    out[..., 0] *= 1.10  # leichter Blaustich (BGR)
    out += rng.normal(0, 8, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def ueberbelichtung(img, rng):
    return cv2.convertScaleAbs(img, alpha=1.4, beta=60)


def nebel(img, rng):
    h, w = img.shape[:2]
    # Nebel nimmt zum Horizont (oben) hin zu
    grad = np.linspace(0.65, 0.35, h, dtype=np.float32)[:, None, None]
    haze = np.full_like(img, 225, dtype=np.float32)
    out = img.astype(np.float32) * (1 - grad) + haze * grad
    return np.clip(out, 0, 255).astype(np.uint8)


def regen(img, rng):
    h, w = img.shape[:2]
    streaks = np.zeros((h, w), dtype=np.uint8)
    n = int(h * w / 600)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    lens = rng.integers(10, 25, n)
    for x, y, ln in zip(xs, ys, lens):
        cv2.line(streaks, (int(x), int(y)), (int(x) + 3, int(y) + int(ln)), 200, 1)
    streaks = cv2.blur(streaks, (3, 3))
    out = cv2.addWeighted(img, 0.85, cv2.cvtColor(streaks, cv2.COLOR_GRAY2BGR), 0.35, 0)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return _gamma(out, 1.2)


def bewegungsunschaerfe(img, rng):
    k = 9
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def rauschen(img, rng):
    out = img.astype(np.float32) + rng.normal(0, 20, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_kompression(img, rng):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


CONDITIONS = {
    "dunkel": dunkel,
    "nacht": nacht,
    "ueberbelichtung": ueberbelichtung,
    "nebel": nebel,
    "regen": regen,
    "bewegungsunschaerfe": bewegungsunschaerfe,
    "rauschen": rauschen,
    "jpeg_kompression": jpeg_kompression,
}


def build_condition_set(src_root: Path, names: dict, cond: str, fn, out_root: Path) -> Path:
    """Erzeugt veränderte Kopie der Testmenge und eine passende data.yaml."""
    rng = np.random.default_rng(RNG_SEED)
    dst = out_root / cond
    img_dst, lbl_dst = dst / "images" / "test", dst / "labels" / "test"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for p in sorted((src_root / "images" / "test").iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(img_dst / p.name), fn(img, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl = src_root / "labels" / "test" / f"{p.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, lbl_dst / lbl.name)
    y = {"path": str(dst.resolve()), "train": "images/test", "val": "images/test", "test": "images/test",
         "names": names}
    (dst / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return dst / "data.yaml"


def example_grid(src_root: Path, out_file: Path, image_name: str | None = None) -> None:
    imgs = sorted((src_root / "images" / "test").iterdir())
    labels_dir = src_root / "labels" / "test"
    pick = None
    if image_name:
        pick = next((p for p in imgs if p.stem == image_name), None)
    if pick is None:  # erstes Bild mit mindestens einem Zeichen
        pick = next((p for p in imgs if (labels_dir / f"{p.stem}.txt").stat().st_size > 0), imgs[0])
    base = cv2.imread(str(pick))
    rng = np.random.default_rng(RNG_SEED)
    tiles = [("original", base)] + [(c, f(base, rng)) for c, f in CONDITIONS.items()]
    th, tw = 270, 460
    cells = []
    for title, im in tiles:
        t = cv2.resize(im, (tw, th))
        cv2.rectangle(t, (0, 0), (tw, 28), (0, 0, 0), -1)
        cv2.putText(t, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(t)
    while len(cells) % 3:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="data.yaml des Originaldatensatzes")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--name", default="robust")
    ap.add_argument("--beispielbild", default=None, help="Bildname ohne Endung für die Beispielgrafik")
    args = ap.parse_args()

    args.data, args.weights = str(resolve_path(args.data)), str(resolve_path(args.weights))
    meta = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    src_root = Path(meta["path"])
    out_root = DATA_DIR / "processed" / "robustheit" / src_root.name

    example_grid(src_root, FIGURES_DIR / "robustheit_beispiele.jpg", args.beispielbild)

    rows = []
    base = evaluate(args.weights, args.data, args.imgsz, "test", args.device, f"{args.name}_original")
    rows.append(("original", base))
    for cond in args.conditions:
        print(f"\n=== Bedingung: {cond}")
        y = build_condition_set(src_root, meta["names"], cond, CONDITIONS[cond], out_root)
        rows.append((cond, evaluate(args.weights, y, args.imgsz, "test", args.device, f"{args.name}_{cond}")))

    out = METRICS_DIR / f"robustheit_{args.name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["bedingung", "precision", "recall", "mAP50", "mAP50_95", "delta_mAP50", "delta_mAP50_95"])
        for cond, e in rows:
            wr.writerow([cond, f"{e['precision']:.4f}", f"{e['recall']:.4f}", f"{e['mAP50']:.4f}",
                         f"{e['mAP50_95']:.4f}", f"{e['mAP50'] - base['mAP50']:+.4f}",
                         f"{e['mAP50_95'] - base['mAP50_95']:+.4f}"])
    print(f"\nErgebnis: {out}")
    print(f"Beispielgrafik: {FIGURES_DIR / 'robustheit_beispiele.jpg'}")


if __name__ == "__main__":
    main()
