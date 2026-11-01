"""Misst die Inferenzgeschwindigkeit (Latenz, FPS) für Modell- und Bildgrößenvarianten.

Grundlage für die Tabelle "Genauigkeit versus Echtzeitfähigkeit" (Forschungsfrage 3).
Gemessen wird nach einigen Aufwärmdurchläufen auf echten Testbildern, getrennt nach
Vorverarbeitung, Inferenz und Nachverarbeitung sowie Ende-zu-Ende.

Aufruf:
  python src/benchmark_speed.py --weights runs/train/E0_baseline_n1280/weights/best.pt ^
         runs/train/E3_s1280/weights/best.pt --imgsz 640 960 1280 --devices 0 cpu ^
         --images data/processed/gtsdb_yolo_super/images/test
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2

from common import METRICS_DIR, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1280])
    ap.add_argument("--devices", nargs="+", default=["0"])
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    from ultralytics import YOLO

    args.weights = [str(resolve_path(w)) for w in args.weights]
    files = sorted(resolve_path(args.images).glob("*.*"))[: args.n]
    frames = [cv2.imread(str(p)) for p in files]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit("Keine Bilder gefunden")

    rows = []
    for wpath in args.weights:
        model = YOLO(wpath)
        for dev in args.devices:
            for sz in args.imgsz:
                kw = dict(imgsz=sz, device=dev, verbose=False, conf=0.25)
                for i in range(args.warmup):
                    model.predict(frames[i % len(frames)], **kw)
                pre, inf, post, e2e = [], [], [], []
                for f in frames:
                    t0 = time.perf_counter()
                    r = model.predict(f, **kw)[0]
                    _ = r.boxes.xyxy.cpu()
                    e2e.append((time.perf_counter() - t0) * 1000)
                    pre.append(r.speed["preprocess"])
                    inf.append(r.speed["inference"])
                    post.append(r.speed["postprocess"])
                e2e_sorted = sorted(e2e)
                row = {
                    "gewichte": Path(wpath).parts[-3] if len(Path(wpath).parts) >= 3 else wpath,
                    "device": dev, "imgsz": sz, "bilder": len(frames),
                    "pre_ms": round(statistics.mean(pre), 2), "inferenz_ms": round(statistics.mean(inf), 2),
                    "post_ms": round(statistics.mean(post), 2), "e2e_ms_mittel": round(statistics.mean(e2e), 2),
                    "e2e_ms_p95": round(e2e_sorted[int(0.95 * (len(e2e_sorted) - 1))], 2),
                    "fps": round(1000 / statistics.mean(e2e), 1),
                }
                rows.append(row)
                print(row)

    out = METRICS_DIR / "geschwindigkeit.csv"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nGespeichert: {out}")


if __name__ == "__main__":
    main()
