"""Evaluiert ein trainiertes Modell auf einer Datenaufteilung (Standard: test).

Speichert Gesamtmetriken, Metriken je Klasse und die Ultralytics-Plots
(Confusion Matrix, PR-Kurve, Beispielbilder) für Kapitel 6 des Berichts.

Aufruf:
  python src/evaluate_model.py --weights runs/train/baseline_n1280/weights/best.pt ^
         --data data/processed/gtsdb_yolo_super/data.yaml --imgsz 1280 --name baseline_n1280
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import METRICS_DIR, RUNS_DIR, resolve_path, save_json, system_info


def evaluate(weights: str | Path, data: str | Path, imgsz: int = 1280, split: str = "test",
             device: str | None = None, name: str = "eval", batch: int = 8, conf: float = 0.001,
             save_csv: bool = True) -> dict:
    from ultralytics import YOLO

    weights, data = resolve_path(weights), resolve_path(data)
    model = YOLO(str(weights))
    kwargs = dict(data=str(data), split=split, imgsz=imgsz, batch=batch, conf=conf, plots=True,
                  project=str(RUNS_DIR / "eval"), name=name, exist_ok=True, verbose=False)
    if device is not None:
        kwargs["device"] = device
    m = model.val(**kwargs)

    names = m.names
    summary = {
        "name": name,
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP75": float(m.box.map75),
        "mAP50_95": float(m.box.map),
        "speed_ms_pro_bild": {k: round(float(v), 3) for k, v in m.speed.items()},
        "plots_dir": str(m.save_dir),
        "system": system_info(),
    }
    per_class = []
    for i, cls in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class.append({"klasse_id": int(cls), "klasse": names[int(cls)], "precision": float(p),
                          "recall": float(r), "AP50": float(ap50), "AP50_95": float(ap)})
    summary["je_klasse"] = per_class

    if save_csv:
        save_json(summary, METRICS_DIR / f"eval_{name}_{split}.json")
        with (METRICS_DIR / f"eval_{name}_{split}_je_klasse.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["klasse_id", "klasse", "precision", "recall", "AP50", "AP50_95"],
                                delimiter=";")
            wr.writeheader()
            wr.writerows(per_class)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default=None, help="0 = erste GPU, cpu = Prozessor")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="eval")
    args = ap.parse_args()

    s = evaluate(args.weights, args.data, args.imgsz, args.split, args.device, args.name, args.batch)
    print(f"\n{s['name']} ({s['split']}): P={s['precision']:.3f} R={s['recall']:.3f} "
          f"mAP50={s['mAP50']:.3f} mAP50-95={s['mAP50_95']:.3f}")
    print(f"Geschwindigkeit (ms/Bild): {s['speed_ms_pro_bild']}")
    print(f"Plots: {s['plots_dir']}")


if __name__ == "__main__":
    main()
