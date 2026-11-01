"""Führt die Versuchsreihe aus configs/experiments.yaml aus und fasst sie zusammen.

Für jedes Experiment: Training -> Auswertung auf der Testmenge -> Zeile in
results/metrics/experimente.csv (Vergleichstabelle für Kapitel 5/6 des Berichts).

Aufruf:
  python src/run_experiments.py --config configs/experiments.yaml
  python src/run_experiments.py --config configs/experiments.yaml --only E0_baseline_n1280 E1_n640
Bereits trainierte Experimente werden übersprungen (vorhandene best.pt), außer mit --force.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from common import METRICS_DIR, ROOT, RUNS_DIR, resolve_path
from evaluate_model import evaluate
import traceback

from train_yolo import resume, train

FIELDS = ["name", "note", "model", "imgsz", "data", "precision", "recall", "mAP50", "mAP75", "mAP50_95",
          "inferenz_ms", "gesamt_ms", "fps_modell", "trainingsdauer_min", "weights"]


def resolve(p: str) -> str:
    return str(resolve_path(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments.yaml"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="überschreibt epochs für alle (z. B. Schnelltest)")
    ap.add_argument("--workers", type=int, default=None, help="überschreibt DataLoader-Worker (0 bei Speicherproblemen)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {})
    rows = []

    out = METRICS_DIR / "experimente.csv"

    def write_table() -> None:
        existing = []
        if out.exists():
            with out.open(encoding="utf-8") as f:
                done = {x["name"] for x in rows}
                existing = [r for r in csv.DictReader(f, delimiter=";") if r["name"] not in done]
        with out.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
            wr.writeheader()
            wr.writerows(existing + rows)

    for exp in cfg["experiments"]:
        name = exp["name"]
        if args.only and name not in args.only:
            continue
        params = {**defaults, **{k: v for k, v in exp.items() if k not in ("name", "note")}}
        data = resolve(params.pop("data"))
        device = str(params.pop("device")) if params.get("device") is not None else None
        params.pop("device", None)
        if args.epochs:
            params["epochs"] = args.epochs
        if args.workers is not None:
            params["workers"] = args.workers

        wdir = RUNS_DIR / "train" / name / "weights"
        best, last = wdir / "best.pt", wdir / "last.pt"
        train_json = METRICS_DIR / f"train_{name}.json"
        try:
            if train_json.exists() and best.exists() and not args.force:
                print(f"[{name}] vorhanden, Training uebersprungen")
                t = json.loads(train_json.read_text(encoding="utf-8"))
            elif last.exists() and not train_json.exists() and not args.force:
                print(f"\n=== [{name}] abgebrochenes Training wird fortgesetzt")
                t = resume(name, data, device, workers=params.get("workers"))
            else:
                print(f"\n=== [{name}] Training: {params}")
                t = train(data, name, device, **params)

            e = evaluate(best, data, imgsz=params["imgsz"], split="test", device=device, name=name)
        except Exception:
            print(f"\n!!! [{name}] FEHLER, Experiment wird uebersprungen:")
            traceback.print_exc()
            continue
        sp = e["speed_ms_pro_bild"]
        total = sum(sp.values())
        rows.append({
            "name": name, "note": exp.get("note", ""), "model": params["model"], "imgsz": params["imgsz"],
            "data": Path(data).parent.name, "precision": round(e["precision"], 4), "recall": round(e["recall"], 4),
            "mAP50": round(e["mAP50"], 4), "mAP75": round(e["mAP75"], 4), "mAP50_95": round(e["mAP50_95"], 4),
            "inferenz_ms": round(sp.get("inference", 0), 2), "gesamt_ms": round(total, 2),
            "fps_modell": round(1000 / total, 1) if total else None,
            "trainingsdauer_min": t.get("trainingsdauer_min", t.get("trainingsdauer_min_fortsetzung")),
            "weights": str(best),
        })
        write_table()

    print(f"\nVergleichstabelle: {out}")
    for r in rows:
        print(f"  {r['name']:22s} mAP50={r['mAP50']:.3f} mAP50-95={r['mAP50_95']:.3f} "
              f"{r['gesamt_ms']} ms/Bild ({r['fps_modell']} FPS)")


if __name__ == "__main__":
    main()
