"""Trainiert ein Ultralytics-YOLO-Modell auf dem konvertierten GTSDB-Datensatz.

Wichtige Voreinstellungen (fachlich begründet, im Bericht Kapitel 5 erläutern):
  - imgsz 1280: GTSDB-Bilder sind 1360 x 800 px, Zeichen teils nur 16 px groß.
    Bei 640 würden sie auf ca. 8 px schrumpfen.
  - fliplr 0.0: horizontales Spiegeln ist bei Verkehrszeichen unzulässig, weil es die
    Bedeutung ändert (z. B. "Kurve links" -> "Kurve rechts", "Rechts vorbei" -> "Links vorbei").
    Ultralytics spiegelt standardmäßig mit 50 % Wahrscheinlichkeit.
  - seed 42 und deterministic True für Reproduzierbarkeit.

Aufruf (Baseline):
  python src/train_yolo.py --data data/processed/gtsdb_yolo_super/data.yaml --name baseline_n1280
Mit anschließender Testauswertung:
  python src/train_yolo.py --data ... --name baseline_n1280 --eval-test
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from common import METRICS_DIR, RUNS_DIR, WEIGHTS_DIR, ensure_dirs, resolve_path, save_json, system_info

DEFAULTS = dict(
    model="yolo26n.pt",
    imgsz=1280,
    epochs=100,
    batch=8,
    patience=30,
    seed=42,
    deterministic=True,
    fliplr=0.0,
    workers=4,
)


def robust_augmentations() -> list:
    """Zusätzliche Albumentations-Transformationen gegen die Schwachstellen aus den Robustheitstests
    (Bewegungsunschärfe, Sensorrauschen, starke Kompression, Dunkelheit). Die Parameter decken bewusst
    einen breiten Bereich ab und sind nicht identisch mit den festen Testbedingungen in robustness_tests.py."""
    import albumentations as A

    return [
        A.MotionBlur(blur_limit=(3, 11), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.2),
        A.ImageCompression(quality_range=(10, 60), p=0.2),
        A.RandomGamma(gamma_limit=(100, 300), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.2), contrast_limit=(-0.3, 0.2), p=0.2),
    ]


def robust_farbe_augmentations() -> list:
    """Preset robust plus Farbstich und Blendung (Ergebnis der Vergleichsfahrten 04./05.10.2026):
    E9 uebersah nachts Schilder im roten Licht von Bremsleuchten und Ampeln sowie bei Scheinwerferblendung.
    RGBShift verschiebt das ganze Bild Richtung Rot/Orange (Natriumlicht, Bremslicht), RandomSunFlare
    setzt helle Lichtflecken mit Streulicht in die obere Bildhaelfte."""
    import albumentations as A

    return robust_augmentations() + [
        A.RGBShift(r_shift_limit=(0, 50), g_shift_limit=(-20, 10), b_shift_limit=(-40, 0), p=0.2),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.6), src_radius=160, num_flare_circles_range=(2, 6), p=0.1),
    ]


AUG_PRESETS = {"robust": robust_augmentations, "robust_farbe": robust_farbe_augmentations}


def train(data: str | Path, name: str, device: str | None = None, **overrides) -> dict:
    """Trainiert ein Modell und gibt eine Zusammenfassung zurück (auch als JSON gespeichert)."""
    from ultralytics import YOLO

    ensure_dirs()
    data = resolve_path(data)
    cfg = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    model_name = cfg.pop("model")
    preset = cfg.pop("aug_preset", None)
    if preset:
        cfg["augmentations"] = AUG_PRESETS[preset]()
    # Vortrainierte Gewichte (und die Datei für den AMP-Check) landen im Arbeitsverzeichnis,
    # nicht im OneDrive-Projektordner
    cwd = os.getcwd()
    os.chdir(WEIGHTS_DIR)
    try:
        model = YOLO(model_name)
        kwargs = dict(data=str(data), project=str(RUNS_DIR / "train"), name=name, exist_ok=True, plots=True, **cfg)
        if device is not None:
            kwargs["device"] = device

        t0 = time.time()
        results = model.train(**kwargs)
        duration_min = (time.time() - t0) / 60
    finally:
        os.chdir(cwd)

    save_dir = Path(model.trainer.save_dir)
    rd = getattr(results, "results_dict", {}) or {}
    summary = {
        "name": name,
        "model": model_name,
        "data": str(data),
        "parameter": {**{k: v for k, v in cfg.items() if k != "augmentations"}, "aug_preset": preset, "device": device},
        "trainingsdauer_min": round(duration_min, 1),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "save_dir": str(save_dir),
        "val_precision": rd.get("metrics/precision(B)"),
        "val_recall": rd.get("metrics/recall(B)"),
        "val_mAP50": rd.get("metrics/mAP50(B)"),
        "val_mAP50_95": rd.get("metrics/mAP50-95(B)"),
        "system": system_info(),
    }
    save_json(summary, METRICS_DIR / f"train_{name}.json")
    return summary


def resume(name: str, data: str | Path, device: str | None = None, workers: int | None = None) -> dict:
    """Setzt ein abgebrochenes Training ab weights/last.pt fort (z. B. nach Speicherfehler)."""
    from ultralytics import YOLO

    ensure_dirs()
    last = RUNS_DIR / "train" / name / "weights" / "last.pt"
    cwd = os.getcwd()
    os.chdir(WEIGHTS_DIR)
    try:
        model = YOLO(str(last))
        kwargs = {"resume": True}
        if workers is not None:
            kwargs["workers"] = workers
        if device is not None:
            kwargs["device"] = device
        t0 = time.time()
        results = model.train(**kwargs)
        duration_min = (time.time() - t0) / 60
    finally:
        os.chdir(cwd)
    save_dir = Path(model.trainer.save_dir)
    rd = getattr(results, "results_dict", {}) or {}
    summary = {
        "name": name, "data": str(resolve_path(data)), "fortgesetzt": True,
        "parameter": {k: getattr(model.trainer.args, k, None) for k in
                      ("model", "imgsz", "epochs", "batch", "patience", "seed", "fliplr", "workers", "optimizer", "lr0")},
        "trainingsdauer_min_fortsetzung": round(duration_min, 1),
        "best_weights": str(save_dir / "weights" / "best.pt"), "save_dir": str(save_dir),
        "val_precision": rd.get("metrics/precision(B)"), "val_recall": rd.get("metrics/recall(B)"),
        "val_mAP50": rd.get("metrics/mAP50(B)"), "val_mAP50_95": rd.get("metrics/mAP50-95(B)"),
        "system": system_info(),
    }
    save_json(summary, METRICS_DIR / f"train_{name}.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="baseline_n1280")
    ap.add_argument("--model", default=None, help="z. B. yolo26n.pt, yolo26s.pt, yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--mosaic", type=float, default=None)
    ap.add_argument("--device", default=None, help="0 = erste GPU, cpu = Prozessor")
    ap.add_argument("--workers", type=int, default=None, help="bei Windows-Problemen 0 setzen")
    ap.add_argument("--eval-test", action="store_true", help="nach dem Training auf der Testmenge auswerten")
    args = ap.parse_args()

    s = train(args.data, args.name, args.device, model=args.model, imgsz=args.imgsz, epochs=args.epochs,
              batch=args.batch, patience=args.patience, lr0=args.lr0, mosaic=args.mosaic, workers=args.workers)
    print(f"\nTraining fertig in {s['trainingsdauer_min']} min, Gewichte: {s['best_weights']}")
    print(f"Validierung: mAP50={s['val_mAP50']}, mAP50-95={s['val_mAP50_95']}")

    if args.eval_test:
        from evaluate_model import evaluate
        e = evaluate(s["best_weights"], args.data, imgsz=s["parameter"]["imgsz"], split="test",
                     device=args.device, name=args.name)
        print(f"Test: mAP50={e['mAP50']:.3f}, mAP50-95={e['mAP50_95']:.3f}")


if __name__ == "__main__":
    main()
