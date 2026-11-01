"""Gemeinsame Konstanten und Hilfsfunktionen für das Projekt DLBAIPCV01_D.

Enthält die 43 GTSDB/GTSRB-Klassen, die Zuordnung zu den vier Obergruppen
(prohibitory, danger, mandatory, other) und Pfadkonstanten.
Die Klassennamen sind bewusst ASCII (snake_case), weil OpenCV beim Einzeichnen
keine Umlaute darstellen kann. Die deutschen Bezeichnungen stehen in CLASS_NAMES_DE.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 02_Code (Code und kleine Ergebnisse, liegt in OneDrive)


def _work_dir() -> Path:
    """Arbeitsverzeichnis für große Dateien (Datensätze, Trainingsläufe, Videos, Gewichte).

    Liegt bewusst außerhalb von OneDrive. Reihenfolge: Umgebungsvariable DLBAIPCV01_WORK,
    sonst C:\\IU\\DLBAIPCV01, falls vorhanden, sonst der Code-Ordner selbst.
    """
    env = os.environ.get("DLBAIPCV01_WORK")
    if env:
        return Path(env)
    default = Path(r"C:\IU\DLBAIPCV01")
    return default if default.exists() else ROOT


WORK_DIR = _work_dir()
DATA_DIR = WORK_DIR / "data"
VIDEOS_DIR = WORK_DIR / "videos"
RUNS_DIR = WORK_DIR / "runs"
WEIGHTS_DIR = WORK_DIR / "weights"
RESULTS_DIR = ROOT / "results"  # CSV, JSON und Grafiken für den Bericht bleiben im Projektordner
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"


def resolve_path(p: str | Path) -> Path:
    """Löst relative Pfade auf: data/, videos/, runs/, weights/ liegen im Arbeitsverzeichnis,
    alles andere (z. B. configs/) im Code-Ordner. Absolute oder existierende Pfade bleiben unverändert."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p.resolve()
    if p.parts and p.parts[0] in ("data", "videos", "runs", "weights"):
        return WORK_DIR / p
    return ROOT / p

# 43 Klassen-IDs wie in GTSDB/GTSRB (ReadMe des Datensatzes)
CLASS_NAMES = [
    "speed_20", "speed_30", "speed_50", "speed_60", "speed_70", "speed_80",
    "end_speed_80", "speed_100", "speed_120", "no_overtaking",
    "no_overtaking_trucks", "priority_next_intersection", "priority_road",
    "give_way", "stop", "no_vehicles", "no_trucks", "no_entry",
    "danger_general", "curve_left", "curve_right", "double_curve", "bumpy_road",
    "slippery_road", "road_narrows_right", "road_works", "traffic_signals",
    "pedestrians", "children", "cyclists", "ice_snow", "wild_animals",
    "end_all_restrictions", "turn_right_ahead", "turn_left_ahead", "ahead_only",
    "straight_or_right", "straight_or_left", "keep_right", "keep_left",
    "roundabout", "end_no_overtaking", "end_no_overtaking_trucks",
]

CLASS_NAMES_DE = [
    "Zulässige Höchstgeschwindigkeit 20", "Zulässige Höchstgeschwindigkeit 30",
    "Zulässige Höchstgeschwindigkeit 50", "Zulässige Höchstgeschwindigkeit 60",
    "Zulässige Höchstgeschwindigkeit 70", "Zulässige Höchstgeschwindigkeit 80",
    "Ende Höchstgeschwindigkeit 80", "Zulässige Höchstgeschwindigkeit 100",
    "Zulässige Höchstgeschwindigkeit 120", "Überholverbot",
    "Überholverbot für Kfz über 3,5 t", "Vorfahrt an der nächsten Kreuzung",
    "Vorfahrtstraße", "Vorfahrt gewähren", "Halt. Vorfahrt gewähren",
    "Verbot für Fahrzeuge aller Art", "Verbot für Kfz über 3,5 t",
    "Verbot der Einfahrt", "Gefahrstelle", "Kurve links", "Kurve rechts",
    "Doppelkurve", "Unebene Fahrbahn", "Schleudergefahr",
    "Verengte Fahrbahn rechts", "Arbeitsstelle", "Lichtzeichenanlage",
    "Fußgänger", "Kinder", "Radverkehr", "Schnee- oder Eisglätte", "Wildwechsel",
    "Ende sämtlicher Streckenverbote", "Vorgeschriebene Fahrtrichtung rechts",
    "Vorgeschriebene Fahrtrichtung links", "Vorgeschriebene Fahrtrichtung geradeaus",
    "Geradeaus oder rechts", "Geradeaus oder links", "Rechts vorbei", "Links vorbei",
    "Kreisverkehr", "Ende des Überholverbots", "Ende des Überholverbots für Kfz über 3,5 t",
]

# Obergruppen nach der GTSDB-Wettbewerbseinteilung; alle übrigen Klassen = other
SUPER_CLASSES = ["prohibitory", "danger", "mandatory", "other"]
_PROHIBITORY = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
_DANGER = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
_MANDATORY = {33, 34, 35, 36, 37, 38, 39, 40}


def to_super_class(class_id: int) -> int:
    """Bildet eine GTSDB-Klassen-ID (0..42) auf eine Obergruppe (0..3) ab."""
    if class_id in _PROHIBITORY:
        return 0
    if class_id in _DANGER:
        return 1
    if class_id in _MANDATORY:
        return 2
    return 3


def names_for_mode(mode: str) -> list[str]:
    if mode == "super":
        return SUPER_CLASSES
    if mode == "full":
        return CLASS_NAMES
    raise ValueError(f"Unbekannter Modus: {mode} (erlaubt: super, full)")


def ensure_dirs() -> None:
    for d in (METRICS_DIR, FIGURES_DIR, RUNS_DIR, WEIGHTS_DIR, DATA_DIR, VIDEOS_DIR / "input", VIDEOS_DIR / "output"):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def system_info() -> dict:
    """Hardware- und Softwarestand für die Reproduzierbarkeit im Bericht."""
    info = {"python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor(),
            "arbeitsverzeichnis": str(WORK_DIR)}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = None
    return info
