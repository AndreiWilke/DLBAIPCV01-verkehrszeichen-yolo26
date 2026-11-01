"""Prüft die Entwicklungsumgebung und speichert den Stand für den Bericht.

Aufruf:  python src/check_setup.py
Ergebnis: results/metrics/systeminfo.json
"""

from common import METRICS_DIR, ensure_dirs, save_json, system_info


def main() -> None:
    ensure_dirs()
    info = system_info()
    for k, v in info.items():
        print(f"{k:16s}: {v}")
    if info.get("torch") and not info.get("cuda_available"):
        print("\nHinweis: PyTorch läuft ohne CUDA. Für das Training auf der GPU PyTorch mit CUDA "
              "installieren (siehe README, Schritt 2).")
    save_json(info, METRICS_DIR / "systeminfo.json")
    print(f"\nGespeichert: {METRICS_DIR / 'systeminfo.json'}")


if __name__ == "__main__":
    main()
