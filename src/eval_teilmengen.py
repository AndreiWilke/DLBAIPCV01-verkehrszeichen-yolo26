"""Recall auf der eigenen Strecke getrennt nach Schildstellen und Zufallspunkten (02.11.2026).

Die 50 Schildstellen (Bilder S00 bis S49) wurden dort gewählt, wo E3 oder E9 bei Tag ein Schild erkannt hatten,
die 30 Zufallspunkte (Z00 bis Z29) unabhängig von Modellerkennungen. Der Vergleich zeigt, wie stark die Auswahl
den Recall verzerrt. Grundwahrheit und Zuordnung wie in eval_strecke.py (IoU ab 0,5, Klasse muss stimmen).

Aufruf (im Ordner 02_Code, nach eval_strecke.py):  python src/eval_teilmengen.py --modelle E3 E9 E11
"""
from __future__ import annotations

import argparse

from common import DATA_DIR, METRICS_DIR, RUNS_DIR
from eval_strecke import BED, GEWICHTE, lies_cvat, schreibe_csv, zuordnen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelle", nargs="+", default=["E3", "E9", "E11"])
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    basis = DATA_DIR / "eigene_annotation"
    boxen = lies_cvat(basis / "strecke1" / "cvat_annotation_api.txt")
    bilder = sorted((basis / "strecke1_gt_original" / "images" / "test").glob("*.jpg"))
    assert len(bilder) == 240, len(bilder)
    gt = [boxen.get(i, []) for i in range(len(bilder))]
    teil = [b.name[0] for b in bilder]                  # "S" Schildstelle, "Z" Zufallspunkt
    bed = [b.name.split("_")[1] for b in bilder]

    from ultralytics import YOLO
    zeilen = []
    for m in a.modelle:
        model = YOLO(str(RUNS_DIR / "train" / GEWICHTE[m] / "weights" / "best.pt"))
        preds = []
        for b in bilder:
            r = model.predict(str(b), imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
            preds.append([(*xy, c, int(k)) for xy, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())])
        for t, name in [("S", "schildstellen"), ("Z", "zufallspunkte")]:
            for c in BED + ["alle"]:
                idx = [i for i in range(len(bilder)) if teil[i] == t and c in ("alle", bed[i])]
                tp = fp = ngt = 0
                for i in idx:
                    x, f, _ = zuordnen(gt[i], preds[i]); tp += x; fp += f; ngt += len(gt[i])
                zeilen.append([m, name, c, len(idx), ngt, tp, fp, tp / ngt if ngt else 0.0])
        print(m, "fertig")
    schreibe_csv(METRICS_DIR / "cvat_strecke1_teilmengen.csv",
                 ["Modell", "Teilmenge", "Bedingung", "Bilder", "GT", "TP", "FP", "Recall"], zeilen)
    for z in zeilen:
        print(z)


if __name__ == "__main__":
    main()
