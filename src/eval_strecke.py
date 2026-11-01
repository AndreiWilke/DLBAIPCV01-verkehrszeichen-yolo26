"""Precision und Recall auf der eigenen Annotation von Strecke 1 (CVAT) je Bedingung und Modell.

Grundwahrheit: manuelle Annotation in CVAT (Task 2646403), 240 Bilder = 80 Streckenpunkte x Tag/Dämmerung/Nacht,
Vorannotation durch E3+E9+E11, jede Box von Hand geprüft und ergänzt. Eingelesen aus
data/eigene_annotation/strecke1/cvat_annotation_api.txt (über die CVAT-API ausgelesen) und als YOLO-Labels geschrieben.

Auswertung:
  a) Betriebspunkt der Pipeline: conf 0,4 (zusätzlich 0,25), IoU ab 0,5, Klasse muss stimmen (gierige Zuordnung
     nach Konfidenz); P, R, F1 je Bedingung, dazu Recall ohne Klassenprüfung und Recall nach Zeichengröße
  b) mAP50 und mAP50-95 je Bedingung mit Ultralytics val (conf 0,001)
Ausgewertet werden die Originalbilder (nicht verpixelt); die Annotation gilt für beide Fassungen.

Aufruf (im Ordner 02_Code):  python src/eval_strecke.py --modelle E3 E9 E11
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import yaml

from common import DATA_DIR, METRICS_DIR, RUNS_DIR, SUPER_CLASSES

GEWICHTE = {"E3": "E3_s1280", "E9": "E9_s1280_robust", "E11": "E11_s1280_robust_farbe", "E0": "E0_baseline"}
KUERZEL = {"p": "prohibitory", "d": "danger", "m": "mandatory", "o": "other"}
BED = ["tag", "daemmerung", "nacht"]
GROESSE = [(0, 20), (20, 32), (32, 64), (64, 10_000)]
W, H = 1920, 1080


def lies_cvat(p: Path) -> dict[int, list[tuple]]:
    boxen, summe, fsum, n = {}, 0, 0, 0
    for zeile in p.read_text(encoding="utf-8").splitlines():
        if not zeile.strip() or zeile.startswith("#"):
            continue
        for teil in zeile.split("|"):
            t = teil.split()
            f, k, x1, y1, x2, y2 = int(t[0]), t[1], *map(float, t[2:6])
            boxen.setdefault(f, []).append((x1, y1, x2, y2, SUPER_CLASSES.index(KUERZEL[k])))
            summe += x1 + y1 + x2 + y2; fsum += f; n += 1
    assert (n, int(summe), fsum) == (415, 1298010, 45217), f"Prüfsumme falsch: {n} {summe} {fsum}"
    return boxen


def schreibe_labels(bilder: list[Path], boxen: dict, lbl_dir: Path) -> None:
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(bilder):
        z = [f"{k} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
             for x1, y1, x2, y2, k in boxen.get(i, [])]
        (lbl_dir / f"{b.stem}.txt").write_text("\n".join(z) + ("\n" if z else ""), encoding="utf-8")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def zuordnen(gt: list, pred: list, klasse: bool = True) -> tuple[int, int, list[bool]]:
    """pred: (x1,y1,x2,y2,conf,cls) absteigend nach conf. Gibt TP, FP und Trefferliste je GT zurück."""
    frei = [True] * len(gt); tp = fp = 0
    for p in sorted(pred, key=lambda x: -x[4]):
        best, bj = 0.5, -1
        for j, g in enumerate(gt):
            if frei[j] and (not klasse or g[4] == p[5]):
                v = iou(p, g)
                if v >= best:
                    best, bj = v, j
        if bj >= 0:
            frei[bj] = False; tp += 1
        else:
            fp += 1
    return tp, fp, [not f for f in frei]


def zahl(x) -> str:
    return str(round(x, 3)).replace(".", ",") if isinstance(x, float) else str(x)


def schreibe_csv(p: Path, kopf: list, zeilen: list) -> None:
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(kopf)
        w.writerows([[zahl(x) for x in z] for z in zeilen])
    print("->", p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelle", nargs="+", default=["E3", "E9", "E11"])
    ap.add_argument("--confs", nargs="+", type=float, default=[0.4, 0.25])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--bilder", choices=["original", "anonym"], default="original")
    ap.add_argument("--ohne-map", action="store_true")
    a = ap.parse_args()

    basis = DATA_DIR / "eigene_annotation"
    boxen = lies_cvat(basis / "strecke1" / "cvat_annotation_api.txt")
    # Auswertungsordner im YOLO-Aufbau: images/test + labels/test
    ziel = basis / ("strecke1_gt_original" if a.bilder == "original" else "strecke1_gt_anonym")
    quelle = basis / "Archiv" / "strecke1_bilder_original" if a.bilder == "original" else basis / "strecke1" / "images" / "test"
    img_dir = ziel / "images" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    bilder = sorted(quelle.glob("*.jpg"))
    assert len(bilder) == 240, len(bilder)
    for b in bilder:
        if not (img_dir / b.name).exists():
            shutil.copy2(b, img_dir / b.name)
    bilder = sorted(img_dir.glob("*.jpg"))
    schreibe_labels(bilder, boxen, ziel / "labels" / "test")
    bed = [b.name.split("_")[1] for b in bilder]
    for c in BED + ["alle"]:
        (ziel / f"{c}.txt").write_text("\n".join(str(b) for b, x in zip(bilder, bed) if c in ("alle", x)) + "\n", encoding="utf-8")
        y = {"path": str(ziel), "train": f"{c}.txt", "val": f"{c}.txt", "test": f"{c}.txt",
             "names": {i: n for i, n in enumerate(SUPER_CLASSES)}}
        (ziel / f"{c}.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")

    gt = [boxen.get(i, []) for i in range(len(bilder))]
    print("GT-Boxen je Bedingung:", {c: sum(len(g) for g, x in zip(gt, bed) if x == c) for c in BED},
          "| Bilder ohne Zeichen:", {c: sum(1 for g, x in zip(gt, bed) if x == c and not g) for c in BED})

    from ultralytics import YOLO
    pr, groesse, kl, mp = [], [], [], []
    for m in a.modelle:
        w = RUNS_DIR / "train" / GEWICHTE[m] / "weights" / "best.pt"
        model = YOLO(str(w))
        preds = []
        for b in bilder:
            r = model.predict(str(b), imgsz=a.imgsz, conf=min(a.confs), device=a.device, verbose=False)[0]
            preds.append([(*xy, c, int(k)) for xy, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())])
        for conf in a.confs:
            for c in BED + ["alle"]:
                idx = [i for i, x in enumerate(bed) if c in ("alle", x)]
                tp = fp = tpa = ngt = 0
                for i in idx:
                    p = [q for q in preds[i] if q[4] >= conf]
                    t, f, _ = zuordnen(gt[i], p); tp += t; fp += f; ngt += len(gt[i])
                    tpa += zuordnen(gt[i], p, klasse=False)[0]
                P = tp / (tp + fp) if tp + fp else 0.0; R = tp / ngt if ngt else 0.0
                F1 = 2 * P * R / (P + R) if P + R else 0.0
                pr.append([m, conf, c, ngt, tp, fp, ngt - tp, P, R, F1, tpa / ngt if ngt else 0.0])
                if conf == a.confs[0]:
                    for lo, hi in GROESSE:
                        n = h = 0
                        for i in idx:
                            p = [q for q in preds[i] if q[4] >= conf]
                            treffer = zuordnen(gt[i], p)[2]
                            for g, ok in zip(gt[i], treffer):
                                s = max(g[2] - g[0], g[3] - g[1])
                                if lo <= s < hi:
                                    n += 1; h += ok
                        groesse.append([m, c, f"{lo}-{hi if hi < 10_000 else 'max'}", n, h, h / n if n else 0.0])
                    if c == "alle":
                        for k, name in enumerate(SUPER_CLASSES):
                            n = h = fpk = 0
                            for i in idx:
                                p = [q for q in preds[i] if q[4] >= conf]
                                gk = [g for g in gt[i] if g[4] == k]; pk = [q for q in p if q[5] == k]
                                t, f, _ = zuordnen(gk, pk); n += len(gk); h += t; fpk += f
                            kl.append([m, name, n, h, fpk, h / (h + fpk) if h + fpk else 0.0, h / n if n else 0.0])
        if not a.ohne_map:
            for c in BED + ["alle"]:
                v = model.val(data=str(ziel / f"{c}.yaml"), split="test", imgsz=a.imgsz, batch=4, conf=0.001,
                              device=a.device, plots=False, verbose=False, project=str(RUNS_DIR / "val_strecke1"),
                              name=f"{m}_{c}_{a.bilder}", exist_ok=True)
                mp.append([m, c, float(v.box.mp), float(v.box.mr), float(v.box.map50), float(v.box.map)])
        print(m, "fertig")

    s = "" if a.bilder == "original" else "_anonym"
    schreibe_csv(METRICS_DIR / f"cvat_strecke1_pr{s}.csv",
                 ["Modell", "conf", "Bedingung", "GT", "TP", "FP", "FN", "Precision", "Recall", "F1", "Recall_ohne_Klasse"], pr)
    schreibe_csv(METRICS_DIR / f"cvat_strecke1_groesse{s}.csv", ["Modell", "Bedingung", "Groesse_px", "GT", "gefunden", "Recall"], groesse)
    schreibe_csv(METRICS_DIR / f"cvat_strecke1_klassen{s}.csv", ["Modell", "Klasse", "GT", "TP", "FP", "Precision", "Recall"], kl)
    if mp:
        schreibe_csv(METRICS_DIR / f"cvat_strecke1_map{s}.csv", ["Modell", "Bedingung", "P_val", "R_val", "mAP50", "mAP50-95"], mp)
    for z in pr:
        print(";".join(zahl(x) for x in z))
    for z in mp:
        print(";".join(zahl(x) for x in z))


if __name__ == "__main__":
    main()
