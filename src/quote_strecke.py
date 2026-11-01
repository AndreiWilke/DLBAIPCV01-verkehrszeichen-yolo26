"""Schildstellen-Quote je Bedingung und Modell auf Strecke 1.

Referenz: Schildstellen der Tagfahrt (results/metrics/route/schildstellen_strecke1.csv, aus E3 und E9 bei Tag).
Treffer: mindestens eine Erkennung (conf ab 0,4) im über den Streckenabgleich zugeordneten Zeitfenster
der jeweiligen Fahrt, Toleranz ±1,5 s. Grundlage sind die Erkennungsverläufe je Modell
(WORK_DIR/analyse/<Video>/det_<Modell>.csv, jedes 10. Bild, aus analyse/det_verlauf.py).

Aufruf:  python src/quote_strecke.py --modelle E3 E9 E11
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from common import ROOT, WORK_DIR

FAHRTEN = [("IMG_9733", "tag_s", "Tag"), ("IMG_9726", "daemmerung_s", "Dämmerung"), ("IMG_9730", "nacht_s", "Nacht")]


def lies(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def z(s):
    return float(str(s).replace(",", "."))


def ereignisse(t, m, gap=2, minlen=2):
    ev, run, miss = [], [], 0
    for ti, mk in zip(t, m):
        if mk:
            run.append(ti); miss = 0
        else:
            miss += 1
            if miss > gap and run:
                if len(run) >= minlen:
                    ev.append((run[0], run[-1]))
                run = []
    if len(run) >= minlen:
        ev.append((run[0], run[-1]))
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelle", nargs="+", default=["E3", "E9", "E11"])
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--toleranz", type=float, default=1.5)
    ap.add_argument("--ziel", default=str(ROOT / "results/metrics/erkennungsquote_strecke1_modelle.csv"))
    ap.add_argument("--referenz-modelle", nargs="*", default=None,
                    help="Referenz neu aus der Tagfahrt bilden (Vereinigung dieser Modelle) statt schildstellen_strecke1.csv")
    a = ap.parse_args()

    zu = lies(ROOT / "results/metrics/route/zeitabgleich_strecke1.csv")
    tag = np.array([z(r["tag_s"]) for r in zu])
    sp = {c: np.array([z(r[c]) for r in zu]) for _, c, _ in FAHRTEN}
    t0, t1 = tag.min(), tag.max()
    if a.referenz_modelle:
        dd = [lies(WORK_DIR / "analyse" / "IMG_9733" / f"det_{m}.csv") for m in a.referenz_modelle]
        tt = np.array([z(r["sek"]) for r in dd[0]])
        uni = np.any([[z(r["max_conf"]) >= a.conf for r in d] for d in dd], axis=0)
        sel = (tt >= t0) & (tt <= t1)
        ref = ereignisse(tt[sel], uni[sel])
        print(f"Referenz aus {'+'.join(a.referenz_modelle)}: {len(ref)} Schildstellen")
    else:
        ref = [(z(r["tag_start_s"]), z(r["tag_ende_s"])) for r in lies(ROOT / "results/metrics/route/schildstellen_strecke1.csv")]

    zeilen, treffer = [], {}
    for v, col, name in FAHRTEN:
        for m in a.modelle:
            d = lies(WORK_DIR / "analyse" / v / f"det_{m}.csv")
            ts = np.array([z(r["sek"]) for r in d]); mk = np.array([z(r["max_conf"]) >= a.conf for r in d])
            mp = lambda x: float(np.interp(x, tag, sp[col]))
            h = np.array([mk[(ts >= mp(s) - a.toleranz) & (ts <= mp(e) + a.toleranz)].any() for s, e in ref])
            treffer[(name, m)] = h
            qs, qe = mp(t0), mp(t1)
            refq = [(mp(s) - a.toleranz, mp(e) + a.toleranz) for s, e in ref]
            qev = [q for q in ereignisse(ts, mk) if q[0] >= qs and q[1] <= qe]
            extra = [q for q in qev if not any(q[0] <= rb and q[1] >= ra for ra, rb in refq)]
            sel = (ts >= qs) & (ts <= qe)
            zeilen.append([name, m, len(ref), int(h.sum()), round(100 * h.mean(), 1), len(qev), len(extra),
                           round(100 * mk[sel].mean(), 1)])
    with open(a.ziel, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Bedingung", "Modell", "Ref_Schildstellen", "gefunden", "Quote_%", "Ereignisse_im_Abschnitt",
                     "ohne_Tag-Entsprechung", "Bilder_mit_Erkennung_%"])
        w.writerows([[str(x).replace(".", ",") if isinstance(x, float) else x for x in r] for r in zeilen])
    for r in zeilen:
        print(";".join(map(str, r)))
    ms = a.modelle
    for name in ("Dämmerung", "Nacht"):
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                x, y = treffer[(name, ms[i])], treffer[(name, ms[j])]
                print(f"{name}: nur {ms[i]} {int((x & ~y).sum())}, nur {ms[j]} {int((y & ~x).sum())}, beide {int((x & y).sum())}")


if __name__ == "__main__":
    main()
