"""Demo-Video: dieselbe Stelle von Strecke 1 bei Tag, Dämmerung und Nacht nebeneinander (2x2-Raster).

Fassung 4 (02.11.2026, sonst wie Fassung 3 vom 05.10.2026):
- Alle drei Fahrten laufen in Originalgeschwindigkeit (30 FPS, kein Zeitstrecken). Synchronisiert wird nur
  der Startpunkt jedes Abschnitts über den bildbasierten Streckenabgleich (zeitabgleich_strecke1.csv).
  Gewählt werden Abschnitte, in denen alle drei Fahrten etwa gleich schnell fahren (Geschwindigkeits-
  verhältnis 0,85 bis 1,2), damit die Bilder bis zum Abschnittsende zusammenpassen; unter diesen die mit
  den meisten Schildstellen.
- Erkennung direkt im Skript mit dem gewählten Modell (conf ab 0,4) auf den Originalvideos; Boxen und
  Beschriftungen werden nach dem Verkleinern gezeichnet, damit sie lesbar bleiben.
- Lupe im vierten Feld: die aktuell sicherste Erkennung jeder Fahrt vergrößert.
- Offline gerendert: keine Threads und keine zeitliche Stabilisierung wie in run_video_pipeline.py.
- Datenschutz: Kennzeichen (YOLO11-Modell morsetechlab/yolov11-license-plate-detection, AGPL-3.0) und
  Gesichter (OpenCV YuNet) werden in voller Auflösung erkannt und verpixelt, Boxen 10 Frames gehalten.
  Seit 02.11.2026 ohne Ausnahme: auch Bereiche, die ein erkanntes Verkehrszeichen überdecken, werden verpixelt.

Aufruf (im Ordner 02_Code):
  python src/make_demo_vergleich.py --modell E11 --abschnitte 5 --sekunden 10
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import METRICS_DIR, ROOT, RUNS_DIR, VIDEOS_DIR, WEIGHTS_DIR, resolve_path

FAHRTEN = [("IMG_9733", "tag_s", "Tag", "05.10.2026 13:15:10"),
           ("IMG_9726", "daemmerung_s", "Dämmerung", "04.10.2026 18:54:47"),
           ("IMG_9730", "nacht_s", "Nacht", "04.10.2026 21:55:16")]
GEWICHTE = {"E11": "train/E11_s1280_robust_farbe/weights/best.pt", "E9": "train/E9_s1280_robust/weights/best.pt",
            "E3": "train/E3_s1280/weights/best.pt"}
KLASSEN = ["Verbot", "Gefahr", "Gebot", "Sonstige"]
FARBEN = [(40, 40, 230), (0, 170, 255), (230, 140, 30), (60, 200, 60)]  # BGR je Klasse
W, H = 1920, 1080
TW, TH = 960, 540


def de(x, n=1):
    return f"{float(x):.{n}f}".replace(".", ",")


def zahl(s):
    return float(str(s).replace(",", "."))


_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        cands = (["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"] if bold else
                 ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"])
        _fonts[key] = next((ImageFont.truetype(f, size) for f in cands if Path(f).exists()), ImageFont.load_default())
    return _fonts[key]


def texte(img, items):
    """items: Liste (xy, text, size, bold, fill, hintergrund) in einem PIL-Durchgang."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for xy, s, size, bold, fill, bg in items:
        f = font(size, bold)
        if bg is not None:
            l, t, r, b = d.textbbox(xy, s, font=f)
            d.rectangle((l - 6, t - 4, r + 6, b + 4), fill=bg)
        d.text(xy, s, font=f, fill=fill)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class Anonymisierer:
    def __init__(self, halten=10):
        from ultralytics import YOLO
        self.kz = YOLO(str(WEIGHTS_DIR / "anonym" / "kennzeichen_yolo11s.pt"))
        self.face = cv2.FaceDetectorYN.create(str(WEIGHTS_DIR / "anonym" / "gesicht_yunet_2023mar.onnx"), "", (1920, 1080), 0.5, 0.3, 5000)
        self.halten, self.mem, self.n_kz, self.n_gesicht = halten, {}, 0, 0

    def __call__(self, key, img, schilder):
        h, w = img.shape[:2]
        boxes = [tuple(b) for b in self.kz.predict(img, imgsz=1280, conf=0.2, device=0, verbose=False)[0].boxes.xyxy.tolist()]
        self.n_kz += len(boxes)
        self.face.setInputSize((w, h))
        _, faces = self.face.detect(img)
        if faces is not None:
            for x, y, fw, fh in faces[:, :4]:
                boxes.append((x, y, x + fw, y + fh)); self.n_gesicht += 1
        alt = [(b, a - 1) for b, a in self.mem.get(key, []) if a > 1]
        self.mem[key] = [(b, self.halten) for b in boxes] + alt
        for (x1, y1, x2, y2), _ in self.mem[key]:
            dx, dy = (x2 - x1) * 0.25 + 6, (y2 - y1) * 0.35 + 6
            a, b, c, d = int(max(0, x1 - dx)), int(max(0, y1 - dy)), int(min(w, x2 + dx)), int(min(h, y2 + dy))
            if c - a < 2 or d - b < 2:
                continue
            roi = img[b:d, a:c]
            k = max(2, min(roi.shape[:2]) // 5)
            small = cv2.resize(roi, (max(1, roi.shape[1] // k), max(1, roi.shape[0] // k)), interpolation=cv2.INTER_AREA)
            img[b:d, a:c] = cv2.resize(small, (c - a, d - b), interpolation=cv2.INTER_NEAREST)
        return img


def karte(zeilen, sub=None):
    img = np.full((H, W, 3), 25, np.uint8)
    items, y = [], 360
    for i, z in enumerate(zeilen):
        items.append(((160, y), z, 46 if i == 0 else 34, i == 0, (255, 255, 255), None))
        y += 90 if i == 0 else 56
    if sub:
        items.append(((160, H - 120), sub, 26, False, (190, 190, 190), None))
    return texte(img, items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zuordnung", default=str(ROOT / "results/metrics/route/zeitabgleich_strecke1.csv"))
    ap.add_argument("--pruefliste", default=str(ROOT / "results/metrics/route/pruefliste_strecke1.csv"))
    ap.add_argument("--quote", default=str(ROOT / "results/metrics/erkennungsquote_strecke1_modelle.csv"))
    ap.add_argument("--modell", default="E11")
    ap.add_argument("--vergleich", nargs="+", default=["E11", "E9", "E3"])
    ap.add_argument("--abschnitte", type=int, default=5)
    ap.add_argument("--sekunden", type=float, default=10)
    ap.add_argument("--verhaeltnis", nargs=2, type=float, default=[0.85, 1.2])
    ap.add_argument("--ziel", default=str(VIDEOS_DIR / "output" / "demo_strecke1_vergleich.mp4"))
    ap.add_argument("--crf", type=int, default=21)
    args = ap.parse_args()

    with open(args.zuordnung, encoding="utf-8-sig") as f:
        zu = list(csv.DictReader(f, delimiter=";"))
    tag = np.array([zahl(r["tag_s"]) for r in zu])
    sp = {c: np.array([zahl(r[c]) for r in zu]) for _, c, _, _ in FAHRTEN}
    mp = lambda t, c: float(np.interp(t, tag, sp[c]))
    with open(args.pruefliste, encoding="utf-8-sig") as f:
        ref = np.array(sorted(zahl(r["sek"]) for r in csv.DictReader(f, delimiter=";") if r["art"].startswith("ref")))
    with open(args.quote, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    q = {(r["Bedingung"], r["Modell"]): f"{zahl(r['Quote_%']):.0f}" for r in rows}
    n_ref = rows[0]["Ref_Schildstellen"]

    # Abschnitte: alle drei Fahrten etwa gleich schnell, dann die meisten Schildstellen
    L, lo, hi = args.sekunden, *args.verhaeltnis
    kand = []
    for s in np.arange(tag.min() + 1, tag.max() - L - 1, 0.5):
        v = [(mp(s + L, c) - mp(s, c)) / L for _, c, _, _ in FAHRTEN[1:]]
        if all(lo <= x <= hi for x in v):
            kand.append((int(((ref >= s + 1) & (ref <= s + L - 1)).sum()), -max(abs(x - 1) for x in v), float(s), v))
    kand.sort(reverse=True)
    wahl = []
    for n, _, s, v in kand:
        if len(wahl) >= args.abschnitte:
            break
        if all(abs(s - w[0]) >= L + 2 for w in wahl):
            wahl.append((s, v, n))
    wahl.sort()
    print("Abschnitte:", [(round(s, 1), [round(x, 2) for x in v], n) for s, v, n in wahl])

    from ultralytics import YOLO
    modell = YOLO(str(RUNS_DIR / GEWICHTE[args.modell]))
    anon = Anonymisierer()
    caps = {v: cv2.VideoCapture(str(VIDEOS_DIR / "input" / f"{v}.MOV")) for v, _, _, _ in FAHRTEN}

    import imageio_ffmpeg
    ziel = resolve_path(args.ziel)
    wr = imageio_ffmpeg.write_frames(str(ziel), (W, H), fps=30, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=8,
                                     output_params=["-crf", str(args.crf), "-preset", "medium"])
    wr.send(None)
    put = lambda img: wr.send(np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    titel = karte(["YOLO-Verkehrszeichenerkennung: eine Strecke, drei Lichtverhältnisse",
                   f"Modell {args.modell} (YOLO26s, 1280 px, robust trainiert), offline gerendert",
                   "Tag, Dämmerung und Nacht an derselben Stelle, alle in Originalgeschwindigkeit",
                   "Rechts unten: Lupe auf das jeweils sicherste erkannte Zeichen"],
                  "DLBAIPCV01_D, Berlin. Kennzeichen und Gesichter automatisch verpixelt.")
    for _ in range(30 * 5):
        put(titel)

    protokoll = []
    for k, (s, v, n) in enumerate(wahl, 1):
        starts = {c: mp(s, c) for _, c, _, _ in FAHRTEN}
        for vid, c, _, _ in FAHRTEN:
            caps[vid].set(cv2.CAP_PROP_POS_FRAMES, int(round(starts[c] * 30)))
        for j in range(int(L * 30)):
            grid = np.zeros((H, W, 3), np.uint8)
            lupen = []
            for idx, (vid, c, name, start) in enumerate(FAHRTEN):
                ok, fr = caps[vid].read()
                if not ok:
                    fr = np.zeros((1080, 1920, 3), np.uint8)
                r = modell.predict(fr, imgsz=1280, conf=0.4, device=0, verbose=False)[0]
                det = [(b, float(cf), int(cl)) for b, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
                fr = anon(vid, fr, [d[0] for d in det])
                if det:
                    b, cf, cl = max(det, key=lambda d: d[1])
                    cx, cy, sz = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(70, 2.6 * max(b[2] - b[0], b[3] - b[1]))
                    x0, y0 = int(max(0, cx - sz / 2)), int(max(0, cy - sz / 2))
                    crop = fr[y0:y0 + int(sz), x0:x0 + int(sz)]
                    lupen.append((name, cl, cf, crop.copy() if crop.size else None))
                else:
                    lupen.append((name, None, None, None))
                tile = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA)
                items = []
                for (x1, y1, x2, y2), cf, cl in det:
                    p1, p2 = (int(x1 / 2) - 3, int(y1 / 2) - 3), (int(x2 / 2) + 3, int(y2 / 2) + 3)
                    cv2.rectangle(tile, p1, p2, FARBEN[cl], 3)
                    ty = p1[1] - 30 if p1[1] > 40 else p2[1] + 6
                    items.append(((p1[0], ty), f"{KLASSEN[cl]} {de(cf, 2)}", 20, True, (255, 255, 255), tuple(FARBEN[cl][::-1])))
                tv = starts[c] + j / 30
                uhr = (datetime.strptime(start, "%d.%m.%Y %H:%M:%S") + timedelta(seconds=tv)).strftime("%d.%m. %H:%M:%S")
                items.append(((14, 10), f"{name}  {uhr}", 28, True, (255, 255, 255), (0, 0, 0)))
                tile = texte(tile, items)
                rr, cc = divmod(idx, 2)
                grid[rr * TH:(rr + 1) * TH, cc * TW:(cc + 1) * TW] = tile
            # Feld rechts unten: Kopf, Lupen, Quoten
            panel = np.full((TH, TW, 3), 30, np.uint8)
            LS = 270
            for i, (name, cl, cf, crop) in enumerate(lupen):
                x0 = 30 + i * (LS + 30)
                if crop is not None:
                    panel[90:90 + LS, x0:x0 + LS] = cv2.resize(crop, (LS, LS), interpolation=cv2.INTER_CUBIC)
                    cv2.rectangle(panel, (x0, 90), (x0 + LS - 1, 90 + LS - 1), FARBEN[cl], 3)
                else:
                    cv2.rectangle(panel, (x0, 90), (x0 + LS - 1, 90 + LS - 1), (80, 80, 80), 2)
            items = [((30, 20), f"Abschnitt {k} von {len(wahl)}   Modell {args.modell}", 30, True, (255, 255, 255), None)]
            for i, (name, cl, cf, crop) in enumerate(lupen):
                x0 = 30 + i * (LS + 30)
                items.append(((x0, 62), name, 22, True, (255, 255, 255), None))
                items.append(((x0, 90 + LS + 8), f"{KLASSEN[cl]} {de(cf, 2)}" if cl is not None else "keine Erkennung", 22, False,
                              (230, 230, 230), None))
            items.append(((30, 420), f"Wiedergefundene Schildstellen ({n_ref} aus der Tagfahrt): Tag / Dämmerung / Nacht", 20, True, (255, 255, 255), None))
            for i, mm in enumerate(args.vergleich):
                items.append(((30, 452 + i * 26), f"{mm}:  {q.get(('Tag', mm))} %  /  {q.get(('Dämmerung', mm))} %  /  {q.get(('Nacht', mm))} %", 20, False,
                              (230, 230, 230), None))
            grid[TH:, TW:] = texte(panel, items)
            put(grid)
        protokoll.append({"abschnitt": k, "tag_start_s": s, "verhaeltnis_daemmerung_nacht": v, "schildstellen": n,
                          "start_s": {c: round(starts[c], 2) for _, c, _, _ in FAHRTEN}})
        print(f"Abschnitt {k}: Tag {s:.1f} s, Verhältnis {[round(x, 2) for x in v]}", flush=True)

    summ = {name: json.loads((METRICS_DIR / f"video_{vid}_{args.modell}_summary.json").read_text(encoding="utf-8"))
            for vid, _, name, _ in FAHRTEN}
    ende = karte(["Ergebnis Strecke 1",
                  *[f"Wiedergefundene Schildstellen {mm}: {q.get(('Tag', mm))} / {q.get(('Dämmerung', mm))} / {q.get(('Nacht', mm))} %"
                    for mm in args.vergleich],
                  "Latenz " + args.modell + " je Bild: " + ", ".join(f"{n} {de(summ[n]['latenz_ms_mittel'])} ms" for _, _, n, _ in FAHRTEN)],
                 "Tag / Dämmerung / Nacht. Referenz modellbasiert, Annotation in CVAT folgt.")
    for _ in range(30 * 5):
        put(ende)
    wr.close()
    meta = {"fassung": 4, "modell": args.modell, "abschnitte": protokoll,
            "anonymisierung": {"kennzeichen_boxen": anon.n_kz, "gesicht_boxen": anon.n_gesicht}}
    ziel.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Demo-Video: {ziel}", meta["anonymisierung"])


if __name__ == "__main__":
    main()
