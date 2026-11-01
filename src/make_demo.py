"""Schneidet aus den Ausgabevideos der Video-Pipeline ein kurzes Demo-Video (H.264).

Aus jedem Ergebnisvideo werden die Abschnitte mit den meisten bestätigten Erkennungen
(Spalte "bestaetigt" im Frame-Protokoll) gewählt, mit Titelkarten verbunden und mit
ffmpeg (imageio-ffmpeg) als H.264 gespeichert, damit das Video überall abspielbar ist.

Aufruf:
  python src/make_demo.py --namen IMG_9719_E9 IMG_9723_E9 --titel "Tag, Stadtverkehr" "Nacht" ^
         --abschnitte 6 5 --sekunden 6 --ziel videos/output/demo_E9_kurz.mp4
"""

from __future__ import annotations

import argparse
import csv
import json

import cv2
import numpy as np

from common import METRICS_DIR, VIDEOS_DIR, resolve_path


def best_windows(frames_csv, n_frames: int, win: int, count: int) -> list[tuple[int, int]]:
    score = np.zeros(n_frames)
    with open(frames_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            i = int(r["frame"])
            if i < n_frames and r.get("status") == "verarbeitet" and r.get("bestaetigt"):
                score[i] = 1 + len(r["bestaetigt"].split("|"))
    csum = np.concatenate([[0], np.cumsum(score)])
    sums = csum[win:] - csum[:-win]
    chosen = []
    order = np.argsort(-sums)
    for s in order:
        if sums[s] <= 0 or len(chosen) >= count:
            break
        if all(abs(s - c) >= win for c in chosen):
            chosen.append(int(s))
    return sorted((c, c + win) for c in chosen)


def title_card(text: str, sub: str, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 25, np.uint8)
    for t, y, sc in ((text, h // 2 - 20, 2.0), (sub, h // 2 + 50, 1.0)):
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        cv2.putText(img, t, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namen", nargs="+", required=True, help="Namen wie bei run_video_pipeline --name")
    ap.add_argument("--titel", nargs="+", required=True)
    ap.add_argument("--abschnitte", nargs="+", type=int, required=True)
    ap.add_argument("--sekunden", type=float, default=6.0)
    ap.add_argument("--ziel", default=str(VIDEOS_DIR / "output" / "demo_kurz.mp4"))
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    import imageio_ffmpeg

    ziel = resolve_path(args.ziel)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fps, size = 30, (1920, 1080)
    writer = imageio_ffmpeg.write_frames(str(ziel), size, fps=fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=8,
                                         output_params=["-crf", str(args.crf), "-preset", "medium"])
    writer.send(None)

    def put(img):
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size)
        writer.send(np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    for _ in range(int(fps * 3)):
        put(title_card("YOLO-Verkehrszeichenerkennung", "DLBAIPCV01_D, Modell E9 (YOLO26s, robust trainiert)", size))
    protokoll = []
    for name, titel, n_abs in zip(args.namen, args.titel, args.abschnitte):
        video = VIDEOS_DIR / "output" / f"{name}_yolo.mp4"
        summ = json.loads((METRICS_DIR / f"video_{name}_summary.json").read_text(encoding="utf-8"))
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wins = best_windows(METRICS_DIR / f"video_{name}_frames.csv", n, int(args.sekunden * fps), n_abs)
        sub = (f"Latenz im Mittel {summ['latenz_ms_mittel']} ms, P95 {summ['latenz_ms_p95']} ms, "
               f"{summ['fps_verarbeitung']} FPS")
        for _ in range(int(fps * 2.5)):
            put(title_card(titel, sub, size))
        for a, b in wins:
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            for _ in range(a, b):
                ok, fr = cap.read()
                if not ok:
                    break
                put(fr)
        cap.release()
        protokoll.append({"name": name, "abschnitte_frames": wins})
    writer.close()
    (ziel.with_suffix(".json")).write_text(json.dumps(protokoll, indent=1), encoding="utf-8")
    print(f"Demo-Video: {ziel}")


if __name__ == "__main__":
    main()
