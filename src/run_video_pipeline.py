"""Video-Pipeline zur (simulierten) Echtzeit-Verkehrszeichenerkennung.

Architektur im Echtzeitmodus (--realtime), angelehnt an eine Fahrzeugkamera:
  Kamera-Thread   liest das Video im Takt der Aufnahme (z. B. 30 FPS) und legt immer nur das
                  NEUESTE Bild bereit (Puffer mit einem Platz). Bilder, die die Erkennung nicht
                  rechtzeitig abholt, gelten als verworfen.
  Erkennung       holt das jeweils neueste Bild, führt YOLO aus und zeichnet das Ergebnis.
  Anzeige/Writer  schreibt im Aufnahmetakt jeweils das zuletzt fertige Ergebnisbild (wie ein
                  Display im Fahrzeug); Schreiben und Dekodieren belasten die Erkennung nicht.

Liest ein Fahrvideo (oder eine Kamera) frameweise mit OpenCV, führt die YOLO-Inferenz aus,
zeichnet Bounding Boxes, Klasse und Konfidenz ein und schreibt ein neues MP4-Video.
Zusätzlich:
  --realtime     simuliert eine echte Kamera: Frames kommen im Takt der Quell-FPS an.
                 Ist die Verarbeitung zu langsam, werden Frames verworfen (wie im Fahrzeug).
  --stabilize    zeitliche Stabilisierung: ein Zeichen gilt erst als "bestätigt", wenn es in
                 K der letzten N Frames erkannt wurde (reduziert Flackern und Fehlalarme).
Protokolle für den Bericht:
  results/metrics/video_<name>_frames.csv   Latenz und Detektionen je Frame
  results/metrics/video_<name>_summary.json FPS, Latenz (Mittel, Median, P95), verworfene Frames

Aufruf:
  python src/run_video_pipeline.py --weights runs/train/E0_baseline_n1280/weights/best.pt ^
         --source videos/input/fahrt_tag.mp4 --name fahrt_tag --realtime
  Kamera statt Datei: --source 0     Live-Anzeige: --show  (Taste q beendet)
"""

from __future__ import annotations

import argparse
import csv
import queue
import statistics
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from common import METRICS_DIR, VIDEOS_DIR, resolve_path, save_json, system_info

PALETTE = [(56, 56, 255), (0, 194, 255), (255, 149, 0), (0, 200, 83), (200, 0, 200), (255, 255, 0)]


def draw(frame, boxes, names, fps_avg, lat_ms, frame_idx, dropped, confirmed):
    for (x1, y1, x2, y2), cls, conf in boxes:
        c = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"{names[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), c, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    info = [f"FPS {fps_avg:5.1f}", f"Latenz {lat_ms:5.1f} ms", f"Frame {frame_idx}"]
    if dropped is not None:
        info.append(f"verworfen {dropped}")
    if confirmed is not None:
        info.append("bestaetigt: " + (", ".join(names[c] for c in sorted(confirmed)) or "-"))
    pad_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for t in info) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pad_w, 26 * len(info) + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(info):
        cv2.putText(frame, t, (10, 26 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def gpu_clock_reader():
    """Liefert eine Funktion, die den aktuellen GPU-Takt (MHz) zurückgibt, oder None."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="Videodatei oder Kameraindex (0)")
    ap.add_argument("--name", default=None, help="Name für Ausgabedateien")
    ap.add_argument("--out", default=None, help="Ausgabevideo (Standard videos/output/<name>_yolo.mp4)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true", help="FP16-Inferenz auf der GPU")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--stabilize", nargs=2, type=int, metavar=("K", "N"), default=None,
                    help="z. B. 2 3 = bestätigt, wenn in 2 der letzten 3 Frames erkannt")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=10, help="Aufwärm-Inferenzen vor der Messung")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    args.weights = str(resolve_path(args.weights))
    src = int(args.source) if str(args.source).isdigit() else str(resolve_path(args.source))
    name = args.name or (Path(str(args.source)).stem if not isinstance(src, int) else "kamera")
    out_path = resolve_path(args.out) if args.out else VIDEOS_DIR / "output" / f"{name}_yolo.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Quelle nicht lesbar: {args.source}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    model = YOLO(args.weights)
    names = model.names
    pkw = dict(imgsz=args.imgsz, conf=args.conf, verbose=False, half=args.half)
    if args.device is not None:
        pkw["device"] = args.device
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(args.warmup):
        model.predict(dummy, **pkw)
    clock = gpu_clock_reader()

    history: deque = deque(maxlen=args.stabilize[1] if args.stabilize else 1)
    lat_hist: deque = deque(maxlen=30)
    frame_rows, latencies, clocks = [], [], []
    cls_counter: Counter = Counter()
    state = {"annotated": None, "processed": 0}
    lock = threading.Lock()

    def process(idx: int, frame: np.ndarray, dropped: int | None) -> np.ndarray:
        t0 = time.perf_counter()
        r = model.predict(frame, **pkw)[0]
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy().astype(int) if len(b) else np.zeros((0, 4), int)
        cls = b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, int)
        conf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)
        lat_hist.append(lat_ms)
        if clock:
            clocks.append(clock())
        confirmed = None
        if args.stabilize:
            history.append(set(cls.tolist()))
            confirmed = {c for c in set().union(*history) if sum(c in s for s in history) >= args.stabilize[0]}
        for c in cls:
            cls_counter[names[int(c)]] += 1
        fps_avg = 1000 / (sum(lat_hist) / len(lat_hist))
        boxes = [((int(x1), int(y1), int(x2), int(y2)), int(c), float(p))
                 for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf)]
        annotated = draw(frame, boxes, names, fps_avg, lat_ms, idx, dropped, confirmed)
        frame_rows.append({
            "frame": idx, "status": "verarbeitet", "latenz_ms": round(lat_ms, 2),
            "pre_ms": round(r.speed.get("preprocess", 0), 2), "inf_ms": round(r.speed.get("inference", 0), 2),
            "post_ms": round(r.speed.get("postprocess", 0), 2), "anzahl": len(boxes),
            "klassen": "|".join(names[c] for _, c, _ in boxes),
            "konfidenzen": "|".join(f"{p:.2f}" for _, _, p in boxes),
            "bestaetigt": "|".join(names[c] for c in sorted(confirmed)) if confirmed is not None else "",
        })
        return annotated

    t_start = time.perf_counter()
    total = 0
    if args.realtime:
        slot: dict = {"frame": None, "idx": -1}
        done = threading.Event()
        ticks: queue.Queue = queue.Queue()

        def camera():
            nonlocal total
            i = 0
            while True:
                ok, fr = cap.read()
                if not ok or (args.max_frames and i >= args.max_frames):
                    break
                if not isinstance(src, int):  # Datei im Aufnahmetakt ausgeben
                    delay = t_start + i / fps_src - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                with lock:
                    slot["frame"], slot["idx"] = fr, i
                ticks.put(i)
                i += 1
            total = i
            done.set()
            ticks.put(None)

        def display():
            while True:
                i = ticks.get()
                if i is None:
                    break
                with lock:
                    img = state["annotated"]
                if img is not None:
                    writer.write(img)

        th_cam = threading.Thread(target=camera, daemon=True)
        th_disp = threading.Thread(target=display, daemon=True)
        th_cam.start()
        th_disp.start()
        last_idx = -1
        while True:
            with lock:
                fr, idx = slot["frame"], slot["idx"]
                slot["frame"] = None
            if fr is None:
                if done.is_set():
                    break
                time.sleep(0.001)
                continue
            dropped = idx - last_idx - 1 if last_idx >= 0 else idx
            state.setdefault("dropped", 0)
            state["dropped"] += max(0, dropped)
            ann = process(idx, fr, state["dropped"])
            with lock:
                state["annotated"] = ann
            state["processed"] += 1
            last_idx = idx
            if args.show:
                cv2.imshow("Verkehrszeichenerkennung", ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        th_cam.join()
        th_disp.join()
        dropped_total = total - state["processed"]
    else:
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok or (args.max_frames and i >= args.max_frames):
                break
            ann = process(i, fr, None)
            writer.write(ann)
            state["processed"] += 1
            if args.show:
                cv2.imshow("Verkehrszeichenerkennung", ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            i += 1
        total, dropped_total = i, 0

    wall = time.perf_counter() - t_start
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    fields = ["frame", "status", "latenz_ms", "pre_ms", "inf_ms", "post_ms", "anzahl", "klassen",
              "konfidenzen", "bestaetigt"]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / f"video_{name}_frames.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        wr.writeheader()
        wr.writerows(sorted(frame_rows, key=lambda r: r["frame"]))

    processed = state["processed"]
    lat_sorted = sorted(latencies)
    summary = {
        "quelle": str(args.source), "gewichte": args.weights, "imgsz": args.imgsz, "conf": args.conf,
        "half": args.half, "quelle_fps": round(fps_src, 2), "aufloesung": f"{w}x{h}",
        "frames_gesamt": total, "frames_verarbeitet": processed, "frames_verworfen": dropped_total,
        "anteil_verarbeitet": round(processed / total, 3) if total else None,
        "latenz_ms_mittel": round(statistics.mean(latencies), 2) if latencies else None,
        "latenz_ms_median": round(statistics.median(latencies), 2) if latencies else None,
        "latenz_ms_p95": round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 2) if latencies else None,
        "fps_verarbeitung": round(processed / sum(latencies) * 1000, 1) if latencies else None,
        "fps_ende_zu_ende": round(processed / wall, 1) if wall else None,
        "echtzeitfaehig": (processed / sum(latencies) * 1000 >= fps_src) if latencies else None,
        "gpu_takt_mhz_mittel": round(statistics.mean(clocks)) if clocks else None,
        "realtime_simulation": args.realtime, "stabilisierung": args.stabilize,
        "detektionen_je_klasse": dict(cls_counter), "ausgabevideo": str(out_path), "system": system_info(),
    }
    save_json(summary, METRICS_DIR / f"video_{name}_summary.json")
    print(f"\nAusgabevideo: {out_path}")
    for k in ("frames_gesamt", "frames_verarbeitet", "frames_verworfen", "latenz_ms_mittel", "latenz_ms_p95",
              "fps_verarbeitung", "fps_ende_zu_ende", "gpu_takt_mhz_mittel", "echtzeitfaehig"):
        print(f"  {k}: {summary[k]}")


if __name__ == "__main__":
    main()
