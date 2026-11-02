"""Verpixelt Kennzeichen und Gesichter in den 240 Testbildern vor dem Upload nach CVAT (Original vom 05.10.2026,
im Arbeitsverzeichnis ausgefuehrt). Bereiche, die einen Schild-Vorschlag ueberlappen, blieben dabei unverpixelt.
Die Originalbilder werden in data/eigene_annotation/Archiv/strecke1_bilder_original gesichert.
"""
import cv2, numpy as np, zipfile, shutil
from pathlib import Path
from ultralytics import YOLO
base = Path(r"data\eigene_annotation\strecke1"); img_dir = base / "images" / "test"; lbl_dir = base / "labels" / "test"
orig = Path(r"data\eigene_annotation\Archiv\strecke1_bilder_original"); orig.mkdir(parents=True, exist_ok=True)
kz = YOLO(r"weights\anonym\kennzeichen_yolo11s.pt")
fd = cv2.FaceDetectorYN.create(r"weights\anonym\gesicht_yunet_2023mar.onnx", "", (1920, 1080), 0.6, 0.3, 5000)
def iou_any(b, signs):
    for s in signs:
        ix = max(0, min(b[2], s[2]) - max(b[0], s[0])); iy = max(0, min(b[3], s[3]) - max(b[1], s[1]))
        if ix * iy > 0: return True
    return False
n_kz = n_g = n_skip = 0; files = sorted(img_dir.glob("*.jpg"))
for f in files:
    if not (orig / f.name).exists(): shutil.copy2(f, orig / f.name)
    img = cv2.imread(str(orig / f.name)); h, w = img.shape[:2]
    signs = []
    lf = lbl_dir / f"{f.stem}.txt"
    if lf.exists():
        for line in lf.read_text().split("\n"):
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                signs.append(((cx - bw/2) * w, (cy - bh/2) * h, (cx + bw/2) * w, (cy + bh/2) * h))
    boxes = [tuple(b) for b in kz.predict(img, imgsz=1280, conf=0.25, device=0, verbose=False)[0].boxes.xyxy.tolist()]
    n_kz += len(boxes)
    fd.setInputSize((w, h)); _, faces = fd.detect(img)
    if faces is not None:
        for x, y, fw, fh in faces[:, :4]: boxes.append((x, y, x + fw, y + fh)); n_g += 1
    for (x1, y1, x2, y2) in boxes:
        dx, dy = (x2 - x1) * 0.2 + 4, (y2 - y1) * 0.3 + 4
        a, b, c, d = int(max(0, x1 - dx)), int(max(0, y1 - dy)), int(min(w, x2 + dx)), int(min(h, y2 + dy))
        if iou_any((a, b, c, d), signs): n_skip += 1; continue
        if c - a < 2 or d - b < 2: continue
        roi = img[b:d, a:c]; k = max(1, min(roi.shape[:2]) // 6)
        img[b:d, a:c] = cv2.resize(cv2.resize(roi, (max(1, roi.shape[1] // k), max(1, roi.shape[0] // k))), (c - a, d - b), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
z = base / "bilder.zip"
if z.exists(): shutil.move(str(z), orig / "bilder_original.zip")
with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zz:
    for f in files: zz.write(f, f.name)
print(f"{len(files)} Bilder anonymisiert: {n_kz} Kennzeichen, {n_g} Gesichter, {n_skip} Boxen wegen Überlappung mit Schild-Vorschlag ausgelassen")
