"""Fasst mehrere mit extract_frames.py erzeugte Ordner zu einer gemeinsamen Annotationsmenge zusammen.

Ergebnis: ein Ordner mit images/test, labels/test, data.yaml sowie bilder.zip und
vorannotation_yolo11.zip für EINE gemeinsame CVAT-Aufgabe.

Aufruf:
  python src/merge_frames.py --quellen data/eigene_annotation/IMG_9718 data/eigene_annotation/IMG_9719 ^
         --ziel data/eigene_annotation/videotest_01-10
"""

from __future__ import annotations

import argparse
import shutil
import zipfile

import yaml

from common import SUPER_CLASSES, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quellen", nargs="+", required=True)
    ap.add_argument("--ziel", required=True)
    args = ap.parse_args()

    ziel = resolve_path(args.ziel)
    img_dir, lbl_dir = ziel / "images" / "test", ziel / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    bilder = []
    for q in args.quellen:
        q = resolve_path(q)
        for img in sorted((q / "images" / "test").glob("*.jpg")):
            shutil.copy2(img, img_dir / img.name)
            lbl = q / "labels" / "test" / f"{img.stem}.txt"
            (lbl_dir / f"{img.stem}.txt").write_text(lbl.read_text(encoding="utf-8") if lbl.exists() else "",
                                                    encoding="utf-8")
            bilder.append(img_dir / img.name)

    y = {"path": str(ziel.resolve()), "train": "images/test", "val": "images/test", "test": "images/test",
         "names": {i: n for i, n in enumerate(SUPER_CLASSES)}}
    (ziel / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    with zipfile.ZipFile(ziel / "bilder.zip", "w", zipfile.ZIP_STORED) as z:
        for f in bilder:
            z.write(f, f.name)
    with zipfile.ZipFile(ziel / "vorannotation_yolo11.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(SUPER_CLASSES) + "\n")
        z.writestr("obj.data", f"classes = {len(SUPER_CLASSES)}\nnames = data/obj.names\ntrain = data/train.txt\n")
        z.writestr("train.txt", "\n".join(f"data/obj_train_data/{f.name}" for f in bilder) + "\n")
        for f in bilder:
            z.writestr(f"obj_train_data/{f.stem}.txt", (lbl_dir / f"{f.stem}.txt").read_text(encoding="utf-8"))
    n_box = sum(len([l for l in (lbl_dir / f"{f.stem}.txt").read_text().splitlines() if l.strip()]) for f in bilder)
    print(f"{len(bilder)} Bilder mit {n_box} vorgeschlagenen Boxen in {ziel}")


if __name__ == "__main__":
    main()
