"""Übernimmt die in CVAT korrigierte Annotation (Export im Format "YOLO 1.1") als eigene Testmenge.

Aufruf:
  python src/import_cvat.py --zip <Pfad>/task_export.zip --ziel data/eigene_annotation/fahrt_tag_stadt
Danach:
  python src/evaluate_model.py --weights runs/train/E3_s1280/weights/best.pt ^
         --data data/eigene_annotation/fahrt_tag_stadt/data.yaml --name eigenvideo_tag_E3
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from common import resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--ziel", required=True, help="Ordner, den extract_frames.py angelegt hat")
    args = ap.parse_args()

    ziel = resolve_path(args.ziel)
    lbl_dir = ziel / "labels" / "test"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    bilder = {p.stem for p in (ziel / "images" / "test").glob("*.jpg")}
    n, boxen = 0, 0
    with zipfile.ZipFile(resolve_path(args.zip)) as z:
        names = z.read("obj.names").decode("utf-8").split()
        if names != ["prohibitory", "danger", "mandatory", "other"]:
            print(f"WARNUNG: Klassenreihenfolge in CVAT ist {names}; sie muss prohibitory, danger, mandatory, other sein")
        for info in z.infolist():
            p = Path(info.filename)
            if p.suffix == ".txt" and p.parent.name.startswith("obj_") and p.stem in bilder:
                text = z.read(info).decode("utf-8")
                (lbl_dir / p.name).write_text(text, encoding="utf-8")
                n += 1
                boxen += sum(1 for line in text.splitlines() if line.strip())
    print(f"{n} Label-Dateien mit {boxen} Boxen übernommen nach {lbl_dir}")


if __name__ == "__main__":
    main()
