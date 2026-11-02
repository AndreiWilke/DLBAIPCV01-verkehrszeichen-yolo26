# Code-Grundgerüst: YOLO-Verkehrszeichenerkennung (DLBAIPCV01_D)

Lauffähiges Python-Projekt für Datenaufbereitung, Training, Evaluation, Robustheitstests
und Video-Pipeline. Alle Befehle werden im Ordner `02_Code` in der PowerShell (VS Code-Terminal) ausgeführt.

Getestet am 29.09.2026 mit synthetischen Daten: in der Cloud (Python 3.11, OpenCV 4.13, CPU) und auf dem Zielrechner (Python 3.12, OpenCV 5.0, RTX 3060).
Zielrechner: Windows 11, Python 3.12, NVIDIA RTX 3060 (12 GB), Ryzen 7 5800X, 16 GB RAM.

## Ordnerstruktur

Code und kleine Ergebnisse liegen im Projektordner (OneDrive). Große Dateien liegen im
**Arbeitsverzeichnis `C:\IU\DLBAIPCV01`** außerhalb von OneDrive. Relative Pfade in den Befehlen
(`data/...`, `runs/...`, `videos/...`, `weights/...`) werden automatisch dorthin aufgelöst
(`common.resolve_path`). Ein anderes Arbeitsverzeichnis lässt sich über die Umgebungsvariable
`DLBAIPCV01_WORK` setzen.

```
02_Code/  (OneDrive)
├── configs/experiments.yaml      Versuchsreihe (Hyperparameter, One-Factor-at-a-Time)
├── results/metrics/              CSV/JSON für Tabellen im Bericht
├── results/figures/              Grafiken für den Bericht
├── results/screenshots/          Screenshots aus dem Demo-Video
├── report/                       Berichtsentwurf
└── src/
    ├── common.py                 Klassen, Obergruppen, Pfade, Systeminfo
    ├── check_setup.py            Umgebung prüfen
    ├── convert_gtsdb_to_yolo.py  GTSDB -> YOLO-Format + Datensatzstatistik
    ├── train_yolo.py             Einzeltraining
    ├── run_experiments.py        Versuchsreihe inkl. Testauswertung -> experimente.csv
    ├── evaluate_model.py         Precision, Recall, mAP50, mAP50-95, je Klasse, Confusion Matrix
    ├── robustness_tests.py       Licht- und Wetterbedingungen simulieren und auswerten
    ├── benchmark_speed.py        Latenz und FPS je Modell, Bildgröße, GPU/CPU
    ├── run_video_pipeline.py     Video-Pipeline mit Echtzeitsimulation und Stabilisierung
    └── extract_frames.py         Frames aus eigenem Video für die Annotation

C:\IU\DLBAIPCV01\  (Arbeitsverzeichnis, nicht in OneDrive)
├── .venv/                        Python-Umgebung
├── data/raw/                     GTSDB-Rohdaten (FullIJCNN2013)
├── data/processed/               YOLO-Datensätze
├── data/eigene_annotation/       annotierte Frames aus eigenen Fahrvideos
├── videos/input/                 eigene Fahrvideos (MP4)
├── videos/output/                Ergebnisvideos mit Bounding Boxes
├── runs/                         Trainings- und Evaluationsläufe (Gewichte, Plots)
└── weights/                      vortrainierte Ausgangsgewichte (yolo26n.pt usw.)
```

Python-Cache-Dateien werden über die Benutzervariable `PYTHONPYCACHEPREFIX` nach
`C:\IU\DLBAIPCV01\.pycache` umgeleitet, damit im Projektordner keine `__pycache__`-Ordner entstehen.

## Schritt 1 und 2: Umgebung (erledigt am 29.09.2026)

Die virtuelle Umgebung liegt bewusst **außerhalb von OneDrive** unter `C:\IU\DLBAIPCV01\.venv`,
weil PyTorch mit CUDA rund 5 GB und mehrere tausend Dateien umfasst und OneDrive sonst dauerhaft synchronisiert.

Installierter Stand (Details in `requirements_lock.txt`, Hardware in `results/metrics/systeminfo.json`):

| Komponente | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA / cuDNN | 12.8 / 9.19 |
| Ultralytics | 8.4.165 |
| OpenCV | 5.0.0 |
| GPU | NVIDIA GeForce RTX 3060, 12 GB |

Funktionstest auf der GPU mit synthetischen Daten bestanden: Konvertierung, Training (imgsz 1280, AMP, 4 DataLoader-Worker),
Testauswertung und Video-Pipeline mit 1080p (Latenz im Mittel 15,7 ms, P95 19,3 ms).

Umgebung in jeder neuen PowerShell aktivieren:

```powershell
cd "<Projektordner>/DLBAIPCV01_Projektpaket/02_Code"
C:\IU\DLBAIPCV01\.venv\Scripts\Activate.ps1
# falls die Aktivierung blockiert wird, einmalig:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

In VS Code: Strg+Umschalt+P, "Python: Select Interpreter", Pfad `C:\IU\DLBAIPCV01\.venv\Scripts\python.exe`.

Neuaufbau der Umgebung (falls nötig):

```powershell
py -3.12 -m venv C:\IU\DLBAIPCV01\.venv
C:\IU\DLBAIPCV01\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python src/check_setup.py          # muss cuda_available: True und die RTX 3060 zeigen
pip freeze > requirements_lock.txt
```

## Schritt 3: GTSDB herunterladen und konvertieren

1. `FullIJCNN2013.zip` von der Archivseite des Datensatzes laden
   (https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html,
   verlinkt von https://benchmark.ini.rub.de/gtsdb_dataset.html).
2. Entpacken nach `C:\IU\DLBAIPCV01\data\raw\FullIJCNN2013\` (900 `.ppm`-Bilder, `gt.txt`, `ReadMe.txt`).
3. Konvertieren:

```powershell
python src/convert_gtsdb_to_yolo.py --src data/raw/FullIJCNN2013 --mode super   # 4 Obergruppen
python src/convert_gtsdb_to_yolo.py --src data/raw/FullIJCNN2013 --mode full    # 43 Klassen
```

Aufteilung: Bilder 00000 bis 00599 (offizieller Trainingsteil) werden in train (85 %) und val (15 %, Seed 42)
geteilt, Bilder 00600 bis 00899 bilden ausschließlich die Testmenge. Die Statistik
(`results/metrics/datensatz_*`) liefert Klassenverteilung und Zeichengrößen für Kapitel 4.

## Schritt 4: Baseline trainieren

```powershell
python src/train_yolo.py --data data/processed/gtsdb_yolo_super/data.yaml --name E0_baseline_n1280 --device 0 --eval-test
```

Voreinstellungen: YOLO26n, imgsz 1280, 100 Epochen, Batch 8, Patience 30, Seed 42, **fliplr 0.0**.
Horizontales Spiegeln (Ultralytics-Standard 0.5) ist bei Verkehrszeichen fachlich falsch, weil es
Richtungszeichen vertauscht. Das Experiment E6 zeigt den Effekt als Negativkontrolle.
Bei DataLoader-Fehlern unter Windows `--workers 0` ergänzen.

## Schritt 5: Versuchsreihe (Hyperparameter-Optimierung)

```powershell
python src/run_experiments.py --config configs/experiments.yaml
# nur einzelne: --only E0_baseline_n1280 E1_n640     Schnelltest: --epochs 3
```

Ergebnis: `results/metrics/experimente.csv` (mAP, Precision, Recall, ms/Bild, FPS je Variante).

## Schritt 6: Robustheit unter Licht- und Wetterbedingungen

```powershell
python src/robustness_tests.py --weights runs/train/E0_baseline_n1280/weights/best.pt --data data/processed/gtsdb_yolo_super/data.yaml --imgsz 1280 --device 0
```

Bedingungen: dunkel, nacht, ueberbelichtung, nebel, regen, bewegungsunschaerfe, rauschen, jpeg_kompression.
Ergebnis: `results/metrics/robustheit_robust.csv` und `results/figures/robustheit_beispiele.jpg`.

## Schritt 7: Geschwindigkeit (Genauigkeit versus Echtzeit)

```powershell
python src/benchmark_speed.py --weights runs/train/E0_baseline_n1280/weights/best.pt runs/train/E3_s1280/weights/best.pt --imgsz 640 960 1280 --devices 0 cpu --images data/processed/gtsdb_yolo_super/images/test
```

## Schritt 8: Eigenes Fahrvideo

```powershell
# Demo-Video mit Echtzeitsimulation und zeitlicher Stabilisierung
python src/run_video_pipeline.py --weights runs/train/E0_baseline_n1280/weights/best.pt --source videos/input/fahrt_tag.mp4 --name fahrt_tag --realtime --stabilize 2 3 --device 0

# Frames extrahieren und vom Modell vorannotieren lassen (150 Frames: 2/3 mit, 1/3 ohne Erkennung)
python src/extract_frames.py --video videos/input/fahrt_tag.mp4 --every-sec 1 --vorannotieren runs/train/E3_s1280/weights/best.pt --auswahl 150 --device 0
# in CVAT prüfen und korrigieren (siehe unten), Export "YOLO 1.1", dann:
python src/import_cvat.py --zip <Pfad zum CVAT-Export>.zip --ziel data/eigene_annotation/fahrt_tag
python src/evaluate_model.py --weights runs/train/E3_s1280/weights/best.pt --data data/eigene_annotation/fahrt_tag/data.yaml --name eigenvideo_tag_E3
```

### Mehrere Videos zu einer CVAT-Aufgabe zusammenfassen und Demo-Video schneiden

```powershell
python src/merge_frames.py --quellen data/eigene_annotation/einzeln/IMG_9718 data/eigene_annotation/einzeln/IMG_9719 --ziel data/eigene_annotation/videotest_01-10
python src/make_demo.py --namen IMG_9719_E9 IMG_9723_E9 --titel "Tag, Stadtverkehr Berlin" "Nacht" --abschnitte 6 5 --sekunden 6 --ziel videos/output/demo_E9_kurz.mp4
```

`make_demo.py` wählt die Abschnitte mit den meisten bestätigten Erkennungen und speichert als H.264
(benötigt `pip install imageio-ffmpeg`).

Hinweis Echtzeitmodus (02.10.2026): Die erste Fassung der Pipeline las, erkannte und schrieb nacheinander in einer
Schleife. Dekodieren und Schreiben belasteten damit das Zeitbudget, die GPU taktete bei der stoßweisen Last herunter
(etwa 825 statt 2.030 MHz), und 67 % der Frames wurden verworfen. Seit der Umstellung auf Kamera-Thread,
Erkennungsschleife und Anzeige-Thread werden über 99 % der Frames verarbeitet.

### Modellgestützte Annotation in CVAT

1. In CVAT eine Aufgabe anlegen, Labels in genau dieser Reihenfolge: `prohibitory`, `danger`, `mandatory`, `other`.
2. `bilder.zip` aus dem Ausgabeordner als Daten hochladen.
3. Menü der Aufgabe: "Upload annotations", Format "YOLO 1.1", Datei `vorannotation_yolo11.zip`.
4. Jeden Frame prüfen: falsche Boxen löschen, Klassen korrigieren, Rahmen anpassen, **übersehene Zeichen ergänzen**.
   Ohne diese Prüfung würde das Modell an seinen eigenen Vorschlägen gemessen (Zirkelschluss).
5. Export "YOLO 1.1" und mit `import_cvat.py` übernehmen.


## Wiederholungsläufe mit weiteren Seeds (01.11.2026)

E0, E3 und E9 wurden mit den Seeds 1 und 2 wiederholt (Originalläufe: Seed 42), um die Streuung zwischen
Trainingsläufen abzuschätzen. Die Datenaufteilung bleibt gleich, nur der Trainings-Seed ändert sich.

```powershell
python src/run_experiments.py --config configs/experiments_seeds.yaml --workers 2
python src/robustness_tests.py --weights runs/train/E3_s1280_seed1/weights/best.pt --data data/processed/gtsdb_yolo_super/data.yaml --imgsz 1280 --device 0 --name E3_seed1
```

Ergebnis: zusätzliche Zeilen `*_seed1` und `*_seed2` in `results/metrics/experimente.csv` sowie
`results/metrics/robustheit_E3_seed1.csv` usw.

## Streckenabgleich der eigenen Fahrten

`src/route_features.py` berechnet je Video alle 0,5 s einen ResNet18-Merkmalsvektor (Ergebnis
`results/metrics/route/route_<video>.npz`), `src/route_match.py` die Distanzmatrizen gegen die Tagfahrt. Beide Skripte
liefen im Arbeitsverzeichnis. Den Viterbi-Pfad durch die gefilterte Matrix berechnete ein einmaliges Skript, das nicht
gesichert wurde. Sein Ergebnis liegt als `results/metrics/route/zeitabgleich_strecke1.csv` vor.

## Eigene Testmenge

`src/anonym_cvat.py` verpixelte die 240 Testbilder vor dem Upload nach CVAT, `src/det_verlauf.py` schrieb den
Erkennungsverlauf je Fahrt (`results/metrics/route/det_*.csv`). Die Grundwahrheit liegt als
`results/cvat/cvat_annotation_api.txt` bei (nur Rahmenkoordinaten und Klassen, keine Bilder). `src/eval_teilmengen.py`
vergleicht den Recall an den 50 Schildstellen mit den 30 Zufallspunkten (`results/metrics/cvat_strecke1_teilmengen.csv`).

## Hinweis Albumentations (30.09.2026)

Für den Verbesserungsschritt (E9, E10, `aug_preset: robust`) wurde `albumentations` nachinstalliert. Das Paket zieht
`opencv-python-headless` mit, das `opencv-python` überschreibt (dann fehlt `cv2.imshow`). Danach deshalb:

```powershell
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python==5.0.0.93
```

Die Experimente E0 bis E8 liefen ohne Albumentations, also ohne die Ultralytics-Standardtransformationen (Blur, CLAHE usw. mit je 1 %).

Hinweis Datenschutz: Die Aufgabenstellung verlangt die Anonymisierung personenbezogener Daten.
Kennzeichen und Gesichter vor Upload des Demo-Videos und vor Verwendung von Screenshots unkenntlich machen.

## Hinweis zu diesem Repository

Code, Konfigurationen und Messdateien zum Projektbericht DLBAIPCV01_D (IU Internationale Hochschule, Projekt: Computer Vision). Die eigenen Fahrvideos, Bildausschnitte daraus, Modellgewichte und Trainingsläufe sind nicht enthalten (Datenschutz und Dateigröße). Die Auswertungen in `results/metrics` sind vollständig.
