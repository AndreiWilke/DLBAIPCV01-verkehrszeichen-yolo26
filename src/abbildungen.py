"""Erzeugt Abb. 1 bis 3 des Projektberichts aus results/metrics nach results/figures (Aufruf im Ordner 02_Code:
python src/abbildungen.py)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
M, F = os.path.join(B, "metrics"), os.path.join(B, "figures")
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
FARBE = {"E3": "#9AA5B1", "E9": "#3E7CB1", "E11": "#D1495B"}

def lies(name, sep=";"):
    with open(os.path.join(M, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=sep))

from matplotlib.ticker import FuncFormatter
KOMMA = FuncFormatter(lambda v, p: f"{v:.1f}".replace(".", ","))

def zahl(s):
    return float(str(s).replace(",", "."))

# Abbildung 1: Pipeline
fig, ax = plt.subplots(figsize=(6.3, 2.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, kopf, text, farbe="#EEF3F8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=farbe, ec="#33415C", lw=0.9))
    ax.text(x + w / 2, y + h - 3.2, kopf, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(x + w / 2, y + (h - 5) / 2, text, ha="center", va="center", fontsize=7.4, linespacing=1.25)
def pfeil(x1, y1, x2, y2, txt=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#33415C", lw=1.0))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.8, txt, ha="center", fontsize=6.8, color="#33415C")
box(0.5, 20, 18, 17, "Kamera-Thread", "liest das Video\nim Aufnahmetakt\n(30 Bilder/s,\n1.920 × 1.080 px)")
box(25, 20, 18, 17, "Puffer", "hält nur das\nneueste Bild;\nnicht abgeholte\nBilder: verworfen", "#FFF4E0")
box(49.5, 20, 22, 17, "Erkennungsschleife", "YOLO26s (E11)\nEingabe 1.280 px\nconf 0,4\nLatenzmessung je Bild")
box(78, 20, 21, 17, "Anzeige-Thread", "schreibt im Takt\ndas zuletzt fertige\nErgebnisbild\n(Video, Anzeige)")
pfeil(19.3, 28.5, 24.2, 28.5); pfeil(43.8, 28.5, 48.7, 28.5); pfeil(72.3, 28.5, 77.2, 28.5)
box(49.5, 1, 22, 13, "Stabilisierung", "Obergruppe bestätigt,\nwenn in 2 der letzten\n3 Bilder erkannt", "#EAF5EA")
box(78, 1, 21, 13, "Protokoll", "Latenz, Detektionen,\nverworfene Bilder\n(CSV und JSON)", "#F2F2F2")
pfeil(60.5, 19.2, 60.5, 14.8); pfeil(72.3, 7.5, 77.2, 7.5)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(F, "bericht_abb1_pipeline.png"), dpi=300); plt.close(fig)

# Abbildung 2: Robustheit
bed = ["original", "dunkel", "nacht", "ueberbelichtung", "nebel", "regen", "bewegungsunschaerfe", "rauschen", "jpeg_kompression"]
lab = ["Original", "Dunkel", "Nacht", "Überbe-\nlichtung", "Nebel", "Regen", "Bewegungs-\nunschärfe", "Rauschen", "JPEG Q15"]
fig, ax = plt.subplots(figsize=(6.3, 2.9))
for k, m in enumerate(["E3", "E9", "E11"]):
    rows = {r["bedingung"]: float(r["mAP50_95"]) for r in lies(f"robustheit_{m}.csv")}
    xs = [i + (k - 1) * 0.27 for i in range(len(bed))]
    ax.bar(xs, [rows[b] for b in bed], width=0.26, color=FARBE[m], label={"E3": "E3 (ohne Robustheits-Augmentierung)", "E9": "E9 (robust)", "E11": "E11 (robust + Farbe/Blendung)"}[m])
ax.set_xticks(range(len(bed))); ax.set_xticklabels(lab, fontsize=7.6)
ax.set_ylabel("mAP50-95"); ax.set_ylim(0.4, 1.0); ax.yaxis.set_major_formatter(KOMMA)
ax.grid(axis="y", color="#DDDDDD", lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7.4, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(F, "bericht_abb2_robustheit.png"), dpi=300); plt.close(fig)

# Abbildung 3: Recall nach Licht und Größe
pr = [r for r in lies("cvat_strecke1_pr.csv") if r["conf"] == "0,4"]
gr = [r for r in lies("cvat_strecke1_groesse.csv") if r["Bedingung"] == "alle"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1, 1.1]})
bedz = ["tag", "daemmerung", "nacht", "alle"]; bl = ["Tag", "Dämmerung", "Nacht", "Gesamt"]
grz = ["0-20", "20-32", "32-64", "64-max"]; gl = ["< 20", "20 bis < 32", "32 bis < 64", "≥ 64"]
for k, m in enumerate(["E3", "E9", "E11"]):
    v = {r["Bedingung"]: zahl(r["Recall"]) for r in pr if r["Modell"] == m}
    a1.bar([i + (k - 1) * 0.27 for i in range(4)], [v[b] for b in bedz], width=0.26, color=FARBE[m], label=m)
    g = {r["Groesse_px"]: zahl(r["Recall"]) for r in gr if r["Modell"] == m}
    a2.bar([i + (k - 1) * 0.27 for i in range(4)], [g[b] for b in grz], width=0.26, color=FARBE[m], label=m)
for a, l, t in [(a1, bl, "a) nach Lichtbedingung"), (a2, gl, "b) nach Zeichengröße (px, längere Seite)")]:
    a.set_xticks(range(4)); a.set_xticklabels(l, fontsize=7.6); a.set_ylim(0, 1); a.yaxis.set_major_formatter(KOMMA)
    a.set_title(t, fontsize=8.5, loc="left"); a.grid(axis="y", color="#DDDDDD", lw=0.6); a.set_axisbelow(True)
a1.set_ylabel("Recall (conf 0,4, IoU ≥ 0,5)")
a1.legend(fontsize=7.4, frameon=False, ncol=3, loc="upper right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(F, "bericht_abb3_recall.png"), dpi=300); plt.close(fig)
print("ok")
