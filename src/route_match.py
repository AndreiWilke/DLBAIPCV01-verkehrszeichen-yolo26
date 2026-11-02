"""Streckenabgleich, Schritt 2: Distanzmatrizen gegen die Tagfahrt (Original vom 05.10.2026).

Kosinus-Distanz der ResNet18-Merkmale, lokal zeilen- und spaltenweise z-normiert (Fenster 10), als Bild und als
D_<merkmal>.npy gespeichert. Den Viterbi-Pfad durch die gefilterte Matrix (Ergebnis route/zeitabgleich_strecke1.csv)
berechnete ein einmaliges Skript in der Arbeitssitzung, das nicht gesichert wurde.
"""
import os
import numpy as np, cv2, sys
ref = "IMG_9733"
R = np.load(rf"analyse\{ref}\route.npz")
out = os.environ.get("ROUTE_FIG_DIR", r"results\figures")
def enhance(D, w=10):
    E = np.zeros_like(D)
    for i in range(D.shape[0]):
        a, b = max(0, i-w), min(D.shape[0], i+w+1)
        loc = D[a:b]; E[i] = (D[i]-loc.mean(0))/(loc.std(0)+1e-6)
    return E
for q in ["IMG_9726","IMG_9730"]:
    Q = np.load(rf"analyse\{q}\route.npz")
    Dc = 1 - Q["cnn"] @ R["cnn"].T                      # Zeilen: Query, Spalten: Tag
    Dt = np.sqrt(np.maximum(0,(Q["tiny"]**2).sum(1)[:,None]+(R["tiny"]**2).sum(1)[None,:]-2*Q["tiny"]@R["tiny"].T)/Q["tiny"].shape[1])
    for nm, D in [("cnn",Dc),("tiny",Dt)]:
        if D is None: continue
        E = enhance(D.T).T; E = enhance(E)
        img = 255*(1-(E-E.min())/(np.ptp(E)+1e-9)); img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(rf"{out}\route_match_{q}_vs_{ref}_{nm}.png", img); np.save(rf"analyse\{q}\D_{nm}.npy", E)
    print(q, Dc.shape)

