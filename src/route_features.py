"""Streckenabgleich, Schritt 1: Merkmale je Video (Original vom 05.10.2026, im Arbeitsverzeichnis ausgefuehrt).

Alle 15 Bilder (0,5 s) ein ResNet18-Merkmalsvektor (ImageNet, bis layer3, 2x4 Pooling, obere 85 % des Bildes)
und ein kleines patch-normiertes Graustufenbild wie bei SeqSLAM. Ausgabe: analyse/<video>/route.npz
(Kopien in results/metrics/route/route_<video>.npz).
Aufruf im Arbeitsverzeichnis: python route_features.py IMG_9733
"""
import cv2, numpy as np, sys, torch, torchvision
name = sys.argv[1]; step = 15
m = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()
feat = torch.nn.Sequential(*list(m.children())[:-3])  # bis layer3
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
cap = cv2.VideoCapture(rf"videos\input\{name}.MOV"); i = 0; F = []; S = []; T = []; batch = []
def flush():
    global batch
    if not batch: return
    x = torch.from_numpy(np.stack(batch)).cuda().permute(0,3,1,2).float()/255.
    with torch.no_grad():
        y = feat((x-mean)/std)                        # B,256,h,w
        y = torch.nn.functional.adaptive_avg_pool2d(y, (2,4)).flatten(1)
        y = torch.nn.functional.normalize(y, dim=1)
    F.append(y.cpu().numpy()); batch = []
while cap.grab():
    if i % step == 0:
        f = cap.retrieve()[1]; h = f.shape[0]; f = f[:int(h*0.85)]
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 31)).astype(np.float32)
        # Patch-Normalisierung 8x8 (SeqSLAM)
        for yy in range(0, 31, 8):
            for xx in range(0, 64, 8):
                p = g[yy:yy+8, xx:xx+8]; g[yy:yy+8, xx:xx+8] = (p - p.mean())/(p.std()+1e-6)
        S.append(g.flatten()); T.append(i/30)
        batch.append(cv2.cvtColor(cv2.resize(f, (448, 224)), cv2.COLOR_BGR2RGB))
        if len(batch) == 64: flush()
    i += 1
flush()
np.savez_compressed(rf"analyse\{name}\route.npz", cnn=np.concatenate(F), tiny=np.array(S), t=np.array(T))
print(name, len(T), "fertig")
