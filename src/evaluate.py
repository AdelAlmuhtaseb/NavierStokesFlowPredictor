import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from models import UNet, FNO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic.npz")
ap.add_argument("--out", default="results/comparison.png")
args = ap.parse_args()

dev = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available()
       else "cpu")

d = np.load(args.data)
X = torch.tensor(d["imgs"]).unsqueeze(1)
Y = torch.tensor(d["fields"])
ntr = int(0.8 * len(X))                      # same split as train.py
Xva, Yva = X[ntr:].to(dev), Y[ntr:].to(dev)

def rel(pred, y):
    return ((pred - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean().item()

models = {"U-Net": (UNet, "results/unet.pt"), "FNO": (FNO, "results/fno.pt")}
preds, rows = {}, []

for name, (cls, path) in models.items():
    if not Path(path).exists():
        print(f"skipping {name}: {path} not found")
        continue
    net = cls().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    with torch.no_grad():
        p = net(Xva)
        errs = [rel(p[:, i], Yva[:, i]) for i in range(3)]
        net(Xva[:1])                          # warm-up
        t0 = time.perf_counter()
        for _ in range(50):
            net(Xva[:1])
        ms = (time.perf_counter() - t0) / 50 * 1000
    preds[name] = p
    rows.append((name, *errs, ms, sum(q.numel() for q in net.parameters())))

table = ["| Model | Pressure error | Velocity u error | Velocity v error | ms/sample | Parameters |",
         "|-------|----------------|------------------|------------------|-----------|------------|"]
for n, e1, e2, e3, ms, npar in rows:
    table.append(f"| {n} | {e1:.3f} | {e2:.3f} | {e3:.3f} | {ms:.1f} | {npar:,} |")
text = "\n".join(table)
print(text)
Path("results/results_table.md").write_text(text + "\n")

# figure: shape, true pressure, then each model's prediction
cols = ["shape", "true"] + list(preds)
fig, ax = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 7))
for r in range(3):
    ax[r, 0].imshow(Xva[r, 0].cpu(), cmap="gray")
    ax[r, 1].imshow(Yva[r, 0].cpu())
    for c, name in enumerate(preds, start=2):
        ax[r, c].imshow(preds[name][r, 0].cpu())
for c, t in enumerate(cols):
    ax[0, c].set_title(t)
for a in ax.flat:
    a.axis("off")
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print("saved", args.out, "and results/results_table.md")