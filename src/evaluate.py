import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import UNet

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic.npz")
ap.add_argument("--weights", default="results/unet.pt")
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

model = UNet().to(dev)
model.load_state_dict(torch.load(args.weights, map_location=dev))
model.eval()

with torch.no_grad():
    pred = model(Xva)
    # relative L2 error for each field separately
    for i, name in enumerate(["pressure", "velocity u", "velocity v"]):
        e = ((pred[:, i] - Yva[:, i]).flatten(1).norm(dim=1)
             / Yva[:, i].flatten(1).norm(dim=1)).mean().item()
        print(f"{name}: relative error {e:.3f}")

    # speed: time per sample, after a warm-up run
    model(Xva[:1])
    t0 = time.perf_counter()
    for _ in range(50):
        model(Xva[:1])
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"inference time: {ms:.1f} ms per sample on {dev}")

# picture: shape, true pressure, predicted pressure, difference
fig, ax = plt.subplots(3, 4, figsize=(10, 7))
for r in range(3):
    ax[r, 0].imshow(Xva[r, 0].cpu(), cmap="gray")
    ax[r, 1].imshow(Yva[r, 0].cpu())
    ax[r, 2].imshow(pred[r, 0].cpu())
    ax[r, 3].imshow((pred[r, 0] - Yva[r, 0]).abs().cpu(), cmap="hot")
for c, t in enumerate(["shape", "true", "predicted", "abs. error"]):
    ax[0, c].set_title(t)
for a in ax.flat:
    a.axis("off")
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print("saved", args.out)