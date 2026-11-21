import argparse
import numpy as np
import torch
from models import UNet

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic.npz")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
args = ap.parse_args()

dev = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available()
       else "cpu")
print("device:", dev)

d = np.load(args.data)
X = torch.tensor(d["imgs"]).unsqueeze(1)
Y = torch.tensor(d["fields"])
n = len(X)
ntr = int(0.8 * n)
Xtr, Ytr = X[:ntr], Y[:ntr]
Xva, Yva = X[ntr:].to(dev), Y[ntr:].to(dev)

model = UNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

def rel_err(pred, y):
    # relative L2 error per sample, averaged
    return ((pred - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean().item()

for epoch in range(args.epochs):
    model.train()
    perm = torch.randperm(ntr)
    total = 0.0
    for i in range(0, ntr, args.batch):
        idx = perm[i:i + args.batch]
        xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        err = rel_err(model(Xva), Yva)
    print(f"epoch {epoch+1}/{args.epochs}  train loss {total/ntr:.5f}  val rel. error {err:.3f}")

torch.save(model.state_dict(), "results/unet.pt")
print("saved results/unet.pt")