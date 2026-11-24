import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
from models import UNet, FNO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unet", choices=["unet", "fno"])
ap.add_argument("--data", default="data/panel_dataset.npz")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--ntrain", type=int, default=0, help="use only this many training samples (0 = all)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
dev = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available()
       else "cpu")
print("device:", dev, "| model:", args.model)

d = np.load(args.data)
X = torch.tensor(d["imgs"])
Y = torch.tensor(d["fields"])
if X.ndim == 3:                       # old single-channel data
    X = X.unsqueeze(1)
n = len(X)
ntr_all = int(0.8 * n)                # last 20% is kept for validation
Xtr, Ytr = X[:ntr_all], Y[:ntr_all]
if args.ntrain:
    Xtr, Ytr = Xtr[:args.ntrain], Ytr[:args.ntrain]
Xva, Yva = X[ntr_all:].to(dev), Y[ntr_all:].to(dev)
in_ch, out_ch = X.shape[1], Y.shape[1]
print("train samples:", len(Xtr), "| validation samples:", len(Xva))

# channel 0 of the input is the body mask, so fluid = 1 - mask.
# Normalise every output channel with its mean and std over fluid points of the training set.
fl_tr = 1 - Xtr[:, :1]
cnt = fl_tr.sum()
mean = (Ytr * fl_tr).sum((0, 2, 3), keepdim=True) / cnt
std = (((Ytr - mean) ** 2 * fl_tr).sum((0, 2, 3), keepdim=True) / cnt).sqrt()
mean_d, std_d = mean.to(dev), std.to(dev)

cls = UNet if args.model == "unet" else FNO
model = cls(in_ch=in_ch, out_ch=out_ch).to(dev)
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

def rel_err(pred_n, x, y):
    """Relative L2 error per output channel, over fluid points only."""
    fl = 1 - x[:, :1]
    pred = pred_n * std_d + mean_d
    num = ((pred - y) * fl).flatten(2).norm(dim=2)
    den = (y * fl).flatten(2).norm(dim=2)
    return (num / den).mean(0)

t0 = time.time()
ntr = len(Xtr)
for epoch in range(args.epochs):
    model.train()
    perm = torch.randperm(ntr)
    total = 0.0
    for i in range(0, ntr, args.batch):
        idx = perm[i:i + args.batch]
        xb, yb = Xtr[idx].to(dev), ((Ytr[idx] - mean) / std).to(dev)
        fl = 1 - xb[:, :1]
        loss = (((model(xb) - yb) ** 2) * fl).sum() / (fl.sum() * out_ch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        e = rel_err(model(Xva), Xva, Yva)
    print(f"epoch {epoch+1}/{args.epochs}  train loss {total/ntr:.4f}  "
          f"val rel. error {e.mean().item():.3f}  (per channel: {', '.join(f'{v:.3f}' for v in e.tolist())})")

secs = time.time() - t0
print(f"training time: {secs:.0f} s")
Path("results").mkdir(exist_ok=True)
tag = f"_n{args.ntrain}" if args.ntrain else ""
path = f"results/{args.model}_{Path(args.data).stem}{tag}.pt"
torch.save(dict(state=model.state_dict(), mean=mean, std=std, in_ch=in_ch, out_ch=out_ch,
                lr=args.lr, ntrain=len(Xtr)), path)
print("saved", path)

# log this run so the experiments can be compared later
log = Path("results/experiments.csv")
new = not log.exists()
with open(log, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["model", "data", "ntrain", "lr", "epochs", "batch", "seed",
                    "val_error_mean", "val_err_u", "val_err_v", "val_err_cp", "train_seconds"])
    w.writerow([args.model, Path(args.data).stem, len(Xtr), args.lr, args.epochs, args.batch, args.seed,
                f"{e.mean().item():.4f}", *[f"{v:.4f}" for v in e.tolist()], f"{secs:.0f}"])
print("logged to", log)