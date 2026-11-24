import argparse
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import UNet, FNO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/panel_dataset.npz")
ap.add_argument("--out", default="results/comparison.png")
args = ap.parse_args()

dev = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available()
       else "cpu")

d = np.load(args.data)
X = torch.tensor(d["imgs"])
Y = torch.tensor(d["fields"])
if X.ndim == 3:
    X = X.unsqueeze(1)
ntr = int(0.8 * len(X))                      # same split as train.py
Xva, Yva = X[ntr:].to(dev), Y[ntr:].to(dev)
fl = 1 - Xva[:, :1]                          # 1 where there is fluid
stem = Path(args.data).stem
chan = ["velocity u", "velocity v", "pressure Cp"]
if Yva.shape[1] != 3:
    chan = [f"channel {i}" for i in range(Yva.shape[1])]

models = {"U-Net": (UNet, f"results/unet_{stem}.pt"), "FNO": (FNO, f"results/fno_{stem}.pt")}
preds, rows = {}, []
for name, (cls, path) in models.items():
    if not Path(path).exists():
        print(f"skipping {name}: {path} not found")
        continue
    ck = torch.load(path, map_location=dev)
    net = cls(in_ch=ck["in_ch"], out_ch=ck["out_ch"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    mean, std = ck["mean"].to(dev), ck["std"].to(dev)
    with torch.no_grad():
        p = net(Xva) * std + mean
        e = (((p - Yva) * fl).flatten(2).norm(dim=2) / (Yva * fl).flatten(2).norm(dim=2)).mean(0)
        net(Xva[:1])                          # warm-up
        t0 = time.perf_counter()
        for _ in range(50):
            net(Xva[:1])
        ms = (time.perf_counter() - t0) / 50 * 1000
    preds[name] = p
    rows.append((name, e.cpu().tolist(), ms, sum(q.numel() for q in net.parameters()), ck["ntrain"]))

# time the panel solver on the same computer (solve + velocity on the grid)
solver_ms = None
if "params" in d.files and X.shape[1] == 3:
    from panel import naca4_points, solve_panels, velocity_at
    G = X.shape[-1]
    gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, G), np.linspace(-1.0, 1.0, G), indexing="xy")
    P = np.column_stack([gx.ravel(), gy.ravel()])
    t0 = time.perf_counter()
    for m, p_, t, al in d["params"][ntr:ntr + 5]:
        sol = solve_panels(naca4_points(m, p_, t), al)
        velocity_at(sol, P)
    solver_ms = (time.perf_counter() - t0) / 5 * 1000

lines = ["| Model | " + " | ".join(f"{c} error" for c in chan) + " | ms/sample | Parameters | Training samples |",
         "|---" * (len(chan) + 4) + "|"]
for name, e, ms, npar, ntrain in rows:
    lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in e) + f" | {ms:.1f} | {npar:,} | {ntrain} |")
text = "\n".join(lines)
print(text)
if solver_ms is not None:
    extra = f"\nPanel solver: {solver_ms:.0f} ms per sample on this computer."
    for name, e, ms, npar, ntrain in rows:
        extra += f"\n{name} is about {solver_ms / ms:.0f}x faster than the solver."
    print(extra)
    text += "\n" + extra
Path("results/results_table.md").write_text(text + "\n")

# figure: shape, true speed, then each model's predicted speed and error
def speed(t):
    return torch.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2).cpu()

true_s = speed(Yva)
cols = ["shape", "true speed"]
for name in preds:
    cols += [name, f"{name} |error|"]
fig, ax = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 7), squeeze=False)
vmax = true_s[:3].max().item()
for r in range(3):
    ax[r, 0].imshow(Xva[r, 0].cpu(), cmap="gray", origin="lower")
    ax[r, 1].imshow(true_s[r], origin="lower", vmin=0, vmax=vmax)
    c = 2
    for name, p in preds.items():
        ps = speed(p)
        ax[r, c].imshow(ps[r], origin="lower", vmin=0, vmax=vmax)
        ax[r, c + 1].imshow((ps[r] - true_s[r]).abs() * (1 - Xva[r, 0].cpu()), origin="lower", cmap="hot")
        c += 2
for c, title in enumerate(cols):
    ax[0, c].set_title(title)
for a in ax.flat:
    a.axis("off")
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print("saved", args.out, "and results/results_table.md")