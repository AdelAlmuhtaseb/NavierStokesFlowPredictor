import csv
import sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = {"panel_g32": 32, "panel_dataset": 64, "panel_g128": 128}   # data name -> grid points per side

files = sys.argv[1:] or ["results/experiments.csv"]
rows = []
for f in files:
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == "unet" and r["data"] in GRID and int(r["ntrain"]) == 1200 \
                    and abs(float(r["lr"]) - 1e-3) < 1e-12 and int(r["batch"]) == 32:
                rows.append(dict(grid=GRID[r["data"]], epochs=int(r["epochs"]),
                                 err=float(r["val_error_mean"]), secs=float(r["train_seconds"])))

def group(epochs):
    g = defaultdict(list)
    for r in rows:
        if r["epochs"] == epochs:
            g[r["grid"]].append((r["err"], r["secs"]))
    return dict(sorted(g.items()))

fig, ax = plt.subplots(figsize=(6.2, 4.4))
for epochs, marker in ((100, "o-"), (300, "s--")):
    g = group(epochs)
    if not g:
        continue
    print(f"U-Net, {epochs} epochs, 1200 training airfoils")
    xs = list(g)
    mean = np.array([np.mean([e for e, _ in g[k]]) for k in xs])
    lo = mean - np.array([min(e for e, _ in g[k]) for k in xs])
    hi = np.array([max(e for e, _ in g[k]) for k in xs]) - mean
    for k in xs:
        errs = [e for e, _ in g[k]]
        secs = np.mean([s for _, s in g[k]])
        print(f"   grid {k:3d} x {k:<3d}  runs={len(errs)}  mean error={np.mean(errs):.4f}  min={min(errs):.4f}  max={max(errs):.4f}  training time={secs:.0f} s")
    ax.errorbar(xs, mean, yerr=[lo, hi], fmt=marker, capsize=3, label=f"{epochs} epochs")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(list(GRID.values()))
ax.set_xticklabels([str(v) for v in GRID.values()])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("grid points per side")
ax.set_ylabel("validation relative error")
ax.set_title("U-Net error against grid resolution")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("results/resolution.png", dpi=150)
print("saved results/resolution.png")