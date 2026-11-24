import csv
import sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sys.argv[1:] or ["results/experiments.csv"]
rows = []
for f in files:
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(model=r["model"], n=int(r["ntrain"]), lr=float(r["lr"]), epochs=int(r["epochs"]),
                             batch=int(r["batch"]), seed=int(r["seed"]), err=float(r["val_error_mean"])))

# full training set, U-Net, batch size 32
base = [r for r in rows if r["model"] == "unet" and r["n"] == 1200 and r["batch"] == 32]

def group(rs, key):
    g = defaultdict(list)
    for r in rs:
        g[r[key]].append(r["err"])
    return dict(sorted(g.items()))

lr_runs = group([r for r in base if r["epochs"] == 100], "lr")
ep_runs = group([r for r in base if abs(r["lr"] - 1e-3) < 1e-12], "epochs")

def show(title, g, name):
    print(title)
    for k, v in g.items():
        print(f"   {name}={k:<8}  runs={len(v)}  mean error={np.mean(v):.4f}  min={min(v):.4f}  max={max(v):.4f}")

show("Learning-rate sweep (100 epochs, 1200 training airfoils)", lr_runs, "lr")
show("Training length at lr = 0.001 (1200 training airfoils)", ep_runs, "epochs")

def draw(ax, g, xlabel, title):
    xs = list(g)
    mean = np.array([np.mean(g[k]) for k in xs])
    below = mean - np.array([min(g[k]) for k in xs])
    above = np.array([max(g[k]) for k in xs]) - mean
    ax.errorbar(xs, mean, yerr=[below, above], fmt="o-", capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("validation relative error")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
if len(lr_runs) >= 2:
    draw(ax[0], lr_runs, "learning rate", "Learning-rate sweep (bars show min to max over seeds)")
if len(ep_runs) >= 2:
    draw(ax[1], ep_runs, "epochs (at lr = 0.001)", "Training length")
plt.tight_layout()
plt.savefig("results/sweeps.png", dpi=150)
print("saved results/sweeps.png")