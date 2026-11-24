import csv
import math
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
with open("results/experiments.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        n, ep, bs = int(r["ntrain"]), int(r["epochs"]), int(r["batch"])
        rows.append(dict(n=n, epochs=ep, lr=float(r["lr"]), model=r["model"], seed=int(r["seed"]),
                         steps=math.ceil(n / bs) * ep, err=float(r["val_error_mean"])))

# only the U-Net runs at lr = 0.001 belong to the training-size study
base = [r for r in rows if r["model"] == "unet" and abs(r["lr"] - 1e-3) < 1e-12]
series = {
    "30 epochs each (smaller sets get fewer steps)": [r for r in base if r["epochs"] == 30],
    "about 1140 steps each (equal training effort)": [r for r in base if abs(r["steps"] - 1140) <= 20],
}

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for (label, rs), marker in zip(series.items(), ("o", "s")):
    groups = defaultdict(list)
    for r in rs:
        groups[r["n"]].append(r["err"])
    ns = sorted(groups)
    if len(ns) < 2:
        continue
    mean = np.array([np.mean(groups[k]) for k in ns])
    below = mean - np.array([min(groups[k]) for k in ns])
    above = np.array([max(groups[k]) for k in ns]) - mean
    slope = np.polyfit(np.log(ns), np.log(mean), 1)[0]
    ax.errorbar(ns, mean, yerr=[below, above], fmt=marker + "-", capsize=3,
                label=f"{label}\nfitted slope {slope:.2f}")
    print(label)
    for k in ns:
        g = groups[k]
        print(f"   n={k:5d}  runs={len(g)}  mean error={np.mean(g):.3f}  min={min(g):.3f}  max={max(g):.3f}")
    print(f"   fitted slope of log(mean error) against log(n): {slope:.2f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of training airfoils")
ax.set_ylabel("validation relative error (mean over channels)")
ax.set_title("U-Net error against training-set size")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/training_size.png", dpi=150)
print("saved results/training_size.png")