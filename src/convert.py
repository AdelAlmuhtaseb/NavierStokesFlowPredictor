import numpy as np
import airfrans as af
from pathlib import Path

out = Path("data/npz")
out.mkdir(parents=True, exist_ok=True)

data, names = af.dataset.load(root="data/Dataset", task="full", train=True)
for arr, name in zip(data, names):
    np.savez_compressed(out / f"{name}.npz", sim=arr)
print("saved", len(data), "simulations")