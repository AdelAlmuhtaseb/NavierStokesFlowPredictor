import argparse
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib.path import Path as MplPath

# AirfRANS point-cloud columns:
# 0,1 position x,y | 2,3 inlet velocity | 4 distance to airfoil
# 5,6 normals | 7,8 velocity u,v | 9 pressure / density
# 10 turbulent viscosity | 11 on-airfoil flag
X_MIN, X_MAX, Y_MIN, Y_MAX = -0.5, 1.5, -1.0, 1.0


def make_grid(G):
    xs = np.linspace(X_MIN, X_MAX, G)
    ys = np.linspace(Y_MIN, Y_MAX, G)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel()])


def convert_sim(sim, P, G):
    """Resample one simulation to a regular grid and return model inputs/targets."""
    uin, vin = sim[0, 2], sim[0, 3]
    U = float(np.hypot(uin, vin))
    alpha = float(np.arctan2(vin, uin))

    keep = (sim[:, 0] > X_MIN - 0.3) & (sim[:, 0] < X_MAX + 0.3) & \
           (sim[:, 1] > Y_MIN - 0.3) & (sim[:, 1] < Y_MAX + 0.3)
    pts = sim[keep]
    _, first = np.unique(np.round(pts[:, :2], 9), axis=0, return_index=True)
    pts = pts[np.sort(first)]

    # Targets: u/U, v/U, Cp = p / (0.5 U^2)
    vals = np.column_stack([pts[:, 7] / U, pts[:, 8] / U, pts[:, 9] / (0.5 * U ** 2)])
    f = LinearNDInterpolator(pts[:, :2], vals)(P)
    n_nan = int(np.isnan(f).any(axis=1).sum())
    f = np.nan_to_num(f)

    surf = sim[sim[:, 11] > 0.5][:, :2]
    c = surf.mean(axis=0)
    order = np.argsort(np.arctan2(surf[:, 1] - c[1], surf[:, 0] - c[0]))
    inside = MplPath(surf[order]).contains_points(P)

    fields = np.zeros((3, G, G), dtype=np.float32)
    imgs = np.zeros((4, G, G), dtype=np.float32)
    for k in range(3):
        fields[k] = np.where(inside, 0.0, f[:, k]).reshape(G, G)
    imgs[0] = inside.reshape(G, G)
    imgs[1] = np.cos(alpha)
    imgs[2] = np.sin(alpha)
    imgs[3] = U / 100.0
    return imgs, fields, U, np.degrees(alpha), n_nan


def save_split(path, imgs, fields, params, names):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        imgs=np.stack(imgs),
        fields=np.stack(fields),
        params=np.asarray(params, dtype=np.float32),
        names=np.asarray(names),
    )
    print(f"saved {path}: inputs {np.stack(imgs).shape}, targets {np.stack(fields).shape}")


if __name__ == "__main__":
    import airfrans as af

    ap = argparse.ArgumentParser(description="Convert AirfRANS official train/test splits without mixing them.")
    ap.add_argument("--root", default="data/Dataset")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--limit", type=int, default=0, help="convert only this many samples from each split")
    ap.add_argument("--combined-out", default="", help="optional legacy combined NPZ output; avoid using it for clean evaluation")
    args = ap.parse_args()

    G = args.grid
    P = make_grid(G)
    combined = {"imgs": [], "fields": [], "params": [], "names": []}
    t0 = time.time()

    for train in (True, False):
        split = "train" if train else "test"
        data, names = af.dataset.load(root=args.root, task="full", train=train)
        if args.limit:
            data, names = data[:args.limit], names[:args.limit]

        collected = {"imgs": [], "fields": [], "params": [], "names": []}
        print(f"\nConverting official {split} split: {len(data)} simulations")
        for k, sim in enumerate(data):
            imgs, fields, U, alpha, n_nan = convert_sim(sim, P, G)
            collected["imgs"].append(imgs)
            collected["fields"].append(fields)
            collected["params"].append([U, alpha])
            collected["names"].append(names[k])
            for key in combined:
                combined[key].append(collected[key][-1])
            if n_nan:
                print(f"  note: {names[k]} has {n_nan} grid points outside the interpolation data")
            if (k + 1) % 25 == 0 or k + 1 == len(data):
                print(f"  {split}: {k+1}/{len(data)} ({time.time()-t0:.0f}s elapsed)")
        save_split(
            Path(args.out_dir) / f"airfrans_{split}_g{G}.npz",
            collected["imgs"], collected["fields"], collected["params"], collected["names"]
        )
        del data

    if args.combined_out:
        save_split(args.combined_out, combined["imgs"], combined["fields"], combined["params"], combined["names"])

    print(f"\nFinished. Official train and test splits remain separate. Total time: {time.time()-t0:.0f}s")
