import argparse
import time
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MplPath
from panel import naca4_points, solve_panels, velocity_at

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1500, help="number of airfoils")
ap.add_argument("--grid", type=int, default=64, help="grid points per side")
ap.add_argument("--out", default="data/panel_dataset.npz")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

G = args.grid
xs = np.linspace(-0.5, 1.5, G)          # window around the airfoil (chord = 1)
ys = np.linspace(-1.0, 1.0, G)
X, Y = np.meshgrid(xs, ys, indexing="xy")   # X[row, col]: rows go along y
P = np.column_stack([X.ravel(), Y.ravel()])

rng = np.random.default_rng(args.seed)
params = np.column_stack([
    rng.uniform(0.0, 0.06, args.n),     # camber m
    rng.uniform(0.2, 0.6, args.n),      # camber position p
    rng.uniform(0.08, 0.20, args.n),    # thickness t
    rng.uniform(-5.0, 15.0, args.n),    # angle of attack (degrees)
]).astype(np.float32)

imgs = np.zeros((args.n, 3, G, G), dtype=np.float32)
fields = np.zeros((args.n, 3, G, G), dtype=np.float32)
t0 = time.time()
for k, (m, p, t, al) in enumerate(params):
    pts = naca4_points(m, p, t)
    sol = solve_panels(pts, al)
    inside = MplPath(pts).contains_points(P).reshape(G, G)
    u, v = velocity_at(sol, P)
    u, v = u.reshape(G, G), v.reshape(G, G)
    cp = 1.0 - (u**2 + v**2)
    imgs[k, 0] = inside
    imgs[k, 1] = np.cos(np.radians(al))
    imgs[k, 2] = np.sin(np.radians(al))
    for c, f in enumerate((u, v, cp)):
        fields[k, c] = np.where(inside, 0.0, f)
    if (k + 1) % 250 == 0:
        print(f"{k+1}/{args.n}  ({time.time()-t0:.0f} s)")

Path(args.out).parent.mkdir(exist_ok=True)
np.savez(args.out, imgs=imgs, fields=fields, params=params)
print("saved", args.out, imgs.shape, fields.shape)
print("field ranges: u [%.2f, %.2f]  v [%.2f, %.2f]  Cp [%.2f, %.2f]" % (
    fields[:, 0].min(), fields[:, 0].max(), fields[:, 1].min(), fields[:, 1].max(),
    fields[:, 2].min(), fields[:, 2].max()))
print("any NaN:", bool(np.isnan(fields).any()))