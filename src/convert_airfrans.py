import argparse
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib.path import Path as MplPath

# Column layout of one AirfRANS simulation (from the airfrans documentation):
#  0,1 position x,y | 2,3 inlet velocity | 4 distance to airfoil | 5,6 normals
#  7,8 velocity u,v | 9 pressure / density | 10 turbulent viscosity | 11 on-airfoil flag
X_MIN, X_MAX, Y_MIN, Y_MAX = -0.5, 1.5, -1.0, 1.0       # same window as the panel dataset


def make_grid(G):
    xs = np.linspace(X_MIN, X_MAX, G)
    ys = np.linspace(Y_MIN, Y_MAX, G)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel()])


def convert_sim(sim, P, G):
    """Resample one simulation (N x 12 point cloud) onto the G x G grid P.
    Returns the 4 input channels and the 3 output channels, plus (speed, angle, #NaN)."""
    uin, vin = sim[0, 2], sim[0, 3]
    U = float(np.hypot(uin, vin))
    alpha = float(np.arctan2(vin, uin))

    # keep only points near the window, so the triangulation is smaller and faster
    keep = (sim[:, 0] > X_MIN - 0.3) & (sim[:, 0] < X_MAX + 0.3) & \
           (sim[:, 1] > Y_MIN - 0.3) & (sim[:, 1] < Y_MAX + 0.3)
    pts = sim[keep]
    _, first = np.unique(np.round(pts[:, :2], 9), axis=0, return_index=True)   # drop duplicate nodes
    pts = pts[np.sort(first)]

    # non-dimensional targets: u/U, v/U, Cp = p / (0.5 U^2)
    vals = np.column_stack([pts[:, 7] / U, pts[:, 8] / U, pts[:, 9] / (0.5 * U ** 2)])
    f = LinearNDInterpolator(pts[:, :2], vals)(P)          # linear interpolation on a Delaunay triangulation
    n_nan = int(np.isnan(f).any(axis=1).sum())
    f = np.nan_to_num(f)

    # body mask: polygon through the airfoil surface nodes, ordered by angle around their centre
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
    imgs[3] = U / 100.0                                     # inlet speed, scaled to about 0.3 - 0.9
    return imgs, fields, U, np.degrees(alpha), n_nan


if __name__ == "__main__":
    import airfrans as af

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Dataset")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--out", default="data/airfrans_g64.npz")
    ap.add_argument("--limit", type=int, default=0, help="convert only this many simulations per split (for a quick test)")
    args = ap.parse_args()

    G = args.grid
    P = make_grid(G)
    all_imgs, all_fields, all_params, all_names = [], [], [], []
    t0 = time.time()
    for train in (True, False):                              # 800 training simulations, then 200 test simulations
        data, names = af.dataset.load(root=args.root, task="full", train=train)
        if args.limit:
            data, names = data[:args.limit], names[:args.limit]
        for k, sim in enumerate(data):
            if train and k == 0:
                s = sim[sim[:, 11] > 0.5]
                print(f"first simulation: {len(sim)} points, airfoil x from {s[:, 0].min():.3f} to {s[:, 0].max():.3f}, "
                      f"y from {s[:, 1].min():.3f} to {s[:, 1].max():.3f}")
            imgs, fields, U, alpha, n_nan = convert_sim(sim, P, G)
            all_imgs.append(imgs)
            all_fields.append(fields)
            all_params.append([U, alpha])
            all_names.append(names[k])
            if n_nan:
                print(f"  note: {names[k]} has {n_nan} grid points outside the data")
            if (k + 1) % 25 == 0:
                print(f"{'train' if train else 'test'} {k+1}/{len(data)}  ({time.time()-t0:.0f} s)")
        del data

    Path(args.out).parent.mkdir(exist_ok=True)
    np.savez(args.out, imgs=np.stack(all_imgs), fields=np.stack(all_fields),
             params=np.array(all_params, dtype=np.float32), names=np.array(all_names))
    F = np.stack(all_fields)
    print("saved", args.out, np.stack(all_imgs).shape, F.shape)
    print("field ranges: u/U [%.2f, %.2f]  v/U [%.2f, %.2f]  Cp [%.2f, %.2f]" % (
        F[:, 0].min(), F[:, 0].max(), F[:, 1].min(), F[:, 1].max(), F[:, 2].min(), F[:, 2].max()))
    print("any NaN:", bool(np.isnan(F).any()))