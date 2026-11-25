import argparse
import sys
import time
import numpy as np
import matplotlib
if "--save" in sys.argv:                     # no window: just write a picture
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Slider
from panel import naca4_points, solve_panels, velocity_at

G = 64                                       # the grid the model was trained on
XS = np.linspace(-0.5, 1.5, G)
YS = np.linspace(-1.0, 1.0, G)
GX, GY = np.meshgrid(XS, YS, indexing="xy")
P = np.column_stack([GX.ravel(), GY.ravel()])
EXTENT = (-0.5, 1.5, -1.0, 1.0)


def load_predictor(path):
    """Load a trained U-Net and return a function: input (3, G, G) array -> output (3, G, G) array."""
    import torch
    from models import UNet
    ck = torch.load(path, map_location="cpu")
    if ck["in_ch"] != 3:
        sys.exit("This demo needs a model trained on the panel dataset (3 input channels).")
    net = UNet(in_ch=ck["in_ch"], out_ch=ck["out_ch"])
    net.load_state_dict(ck["state"])
    net.eval()
    mean, std = ck["mean"], ck["std"]

    def predict(x):
        with torch.no_grad():
            out = net(torch.tensor(x).unsqueeze(0)) * std + mean
        return out[0].numpy()
    return predict


def run_case(predict, m, p, t, alpha):
    """Run the surrogate and the panel solver on one airfoil and compare them."""
    pts = naca4_points(m, p, t)
    inside = MplPath(pts).contains_points(P).reshape(G, G)
    x = np.stack([inside.astype(np.float32),
                  np.full((G, G), np.cos(np.radians(alpha)), dtype=np.float32),
                  np.full((G, G), np.sin(np.radians(alpha)), dtype=np.float32)])

    t0 = time.perf_counter()
    pred = predict(x)
    t_model = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sol = solve_panels(pts, alpha)
    u, v = velocity_at(sol, P)
    t_solver = (time.perf_counter() - t0) * 1000

    truth = np.stack([u.reshape(G, G), v.reshape(G, G), (1 - u ** 2 - v ** 2).reshape(G, G)])
    fl = ~inside
    errs = [np.linalg.norm((pred[k] - truth[k])[fl]) / np.linalg.norm(truth[k][fl]) for k in range(3)]

    speed_true = np.hypot(truth[0], truth[1])
    speed_pred = np.hypot(pred[0], pred[1])
    out = dict(pts=pts, t_model=t_model, t_solver=t_solver, errs=errs, cl=sol["cl"])
    out["true"] = np.where(inside, np.nan, speed_true)
    out["pred"] = np.where(inside, np.nan, speed_pred)
    out["err"] = np.where(inside, np.nan, np.abs(speed_pred - speed_true))
    return out


def make_test_image(path, m, p, t, size=(900, 450)):
    """Draw a NACA airfoil as a black silhouette on a white picture (for trying the image mode)."""
    from PIL import Image, ImageDraw
    pts = naca4_points(m, p, t, n=200)
    W, H = size
    chord = 0.7 * W
    px = 0.15 * W + pts[:, 0] * chord
    py = 0.5 * H - pts[:, 1] * chord
    img = Image.new("L", (W, H), 255)
    ImageDraw.Draw(img).polygon(list(zip(px, py)), fill=0)
    img.save(path)


def mask_from_image(path):
    """Turn a picture of an airfoil into the 64 x 64 silhouette the model expects.

    The picture can have a dark shape on a light background or the other way round, and the airfoil may
    be slightly tilted. The leading edge is taken as the leftmost point and the trailing edge as the
    rightmost point; the picture is rotated and scaled so that the chord has length 1 with the leading
    edge at x = 0, which is where the training airfoils sit."""
    from PIL import Image
    img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    H, W = img.shape
    shape = img < 0.5 * (img.min() + img.max())
    if shape.mean() > 0.5:                      # the shape is the smaller of the two regions
        shape = ~shape
    rows, cols = np.nonzero(shape)
    if len(cols) == 0:
        sys.exit("No shape found in the image.")
    cL, cT = cols.min(), cols.max()
    rL = rows[cols <= cL + 2].mean()            # row of the leading edge
    rT = rows[cols >= cT - 2].mean()            # row of the trailing edge
    d = np.array([cT - cL, -(rT - rL)], dtype=float)    # chord vector, y pointing up
    L = np.hypot(*d)
    if L < 100:
        print(f"Warning: the airfoil is only {L:.0f} pixels long in this picture; use a larger picture for a good result.")
    e1 = d / L
    e2 = np.array([-e1[1], e1[0]])
    gx, gy = P[:, 0], P[:, 1]
    x = cL + L * (gx * e1[0] + gy * e2[0])                  # image column
    y_up = -rL + L * (gx * e1[1] + gy * e2[1])
    r = np.rint(-y_up).astype(int)
    c = np.rint(x).astype(int)
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    inside = np.zeros(len(P), dtype=bool)
    inside[ok] = shape[r[ok], c[ok]]
    return inside.reshape(G, G)


def run_image_case(predict, inside, alpha, naca=None):
    """Predict the flow for a silhouette. If the NACA numbers are known, also run the solver to check."""
    x = np.stack([inside.astype(np.float32),
                  np.full((G, G), np.cos(np.radians(alpha)), dtype=np.float32),
                  np.full((G, G), np.sin(np.radians(alpha)), dtype=np.float32)])
    t0 = time.perf_counter()
    pred = predict(x)
    out = dict(t_model=(time.perf_counter() - t0) * 1000)
    out["speed"] = np.where(inside, np.nan, np.hypot(pred[0], pred[1]))
    out["cp"] = np.where(inside, np.nan, pred[2])
    if naca is not None:
        m, p, t = naca
        t0 = time.perf_counter()
        sol = solve_panels(naca4_points(m, p, t), alpha)
        u, v = velocity_at(sol, P)
        out["t_solver"] = (time.perf_counter() - t0) * 1000
        truth = np.stack([u.reshape(G, G), v.reshape(G, G), (1 - u ** 2 - v ** 2).reshape(G, G)])
        fl = ~inside
        out["errs"] = [np.linalg.norm((pred[k] - truth[k])[fl]) / np.linalg.norm(truth[k][fl]) for k in range(3)]
    return out


def build_image_figure(predict, inside, alpha0, naca=None):
    fig = plt.figure(figsize=(13, 6.4))
    titles = ["Image as the model sees it (64x64)", "Predicted flow speed", "Predicted pressure coefficient Cp"]
    cmaps = ["gray_r", "viridis", "coolwarm"]
    limits = [(0, 1), (0, 2.0), (-2.0, 1.0)]
    ims = []
    for i in range(3):
        ax = fig.add_axes([0.04 + i * 0.325, 0.40, 0.25, 0.52])
        im = ax.imshow(np.zeros((G, G)), origin="lower", extent=EXTENT, vmin=limits[i][0], vmax=limits[i][1],
                       cmap=cmaps[i], interpolation="nearest")
        ax.set_title(titles[i], fontsize=11)
        ax.set_xlabel("x / chord")
        if i == 0:
            ax.set_ylabel("y / chord")
        if i > 0:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ims.append(im)
    ims[0].set_data(inside.astype(float))

    sax = fig.add_axes([0.14, 0.24, 0.45, 0.03])
    s_a = Slider(sax, "angle of attack (deg)", -5.0, 15.0, valinit=alpha0, valfmt="%.1f")
    info = fig.text(0.64, 0.27, "", fontsize=10.5, va="top", family="monospace")
    fig.text(0.04, 0.10, "The model was trained on NACA 4-digit airfoils only, with the chord horizontal and the leading edge on the left.\n"
                         "Other shapes are outside its training data, so its predictions for them can be wrong.",
             fontsize=9, color="gray", va="top")

    def update(_=None):
        r = run_image_case(predict, inside, s_a.val, naca)
        ims[1].set_data(r["speed"])
        ims[2].set_data(r["cp"])
        text = f"Model time: {r['t_model']:.1f} ms"
        if "errs" in r:
            e = r["errs"]
            text += (f"\nPanel solver: {r['t_solver']:.1f} ms  (model {r['t_solver'] / r['t_model']:.0f}x faster)"
                     f"\nRelative error vs solver:\n  u {e[0]:.1%}   v {e[1]:.1%}   Cp {e[2]:.1%}")
        info.set_text(text)
        fig.canvas.draw_idle()

    s_a.on_changed(update)
    update()
    return fig, (s_a,)


def build_figure(predict, m0, p0, t0, a0):
    fig = plt.figure(figsize=(13, 6.8))
    titles = ["Panel solver (truth): flow speed", "U-Net prediction: flow speed", "Absolute error in flow speed"]
    limits = [(0, 2.0), (0, 2.0), (0, 0.5)]
    cmaps = ["viridis", "viridis", "hot"]
    axes, ims, lines = [], [], []
    for i in range(3):
        ax = fig.add_axes([0.04 + i * 0.325, 0.40, 0.25, 0.52])
        im = ax.imshow(np.zeros((G, G)), origin="lower", extent=EXTENT, vmin=limits[i][0], vmax=limits[i][1],
                       cmap=cmaps[i], interpolation="nearest")
        (ln,) = ax.plot([], [], "k-", lw=0.9)
        ax.set_title(titles[i], fontsize=11)
        ax.set_xlabel("x / chord")
        if i == 0:
            ax.set_ylabel("y / chord")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes.append(ax)
        ims.append(im)
        lines.append(ln)

    def make_slider(row, label, lo, hi, val, fmt):
        sax = fig.add_axes([0.14, 0.27 - row * 0.055, 0.45, 0.03])
        return Slider(sax, label, lo, hi, valinit=val, valfmt=fmt)

    s_m = make_slider(0, "camber m", 0.0, 0.06, m0, "%.3f")
    s_p = make_slider(1, "camber position p", 0.2, 0.6, p0, "%.2f")
    s_t = make_slider(2, "thickness t", 0.08, 0.20, t0, "%.3f")
    s_a = make_slider(3, "angle of attack (deg)", -5.0, 15.0, a0, "%.1f")
    info = fig.text(0.64, 0.22, "", fontsize=10.5, va="top", family="monospace")
    fig.text(0.64, 0.045, "Sliders stay inside the training ranges.\nOutside them the model has not been trained.",
             fontsize=9, color="gray", va="top")

    def update(_=None):
        r = run_case(predict, s_m.val, s_p.val, s_t.val, s_a.val)
        for im, key in zip(ims, ("true", "pred", "err")):
            im.set_data(r[key])
        for ln in lines:
            ln.set_data(r["pts"][:, 0], r["pts"][:, 1])
        e = r["errs"]
        info.set_text(f"U-Net:         {r['t_model']:6.1f} ms\n"
                      f"Panel solver:  {r['t_solver']:6.1f} ms\n"
                      f"The U-Net is {r['t_solver'] / r['t_model']:.0f}x faster\n"
                      f"Lift coefficient (solver): {r['cl']:.2f}\n"
                      f"Relative error: u {e[0]:.1%}  v {e[1]:.1%}  Cp {e[2]:.1%}")
        fig.canvas.draw_idle()

    for s in (s_m, s_p, s_t, s_a):
        s.on_changed(update)
    update()
    return fig, (s_m, s_p, s_t, s_a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/unet_panel_dataset.pt")
    ap.add_argument("--image", default="", help="a picture of an airfoil; the model predicts the flow around it")
    ap.add_argument("--make-image", default="", dest="make_image",
                    help="write a test picture of the NACA airfoil given by --m --p --t, then stop")
    ap.add_argument("--m", type=float, default=None, help="camber")
    ap.add_argument("--p", type=float, default=None, help="camber position")
    ap.add_argument("--t", type=float, default=None, help="thickness")
    ap.add_argument("--alpha", type=float, default=6.0, help="angle of attack in degrees")
    ap.add_argument("--save", default="", help="write a picture to this file instead of opening a window")
    args = ap.parse_args()
    m, p, t = (0.02 if args.m is None else args.m), (0.4 if args.p is None else args.p), (0.12 if args.t is None else args.t)

    if args.make_image:
        make_test_image(args.make_image, m, p, t)
        print("wrote", args.make_image, f"(NACA m={m}, p={p}, t={t})")
        sys.exit(0)

    predict = load_predictor(args.weights)
    predict(np.zeros((3, G, G), dtype=np.float32))          # warm-up, so the first timing is fair
    if args.image:
        naca = (m, p, t) if None not in (args.m, args.p, args.t) else None   # only if all three numbers are given
        fig, sliders = build_image_figure(predict, mask_from_image(args.image), args.alpha, naca)
    else:
        fig, sliders = build_figure(predict, m, p, t, args.alpha)
    if args.save:
        fig.savefig(args.save, dpi=130)
        print("saved", args.save)
    else:
        plt.show()