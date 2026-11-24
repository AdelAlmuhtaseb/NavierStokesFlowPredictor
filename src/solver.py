import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def solve_laplace(dirichlet, values):
    """Solve Laplace(psi) = 0 on a uniform N x N grid with finite differences.

    Fixed points (the outer edge and the inside of the body) are in `dirichlet`,
    and `values` holds psi there. At every other point we use the 5-point stencil
        4*psi[i,j] - psi[i+1,j] - psi[i-1,j] - psi[i,j+1] - psi[i,j-1] = 0.
    This gives a sparse linear system A psi = b, solved directly.
    """
    N = dirichlet.shape[0]
    n = N * N
    idx = np.arange(n).reshape(N, N)
    d = idx[dirichlet]          # fixed points: row of the identity
    f = idx[~dirichlet]         # free points: 5-point stencil (never on the outer edge)

    rows = [d, f]
    cols = [d, f]
    vals = [np.ones(d.size), 4.0 * np.ones(f.size)]
    for shift in (N, -N, 1, -1):    # neighbours i+1, i-1, j+1, j-1
        rows.append(f)
        cols.append(f + shift)
        vals.append(-np.ones(f.size))

    A = sp.csc_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    b = np.where(dirichlet, values, 0.0).ravel()
    return spsolve(A, b).reshape(N, N)


def cylinder_case(N, L=4.0, R=1.0, U=1.0):
    """Flow past a cylinder of radius R. Exact stream function: psi = U*y*(1 - R^2/r^2)."""
    xs = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    r2 = X**2 + Y**2
    body = r2 <= R**2
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.where(body, 0.0, U * Y * (1.0 - R**2 / r2))

    edge = np.zeros((N, N), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True

    dirichlet = body | edge
    values = np.where(edge, exact, 0.0)   # exact psi on the outer edge, psi = 0 on the body
    psi = solve_laplace(dirichlet, values)

    err = (psi - exact)[~dirichlet]
    h = xs[1] - xs[0]
    return dict(N=N, h=h, psi=psi, X=X, Y=Y, body=body, exact=exact,
                linf=np.abs(err).max(), rms=np.sqrt((err**2).mean()))


if __name__ == "__main__":
    rows = []
    for N in (33, 65, 129, 257, 513):
        t0 = time.perf_counter()
        r = cylinder_case(N)
        r["time"] = time.perf_counter() - t0
        rows.append(r)

    print(f"{'N':>5} {'h':>9} {'max error':>11} {'order':>6} {'RMS error':>11} {'order':>6} {'time (s)':>9}")
    for k, r in enumerate(rows):
        if k == 0:
            o1 = o2 = float("nan")
        else:
            o1 = np.log2(rows[k - 1]["linf"] / r["linf"])
            o2 = np.log2(rows[k - 1]["rms"] / r["rms"])
        print(f"{r['N']:>5} {r['h']:>9.4f} {r['linf']:>11.2e} {o1:>6.2f} {r['rms']:>11.2e} {o2:>6.2f} {r['time']:>9.2f}")

    h = np.array([r["h"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    r = rows[-1]
    ax[0].contour(r["X"], r["Y"], r["psi"], levels=np.linspace(-3.5, 3.5, 36), colors="tab:blue", linewidths=0.8)
    ax[0].add_patch(plt.Circle((0, 0), 1.0, color="gray"))
    ax[0].set_aspect("equal")
    ax[0].set_title("Streamlines (psi = const), flow left to right")

    ax[1].loglog(h, [q["linf"] for q in rows], "o-", label="max error")
    ax[1].loglog(h, [q["rms"] for q in rows], "s-", label="RMS error")
    ax[1].loglog(h, rows[0]["linf"] * (h / h[0]), "k--", label="slope 1 (reference)")
    ax[1].loglog(h, rows[0]["linf"] * (h / h[0]) ** 2, "k:", label="slope 2 (reference)")
    ax[1].set_xlabel("grid spacing h")
    ax[1].set_ylabel("error in psi vs exact solution")
    ax[1].set_title("Solver convergence")
    ax[1].legend()
    ax[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/solver_convergence.png", dpi=150)
    print("saved results/solver_convergence.png")