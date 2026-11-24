import numpy as np


def naca4_points(m, p, t, n=100):
    """Closed NACA 4-digit outline, ordered clockwise starting at the trailing edge:
    lower surface (TE -> LE), then upper surface (LE -> TE)."""
    s = np.linspace(0, 1, n + 1)
    x = 0.5 * (1 - np.cos(np.pi * s))
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    if m < 1e-9:
        yc = np.zeros_like(x)
        th = np.zeros_like(x)
    else:
        yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
        dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
        th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    px = np.concatenate([xl[::-1], xu[1:]])
    py = np.concatenate([yl[::-1], yu[1:]])
    pts = np.column_stack([px, py])
    pts[-1] = pts[0]            # the trailing edge is closed
    return pts


def circle_points(N):
    phi = -np.linspace(0, 2*np.pi, N + 1)       # clockwise
    return np.column_stack([np.cos(phi), np.sin(phi)])


def geometry(pts):
    x1, y1 = pts[:-1, 0], pts[:-1, 1]
    x2, y2 = pts[1:, 0], pts[1:, 1]
    dx, dy = x2 - x1, y2 - y1
    return dict(x1=x1, y1=y1, s=np.hypot(dx, dy), th=np.arctan2(dy, dx),
                mx=(x1 + x2) / 2, my=(y1 + y2) / 2)


def influence(g, P):
    """Velocity induced at points P (M x 2) by every panel, for unit source strength
    and unit vortex strength (vortex positive clockwise). Returns four (M x N) arrays."""
    c, s_ = np.cos(g["th"]), np.sin(g["th"])
    dx = P[:, 0, None] - g["x1"][None, :]
    dy = P[:, 1, None] - g["y1"][None, :]
    xl = dx * c + dy * s_                     # point in the panel's own frame
    zl = -dx * s_ + dy * c
    L = g["s"][None, :]
    ln = 0.5 * np.log((xl**2 + zl**2) / ((xl - L)**2 + zl**2))
    beta = np.arctan2(zl, xl - L) - np.arctan2(zl, xl)
    # source: (u, w) = (ln, beta)/2pi        vortex: (u, w) = (beta, -ln)/2pi
    su, sw = ln / (2*np.pi), beta / (2*np.pi)
    vu, vw = beta / (2*np.pi), -ln / (2*np.pi)
    return (su*c - sw*s_, su*s_ + sw*c, vu*c - vw*s_, vu*s_ + vw*c)


def solve_panels(pts, alpha_deg, kutta=True):
    g = geometry(pts)
    N = len(g["s"])
    P = np.column_stack([g["mx"], g["my"]])
    sx, sy, vx, vy = influence(g, P)
    # a panel acting on its own midpoint: put in the exact limits
    i = np.arange(N)
    sx[i, i] = sy[i, i] = vx[i, i] = vy[i, i] = 0.0
    th = g["th"]
    tx, ty, nx, ny = np.cos(th), np.sin(th), -np.sin(th), np.cos(th)
    sx[i, i], sy[i, i] = 0.5 * nx, 0.5 * ny        # source: normal velocity +1/2
    vx[i, i], vy[i, i] = 0.5 * tx, 0.5 * ty        # vortex: tangential velocity +1/2

    An = sx * nx[:, None] + sy * ny[:, None]
    At = sx * tx[:, None] + sy * ty[:, None]
    Bn = (vx * nx[:, None] + vy * ny[:, None]).sum(axis=1)
    Bt = (vx * tx[:, None] + vy * ty[:, None]).sum(axis=1)

    a = np.radians(alpha_deg)
    U = np.array([np.cos(a), np.sin(a)])
    if kutta:
        M = np.zeros((N + 1, N + 1))
        M[:N, :N], M[:N, N] = An, Bn
        M[N, :N] = At[0] + At[N - 1]
        M[N, N] = Bt[0] + Bt[N - 1]
        rhs = np.zeros(N + 1)
        rhs[:N] = -(U[0] * nx + U[1] * ny)
        rhs[N] = -(U[0] * (tx[0] + tx[N-1]) + U[1] * (ty[0] + ty[N-1]))
        sol = np.linalg.solve(M, rhs)
        sigma, gamma = sol[:N], sol[N]
        cond = np.linalg.cond(M)
    else:
        sigma = np.linalg.solve(An, -(U[0] * nx + U[1] * ny))
        gamma, cond = 0.0, np.linalg.cond(An)

    vt = U[0] * tx + U[1] * ty + At @ sigma + Bt * gamma
    return dict(g=g, sigma=sigma, gamma=gamma, cp=1 - vt**2, vt=vt, U=U,
                cl=2 * gamma * g["s"].sum(), cond=cond)


def velocity_at(sol, P):
    """Total velocity at arbitrary points P (M x 2), e.g. a grid around the airfoil."""
    sx, sy, vx, vy = influence(sol["g"], P)
    return (sol["U"][0] + sx @ sol["sigma"] + vx @ np.full(len(sol["sigma"]), sol["gamma"]),
            sol["U"][1] + sy @ sol["sigma"] + vy @ np.full(len(sol["sigma"]), sol["gamma"]))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path

    # 1) Cylinder: compare the velocity field with the exact solution at radius 1.5
    print("1) Cylinder: max error of the velocity field at r = 1.5 vs number of panels")
    phi = np.linspace(0, 2*np.pi, 73)[:-1]
    ring = np.column_stack([1.5*np.cos(phi), 1.5*np.sin(phi)])
    u_exact = 1 - np.cos(2*phi) / 1.5**2
    prev = None
    for N in (16, 32, 64, 128, 256):
        sol = solve_panels(circle_points(N), 0.0, kutta=False)
        u, v = velocity_at(sol, ring)
        err = np.abs(u - u_exact).max()
        order = "" if prev is None else f"order {np.log2(prev/err):.2f}"
        print(f"   N={N:4d}  error {err:.2e}  {order}")
        prev = err

    # 2) NACA 0012: lift against the thin-airfoil reference
    print("2) NACA 0012: lift coefficient vs reference 2*pi*sin(alpha)*(1+0.77*t)")
    for al in (0, 2, 5, 8):
        sol = solve_panels(naca4_points(0.0, 0.4, 0.12), al)
        ref = 2*np.pi*np.sin(np.radians(al))*(1 + 0.77*0.12)
        print(f"   alpha={al} deg  CL={sol['cl']:.3f}  reference={ref:.3f}  condition number of the system={sol['cond']:.0f}")

    # 3) Cambered airfoil
    sol0 = solve_panels(naca4_points(0.02, 0.4, 0.12), 0.0)
    print(f"3) NACA 2412 at alpha=0: CL={sol0['cl']:.3f} (thin-airfoil theory gives about 0.25)")

    # figure: pressure on the surface and the speed field, NACA 2412 at 5 degrees
    pts = naca4_points(0.02, 0.4, 0.12)
    sol = solve_panels(pts, 5.0)
    xs = np.linspace(-0.5, 1.5, 240)
    ys = np.linspace(-0.6, 0.6, 144)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    u, v = velocity_at(sol, P)
    speed = np.hypot(u, v).reshape(X.shape)
    inside = Path(pts).contains_points(P).reshape(X.shape)
    speed[inside] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    mx = sol["g"]["mx"]
    ax[0].plot(mx, sol["cp"], "o-", ms=3)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("x / chord")
    ax[0].set_ylabel("pressure coefficient Cp")
    ax[0].set_title(f"NACA 2412, 5 deg: surface pressure (CL = {sol['cl']:.2f})")
    ax[0].grid(alpha=0.3)
    im = ax[1].pcolormesh(X, Y, speed, shading="auto", cmap="viridis")
    ax[1].fill(pts[:, 0], pts[:, 1], color="white", ec="k", lw=0.8)
    ax[1].set_aspect("equal")
    ax[1].set_title("Flow speed around the airfoil (potential flow)")
    plt.colorbar(im, ax=ax[1], label="speed / free-stream speed")
    plt.tight_layout()
    plt.savefig("results/airfoil_panel.png", dpi=150)
    print("saved results/airfoil_panel.png")