import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def naca4(m, p, t, n=100):
    x = np.linspace(0, 1, n)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    return np.concatenate([xu, xl[::-1]]), np.concatenate([yu, yl[::-1]])

dev = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available()
       else "cpu")

d = np.load("data/synthetic.npz")
X = torch.tensor(d["imgs"]).unsqueeze(1)
lo = np.array([0.0, 0.2, 0.08], dtype=np.float32)
hi = np.array([0.06, 0.6, 0.2], dtype=np.float32)
P = d["params"]
Y = torch.tensor((P - lo) / (hi - lo), dtype=torch.float32)

ntr = int(0.8 * len(X))
Xtr, Ytr = X[:ntr], Y[:ntr]
Xva, Yva = X[ntr:].to(dev), Y[ntr:].to(dev)

net = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 3)).to(dev)
opt = torch.optim.Adam(net.parameters(), 1e-3)

for epoch in range(30):
    net.train()
    perm = torch.randperm(ntr)
    for i in range(0, ntr, 32):
        idx = perm[i:i + 32]
        loss = nn.functional.mse_loss(net(Xtr[idx].to(dev)), Ytr[idx].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch+1}/30  loss {loss.item():.5f}")

net.eval()
with torch.no_grad():
    pred = net(Xva).cpu().numpy() * (hi - lo) + lo
true = Yva.cpu().numpy() * (hi - lo) + lo

err = np.abs(pred - true).mean(0)
for name, e, r in zip(["camber", "camber position", "thickness"], err, hi - lo):
    print(f"{name}: mean abs error {e:.4f} ({100*e/r:.1f}% of range)")

# overlay: true outline vs predicted outline
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
for k in range(3):
    xt, yt_ = naca4(*true[k])
    xp, yp = naca4(*pred[k])
    ax[k].plot(xt, yt_, "b-", label="true")
    ax[k].plot(xp, yp, "r--", label="predicted")
    ax[k].set_aspect("equal")
    ax[k].axis("off")
ax[0].legend()
plt.tight_layout()
plt.savefig("results/shape_overlay.png", dpi=150)
print("saved results/shape_overlay.png")

# the equation for the first validation airfoil, from the PREDICTED parameters
m, p, t = pred[0]
print("\nPredicted airfoil equation (NACA 4-digit form):")
print(f"  thickness: y_t(x) = 5*{t:.3f}*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1036*x^4)")
print(f"  camber line, x < {p:.3f}:  y_c(x) = {m/p**2:.3f}*(2*{p:.3f}*x - x^2)")
print(f"  camber line, x >= {p:.3f}: y_c(x) = {m/(1-p)**2:.3f}*((1 - 2*{p:.3f}) + 2*{p:.3f}*x - x^2)")
print(f"  (true parameters were m={true[0][0]:.3f}, p={true[0][1]:.3f}, t={true[0][2]:.3f})")