import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

SIZE = 64   # small, so it runs fast on a CPU
N = 500

def naca4(m, p, t, n=100):
    x = np.linspace(0, 1, n)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    return np.concatenate([xu, xl[::-1]]), np.concatenate([yu, yl[::-1]])

def render(m, p, t, size=SIZE):
    x, y = naca4(m, p, t)
    px = x*size*0.7 + size*0.15
    py = size/2 - y*size*0.7
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).polygon(list(zip(px, py)), fill=255)
    return np.array(img, dtype=np.float32) / 255

def blur(a, iters=4):
    for _ in range(iters):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
               + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
    return a

def fake_flow(mask):
    # NOT real physics, just smooth fields that depend on the shape
    b = blur(mask)
    p = b
    u = 1 - blur(mask, 8)
    v = np.gradient(blur(mask, 6), axis=0)
    return np.stack([p, u, v]).astype(np.float32)

rng = np.random.default_rng(0)
params = np.stack([rng.uniform(0.0, 0.06, N),
                   rng.uniform(0.2, 0.6, N),
                   rng.uniform(0.08, 0.2, N)], axis=1).astype(np.float32)
imgs = np.stack([render(*row) for row in params])
fields = np.stack([fake_flow(m) for m in imgs])

Path("data").mkdir(exist_ok=True)
np.savez("data/synthetic.npz", imgs=imgs, fields=fields, params=params)
print("saved", imgs.shape, fields.shape, params.shape)