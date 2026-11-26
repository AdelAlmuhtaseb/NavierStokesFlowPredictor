from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from scipy import ndimage

from src.models import UNet

# Keep this coordinate window consistent with src/convert_airfrans.py.
X_MIN, X_MAX = -0.5, 1.5
Y_MIN, Y_MAX = -1.0, 1.0
GRID = 64
DEFAULT_CHECKPOINT = Path("results/unet_airfrans_g64.pt")

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

st.set_page_config(
    page_title="AeroLab | Flow Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------- Styling ---------------------------------

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    .stApp {
        background:
            radial-gradient(ellipse at 8% -8%, rgba(24, 121, 166, .22), transparent 34%),
            radial-gradient(ellipse at 95% 3%, rgba(19, 148, 128, .12), transparent 30%),
            linear-gradient(180deg, #07111d 0%, #081522 55%, #07111d 100%);
        color: #eaf2f8;
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 29, .78); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1b2b 0%, #091522 100%);
        border-right: 1px solid rgba(139, 184, 211, .14);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    .block-container { max-width: 1500px; padding-top: 1.7rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: -.035em; color: #f3f8fc; }
    p, label { color: #cad7e2; }
    .eyebrow {
        color: #73dfd0; text-transform: uppercase; letter-spacing: .17em;
        font-size: .68rem; font-weight: 800;
    }
    .brand-mark {
        display: flex; align-items: center; gap: .7rem; margin-bottom: 1.5rem;
    }
    .brand-icon {
        display: grid; place-items: center; width: 2.45rem; height: 2.45rem;
        border: 1px solid rgba(111, 225, 211, .4); border-radius: 11px;
        background: linear-gradient(145deg, rgba(20, 163, 160, .22), rgba(27, 105, 157, .18));
        color: #8af0df; font-size: 1.2rem;
    }
    .brand-name { color: #f0f6fb; font-size: 1.08rem; font-weight: 850; letter-spacing: -.035em; }
    .brand-sub { color: #7992a8; font-size: .63rem; letter-spacing: .16em; text-transform: uppercase; margin-top: .08rem; }
    .hero {
        position: relative; overflow: hidden; padding: 1.55rem 1.7rem;
        border: 1px solid rgba(130, 185, 211, .18); border-radius: 22px;
        background:
            radial-gradient(circle at 92% 0%, rgba(30, 174, 169, .17), transparent 34%),
            linear-gradient(125deg, rgba(17, 39, 59, .96), rgba(10, 25, 41, .96));
        box-shadow: 0 20px 55px rgba(0, 0, 0, .2);
        margin-bottom: 1.3rem;
    }
    .hero:after {
        content: ""; position: absolute; right: -3rem; top: -5rem; width: 19rem; height: 19rem;
        border: 1px solid rgba(105, 218, 204, .11); border-radius: 50%;
        box-shadow: 0 0 0 2.1rem rgba(105, 218, 204, .025), 0 0 0 4.2rem rgba(105, 218, 204, .018);
        pointer-events: none;
    }
    .hero-top { display:flex; align-items:center; gap:.55rem; flex-wrap:wrap; margin-bottom:.7rem; }
    .hero-tag {
        display:inline-flex; align-items:center; border:1px solid rgba(111, 225, 211, .23);
        border-radius:999px; padding:.29rem .65rem; color:#87e5d7;
        background:rgba(14, 143, 145, .09); font-size:.64rem; font-weight:800;
        letter-spacing:.12em; text-transform:uppercase;
    }
    .hero-title {
        color:#f5f9fc; font-size:clamp(2rem, 4.2vw, 3.3rem);
        font-weight:850; letter-spacing:-.055em; line-height:1.02;
        margin:.25rem 0 .55rem;
    }
    .hero-title span { color:#70dccc; }
    .hero-copy { max-width:760px; color:#a9bdcc; line-height:1.65; font-size:.94rem; }
    .section-head {
        display:flex; align-items:center; gap:.65rem; margin:.35rem 0 .85rem;
        color:#edf5fb; font-weight:800; letter-spacing:-.02em; font-size:1.02rem;
    }
    .section-no {
        display:inline-grid; place-items:center; width:1.65rem; height:1.5rem;
        border:1px solid rgba(105, 213, 199, .28); border-radius:7px;
        color:#70dccc; background:rgba(16, 143, 145, .09);
        font-size:.65rem; letter-spacing:.08em;
    }
    .surface {
        background:linear-gradient(145deg, rgba(17, 37, 56, .94), rgba(10, 24, 39, .94));
        border:1px solid rgba(132, 180, 207, .16); border-radius:17px;
        padding:1.05rem 1.15rem; box-shadow:0 12px 32px rgba(0,0,0,.15);
    }
    .surface-label {
        color:#70dccc; text-transform:uppercase; letter-spacing:.13em;
        font-size:.65rem; font-weight:850;
    }
    .surface-title { color:#f1f7fb; font-size:1.02rem; font-weight:800; margin:.4rem 0; }
    .muted { color:#91aabd; }
    .live-pill {
        display:inline-flex; align-items:center; gap:.45rem; border:1px solid rgba(84, 214, 173, .25);
        border-radius:999px; padding:.42rem .72rem; background:rgba(25, 147, 117, .1);
        color:#8be4c4; font-size:.7rem; font-weight:800; letter-spacing:.08em;
    }
    .live-dot {
        width:.45rem; height:.45rem; border-radius:50%; background:#58d7ad;
        box-shadow:0 0 10px rgba(88, 215, 173, .7); display:inline-block;
    }
    .hint {
        color:#8da5b8; font-size:.78rem; line-height:1.55;
        border-left:2px solid rgba(105, 213, 199, .55); padding-left:.75rem;
    }
    [data-testid="stMetric"] {
        background:linear-gradient(145deg, rgba(17, 38, 57, .92), rgba(11, 26, 42, .92));
        border:1px solid rgba(130, 178, 208, .16); border-radius:14px; padding:.85rem 1rem;
    }
    [data-testid="stMetricLabel"] { color:#91aabd !important; }
    [data-testid="stMetricValue"] { color:#f2f7fb !important; font-weight:800; }
    div[data-testid="stFileUploader"] section {
        border:1px dashed rgba(111, 188, 211, .38); border-radius:14px;
        background:rgba(8, 22, 36, .55);
    }
    div[data-testid="stFileUploader"] section:hover { border-color:#67d8ca; }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background:#72ddcf; border-color:#72ddcf;
    }
    [data-testid="stAlert"] { border-radius:12px; }
    [data-testid="stStatusWidget"] { border-radius:12px; }
    hr { border-color:rgba(132, 180, 207, .15); }
    .footer {
        border-top:1px solid rgba(130,178,208,.14); margin-top:2.2rem;
        padding-top:1rem; color:#6f899f; font-size:.68rem;
        letter-spacing:.08em; text-transform:uppercase;
    }
    @keyframes softRise {
        from { opacity:0; transform:translateY(7px); }
        to { opacity:1; transform:translateY(0); }
    }
    .result-intro { animation:softRise .35s ease-out; }
    @media (max-width: 800px) {
        .block-container { padding-top:1rem; }
        .hero { padding:1.2rem; }
        .hero-title { font-size:2.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Core helpers -------------------------------

def open_uploaded_image(blob: bytes) -> Image.Image:
    """Open an image and composite transparency onto a white background."""
    image = Image.open(io.BytesIO(blob)).convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, image).convert("RGB")


def image_to_mask(image: Image.Image, grid: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a clean, high-contrast, horizontally oriented airfoil silhouette
    into the body-mask channel used by the AirfRANS training pipeline.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if min(gray.shape) < 24:
        raise ValueError("The uploaded image is too small. Please use a larger image.")

    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    if background >= 127:
        candidate = gray < max(70, background - 35)
    else:
        candidate = gray > min(185, background + 35)

    candidate = ndimage.binary_closing(candidate, iterations=2)
    labels, count = ndimage.label(candidate)
    if count == 0:
        raise ValueError("I couldn't find the airfoil. Use a dark, filled airfoil on a light background.")

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(np.argmax(sizes))
    component = ndimage.binary_fill_holes(component)

    if component.sum() > 0.55 * component.size:
        raise ValueError("Too much of the image was detected as foreground. Use a clean airfoil silhouette.")

    rows, cols = np.where(component)
    x0, x1 = int(cols.min()), int(cols.max())
    y0, y1 = int(rows.min()), int(rows.max())
    chord_pixels = float(x1 - x0)
    height_pixels = float(y1 - y0)

    if chord_pixels < 12 or height_pixels < 2:
        raise ValueError("The detected outline is too small to use.")
    if chord_pixels / max(height_pixels, 1.0) < 1.5:
        raise ValueError("Show the airfoil side-on, with its long chord direction horizontal. Crop away labels and axes.")

    x_coords = np.linspace(X_MIN, X_MAX, grid)
    y_coords = np.linspace(Y_MIN, Y_MAX, grid)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    pixel_x = x0 + xx * chord_pixels
    pixel_y = (y0 + y1) / 2.0 - yy * chord_pixels

    sampled = ndimage.map_coordinates(
        component.astype(np.float32),
        [pixel_y, pixel_x],
        order=0,
        mode="constant",
        cval=0.0,
    )
    mask = (sampled > 0.5).astype(np.float32)
    if mask.sum() < 3:
        raise ValueError("The outline disappeared on the model grid. Try a tighter crop around the airfoil.")

    return mask, component.astype(np.uint8)


def naca_thickness_shape(x: np.ndarray) -> np.ndarray:
    """Unit-thickness NACA four-digit half-thickness distribution."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return 5.0 * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


def naca_camber_line(x: np.ndarray, m: float, p: float) -> np.ndarray:
    """NACA four-digit mean camber line, with x measured in chord lengths."""
    x = np.asarray(x, dtype=float)
    if abs(m) < 1e-12:
        return np.zeros_like(x)
    left = x <= p
    y = np.empty_like(x)
    y[left] = (m / p**2) * (2.0 * p * x[left] - x[left] ** 2)
    y[~left] = (m / (1.0 - p) ** 2) * (
        (1.0 - 2.0 * p) + 2.0 * p * x[~left] - x[~left] ** 2
    )
    return y


def naca_camber_slope(x: np.ndarray, m: float, p: float) -> np.ndarray:
    """Derivative of the NACA four-digit mean camber line."""
    x = np.asarray(x, dtype=float)
    if abs(m) < 1e-12:
        return np.zeros_like(x)
    left = x <= p
    slope = np.empty_like(x)
    slope[left] = (2.0 * m / p**2) * (p - x[left])
    slope[~left] = (2.0 * m / (1.0 - p) ** 2) * (p - x[~left])
    return slope


def estimate_naca_parameters(component: np.ndarray) -> dict:
    """
    Estimate an equivalent NACA four-digit profile from a filled binary
    silhouette. This is a geometric fit, not a parameter prediction by the U-Net.
    """
    rows, cols = np.where(component > 0)
    if len(cols) < 100:
        raise ValueError("The detected airfoil outline has too few pixels for shape fitting.")

    x0, x1 = int(cols.min()), int(cols.max())
    chord = float(x1 - x0)
    if chord < 40:
        raise ValueError("Please upload a larger airfoil image (at least 40 pixels across the chord).")

    x_samples = np.linspace(0.02, 0.98, 121)
    upper, lower, kept_x = [], [], []
    y_mid = (float(rows.min()) + float(rows.max())) / 2.0

    for x_chord in x_samples:
        col = int(round(x0 + x_chord * chord))
        col = min(max(col, 0), component.shape[1] - 1)
        occupied = np.flatnonzero(component[:, col] > 0)
        if occupied.size < 2:
            continue
        upper.append((y_mid - float(occupied.min())) / chord)
        lower.append((y_mid - float(occupied.max())) / chord)
        kept_x.append(float(x_chord))

    x = np.asarray(kept_x, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    if len(x) < 30:
        raise ValueError("The outline is too broken or thin to estimate NACA parameters.")

    raw_camber = 0.5 * (upper + lower)
    half_thickness = 0.5 * (upper - lower)

    # Fit the NACA camber curve together with a small image offset and tilt.
    # Treating the offset/tilt as nuisance parameters avoids confusing real
    # camber near the leading edge with a slanted image baseline.
    region = (x >= 0.05) & (x <= 0.95)
    x_region = x[region]
    measured_camber = raw_camber[region]
    if len(x_region) < 30:
        raise ValueError("The outline does not contain enough points to fit its camber line.")

    best_fit = None
    for candidate_p in np.linspace(0.10, 0.90, 161):
        camber_basis = naca_camber_line(x_region, 1.0, float(candidate_p))
        design = np.column_stack(
            [camber_basis, np.ones_like(x_region), x_region - 0.5]
        )
        coefficients, *_ = np.linalg.lstsq(design, measured_camber, rcond=None)
        residual = measured_camber - design @ coefficients
        score = float(np.mean(residual**2))
        if best_fit is None or score < best_fit[0]:
            best_fit = (score, float(candidate_p), coefficients)

    _, camber_position, coefficients = best_fit
    camber_ratio, camber_offset, camber_tilt = (float(value) for value in coefficients)
    camber = raw_camber - (
        camber_offset + camber_tilt * (x - 0.5)
    )
    if abs(camber_ratio) < 0.0015:
        camber_ratio, camber_position = 0.0, 0.4

    thickness_basis = naca_thickness_shape(x_region)
    measured_half_thickness = half_thickness[region]
    denominator = float(np.dot(thickness_basis, thickness_basis))
    if denominator <= 1e-12:
        raise ValueError("The uploaded outline does not contain a measurable thickness profile.")
    thickness_ratio = max(0.0, float(np.dot(thickness_basis, measured_half_thickness) / denominator))

    fitted_camber = naca_camber_line(x[region], camber_ratio, camber_position)
    fitted_half_thickness = thickness_ratio * naca_thickness_shape(x[region])
    camber_rmse = float(np.sqrt(np.mean((camber[region] - fitted_camber) ** 2)))
    thickness_rmse = float(np.sqrt(np.mean((half_thickness[region] - fitted_half_thickness) ** 2)))

    x_fit = np.linspace(0.001, 0.999, 300)
    yt = thickness_ratio * naca_thickness_shape(x_fit)
    yc = naca_camber_line(x_fit, camber_ratio, camber_position)
    theta = np.arctan(naca_camber_slope(x_fit, camber_ratio, camber_position))
    x_upper = x_fit - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x_fit + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    naca_code = None
    if camber_ratio >= -0.0005:
        camber_digit = int(round(max(0.0, camber_ratio) * 100.0))
        position_digit = 0 if camber_digit == 0 else int(round(camber_position * 10.0))
        thickness_digits = int(round(thickness_ratio * 100.0))
        if 0 <= camber_digit <= 9 and 0 <= position_digit <= 9 and 0 <= thickness_digits <= 99:
            naca_code = f"NACA {camber_digit}{position_digit}{thickness_digits:02d}"

    return {
        "m": camber_ratio,
        "p": camber_position,
        "t": thickness_ratio,
        "naca_code": naca_code,
        "camber_rmse": camber_rmse,
        "thickness_rmse": thickness_rmse,
        "x_measured": x,
        "upper_measured": upper - baseline,
        "lower_measured": lower - baseline,
        "x_upper_fit": x_upper,
        "y_upper_fit": y_upper,
        "x_lower_fit": x_lower,
        "y_lower_fit": y_lower,
        "samples": len(x),
    }


def plot_naca_fit(fit: dict):
    """Compare the uploaded silhouette outline with its fitted NACA profile."""
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#0b1927")
    ax.set_facecolor("#0b1927")
    ax.plot(fit["x_measured"], fit["upper_measured"], color="#dceaf4", lw=1.5, label="Image outline")
    ax.plot(fit["x_measured"], fit["lower_measured"], color="#dceaf4", lw=1.5)
    ax.plot(fit["x_upper_fit"], fit["y_upper_fit"], "--", color="#70dccc", lw=1.8, label="NACA fit")
    ax.plot(fit["x_lower_fit"], fit["y_lower_fit"], "--", color="#70dccc", lw=1.8)
    ax.axhline(0, color="#8da6bb", lw=0.6, alpha=0.35)
    ax.set_xlim(0, 1)
    ax.set_xlabel("x / chord", color="#9eb2c5")
    ax.set_ylabel("y / chord", color="#9eb2c5")
    ax.tick_params(colors="#9eb2c5", labelsize=8)
    ax.grid(color="#8da6bb", alpha=0.10, linewidth=0.5)
    ax.legend(facecolor="#102437", edgecolor="#294258", labelcolor="#dceaf4", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#294258")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_bytes: bytes, checkpoint_hash: str):
    """Load and cache the trained project checkpoint."""
    del checkpoint_hash  # The hash is part of the cache key.
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location=DEVICE, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=DEVICE)

    if int(checkpoint["in_ch"]) != 4 or int(checkpoint["out_ch"]) != 3:
        raise ValueError("This app expects the AirfRANS U-Net checkpoint with 4 inputs and 3 outputs.")

    model = UNet(
        in_ch=int(checkpoint["in_ch"]),
        out_ch=int(checkpoint["out_ch"]),
        base=int(checkpoint.get("base", 16)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    model.eval()

    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32, device=DEVICE)
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32, device=DEVICE)
    return model, mean, std


def sync_device() -> None:
    """Wait for asynchronous accelerator work before measuring elapsed time."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def predict(
    mask: np.ndarray,
    alpha_deg: float,
    inlet_speed: float,
    loaded_model,
) -> tuple[np.ndarray, float]:
    """Predict u/U, v/U and Cp for the selected geometry and operating point."""
    model, mean, std = loaded_model
    alpha = np.deg2rad(alpha_deg)
    inputs = np.stack(
        [
            mask,
            np.full_like(mask, np.cos(alpha), dtype=np.float32),
            np.full_like(mask, np.sin(alpha), dtype=np.float32),
            np.full_like(mask, inlet_speed / 100.0, dtype=np.float32),
        ],
        axis=0,
    )
    tensor = torch.from_numpy(inputs).unsqueeze(0).to(DEVICE)

    sync_device()
    start = time.perf_counter()
    with torch.inference_mode():
        normalized = model(tensor)
        prediction = normalized * std + mean
    sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    fields = prediction.squeeze(0).detach().cpu().numpy()
    if fields.shape[0] != 3:
        raise ValueError(f"Expected 3 output channels, but the checkpoint returned {fields.shape[0]}.")
    return fields, elapsed_ms


def plot_fields(fields: np.ndarray, mask: np.ndarray):
    """Render four scientific views: speed, Cp, horizontal velocity and vertical velocity."""
    u_over_u, v_over_u, cp = fields
    speed_ratio = np.sqrt(u_over_u**2 + v_over_u**2)
    x = np.linspace(X_MIN, X_MAX, GRID)
    y = np.linspace(Y_MIN, Y_MAX, GRID)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    fluid = mask < 0.5

    panels = [
        ("FLOW SPEED", speed_ratio, "turbo", "speed / inlet speed"),
        ("PRESSURE COEFFICIENT", cp, "coolwarm", "Cp"),
        ("HORIZONTAL VELOCITY", u_over_u, "RdBu_r", "u / inlet speed"),
        ("VERTICAL VELOCITY", v_over_u, "RdBu_r", "v / inlet speed"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#0b1927")
    for index, (ax, (title, values, cmap, bar_label)) in enumerate(zip(axes.flat, panels)):
        ax.set_facecolor("#0b1927")
        shown = np.ma.masked_where(~fluid, values)
        image = ax.imshow(
            shown,
            extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
            origin="lower",
            cmap=cmap,
            interpolation="bilinear",
            aspect="auto",
        )
        ax.contour(xx, yy, mask, levels=[0.5], colors="#f5f9fc", linewidths=1.1)

        # Streamlines are overlaid on the speed field using the model's predicted
        # velocity components; they are not an additional time-dependent simulation.
        if index == 0:
            u_plot = np.ma.masked_where(~fluid, u_over_u)
            v_plot = np.ma.masked_where(~fluid, v_over_u)
            try:
                ax.streamplot(
                    x, y, u_plot, v_plot, density=1.0,
                    color=(1, 1, 1, 0.52), linewidth=0.48, arrowsize=0.55,
                )
            except (ValueError, RuntimeError):
                pass

        ax.set_title(title, color="#eaf2f8", fontsize=9, fontweight="bold", pad=10)
        ax.set_xlabel("x / chord", color="#9eb2c5", fontsize=8)
        ax.set_ylabel("y / chord", color="#9eb2c5", fontsize=8)
        ax.tick_params(colors="#8da6bb", labelsize=7)
        ax.grid(color="#8da6bb", alpha=0.08, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("#294258")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        colorbar.ax.tick_params(labelsize=7, colors="#b9cad8")
        colorbar.set_label(bar_label, color="#b9cad8", fontsize=8)

    fig.tight_layout(pad=2.0)
    return fig


# ------------------------------- Sidebar controls -------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">
          <div class="brand-icon">✦</div>
          <div>
            <div class="brand-name">AeroLab</div>
            <div class="brand-sub">Flow intelligence</div>
          </div>
        </div>
        <div class="eyebrow">Simulation controls</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Operating point")
    alpha_deg = st.slider(
        "Angle of attack",
        min_value=-10.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        format="%.1f°",
        help="The direction of the incoming flow relative to the airfoil chord.",
    )
    inlet_speed = st.slider(
        "Inlet velocity",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        format="%d m/s",
        help="The reference incoming air speed.",
    )
    st.markdown(
        '<div class="hint">Predictions refresh automatically after you change either control. '
        'No Analyze button is required.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Model configuration")
    checkpoint_upload = st.file_uploader(
        "Optional checkpoint override",
        type=["pt"],
        help="Normally leave this empty. AeroLab automatically uses results/unet_airfrans_g64.pt when it is present.",
    )
    if checkpoint_upload is not None:
        st.success("Custom checkpoint selected")
    elif DEFAULT_CHECKPOINT.is_file():
        st.success("Project checkpoint detected")
        st.caption("Using results/unet_airfrans_g64.pt")
    else:
        st.warning("Checkpoint not found")
        st.caption("Add the project checkpoint in results/ or upload a compatible .pt file.")
    st.caption(f"Compute device: {DEVICE.upper()}")
    st.divider()
    st.markdown(
        '<div class="muted" style="font-size:.73rem;line-height:1.55">'
        'Research prototype · U-Net surrogate trained on AirfRANS examples</div>',
        unsafe_allow_html=True,
    )


# -------------------------------- Main workspace --------------------------------

checkpoint_ready = checkpoint_upload is not None or DEFAULT_CHECKPOINT.is_file()
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-top">
        <span class="hero-tag">AirfRANS dataset</span>
        <span class="hero-tag">64 × 64 inference grid</span>
        <span class="hero-tag">U-Net surrogate</span>
      </div>
      <div class="hero-title">AeroLab <span>Flow Studio</span></div>
      <div class="hero-copy">
        Upload an airfoil silhouette to estimate its NACA four-digit geometry, generate the corresponding equations, and inspect the U-Net pressure and velocity predictions at your selected operating point.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, status_col = st.columns([1.18, 0.82], gap="large")
with input_col:
    st.markdown(
        '<div class="section-head"><span class="section-no">01</span> Airfoil geometry</div>',
        unsafe_allow_html=True,
    )
    uploaded_image = st.file_uploader(
        "Upload a side-on airfoil silhouette",
        type=["png", "jpg", "jpeg", "bmp"],
        help="A dark, filled airfoil on a plain light background works best. Keep the chord horizontal.",
        key="airfoil_upload",
    )
    if uploaded_image is not None:
        try:
            preview = open_uploaded_image(uploaded_image.getvalue())
            st.image(preview, caption=f"Input profile · {uploaded_image.name}", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not open the uploaded image: {exc}")
    else:
        st.markdown(
            """
            <div class="surface" style="min-height:220px;display:flex;align-items:center;justify-content:center;text-align:center">
              <div>
                <div style="font-size:2.4rem;color:#70dccc;margin-bottom:.3rem">＋</div>
                <div class="surface-title">Add an airfoil profile</div>
                <div class="muted" style="font-size:.8rem">PNG or JPG · side-on silhouette · high contrast</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with status_col:
    st.markdown(
        '<div class="section-head"><span class="section-no">02</span> Live operating point</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="surface">
          <div class="surface-label">Current conditions</div>
          <div style="display:flex;gap:2rem;flex-wrap:wrap;margin:1rem 0">
            <div>
              <div class="muted" style="font-size:.68rem;letter-spacing:.1em">ANGLE OF ATTACK</div>
              <div style="font-size:2rem;font-weight:850;color:#f2f7fb">{alpha_deg:.1f}<span style="font-size:1rem;color:#9eb2c5">°</span></div>
            </div>
            <div>
              <div class="muted" style="font-size:.68rem;letter-spacing:.1em">INLET VELOCITY</div>
              <div style="font-size:2rem;font-weight:850;color:#f2f7fb">{inlet_speed}<span style="font-size:1rem;color:#9eb2c5"> m/s</span></div>
            </div>
          </div>
          <hr>
          <div style="display:flex;justify-content:space-between;gap:.8rem;flex-wrap:wrap;align-items:center">
            <span class="muted" style="font-size:.76rem">Prediction mode</span>
            <span class="live-pill"><span class="live-dot"></span> AUTO-UPDATE ON</span>
          </div>
          <div class="muted" style="font-size:.77rem;margin-top:.65rem">
            {("Trained checkpoint ready" if checkpoint_ready else "Waiting for trained checkpoint")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class="hint">
          Change either slider in the sidebar. The model will re-run with the new
          angle and speed while keeping the same uploaded geometry.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------- Automatic prediction refresh -------------------------

if uploaded_image is None:
    for key in (
        "_prediction_signature", "airfoil_mask", "airfoil_component",
        "airfoil_fields", "inference_ms", "prediction_conditions",
        "prediction_error", "airfoil_source_name", "naca_fit", "geometry_fit_error",
    ):
        st.session_state.pop(key, None)
else:
    image_bytes = uploaded_image.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    checkpoint_bytes = checkpoint_upload.getvalue() if checkpoint_upload is not None else (
        DEFAULT_CHECKPOINT.read_bytes() if DEFAULT_CHECKPOINT.is_file() else None
    )
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest() if checkpoint_bytes else "missing"

    # Streamlit reruns the script when a slider changes. This signature makes
    # each distinct geometry/condition/checkpoint combination run exactly once.
    signature = (image_hash, float(alpha_deg), int(inlet_speed), checkpoint_hash)

    if st.session_state.get("_prediction_signature") != signature:
        st.session_state["_prediction_signature"] = signature
        st.session_state.pop("airfoil_fields", None)
        st.session_state.pop("prediction_error", None)
        st.session_state["airfoil_source_name"] = uploaded_image.name

        try:
            with st.spinner("Updating the flow prediction for the new operating point..."):
                source_image = open_uploaded_image(image_bytes)
                mask, component = image_to_mask(source_image)
                st.session_state["airfoil_mask"] = mask
                st.session_state["airfoil_component"] = component
                try:
                    st.session_state["naca_fit"] = estimate_naca_parameters(component)
                    st.session_state.pop("geometry_fit_error", None)
                except Exception as fit_exc:
                    st.session_state.pop("naca_fit", None)
                    st.session_state["geometry_fit_error"] = str(fit_exc)

                if checkpoint_bytes is None:
                    st.session_state["prediction_error"] = (
                        "The geometry is ready, but the trained checkpoint was not found. "
                        "Add results/unet_airfrans_g64.pt or upload a compatible checkpoint in the sidebar."
                    )
                else:
                    loaded = load_model(checkpoint_bytes, checkpoint_hash)
                    fields, inference_ms = predict(mask, alpha_deg, inlet_speed, loaded)
                    st.session_state["airfoil_fields"] = fields
                    st.session_state["inference_ms"] = inference_ms
                    st.session_state["prediction_conditions"] = (alpha_deg, inlet_speed)
        except Exception as exc:
            st.session_state.pop("airfoil_fields", None)
            st.session_state["prediction_error"] = str(exc)

# --------------------------------- Results ---------------------------------

if uploaded_image is None:
    st.write("")
    st.markdown(
        """
        <div class="surface" style="margin-top:1.1rem">
          <div class="surface-label">Prediction workspace</div>
          <div class="surface-title">Ready for your first flow prediction</div>
          <div class="muted" style="font-size:.85rem;line-height:1.65">
            Upload an airfoil silhouette to begin. Once the image and trained checkpoint
            are available, AeroLab will generate the flow fields automatically and refresh
            them whenever you change the operating conditions.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif "prediction_error" in st.session_state:
    st.warning(st.session_state["prediction_error"])

if "airfoil_mask" in st.session_state:
    mask = st.session_state["airfoil_mask"]
    st.write("")
    st.markdown(
        '<div class="section-head"><span class="section-no">04</span> Geometry preprocessing</div>',
        unsafe_allow_html=True,
    )
    geom_col, geom_info = st.columns([1, 1], gap="large")
    with geom_col:
        fig_mask, ax_mask = plt.subplots(figsize=(8, 2.45), facecolor="#0b1927")
        ax_mask.set_facecolor("#0b1927")
        ax_mask.imshow(
            mask,
            extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
            origin="lower",
            cmap="gray_r",
            interpolation="nearest",
            aspect="auto",
        )
        ax_mask.set_xlim(X_MIN, X_MAX)
        ax_mask.set_ylim(-0.45, 0.45)
        ax_mask.set_xlabel("x / chord", color="#9eb2c5")
        ax_mask.set_ylabel("y / chord", color="#9eb2c5")
        ax_mask.tick_params(colors="#9eb2c5", labelsize=8)
        for spine in ax_mask.spines.values():
            spine.set_color("#294258")
        fig_mask.tight_layout()
        st.pyplot(fig_mask, use_container_width=True)
        plt.close(fig_mask)
    with geom_info:
        st.markdown(
            """
            <div class="surface">
              <div class="surface-label">Input mapping</div>
              <div class="surface-title">Silhouette detected and normalized</div>
              <div class="muted" style="font-size:.82rem;line-height:1.65">
                The uploaded profile is converted into the 64 × 64 body-mask channel.
                The selected angle and inlet speed are supplied as additional model inputs.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        g1, g2 = st.columns(2)
        g1.metric("Grid resolution", "64 × 64")
        g2.metric("Body cells", f"{int(mask.sum()):,}")


if "naca_fit" in st.session_state:
    fit = st.session_state["naca_fit"]
    m, p, t = fit["m"], fit["p"], fit["t"]

    st.write("")
    st.markdown(
        '<div class="section-head"><span class="section-no">03</span> NACA shape parameters & equation</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Estimated from the uploaded silhouette using a geometric fit to the NACA four-digit family. "
        "These are best-fit geometric estimates from the image, not outputs of the U-Net. Arbitrary profiles may not belong exactly to the NACA four-digit family."
    )

    p_label = "— (symmetric profile)" if abs(m) < 0.0005 else f"{p * 100:.1f}%"
    c1, c2, c3 = st.columns(3)
    c1.metric("Maximum camber", f"{m * 100:.2f}%")
    c2.metric("Camber position", p_label)
    c3.metric("Maximum thickness", f"{t * 100:.2f}%")

    if fit["naca_code"]:
        st.success(f"Nearest four-digit designation: {fit['naca_code']}")
    else:
        st.info(
            "The fitted profile does not map cleanly to a standard positive-camber NACA four-digit designation. "
            "The signed camber and thickness estimates are still shown."
        )

    fit_col, fit_details = st.columns([1.35, 0.65], gap="large")
    with fit_col:
        fig_fit = plot_naca_fit(fit)
        st.pyplot(fig_fit, use_container_width=True)
        plt.close(fig_fit)
    with fit_details:
        st.markdown(
            """
            <div class="surface">
              <div class="surface-label">Geometry fit diagnostics</div>
              <div class="surface-title">Image outline vs. NACA reconstruction</div>
              <div class="muted" style="font-size:.8rem;line-height:1.65">
                Solid line: measured image outline. Dashed line: reconstructed NACA four-digit profile.
                Lower residuals indicate a closer geometric match.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Camber-line RMSE", f"{fit['camber_rmse'] * 100:.3f}% chord")
        st.metric("Half-thickness RMSE", f"{fit['thickness_rmse'] * 100:.3f}% chord")
        st.metric("Outline samples", f"{fit['samples']}")

    with st.expander("View generated NACA equations", expanded=True):
        st.caption("All coordinates are dimensionless and normalized by chord length (x/c).")
        st.markdown("**Thickness distribution with the estimated thickness parameter**")
        st.latex(
            fr"y_t(x)=5({t:.5f})\left(0.2969\sqrt{{x}}-0.1260x-0.3516x^2+0.2843x^3-0.1015x^4\right)"
        )
        st.markdown("**Mean camber line with the estimated camber and position**")
        if abs(m) < 0.0005:
            st.latex(r"y_c(x)=0")
        else:
            st.latex(
                fr"y_c(x)=\begin{{cases}}"
                fr"\dfrac{{{m:.5f}}}{{{p:.4f}^2}}\left(2({p:.4f})x-x^2\right), & 0\leq x\leq {p:.4f}\\"
                fr"\dfrac{{{m:.5f}}}{{(1-{p:.4f})^2}}\left((1-2({p:.4f}))+2({p:.4f})x-x^2\right), & {p:.4f}<x\leq 1"
                fr"\end{{cases}}"
            )
        st.markdown("**Upper and lower surface coordinates**")
        st.latex(
            r"x_u=x-y_t\sin\theta,\quad y_u=y_c+y_t\cos\theta"
        )
        st.latex(
            r"x_l=x+y_t\sin\theta,\quad y_l=y_c-y_t\cos\theta"
        )
        st.latex(
            r"\theta=\tan^{-1}\left(\dfrac{dy_c}{dx}\right)"
        )
elif "geometry_fit_error" in st.session_state:
    st.write("")
    st.markdown(
        '<div class="section-head"><span class="section-no">03</span> NACA shape parameters & equation</div>',
        unsafe_allow_html=True,
    )
    st.warning(f"Flow prediction can still run, but the NACA shape fit could not be completed: {st.session_state['geometry_fit_error']}")

if "airfoil_fields" in st.session_state:
    fields = st.session_state["airfoil_fields"]
    conditions = st.session_state["prediction_conditions"]

    st.write("")
    st.markdown(
        '<div class="section-head result-intro"><span class="section-no">05</span> Predicted flow field</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="live-pill"><span class="live-dot"></span> Prediction updated for current controls</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Angle of attack", f"{conditions[0]:.1f}°")
    m2.metric("Inlet velocity", f"{conditions[1]} m/s")
    m3.metric("Inference time", f"{st.session_state.get('inference_ms', 0):.2f} ms")

    fig = plot_fields(fields, mask)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Predicted outputs: u/U∞, v/U∞ and Cp. Flow speed is derived from the two velocity components. "
        "Streamlines are drawn from the predicted velocity field. This is a neural surrogate prediction, "
        "not a new Navier–Stokes CFD solve."
    )

st.markdown(
    """
    <div class="footer">
      AEROLAB FLOW STUDIO &nbsp;·&nbsp; NAVIER–STOKES FLOW PREDICTOR
      &nbsp;·&nbsp; Research prototype &nbsp;·&nbsp; AirfRANS-trained U-Net surrogate
    </div>
    """,
    unsafe_allow_html=True,
)
