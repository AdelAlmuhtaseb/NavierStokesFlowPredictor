from __future__ import annotations

import base64
import hashlib
import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
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
PREFERRED_CHECKPOINT = Path("results/unet_airfrans_g64_b32_e300.pt")
FALLBACK_CHECKPOINT = Path("results/unet_airfrans_g64.pt")

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

st.set_page_config(
    page_title="AeroLab Flow Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------- Styling ---------------------------------

# Palette shared by the page styles, the matplotlib figures and the flow diagram.
INK = "#132238"
MUTED = "#52647A"
BLUE = "#1B54C4"
RED = "#C23B4E"
LINE = "#D5DDE6"
PAPER = "#EEF2F6"
SURFACE = "#FFFFFF"
TUNNEL = "#0F2A4A"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=B612:wght@400;700&display=swap');
    :root {{ color-scheme: light; }}
    .stApp {{ background:{PAPER}; color:{INK}; }}
    [data-testid="stHeader"] {{ background:{PAPER}; }}
    .block-container {{ max-width:1320px; padding-top:3.6rem; padding-bottom:2.5rem; }}
    h1, h2, h3, .display {{
        font-family:'B612', 'Source Sans 3', sans-serif !important;
        color:{INK} !important; letter-spacing:-.02em;
    }}
    .stApp p, .stApp label, .stApp li {{ color:{INK}; }}
    .stApp [data-testid="stCaptionContainer"] {{ opacity:1; }}
    .stApp [data-testid="stCaptionContainer"] p {{ color:{MUTED}; }}
    .stApp .katex {{ color:{INK}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {LINE}; }}
    [data-testid="stSidebar"] h3 {{ font-size:.95rem; padding-top:.4rem; }}
    .brand {{ display:flex; align-items:center; gap:.7rem; margin-bottom:1.3rem; }}
    .brand svg {{ flex:none; }}
    .brand-name {{ font-size:1.25rem; font-weight:700; line-height:1.1; }}
    .note {{ color:{MUTED} !important; font-size:.82rem; line-height:1.55; }}

    /* Page header */
    .masthead {{ margin-bottom:1.2rem; }}
    .masthead h1 {{ font-size:clamp(1.7rem, 3vw, 2.35rem); padding:0 0 .35rem; }}
    .masthead p {{ color:{MUTED}; max-width:66ch; margin:0; line-height:1.6; }}

    /* Upload panel and flow diagram share one height on wide screens */
    .st-key-intake {{
        background:{SURFACE}; border:1px solid {LINE}; border-radius:6px; padding:1.1rem 1.2rem;
    }}
    .panel-title {{ font-size:1rem; font-weight:700; margin-bottom:.2rem; }}
    .tunnel {{ background:{TUNNEL}; border-radius:6px; padding:1.1rem 1.3rem .9rem; }}
    .tunnel-head {{ display:flex; justify-content:space-between; gap:1.5rem; flex-wrap:wrap; }}
    .tunnel .panel-title {{ color:#FFFFFF !important; }}
    .readouts {{ display:flex; gap:2rem; }}
    .readout span {{ display:block; color:#A9C1DE; font-size:.8rem; text-align:right; }}
    .readout strong {{
        display:block; color:#FFFFFF; font-family:'B612', sans-serif; font-size:1.7rem;
        font-variant-numeric:tabular-nums; text-align:right; line-height:1.3;
    }}
    .tunnel img {{ display:block; width:100%; height:auto; }}
    .stApp .tunnel p {{ color:#A9C1DE; font-size:.8rem; margin:0; }}

    /* Widgets */
    [data-testid="stFileUploaderDropzone"] {{
        background:{PAPER}; border:1px dashed #8CA0BA; border-radius:6px; color:{INK};
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{ border-color:{BLUE}; }}
    [data-testid="stFileUploaderDropzone"] button {{
        background:{SURFACE}; color:{INK}; border:1px solid {LINE};
    }}
    [data-testid="stFileUploader"] small, [data-testid="stFileUploaderDropzoneInstructions"] * {{ color:{MUTED}; }}
    [data-testid="stFileUploaderFile"] *, [data-testid="stFileChip"] * {{ color:{INK}; }}
    [data-testid="stFileChip"] {{ background:{SURFACE}; }}
    [data-testid="stSlider"] [role="slider"] {{ background:{BLUE}; }}
    [data-testid="stSliderThumbValue"] {{ color:{BLUE}; }}
    [data-baseweb="tab-list"], [role="tablist"] {{ gap:1.6rem; }}
    [role="tab"] p {{ font-size:1rem; font-weight:600; }}
    [role="tab"][aria-selected="true"] p {{ color:{BLUE}; }}
    [data-testid="stMetric"] {{
        background:{SURFACE}; border:1px solid {LINE}; border-radius:6px; padding:.75rem 1rem;
    }}
    .stApp [data-testid="stMetricLabel"] p {{ color:{MUTED}; }}
    [data-testid="stMetricValue"] {{
        color:{INK}; font-family:'B612', sans-serif; font-size:1.55rem;
        font-variant-numeric:tabular-nums;
    }}
    [data-testid="stExpander"] details {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:6px; }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary:hover {{
        background:{SURFACE}; color:{INK};
    }}

    [data-testid="stTooltipIcon"] button, [data-testid="stTooltipIcon"] svg {{ color:{MUTED} !important; }}
    [data-testid="stAlertContainer"] {{ background:#E8F0FC; border-radius:6px; }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ background:#E3F4EA; }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background:#FDF0D3; }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ background:#FBE5E8; }}
    .stApp [data-testid="stAlertContainer"] p {{ color:{INK}; }}

    /* Results */
    .designation {{ font-size:2rem; font-weight:700; margin:.2rem 0 .1rem; }}
    .empty {{
        border:1px dashed #A8B8CC; border-radius:6px; padding:2.2rem 1.5rem; margin-top:.6rem;
        text-align:center;
    }}
    .empty .panel-title {{ margin-bottom:.4rem; }}
    .stApp .empty p {{ color:{MUTED}; max-width:60ch; margin:0 auto; line-height:1.6; }}
    .footer {{
        border-top:1px solid {LINE}; margin-top:2.5rem; padding-top:.9rem;
        color:{MUTED}; font-size:.8rem;
    }}
    @media (max-width:640px) {{
        .block-container {{ padding-left:1rem; padding-right:1rem; }}
        .readout span, .readout strong {{ text-align:left; }}
    }}
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
    baseline = camber_offset + camber_tilt * (x - 0.5)
    camber = raw_camber - baseline
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


def style_axes(ax) -> None:
    """Apply the shared light style to a matplotlib axis."""
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(LINE)


def plot_naca_fit(fit: dict):
    """Compare the uploaded silhouette outline with its fitted NACA profile."""
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
    ax.plot(fit["x_measured"], fit["upper_measured"], color=INK, lw=1.6, label="Image outline")
    ax.plot(fit["x_measured"], fit["lower_measured"], color=INK, lw=1.6)
    ax.plot(fit["x_upper_fit"], fit["y_upper_fit"], "--", color=RED, lw=1.8, label="NACA fit")
    ax.plot(fit["x_lower_fit"], fit["y_lower_fit"], "--", color=RED, lw=1.8)
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("x / chord", color=MUTED)
    ax.set_ylabel("y / chord", color=MUTED)
    style_axes(ax)
    ax.grid(color=LINE, linewidth=0.6)
    ax.legend(facecolor=SURFACE, edgecolor=LINE, labelcolor=INK, fontsize=8)
    fig.tight_layout()
    return fig


def flow_diagram(alpha_deg: float, inlet_speed: float, fit: dict | None) -> str:
    """
    Draw the chosen operating point as an SVG data URI: incoming flow at the
    selected angle (arrow length follows speed) meeting the airfoil outline.
    This illustrates the inputs only. It is not a model prediction.
    """
    nose_x, nose_y, chord = 330.0, 150.0, 360.0

    def path(xs, ys) -> str:
        return " ".join(f"{nose_x + chord * x:.1f},{nose_y - chord * y:.1f}" for x, y in zip(xs, ys))

    if fit is None:
        x = np.linspace(0.0, 1.0, 120)
        half = 0.12 * naca_thickness_shape(x)
        outline = path(np.r_[x, x[::-1]], np.r_[half, -half[::-1]])
        dash = ' stroke-dasharray="6 5"'
    else:
        outline = path(
            np.r_[fit["x_upper_fit"], fit["x_lower_fit"][::-1]],
            np.r_[fit["y_upper_fit"], fit["y_lower_fit"][::-1]],
        )
        dash = ""

    angle = np.deg2rad(alpha_deg)
    dx, dy = float(np.cos(angle)), float(-np.sin(angle))
    length = 60.0 + 1.4 * inlet_speed

    def arrow(x0, y0, x1, y1, color, width) -> str:
        head = [
            (x1, y1),
            (x1 - 11 * dx + 5 * dy, y1 - 11 * dy - 5 * dx),
            (x1 - 11 * dx - 5 * dy, y1 - 11 * dy + 5 * dx),
        ]
        points = " ".join(f"{px:.1f},{py:.1f}" for px, py in head)
        return (
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1 - 8 * dx:.1f}" y2="{y1 - 8 * dy:.1f}" '
            f'stroke="{color}" stroke-width="{width}"/><polygon points="{points}" fill="{color}"/>'
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 300">',
        '<defs><pattern id="g" width="30" height="30" patternUnits="userSpaceOnUse">'
        '<path d="M30 0H0V30" fill="none" stroke="#FFFFFF" stroke-opacity="0.07"/></pattern></defs>',
        '<rect width="720" height="300" fill="url(#g)"/>',
    ]
    for offset in (-92, -46, 46, 92):
        cx, cy = 140.0, nose_y + offset
        parts.append(arrow(cx - dx * length / 2, cy - dy * length / 2,
                           cx + dx * length / 2, cy + dy * length / 2, "#6F9BD6", 1.6))
    # Chord reference line, then the flow vector that arrives at the leading edge.
    parts.append(f'<line x1="70" y1="{nose_y}" x2="705" y2="{nose_y}" stroke="#A9C1DE" '
                 'stroke-width="1" stroke-dasharray="3 6"/>')
    parts.append(arrow(nose_x - dx * length, nose_y - dy * length, nose_x, nose_y, "#FFC857", 2.6))
    radius = min(95.0, length * 0.75)
    if abs(alpha_deg) > 0.01:
        sweep = 0 if alpha_deg > 0 else 1
        parts.append(
            f'<path d="M {nose_x - radius:.1f} {nose_y} A {radius:.1f} {radius:.1f} 0 0 {sweep} '
            f'{nose_x - radius * dx:.1f} {nose_y - radius * dy:.1f}" fill="none" '
            'stroke="#FFC857" stroke-width="1.5"/>'
        )
    parts.append(f'<polygon points="{outline}" fill="#FFFFFF" fill-opacity="0.12" '
                 f'stroke="#FFFFFF" stroke-width="2" stroke-linejoin="round"{dash}/>')
    parts.append("</svg>")
    encoded = base64.b64encode("".join(parts).encode()).decode()
    return f"data:image/svg+xml;base64,{encoded}"


def find_default_checkpoint() -> Path | None:
    """Prefer the newer AirfRANS checkpoint; retain the earlier model as fallback."""
    for candidate in (PREFERRED_CHECKPOINT, FALLBACK_CHECKPOINT):
        if candidate.is_file():
            return candidate
    return None


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
        ("Flow speed", speed_ratio, "turbo", "speed / inlet speed"),
        ("Pressure coefficient", cp, "coolwarm", "Cp"),
        ("Horizontal velocity", u_over_u, "RdBu_r", "u / inlet speed"),
        ("Vertical velocity", v_over_u, "RdBu_r", "v / inlet speed"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6), facecolor=SURFACE)
    for index, (ax, (title, values, cmap, bar_label)) in enumerate(zip(axes.flat, panels)):
        ax.set_facecolor("#C9D3DF")
        shown = np.ma.masked_where(~fluid, values)
        image = ax.imshow(
            shown,
            extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
            origin="lower",
            cmap=cmap,
            interpolation="bilinear",
            aspect="equal",
        )
        ax.contour(xx, yy, mask, levels=[0.5], colors=INK, linewidths=1.1)

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

        ax.set_title(title, color=INK, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("x / chord", color=MUTED, fontsize=8)
        ax.set_ylabel("y / chord", color=MUTED, fontsize=8)
        style_axes(ax)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        colorbar.ax.tick_params(labelsize=7, colors=MUTED)
        colorbar.set_label(bar_label, color=MUTED, fontsize=8)

    fig.tight_layout(pad=2.0)
    return fig



def make_flow_animation(
    fields: np.ndarray,
    mask: np.ndarray,
    alpha_deg: float,
    inlet_speed: float,
    frames: int = 40,
) -> bytes:
    """
    Advect visual tracer particles through the predicted steady velocity field.
    This is a visualization of the model prediction, not an unsteady CFD solve.
    """
    u = np.nan_to_num(fields[0], nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(fields[1], nan=0.0, posinf=0.0, neginf=0.0)
    fluid = mask < 0.5
    x = np.linspace(X_MIN, X_MAX, GRID)
    y = np.linspace(Y_MIN, Y_MAX, GRID)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    speed = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(9.0, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(
        np.ma.masked_where(~fluid, speed),
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        origin="lower",
        cmap="turbo",
        interpolation="bilinear",
        aspect="equal",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("speed / inlet speed", color=INK)
    colorbar.ax.tick_params(colors=MUTED, labelsize=8)

    if np.any(mask > 0.5) and np.any(mask < 0.5):
        ax.contourf(
            xx, yy, mask, levels=[0.5, 1.5],
            colors=["#e8eef3"], alpha=1.0,
        )
        ax.contour(
            xx, yy, mask, levels=[0.5],
            colors=[INK], linewidths=0.9,
        )

    try:
        ax.streamplot(
            x, y,
            np.ma.masked_where(~fluid, u),
            np.ma.masked_where(~fluid, v),
            density=1.0,
            color=(1.0, 1.0, 1.0, 0.28),
            linewidth=0.45,
            arrowsize=0.5,
        )
    except (ValueError, RuntimeError):
        pass

    particle_count = 18
    seed_y = np.linspace(Y_MIN + 0.14, Y_MAX - 0.14, particle_count)
    positions = np.column_stack(
        [np.full(particle_count, X_MIN + 0.02), seed_y]
    ).astype(float)
    respawns = np.zeros(particle_count, dtype=int)
    histories = [[] for _ in range(particle_count)]

    trails = LineCollection(
        [], colors=[(0.10, 0.22, 0.30, 0.62)], linewidths=1.15, zorder=4
    )
    ax.add_collection(trails)
    particles = ax.scatter(
        positions[:, 0], positions[:, 1],
        s=22, c="#ffffff", edgecolors=BLUE,
        linewidths=0.75, zorder=5,
    )

    def sample(field, points, order=1):
        ix = (points[:, 0] - X_MIN) / (X_MAX - X_MIN) * (GRID - 1)
        iy = (points[:, 1] - Y_MIN) / (Y_MAX - Y_MIN) * (GRID - 1)
        return ndimage.map_coordinates(
            field, [iy, ix], order=order, mode="nearest"
        )

    def update(_frame):
        nonlocal positions
        u0 = sample(u, positions)
        v0 = sample(v, positions)
        midpoint = positions + 0.5 * 0.03 * np.column_stack([u0, v0])
        um = sample(u, midpoint)
        vm = sample(v, midpoint)
        candidate = positions + 0.03 * np.column_stack([um, vm])

        in_domain = (
            (candidate[:, 0] >= X_MIN) & (candidate[:, 0] <= X_MAX)
            & (candidate[:, 1] >= Y_MIN) & (candidate[:, 1] <= Y_MAX)
        )
        is_fluid = sample(mask, candidate, order=0) < 0.5
        valid = in_domain & is_fluid & np.isfinite(candidate).all(axis=1)

        for i in range(particle_count):
            if valid[i]:
                positions[i] = candidate[i]
            else:
                respawns[i] += 1
                new_y = seed_y[(i + respawns[i]) % particle_count]
                positions[i] = (X_MIN + 0.02, new_y)
                histories[i].clear()

            histories[i].append(positions[i].copy())
            if len(histories[i]) > 9:
                histories[i].pop(0)

        trails.set_segments(
            [np.asarray(history) for history in histories if len(history) > 1]
        )
        particles.set_offsets(positions)
        return trails, particles

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("x / chord", color=INK)
    ax.set_ylabel("y / chord", color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(
        f"Predicted airflow  |  AoA {alpha_deg:.1f}°  |  inlet {inlet_speed} m/s",
        color=INK, fontsize=11, fontweight="bold", pad=10,
    )
    ax.text(
        0.02, 0.025, "Tracer paths through the steady predicted velocity field",
        transform=ax.transAxes, color=INK, fontsize=7,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.72},
    )
    style_axes(ax)
    fig.tight_layout()

    animation = FuncAnimation(
        fig, update, frames=frames, interval=70, blit=False, repeat=True
    )
    output = io.BytesIO()
    animation.save(output, writer=PillowWriter(fps=14), dpi=82)
    plt.close(fig)
    return output.getvalue()


# ------------------------------- Sidebar controls -------------------------------

default_checkpoint = find_default_checkpoint()

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
          <svg width="46" height="30" viewBox="0 0 46 30" aria-hidden="true">
            <path d="M2 17C6 7 22 6 44 18C24 20 8 24 2 17Z" fill="{BLUE}"/>
            <path d="M3 5C16 0 29 3 42 9" fill="none" stroke="{BLUE}" stroke-width="1.6" stroke-linecap="round"/>
          </svg>
          <div class="brand-name display">AeroLab</div>
        </div>
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
        '<p class="note">The prediction updates as soon as you change either control.</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Model")
    checkpoint_upload = st.file_uploader(
        "Optional checkpoint override",
        type=["pt"],
        help="Normally leave this empty. AeroLab prefers results/unet_airfrans_g64_b32_e300.pt and falls back to results/unet_airfrans_g64.pt.",
    )
    if checkpoint_upload is not None:
        st.success("Custom checkpoint selected")
    elif default_checkpoint is not None:
        st.success("Project checkpoint detected")
        st.caption(f"Using {default_checkpoint.as_posix()}")
    else:
        st.warning("Checkpoint not found")
        st.caption("Add the preferred or fallback checkpoint in results/, or upload a compatible .pt file.")
    st.caption(f"Compute device: {DEVICE.upper()}")


# -------------------------------- Main workspace --------------------------------

st.markdown(
    """
    <div class="masthead">
      <h1>Airfoil Flow Studio</h1>
      <p>Upload an airfoil silhouette to see the predicted velocity and pressure around it,
      along with a NACA four-digit fit of its shape and the matching equations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, diagram_col = st.columns([0.8, 1.2], gap="medium")
with input_col:
    with st.container(key="intake"):
        st.markdown('<div class="panel-title display">Airfoil image</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Upload a side-on airfoil silhouette",
            type=["png", "jpg", "jpeg", "bmp"],
            help="A dark, filled airfoil on a plain light background works best. Keep the chord horizontal.",
            key="airfoil_upload",
        )
        if uploaded_image is not None:
            try:
                preview = open_uploaded_image(uploaded_image.getvalue())
                st.image(preview, caption=uploaded_image.name)
            except Exception as exc:
                st.error(f"Could not open the uploaded image: {exc}")
        else:
            st.markdown(
                '<p class="note">Use a dark, filled shape on a plain light background, seen side-on '
                'with the chord horizontal. Crop away labels and axes.</p>',
                unsafe_allow_html=True,
            )


# ------------------------- Automatic prediction refresh -------------------------

if uploaded_image is None:
    for key in (
        "_prediction_signature", "airfoil_mask", "airfoil_component",
        "airfoil_fields", "inference_ms", "prediction_conditions",
        "prediction_error", "airfoil_source_name", "naca_fit", "geometry_fit_error",
        "flow_animation_gif",
    ):
        st.session_state.pop(key, None)
else:
    image_bytes = uploaded_image.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    checkpoint_bytes = checkpoint_upload.getvalue() if checkpoint_upload is not None else (
        default_checkpoint.read_bytes() if default_checkpoint is not None else None
    )
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest() if checkpoint_bytes else "missing"

    # Streamlit reruns the script when a slider changes. This signature makes
    # each distinct geometry/condition/checkpoint combination run exactly once.
    signature = (image_hash, float(alpha_deg), int(inlet_speed), checkpoint_hash)

    if st.session_state.get("_prediction_signature") != signature:
        st.session_state["_prediction_signature"] = signature
        st.session_state.pop("flow_animation_gif", None)
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
                        "The geometry is ready, but no trained checkpoint was found. "
                        "Add results/unet_airfrans_g64_b32_e300.pt (preferred) or "
                        "results/unet_airfrans_g64.pt (fallback), or upload a compatible checkpoint."
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

# ------------------------------- Operating point -------------------------------

# Filled after processing so the diagram can show the outline fitted to this upload.
with diagram_col:
    diagram_fit = st.session_state.get("naca_fit")
    outline_note = (
        "The outline is the NACA fit of your image."
        if diagram_fit is not None
        else "The dashed outline is a NACA 0012 placeholder."
    )
    st.markdown(
        f"""
        <div class="tunnel">
          <div class="tunnel-head">
            <div class="panel-title display">Operating point</div>
            <div class="readouts">
              <div class="readout"><span>Angle of attack</span><strong>{alpha_deg:.1f}°</strong></div>
              <div class="readout"><span>Inlet velocity</span><strong>{inlet_speed} m/s</strong></div>
            </div>
          </div>
          <img src="{flow_diagram(alpha_deg, inlet_speed, diagram_fit)}"
               alt="Incoming flow at {alpha_deg:.1f} degrees to the airfoil chord">
          <p>Arrows show the incoming flow you set, not a prediction. {outline_note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------- Results ---------------------------------

if uploaded_image is None:
    st.markdown(
        """
        <div class="empty">
          <div class="panel-title display">No flow field yet</div>
          <p>Upload an airfoil image above. You will get the predicted flow field, a shape fit
          with its equations, and a view of the grid the model reads.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.write("")
    if "prediction_error" in st.session_state:
        st.warning(st.session_state["prediction_error"])

    flow_tab, shape_tab, grid_tab = st.tabs(["Flow field", "Shape and equations", "Model input"])

    with flow_tab:
        if "airfoil_fields" in st.session_state:
            fields = st.session_state["airfoil_fields"]
            mask = st.session_state["airfoil_mask"]
            conditions = st.session_state["prediction_conditions"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Angle of attack", f"{conditions[0]:.1f}°")
            m2.metric("Inlet velocity", f"{conditions[1]} m/s")
            m3.metric("Inference time", f"{st.session_state.get('inference_ms', 0):.2f} ms")

            fig = plot_fields(fields, mask)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Predicted outputs: u/U∞, v/U∞ and Cp. Flow speed is derived from the two velocity components. "
                "Streamlines are drawn from the predicted velocity field. This is a neural surrogate prediction, "
                "not a new Navier–Stokes CFD solve."
            )

            with st.expander("Animate predicted airflow", expanded=False):
                st.caption(
                    "Generate a looping particle animation from the predicted velocity field. "
                    "This visualizes the steady model prediction; it is not a time-dependent CFD solve."
                )
                if st.button("Generate airflow animation", key="generate_appv2_flow_animation"):
                    try:
                        with st.spinner("Advecting particles through the predicted flow field..."):
                            st.session_state["flow_animation_gif"] = make_flow_animation(
                                fields, mask, conditions[0], conditions[1]
                            )
                        st.success("Animation is ready.")
                    except Exception as exc:
                        st.error(f"Could not create the animation: {exc}")

                gif_bytes = st.session_state.get("flow_animation_gif")
                if gif_bytes:
                    gif_b64 = base64.b64encode(gif_bytes).decode("ascii")
                    st.markdown(
                        f'<img src="data:image/gif;base64,{gif_b64}" '
                        'style="width:100%;height:auto;border-radius:12px;border:1px solid #294258" />',
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "Download airflow GIF",
                        data=gif_bytes,
                        file_name="aerolab_appv2_predicted_airflow.gif",
                        mime="image/gif",
                        key="download_appv2_flow_animation",
                    )
        else:
            st.info("There is no flow field for this image yet.")

    with shape_tab:
        if "naca_fit" in st.session_state:
            fit = st.session_state["naca_fit"]
            m, p, t = fit["m"], fit["p"], fit["t"]

            if fit["naca_code"]:
                st.markdown(
                    f'<div class="designation display">{fit["naca_code"]}</div>'
                    '<p class="note">Nearest four-digit designation</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "The fitted profile does not map cleanly to a standard positive-camber NACA four-digit designation. "
                    "The signed camber and thickness estimates are still shown."
                )

            p_label = "— (symmetric profile)" if abs(m) < 0.0005 else f"{p * 100:.1f}%"
            c1, c2, c3 = st.columns(3)
            c1.metric("Maximum camber", f"{m * 100:.2f}%")
            c2.metric("Camber position", p_label)
            c3.metric("Maximum thickness", f"{t * 100:.2f}%")

            fit_col, fit_details = st.columns([1.5, 0.5], gap="medium")
            with fit_col:
                fig_fit = plot_naca_fit(fit)
                st.pyplot(fig_fit)
                plt.close(fig_fit)
            with fit_details:
                st.metric("Camber-line RMSE (% chord)", f"{fit['camber_rmse'] * 100:.3f}")
                st.metric("Half-thickness RMSE (% chord)", f"{fit['thickness_rmse'] * 100:.3f}")
                st.metric("Outline samples", f"{fit['samples']}")

            st.caption(
                "Estimated from the uploaded silhouette using a geometric fit to the NACA four-digit family. "
                "These are best-fit geometric estimates from the image, not outputs of the U-Net. Arbitrary profiles "
                "may not belong exactly to the NACA four-digit family. Lower RMSE means a closer match."
            )

            with st.expander("View generated NACA equations", expanded=False):
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
                        fr"\dfrac{{{m:.5f}}}{{{p:.4f}^2}}\left(2({p:.4f})x-x^2\right), & 0\leq x\leq {p:.4f}\\[10pt]"
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
            st.warning(f"Flow prediction can still run, but the NACA shape fit could not be completed: {st.session_state['geometry_fit_error']}")
        else:
            st.info("There is no shape fit for this image yet.")

    with grid_tab:
        if "airfoil_mask" in st.session_state:
            mask = st.session_state["airfoil_mask"]
            g1, g2 = st.columns(2)
            g1.metric("Grid resolution", "64 × 64")
            g2.metric("Body cells", f"{int(mask.sum()):,}")

            fig_mask, ax_mask = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
            ax_mask.imshow(
                mask,
                extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                origin="lower",
                cmap="Blues",
                interpolation="nearest",
                aspect="auto",
            )
            ax_mask.set_xlim(X_MIN, X_MAX)
            ax_mask.set_ylim(-0.45, 0.45)
            ax_mask.set_xlabel("x / chord", color=MUTED)
            ax_mask.set_ylabel("y / chord", color=MUTED)
            style_axes(ax_mask)
            fig_mask.tight_layout()
            st.pyplot(fig_mask)
            plt.close(fig_mask)
            st.caption(
                "The uploaded profile is converted into the 64 × 64 body-mask channel. "
                "The selected angle and inlet speed are supplied as additional model inputs."
            )
        else:
            st.info("The airfoil outline has not been detected in this image.")

st.markdown(
    '<div class="footer">AeroLab Flow Studio is a research prototype. '
    'Predictions come from a U-Net surrogate trained on AirfRANS examples.</div>',
    unsafe_allow_html=True,
)
