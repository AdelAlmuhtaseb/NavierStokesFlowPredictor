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

# Keep this consistent with src/convert_airfrans.py.
X_MIN, X_MAX = -0.5, 1.5
Y_MIN, Y_MAX = -1.0, 1.0
GRID = 64
DEFAULT_CHECKPOINT = Path("results/unet_airfrans_g64.pt")

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

st.set_page_config(
    page_title="AeroLab | Airfoil Flow Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(19, 104, 148, .22), transparent 34%),
          radial-gradient(ellipse at 88% 8%, rgba(21, 143, 125, .13), transparent 30%),
          #07111d;
        color: #e8f0f8;
    }
    [data-testid="stHeader"] { background: rgba(7,17,29,.75); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1b2b 0%, #091522 100%);
        border-right: 1px solid rgba(142,190,220,.14);
    }
    .block-container { padding-top: 2.1rem; max-width: 1440px; }
    .eyebrow {
        color: #62d7c5; text-transform: uppercase; letter-spacing: .18em;
        font-size: .72rem; font-weight: 700;
    }
    .hero-title {
        color: #f3f8fc; font-size: clamp(2.1rem, 4vw, 3.55rem);
        line-height: 1.05; font-weight: 800; letter-spacing: -.045em;
        margin: .35rem 0 .65rem 0;
    }
    .hero-sub { color: #9eb2c5; font-size: 1rem; max-width: 760px; }
    .panel {
        background: linear-gradient(145deg, rgba(18,37,56,.94), rgba(10,25,40,.94));
        border: 1px solid rgba(130,178,208,.17);
        border-radius: 18px; padding: 1.15rem 1.25rem;
        box-shadow: 0 14px 36px rgba(0,0,0,.18);
    }
    .panel-label {
        color: #6edacb; text-transform: uppercase; letter-spacing: .12em;
        font-size: .68rem; font-weight: 800;
    }
    .muted { color: #9eb2c5; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0e9e9a, #1679b8);
        color: white; border: 0; border-radius: 12px;
        font-weight: 750; min-height: 3rem;
        box-shadow: 0 8px 22px rgba(13,139,159,.22);
    }
    div.stButton > button[kind="primary"]:hover {
        border: 1px solid #70e2d1; color: white;
        box-shadow: 0 8px 28px rgba(13,180,176,.3);
    }
    [data-testid="stMetric"] {
        background: rgba(17,37,56,.75); border: 1px solid rgba(130,178,208,.15);
        border-radius: 14px; padding: .85rem 1rem;
    }
    [data-testid="stFileUploader"] {
        background: rgba(10,24,39,.55); border-radius: 14px;
    }
    h1, h2, h3 { letter-spacing: -.025em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def open_uploaded_image(blob: bytes) -> Image.Image:
    """Open image and composite transparency onto a white background."""
    image = Image.open(io.BytesIO(blob)).convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, image).convert("RGB")


def image_to_mask(image: Image.Image, grid: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a clean, high-contrast, horizontally oriented airfoil silhouette
    into the mask channel used by convert_airfrans.py.

    Expected input: a dark filled airfoil on a light background, or the inverse.
    Chord is normalized to x = 0..1 and the profile is centered vertically.
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
        raise ValueError(
            "I couldn't find the airfoil. Use a dark, filled airfoil on a plain light background."
        )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(np.argmax(sizes))
    component = ndimage.binary_fill_holes(component)

    if component.sum() > 0.55 * component.size:
        raise ValueError(
            "Too much of the image was detected as foreground. Use a clean airfoil silhouette "
            "with a contrasting background."
        )

    rows, cols = np.where(component)
    x0, x1 = int(cols.min()), int(cols.max())
    y0, y1 = int(rows.min()), int(rows.max())
    chord_pixels = float(x1 - x0)
    height_pixels = float(y1 - y0)
    if chord_pixels < 12 or height_pixels < 2:
        raise ValueError("The detected outline is too small to use.")
    if chord_pixels / max(height_pixels, 1.0) < 1.5:
        raise ValueError(
            "Show the airfoil side-on, with its long chord direction horizontal. "
            "Crop away labels, axes, and other objects."
        )

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
        raise ValueError(
            "The outline disappeared when mapped to the model grid. Try a tighter crop "
            "around the airfoil."
        )
    return mask, component.astype(np.uint8)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_bytes: bytes, checkpoint_hash: str):
    """Load a trusted project checkpoint once and cache the model."""
    del checkpoint_hash  # Makes a replaced checkpoint invalidate the cache.
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location=DEVICE, weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=DEVICE)

    base = int(checkpoint.get("base", 16))
    model = UNet(
        in_ch=int(checkpoint["in_ch"]),
        out_ch=int(checkpoint["out_ch"]),
        base=base,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    model.eval()

    if int(checkpoint["in_ch"]) != 4 or int(checkpoint["out_ch"]) != 3:
        raise ValueError("This demo expects a 4-input / 3-output AirfRANS U-Net checkpoint.")

    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32, device=DEVICE)
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32, device=DEVICE)
    return model, mean, std


def predict(mask: np.ndarray, alpha_deg: float, inlet_speed: float, loaded_model):
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
    start = time.perf_counter()
    with torch.inference_mode():
        normalized = model(tensor)
        prediction = normalized * std + mean
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    fields = prediction.squeeze(0).detach().cpu().numpy()
    if fields.shape[0] != 3:
        raise ValueError(f"Expected 3 output channels, but the checkpoint returned {fields.shape[0]}.")
    return fields, elapsed_ms


def plot_fields(fields: np.ndarray, mask: np.ndarray):
    """Plot the four quantities produced or derived from the trained model."""
    u_over_u, v_over_u, cp = fields
    speed_ratio = np.sqrt(u_over_u**2 + v_over_u**2)
    xx, yy = np.meshgrid(
        np.linspace(X_MIN, X_MAX, GRID),
        np.linspace(Y_MIN, Y_MAX, GRID),
        indexing="xy",
    )
    fluid = mask < 0.5
    panels = [
        ("FLOW SPEED", np.ma.masked_where(~fluid, speed_ratio), "turbo", "speed / inlet speed"),
        ("PRESSURE COEFFICIENT", np.ma.masked_where(~fluid, cp), "coolwarm", "Cp"),
        ("HORIZONTAL VELOCITY", np.ma.masked_where(~fluid, u_over_u), "RdBu_r", "u / inlet speed"),
        ("VERTICAL VELOCITY", np.ma.masked_where(~fluid, v_over_u), "RdBu_r", "v / inlet speed"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#0c1a29")
    for ax, (title, values, cmap, colorbar_label) in zip(axes.flat, panels):
        ax.set_facecolor("#0c1a29")
        image = ax.imshow(
            values,
            extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
            origin="lower",
            cmap=cmap,
            interpolation="bilinear",
            aspect="auto",
        )
        ax.contour(xx, yy, mask, levels=[0.5], colors="#f4f8fb", linewidths=1.05)
        ax.set_title(title, color="#e8f0f8", fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel("x / chord", color="#9eb2c5", fontsize=8)
        ax.set_ylabel("y / chord", color="#9eb2c5", fontsize=8)
        ax.tick_params(colors="#8da6bb", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#294258")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        colorbar.ax.tick_params(labelsize=7, colors="#b9cad8")
        colorbar.set_label(colorbar_label, color="#b9cad8", fontsize=8)

    fig.tight_layout(pad=2.0)
    return fig


# ---------------------------- Sidebar controls ----------------------------

with st.sidebar:
    st.markdown('<div class="eyebrow">AEROLAB / CONTROL ROOM</div>', unsafe_allow_html=True)
    st.markdown("## Flow conditions")
    alpha_deg = st.slider(
        "Angle of attack", min_value=-10.0, max_value=20.0,
        value=5.0, step=0.5, format="%.1f°"
    )
    inlet_speed = st.slider(
        "Inlet velocity", min_value=10, max_value=100,
        value=50, step=5, format="%d m/s"
    )
    st.caption("Use conditions represented in the training data. Predictions outside its range may be unreliable.")
    st.divider()
    st.markdown("### Trained model")
    checkpoint_upload = st.file_uploader(
        "Upload teammate's checkpoint (.pt)",
        type=["pt"],
        help="You can also place the checkpoint at results/unet_airfrans_g64.pt in the project.",
    )
    st.caption(f"Compute device: **{DEVICE.upper()}**")
    st.divider()
    st.markdown(
        '<div class="muted" style="font-size:.78rem">Input format: clean side-on airfoil silhouette. '
        'A dark shape on a plain light background works best.</div>',
        unsafe_allow_html=True,
    )


# ------------------------------- Main screen -------------------------------

st.markdown('<div class="eyebrow">NEURAL CFD SURROGATE · AIRFRANS</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Airfoil Flow Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload an airfoil profile, set the incoming flow, and inspect the predicted '
    'velocity and pressure fields from your trained U-Net model.</div>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.12, 0.88], gap="large")
with left:
    st.markdown("### 01 / Geometry input")
    uploaded_image = st.file_uploader(
        "Drop an airfoil image here",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Use a side-view silhouette with the chord horizontal. Avoid screenshots containing labels or axes.",
        key="airfoil_upload",
    )
    if uploaded_image is not None:
        preview = open_uploaded_image(uploaded_image.getvalue())
        st.image(preview, caption="Uploaded profile", use_container_width=True)
    else:
        st.markdown(
            """
            <div class="panel" style="min-height:220px;display:flex;align-items:center;justify-content:center;text-align:center">
              <div>
                <div style="font-size:2.6rem;color:#69d5c7">＋</div>
                <div style="font-weight:750;color:#edf5fb">Upload an airfoil profile</div>
                <div class="muted" style="font-size:.82rem;margin-top:.35rem">
                  PNG or JPG · side-on view · plain contrasting background
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown("### 02 / Run setup")
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-label">Operating point</div>
          <div style="display:flex;gap:2rem;margin-top:1rem">
            <div><div class="muted" style="font-size:.75rem">ANGLE OF ATTACK</div>
              <div style="font-size:1.8rem;font-weight:800">{alpha_deg:.1f}°</div></div>
            <div><div class="muted" style="font-size:.75rem">INLET SPEED</div>
              <div style="font-size:1.8rem;font-weight:800">{inlet_speed} <span style="font-size:.9rem;color:#9eb2c5">m/s</span></div></div>
          </div>
          <hr style="border-color:rgba(130,178,208,.16);margin:1rem 0">
          <div class="panel-label">Model status</div>
          <div style="margin-top:.45rem;color:#b7c9d8;font-size:.9rem">
            Waiting for image analysis
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    run_clicked = st.button(
        "▶  ANALYZE AIRFOIL",
        type="primary",
        use_container_width=True,
        disabled=uploaded_image is None,
    )
    st.caption("The image is mapped to the model's 64 × 64 input grid.")

if run_clicked and uploaded_image is not None:
    try:
        with st.spinner("Detecting the outline and preparing the model input..."):
            source_image = open_uploaded_image(uploaded_image.getvalue())
            mask, detected_component = image_to_mask(source_image)
        st.session_state["airfoil_mask"] = mask
        st.session_state["airfoil_component"] = detected_component
        st.session_state["airfoil_source_name"] = uploaded_image.name

        checkpoint_bytes = None
        if checkpoint_upload is not None:
            checkpoint_bytes = checkpoint_upload.getvalue()
        elif DEFAULT_CHECKPOINT.exists():
            checkpoint_bytes = DEFAULT_CHECKPOINT.read_bytes()

        if checkpoint_bytes:
            digest = hashlib.sha256(checkpoint_bytes).hexdigest()
            with st.spinner("Running the trained U-Net..."):
                loaded = load_model(checkpoint_bytes, digest)
                fields, inference_ms = predict(mask, alpha_deg, inlet_speed, loaded)
            st.session_state["airfoil_fields"] = fields
            st.session_state["inference_ms"] = inference_ms
            st.session_state["prediction_conditions"] = (alpha_deg, inlet_speed)
            st.success("Prediction complete.")
        else:
            st.session_state.pop("airfoil_fields", None)
            st.info(
                "The outline was detected successfully. Real flow prediction will be enabled when "
                "the trained checkpoint is added, or when you upload the .pt file in the sidebar."
            )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

# Keep displaying the most recent result after Streamlit reruns.
if "airfoil_mask" in st.session_state:
    mask = st.session_state["airfoil_mask"]
    st.write("")
    st.markdown("### 03 / Detected geometry")
    geometry_col, detail_col = st.columns([1, 1])
    with geometry_col:
        fig_mask, ax_mask = plt.subplots(figsize=(8, 2.5), facecolor="#0c1a29")
        ax_mask.set_facecolor("#0c1a29")
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
        fig_mask.tight_layout()
        st.pyplot(fig_mask, use_container_width=True)
        plt.close(fig_mask)
    with detail_col:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-label">Geometry preprocessing</div>
              <div style="font-size:1.05rem;font-weight:750;margin:.55rem 0">Outline detected</div>
              <div class="muted" style="font-size:.86rem;line-height:1.6">
                The dominant silhouette has been mapped into the same normalized coordinate
                window used by the AirfRANS converter. The model receives a 64 × 64 body mask
                plus the selected flow conditions.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Grid resolution", "64 × 64")
        st.metric("Detected body cells", f"{int(mask.sum()):,}")

if "airfoil_fields" in st.session_state:
    fields = st.session_state["airfoil_fields"]
    conditions = st.session_state.get("prediction_conditions", (alpha_deg, inlet_speed))
    st.write("")
    st.markdown("### 04 / Predicted flow field")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Angle of attack", f"{conditions[0]:.1f}°")
    metric2.metric("Inlet velocity", f"{conditions[1]} m/s")
    metric3.metric("Inference time", f"{st.session_state.get('inference_ms', 0):.1f} ms")

    fig = plot_fields(fields, st.session_state["airfoil_mask"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Velocity channels are dimensionless (u/U∞ and v/U∞); speed is derived from them. "
        "Cp is the predicted pressure coefficient. These are model predictions, not a new CFD solve."
    )
elif "airfoil_mask" in st.session_state:
    st.write("")
    st.markdown("### 04 / Prediction workspace")
    st.markdown(
        """
        <div class="panel">
          <div class="panel-label">Waiting for trained weights</div>
          <div style="font-size:1.05rem;font-weight:750;margin:.55rem 0">
            Geometry is ready. The predictor needs your trained checkpoint.
          </div>
          <div class="muted" style="font-size:.87rem;line-height:1.65">
            Add the checkpoint named <b>unet_airfrans_g64.pt</b> inside the results folder,
            or upload the .pt file in the sidebar. Then click <b>Analyze Airfoil</b> again
            to generate the real velocity and pressure plots.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown(
    '<div style="border-top:1px solid rgba(130,178,208,.16);padding-top:1rem;color:#7891a7;'
    'font-size:.74rem">AEROLAB · NAVIER–STOKES FLOW PREDICTOR · Research prototype</div>',
    unsafe_allow_html=True,
)
