# Navier–Stokes Flow Predictor

A research prototype for predicting **two-dimensional airflow around an airfoil** with a neural-network surrogate trained on the [AirfRANS](https://airfrans.readthedocs.io/) dataset. AirfRANS contains Reynolds-averaged Navier–Stokes (RANS) simulation data.

The project includes **AeroLab Flow Studio**, a Streamlit application that accepts an airfoil silhouette and an operating point, then displays predicted velocity and pressure fields. It also estimates an equivalent NACA four-digit geometry from the uploaded outline.

> **Scientific scope:** the U-Net predicts fields learned from precomputed CFD examples. It does not run a new Navier–Stokes solver for each upload, and its particle animation is a visualization of the predicted steady velocity field rather than an unsteady CFD simulation.

## Features

### AeroLab Flow Studio (`appv2.py`)

- Upload a side-on airfoil silhouette (PNG, JPG, JPEG or BMP).
- Set the angle of attack (−10° to 20°) and inlet speed (10–100 m/s).
- Automatically refresh the prediction when the image or operating point changes.
- View predicted flow speed, pressure coefficient, horizontal velocity and vertical velocity, with streamlines derived from the predicted velocity field.
- Inspect the 64 × 64 body-mask grid supplied to the model.
- Generate an experimental particle-tracer animation in the Flow field tab. The animation illustrates particle paths through the steady predicted velocity field; it is not a time-accurate simulation.
- View the image-based NACA fit, its geometric error measures and the corresponding equations.

### Airfoil geometry estimation

The Shape and equations tab fits a standard NACA four-digit profile to the uploaded silhouette and reports:

- **m:** maximum camber as a fraction of chord;
- **p:** chordwise location of maximum camber;
- **t:** maximum thickness as a fraction of chord;
- the nearest NACA four-digit designation when the fitted values map to one; and
- camber-line and half-thickness fitting RMSE values.

This is a **geometric best-fit calculation performed separately from the U-Net**. It is not a neural-network prediction of the NACA parameters. An arbitrary airfoil may not belong to the NACA four-digit family, so the reported designation and parameters are approximations.

## How the flow model works

The AirfRANS point-cloud data is interpolated onto a regular **64 × 64** grid over:

- x/c from −0.5 to 1.5;
- y/c from −1.0 to 1.0.

The U-Net receives four input channels:

1. airfoil/body mask;
2. cos(α), where α is the angle of attack;
3. sin(α); and
4. inlet speed normalized by 100 m/s.

It predicts three fields:

- **u/U∞:** normalized horizontal velocity;
- **v/U∞:** normalized vertical velocity; and
- **Cp:** pressure coefficient, using the AirfRANS pressure target normalized by the dynamic-pressure scale.

The displayed flow-speed magnitude is derived from the two predicted velocity components:

**speed / U∞ = √[(u / U∞)² + (v / U∞)²]**

The application prefers the checkpoint `results/unet_airfrans_g64_b32_e300.pt` and falls back to `results/unet_airfrans_g64.pt`. A compatible checkpoint can also be selected in the sidebar. The app expects a four-input, three-output AirfRANS U-Net checkpoint.

## Quick start

### 1. Get the project and install dependencies

```bash
git clone https://github.com/AdelAlmuhtaseb/NavierStokesFlowPredictor.git
cd NavierStokesFlowPredictor
python -m pip install -r requirements.txt
```

### 2. Run AeroLab

```bash
python -m streamlit run appv2.py
```

Streamlit prints a local address (normally http://localhost:8501) to open in your browser.

### 3. Upload an airfoil

For best results, use a **filled, high-contrast airfoil silhouette** on a plain background. The airfoil should be viewed side-on, with its chord horizontal. Crop away labels, axes and other marks. A clean silhouette helps the image-processing step identify the body mask.

The app looks for the preferred checkpoint and then the fallback checkpoint in the `results/` directory. When neither is available, add a compatible `.pt` checkpoint using the optional checkpoint uploader in the sidebar.

## Training and evaluation

The repository includes scripts for downloading AirfRANS, converting its official splits, training a U-Net or FNO model, and evaluating a saved U-Net checkpoint.

Install the project requirements first:

```bash
python -m pip install -r requirements.txt
```

### 1. Download AirfRANS

```bash
python src/download.py
```

### 2. Convert the official splits separately

```bash
python src/convert_airfrans.py --root data/Dataset --out-dir data --grid 64
```

This creates:

- `data/airfrans_train_g64.npz` — the official 800-simulation training split;
- `data/airfrans_test_g64.npz` — the official 200-simulation test split.

The converter keeps these files separate by default. Do not combine the official splits when reporting an independent test result.

### 3. Train using only the official training split

```bash
python src/train.py --model unet --data data/airfrans_train_g64.npz --epochs 100 --batch 32 --lr 0.001 --sched cosine --base 16 --name unet_airfrans_g64_independent
```

The training script uses 80% of the supplied training file for training and the remaining 20% for validation. With the full 800-case training split, that is 640 training cases and 160 validation cases. The official test file is not needed during training or model selection.

To train the optional Fourier Neural Operator baseline, use `--model fno`. The Streamlit app currently loads the U-Net checkpoint format.

### 4. Evaluate once on the untouched official test split

```bash
python src/evaluate_airfrans_test.py --data data/airfrans_test_g64.npz --checkpoint results/unet_airfrans_g64_independent.pt
```

The evaluator reports per-simulation relative L2 errors over fluid cells for u/U∞, v/U∞ and Cp. It prints the mean, median and 90th percentile for each channel, a three-channel macro-average, and batch-average inference time per sample. It also writes:

```
results/airfrans_independent_test.csv
```

Use these metrics with the exact checkpoint and test protocol that produced them. Relative L2 error is an **error metric**, not an accuracy percentage. The earlier 7.14% result came from a workflow that used the official test cases for validation during training; it should be treated as a validation result, not an independent test result.

## Limitations and responsible interpretation

- The model predicts **2D** fields on a fixed 64 × 64 grid; it does not model a full three-dimensional flow.
- Predictions are learned from the coverage and assumptions of the AirfRANS training data. Results for geometries or operating conditions outside that coverage may be unreliable.
- The app does not directly calculate lift, drag, stall angle, or structural stability.
- The particle animation visualizes paths through one predicted steady velocity field. It does not represent transient vortex shedding or solve the time-dependent Navier–Stokes equations.
- NACA parameters are estimated from the uploaded image and depend on image quality, orientation, scale and how closely the geometry resembles a NACA four-digit profile.
- Inference time depends on the device and software environment; compare timings only under clearly stated conditions.

## Main files

| Path | Purpose |
|---|---|
| `appv2.py` | AeroLab Streamlit application |
| `src/models.py` | U-Net and Fourier Neural Operator definitions |
| `src/download.py` | AirfRANS dataset download |
| `src/convert_airfrans.py` | Conversion of official AirfRANS train/test splits to 64 × 64 grids |
| `src/train.py` | Model training and validation |
| `src/evaluate_airfrans_test.py` | Independent evaluation on the official test split |
| `results/` | Model checkpoints, experiment logs and evaluation output |

## Dataset

The flow surrogate is trained using [AirfRANS](https://airfrans.readthedocs.io/), a benchmark of aerodynamic flows around airfoils. Please refer to the dataset documentation and its associated publication for dataset details and usage conditions.
