# Airfoil Flow Predictor demo

This Streamlit app provides an upload interface for the Navier-Stokes Flow Predictor project.

## What works before training finishes

- Upload a clean, side-on airfoil silhouette (PNG/JPG/BMP).
- Detect the dominant outline and map it to the project's 64 x 64 input grid.
- Set angle of attack and inlet speed.
- Preview the model-ready geometry mask.

The application does not invent flow fields. Real velocity and pressure predictions are enabled only when the trained U-Net checkpoint is available.

## Run locally

From the repository root, install requirements and launch the app:

    python -m pip install -r requirements.txt
    streamlit run app.py

Streamlit will print a local URL to open in your browser.

## Add the trained model

After the AirfRANS training run, place the checkpoint here:

    results/unet_airfrans_g64.pt

Alternatively, upload the .pt checkpoint using the sidebar control in the app.

Then upload an airfoil image and select Analyze Airfoil. The app displays:
- Speed magnitude divided by inlet speed.
- Pressure coefficient Cp.
- Horizontal velocity u/U-infinity.
- Vertical velocity v/U-infinity.

## Input image guidance

For this first version, use a tightly cropped image containing one airfoil only:
- Side-on view, with the chord horizontal.
- Dark filled shape on a plain light background (or the inverse).
- No labels, axes, dimensions, or other objects.

The image is normalized to a chord of one and centered vertically. This is a prototype preprocessing assumption; it is not a general-purpose airfoil-recognition system. Airfoil shapes and flow conditions outside the training distribution may produce unreliable predictions.

## Scientific note

The visualizations are neural-network predictions based on the trained AirfRANS checkpoint, not fresh OpenFOAM/CFD calculations. Validate the checkpoint on the untouched official test split before reporting benchmark accuracy.
