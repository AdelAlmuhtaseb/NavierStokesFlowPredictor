import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from models import UNet


def relative_l2_by_sample(pred, target, fluid):
    """Return per-sample relative L2 error for each output channel."""
    numerator = (((pred - target) * fluid).flatten(2).norm(dim=2))
    denominator = ((target * fluid).flatten(2).norm(dim=2)).clamp_min(1e-12)
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AirfRANS U-Net on an untouched official test split."
    )
    parser.add_argument("--data", default="data/airfrans_test_g64.npz")
    parser.add_argument("--checkpoint", default="results/unet_airfrans_g64_independent.pt")
    parser.add_argument("--out", default="results/airfrans_independent_test.csv")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    data = np.load(args.data, allow_pickle=True)
    X = torch.as_tensor(data["imgs"], dtype=torch.float32, device=device)
    Y = torch.as_tensor(data["fields"], dtype=torch.float32, device=device)
    if X.ndim == 3:
        X = X.unsqueeze(1)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet(
        in_ch=int(checkpoint["in_ch"]),
        out_ch=int(checkpoint["out_ch"]),
        base=int(checkpoint.get("base", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()

    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32, device=device)
    fluid = 1.0 - X[:, :1]

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        normalized = model(X)
        prediction = normalized * std + mean
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(X)

    errors = relative_l2_by_sample(prediction, Y, fluid).cpu().numpy()
    names = data["names"] if "names" in data.files else np.arange(len(X))
    channels = ["u_over_U", "v_over_U", "Cp"]
    mean_by_channel = errors.mean(axis=0)
    median_by_channel = np.median(errors, axis=0)
    p90_by_channel = np.percentile(errors, 90, axis=0)
    mean_all = float(mean_by_channel.mean())

    print(f"Device: {device}")
    print(f"Official untouched test samples: {len(X)}")
    print(f"Mean inference time: {elapsed_ms:.3f} ms/sample (whole test batch / N)")
    print("\nRelative L2 error (mean across test simulations):")
    for i, channel in enumerate(channels):
        print(
            f"  {channel}: mean={mean_by_channel[i]*100:.3f}% | "
            f"median={median_by_channel[i]*100:.3f}% | "
            f"90th percentile={p90_by_channel[i]*100:.3f}%"
        )
    print(f"\nMacro-average across 3 channels: {mean_all*100:.3f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "simulation", "relative_l2_u_over_U", "relative_l2_v_over_U",
            "relative_l2_Cp", "mean_relative_l2_three_channels"
        ])
        for name, row in zip(names, errors):
            writer.writerow([str(name), *[f"{v:.8f}" for v in row], f"{row.mean():.8f}"])
        writer.writerow([
            "MEAN", *[f"{v:.8f}" for v in mean_by_channel], f"{mean_all:.8f}"
        ])
        writer.writerow([
            "MEDIAN", *[f"{v:.8f}" for v in median_by_channel],
            f"{np.median(errors.mean(axis=1)):.8f}"
        ])
        writer.writerow([
            "P90", *[f"{v:.8f}" for v in p90_by_channel],
            f"{np.percentile(errors.mean(axis=1), 90):.8f}"
        ])
    print(f"\nSaved per-simulation errors to {out_path}")


if __name__ == "__main__":
    main()
