import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
from models import UNet

def relative_l2_by_sample(pred, target, fluid):
    numerator = (((pred - target) * fluid).flatten(2).norm(dim=2))
    denominator = ((target * fluid).flatten(2).norm(dim=2)).clamp_min(1e-12)
    return numerator / denominator

def main():
    parser = argparse.ArgumentParser(description="Evaluate a U-Net on the untouched official AirfRANS test split.")
    parser.add_argument("--data", default="data/airfrans_test_g64.npz")
    parser.add_argument("--checkpoint", default="results/unet_airfrans_g64_independent.pt")
    parser.add_argument("--out", default="results/airfrans_independent_test.csv")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = np.load(args.data, allow_pickle=True)
    X = torch.as_tensor(data["imgs"], dtype=torch.float32, device=device)
    Y = torch.as_tensor(data["fields"], dtype=torch.float32, device=device)
    if X.ndim == 3:
        X = X.unsqueeze(1)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet(in_ch=int(ck["in_ch"]), out_ch=int(ck["out_ch"]), base=int(ck.get("base", 16))).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    mean = torch.as_tensor(ck["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(ck["std"], dtype=torch.float32, device=device)
    fluid = 1.0 - X[:, :1]
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        pred = model(X) * std + mean
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    ms = (time.perf_counter() - start) * 1000 / len(X)
    errors = relative_l2_by_sample(pred, Y, fluid).cpu().numpy()
    names = data["names"] if "names" in data.files else np.arange(len(X))
    labels = ["u/U", "v/U", "Cp"]
    means, medians, p90s = errors.mean(0), np.median(errors, axis=0), np.percentile(errors, 90, axis=0)
    print(f"Device: {device}; official test cases: {len(X)}; batch-average inference: {ms:.3f} ms/sample")
    for i, label in enumerate(labels):
        print(f"{label}: mean {means[i]*100:.3f}% | median {medians[i]*100:.3f}% | P90 {p90s[i]*100:.3f}%")
    print(f"Macro-average across three fields: {means.mean()*100:.3f}%")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["simulation", "relative_l2_u_over_U", "relative_l2_v_over_U", "relative_l2_Cp", "mean_relative_l2"])
        for name, row in zip(names, errors):
            w.writerow([str(name), *[f"{v:.8f}" for v in row], f"{row.mean():.8f}"])
        w.writerow(["MEAN", *[f"{v:.8f}" for v in means], f"{means.mean():.8f}"])
        w.writerow(["MEDIAN", *[f"{v:.8f}" for v in medians], f"{np.median(errors.mean(1)):.8f}"])
        w.writerow(["P90", *[f"{v:.8f}" for v in p90s], f"{np.percentile(errors.mean(1), 90):.8f}"])
    print(f"Saved per-simulation errors to {out}")

if __name__ == "__main__":
    main()
