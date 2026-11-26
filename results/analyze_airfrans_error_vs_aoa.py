#!/usr/bin/env python3
"""Analyze AirfRANS independent-test errors against angle of attack.

Reads the existing NPZ test split and per-simulation evaluation CSV.
Writes all new outputs to a fresh folder; it does not modify inputs.
Run from the repository root:
    python results\\analyze_airfrans_error_vs_aoa.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

CHANNELS = [
    ("relative_l2_u_over_U", "Horizontal velocity u/U"),
    ("relative_l2_v_over_U", "Vertical velocity v/U"),
    ("relative_l2_Cp", "Pressure coefficient Cp"),
]


def fresh_directory(base: Path) -> Path:
    """Create a new output directory without overwriting an existing one."""
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}_run{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def load_error_csv(path: Path):
    rows = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"simulation", *(c[0] for c in CHANNELS)}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")
        for row in reader:
            name = row["simulation"].strip()
            if not name or name.upper() in {"MEAN", "MEDIAN", "P90"}:
                continue
            rows[name] = [float(row[c[0]]) * 100.0 for c in CHANNELS]
    return rows


def safe_spearman(x, y):
    try:
        result = spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float("nan"), float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Plot and summarize AirfRANS test errors versus angle of attack."
    )
    parser.add_argument(
        "--data",
        default=r"data\\independent_test_20261126\\airfrans_test_g64.npz",
        help="Independent AirfRANS test NPZ file",
    )
    parser.add_argument(
        "--errors",
        default=r"results\\airfrans_independent_test_20261126_run1.csv",
        help="Per-simulation CSV produced by evaluate_airfrans_test.py",
    )
    parser.add_argument(
        "--out",
        default=r"results\\angle_of_attack_diagnostic_20261126",
        help="Base output directory (a new suffix is used if it already exists)",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=5.0,
        help="Angle-of-attack bin width in degrees (default: 5)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    errors_path = Path(args.errors)
    if not data_path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {data_path}")
    if not errors_path.is_file():
        raise FileNotFoundError(f"Per-simulation error CSV not found: {errors_path}")
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")

    data = np.load(data_path, allow_pickle=True)
    if "params" not in data.files or "names" not in data.files:
        raise ValueError("The NPZ must contain 'params' and 'names' arrays.")

    params = np.asarray(data["params"], dtype=np.float64)
    names = np.asarray(data["names"]).astype(str)
    if params.ndim != 2 or params.shape[1] < 2:
        raise ValueError(f"Unexpected params shape: {params.shape}; expected (N, 2).")
    if len(names) != len(params):
        raise ValueError("The number of names and parameter rows does not match.")

    # src/convert_airfrans.py stores params as [inlet speed U, angle of attack in degrees].
    inlet_speed = params[:, 0]
    aoa_deg = params[:, 1]
    csv_errors = load_error_csv(errors_path)

    missing_names = [name for name in names if name not in csv_errors]
    if missing_names:
        preview = ", ".join(missing_names[:5])
        raise ValueError(
            f"{len(missing_names)} test simulations were not found in the error CSV. "
            f"Examples: {preview}"
        )

    # Match rows by simulation name, not merely by their order in the files.
    err_pct = np.asarray([csv_errors[name] for name in names], dtype=np.float64)
    combined_pct = err_pct.mean(axis=1)
    out_dir = fresh_directory(Path(args.out))

    joined_path = out_dir / "simulation_errors_with_angle.csv"
    with joined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset_index", "simulation", "angle_of_attack_deg", "inlet_speed_U",
            "relative_l2_u_over_U_percent", "relative_l2_v_over_U_percent",
            "relative_l2_Cp_percent", "mean_three_channel_error_percent",
        ])
        for i, name in enumerate(names):
            writer.writerow([
                i, name, f"{aoa_deg[i]:.6f}", f"{inlet_speed[i]:.6f}",
                *[f"{v:.6f}" for v in err_pct[i]], f"{combined_pct[i]:.6f}",
            ])

    width = float(args.bin_width)
    low = np.floor(np.nanmin(aoa_deg) / width) * width
    high = np.ceil(np.nanmax(aoa_deg) / width) * width + width
    edges = np.arange(low, high + width * 1e-9, width)
    if len(edges) < 2:
        edges = np.array([low, low + width])

    bin_rows = []
    labels = ["u_over_U", "v_over_U", "Cp", "three_channel_mean"]
    for j in range(len(edges) - 1):
        left, right = edges[j], edges[j + 1]
        mask = (
            (aoa_deg >= left) & (aoa_deg <= right)
            if j == len(edges) - 2
            else (aoa_deg >= left) & (aoa_deg < right)
        )
        if not np.any(mask):
            continue
        values = np.column_stack([err_pct[mask], combined_pct[mask]])
        row = {
            "angle_bin_deg": f"[{left:g}, {right:g}{']' if j == len(edges)-2 else ')'}",
            "count": int(mask.sum()),
        }
        for k, label in enumerate(labels):
            row[f"{label}_mean_percent"] = float(values[:, k].mean())
            row[f"{label}_median_percent"] = float(np.median(values[:, k]))
            row[f"{label}_p90_percent"] = float(np.percentile(values[:, k], 90))
        bin_rows.append(row)

    summary_path = out_dir / "angle_bin_summary.csv"
    fields = ["angle_bin_deg", "count"]
    for label in labels:
        fields += [
            f"{label}_mean_percent",
            f"{label}_median_percent",
            f"{label}_p90_percent",
        ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bin_rows)

    plot_data = [
        (err_pct[:, 0], "Horizontal velocity u/U", "tab:blue"),
        (err_pct[:, 1], "Vertical velocity v/U", "tab:orange"),
        (err_pct[:, 2], "Pressure coefficient Cp", "tab:green"),
        (combined_pct, "Mean of three channel errors", "tab:purple"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (values, title, color) in zip(axes.flat, plot_data):
        rho, p_value = safe_spearman(aoa_deg, values)
        ax.scatter(
            aoa_deg, values, s=25, alpha=0.75, color=color,
            edgecolors="none", label="Individual simulation",
        )
        means_x, means_y = [], []
        for j in range(len(edges) - 1):
            left, right = edges[j], edges[j + 1]
            mask = (
                (aoa_deg >= left) & (aoa_deg <= right)
                if j == len(edges) - 2
                else (aoa_deg >= left) & (aoa_deg < right)
            )
            if np.any(mask):
                means_x.append((left + right) / 2.0)
                means_y.append(float(values[mask].mean()))
        ax.plot(
            means_x, means_y, marker="o", linewidth=2, color="black",
            label=f"{width:g} degree bin mean",
        )
        ax.set_title(title)
        ax.set_xlabel("Angle of attack (degrees)")
        ax.set_ylabel("Relative L2 error (%)")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.02, 0.97, f"Spearman rho = {rho:.3f}\\np = {p_value:.3g}",
            transform=ax.transAxes, ha="left", va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("AirfRANS independent-test error versus angle of attack", fontsize=14)
    plot_path = out_dir / "error_vs_angle_of_attack.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    report_path = out_dir / "diagnostic_report.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("AirfRANS independent-test error versus angle of attack\\n")
        handle.write("====================================================\\n\\n")
        handle.write(f"Test simulations analyzed: {len(names)}\\n")
        handle.write(
            f"Angle-of-attack range: {aoa_deg.min():.3f} to "
            f"{aoa_deg.max():.3f} degrees\\n"
        )
        handle.write(f"Angle bin width: {width:g} degrees\\n\\n")
        handle.write("Spearman rank correlations (angle of attack versus error):\\n")
        for i, (_, label) in enumerate(CHANNELS):
            rho, p_value = safe_spearman(aoa_deg, err_pct[:, i])
            handle.write(f"  {label}: rho={rho:.4f}, p={p_value:.4g}\\n")
        rho, p_value = safe_spearman(aoa_deg, combined_pct)
        handle.write(f"  Three-channel mean: rho={rho:.4f}, p={p_value:.4g}\\n\\n")
        handle.write("Angle-bin averages are in angle_bin_summary.csv.\\n")
        handle.write(
            "These plots are descriptive diagnostics; correlation does not prove causation.\\n"
        )

    print(f"Analyzed {len(names)} test simulations.")
    print(f"Angle of attack range: {aoa_deg.min():.3f} to {aoa_deg.max():.3f} degrees")
    print("\nSpearman correlation (angle of attack versus error):")
    for i, (_, label) in enumerate(CHANNELS):
        rho, p_value = safe_spearman(aoa_deg, err_pct[:, i])
        print(f"  {label}: rho={rho:.3f}, p={p_value:.4g}")
    rho, p_value = safe_spearman(aoa_deg, combined_pct)
    print(f"  Three-channel mean: rho={rho:.3f}, p={p_value:.4g}")
    print("\nMean errors by angle bin:")
    for row in bin_rows:
        print(
            f"  {row['angle_bin_deg']:>12} deg (n={row['count']:3d}): "
            f"u={row['u_over_U_mean_percent']:.2f}%, "
            f"v={row['v_over_U_mean_percent']:.2f}%, "
            f"Cp={row['Cp_mean_percent']:.2f}%, "
            f"combined={row['three_channel_mean_mean_percent']:.2f}%"
        )
    print(f"\\nNew outputs saved in: {out_dir}")
    print(f"  Plot: {plot_path.name}")
    print(f"  Bin summary: {summary_path.name}")
    print(f"  Per-simulation data: {joined_path.name}")
    print(f"  Text report: {report_path.name}")
    print("Original test data, checkpoint, and existing CSV files were only read.")


if __name__ == "__main__":
    main()
