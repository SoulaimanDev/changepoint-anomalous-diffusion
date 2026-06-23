#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate multi-seed v2 results as mean +/- sample standard deviation.

The script scans run folders created by:

    scripts/run_v2_single_seed_L100.py

and writes:

    results/multiseed_v2_L100/summary_mean_std.csv
    results/multiseed_v2_L100/all_test_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "false_negative_rate",
    "all_changepoint_mae",
    "all_changepoint_rmse",
    "true_positive_mae",
    "true_positive_rmse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate v2 multi-seed test metrics.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "multiseed_v2_L100",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.results_dir.glob("*/test_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No test_metrics.csv files found under {args.results_dir}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["run_dir"] = str(path.parent)
        frames.append(frame)
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(args.results_dir / "all_test_metrics.csv", index=False)

    rows = []
    for architecture, group in all_metrics.groupby("architecture", sort=True):
        row = {
            "architecture": architecture,
            "architecture_display": group["architecture_display"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(s)) for s in sorted(group["seed"].unique())),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_mean_std"] = (
                f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"
                if len(values) > 1
                else f"{row[f'{metric}_mean']:.4f} +/- NA"
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.results_dir / "summary_mean_std.csv", index=False)
    display_columns = ["architecture_display", "n_seeds"] + [
        f"{metric}_mean_std" for metric in METRIC_COLUMNS
    ]
    print(summary[display_columns].to_string(index=False))
    print("Saved:", args.results_dir / "all_test_metrics.csv")
    print("Saved:", args.results_dir / "summary_mean_std.csv")


if __name__ == "__main__":
    main()
