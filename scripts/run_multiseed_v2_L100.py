#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the L=100 v2 multi-seed protocol for CNN-LSTM-v2 and ConvTransformer-v2.

This launcher calls scripts/run_v2_single_seed_L100.py once per
architecture/seed pair. It does not change the scientific protocol: each run
trains on the training split, selects the threshold only on validation, and
evaluates the selected configuration on test.

No results are invented by this script; it only orchestrates executions and
then optionally aggregates the CSV files produced by completed runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_v2_single_seed_L100.py"
AGGREGATE_SCRIPT = PROJECT_ROOT / "scripts" / "aggregate_multiseed_v2_results.py"


DEFAULT_SEEDS = [1, 2, 3, 42, 123]
DEFAULT_ARCHITECTURES = ["cnn_lstm", "convtransformer"]


def parse_csv_list(value: str, *, cast=str) -> list:
    """Parse a comma-separated CLI value."""
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential launcher for the v2 L=100 multi-seed study."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing train/val/test L=100 HDF5 files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "multiseed_v2_L100",
        help="Output folder for the real multi-seed runs.",
    )
    parser.add_argument(
        "--architectures",
        default=",".join(DEFAULT_ARCHITECTURES),
        help="Comma-separated architectures: cnn_lstm,convtransformer.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds, for example: 1,2,3,42,123.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--reduce-lr-patience", type=int, default=6)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional balanced subset. Use only for smoke checks, not final results.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional balanced validation subset. Use only for smoke checks.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional balanced test subset. Use only for smoke checks.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="Optional fixed seed for subset selection across all model seeds.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose test_metrics.csv already exists.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not run the final aggregation step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, architecture: str, seed: int, output_dir: Path) -> list[str]:
    """Build the command for one architecture/seed run."""
    command = [
        sys.executable,
        str(SINGLE_RUN_SCRIPT),
        "--architecture",
        architecture,
        "--seed",
        str(seed),
        "--data-dir",
        str(args.data_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--reduce-lr-patience",
        str(args.reduce_lr_patience),
        "--output-dir",
        str(output_dir),
    ]
    optional_limits = {
        "--max-train-samples": args.max_train_samples,
        "--max-val-samples": args.max_val_samples,
        "--max-test-samples": args.max_test_samples,
        "--subset-seed": args.subset_seed,
    }
    for flag, value in optional_limits.items():
        if value is not None:
            command.extend([flag, str(value)])
    return command


def print_command(command: list[str]) -> None:
    """Print a command in a copy-paste-friendly way."""
    print(" ".join(f'"{part}"' if " " in part else part for part in command), flush=True)


def main() -> None:
    args = parse_args()
    architectures = parse_csv_list(args.architectures, cast=str)
    seeds = parse_csv_list(args.seeds, cast=int)

    invalid = sorted(set(architectures) - set(DEFAULT_ARCHITECTURES))
    if invalid:
        raise ValueError(f"Unknown architecture(s): {invalid}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    for architecture in architectures:
        for seed in seeds:
            output_dir = args.results_dir / f"{architecture}_seed{seed}"
            metrics_file = output_dir / "test_metrics.csv"
            if args.skip_existing and metrics_file.exists():
                print(f"Skipping existing run: {output_dir}", flush=True)
                continue

            command = build_command(args, architecture, seed, output_dir)
            print("\nRunning:", flush=True)
            print_command(command)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if not args.no_aggregate:
        aggregate_command = [
            sys.executable,
            str(AGGREGATE_SCRIPT),
            "--results-dir",
            str(args.results_dir),
        ]
        print("\nAggregating:", flush=True)
        print_command(aggregate_command)
        if not args.dry_run:
            subprocess.run(aggregate_command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
