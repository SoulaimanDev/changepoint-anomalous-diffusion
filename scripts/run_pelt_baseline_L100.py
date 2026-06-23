#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a classical PELT baseline for L=100 changepoint detection/localization.

Protocol:
    1. Load the same validation and test HDF5 splits used by the neural models.
    2. Build simple unsupervised features from dx(t), without labels.
    3. Select the PELT penalty on validation only.
    4. Evaluate the selected penalty once on test.
    5. Save CSV/YAML artifacts under results/pelt_baseline_L100/.

This script does not train a neural network and does not modify frozen v2
results. PELT is used as an interpretable classical baseline, not as an
AnDi-equivalent method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LENGTH = 100
DX_LENGTH = LENGTH - 1
MIN_SEGMENT_LENGTH = 20
VALID_DX_MIN = MIN_SEGMENT_LENGTH - 1
VALID_DX_MAX = LENGTH - MIN_SEGMENT_LENGTH - 1

MODEL_NAMES = ["ATTM", "CTRW", "FBM", "LW", "SBM"]
MODEL_MAP = {i: name for i, name in enumerate(MODEL_NAMES)}
TRANSITIONS = [(m1, m2) for m1 in MODEL_NAMES for m2 in MODEL_NAMES if m1 != m2]
TRANSITION_ORDER = [f"{m1} -> {m2}" for m1, m2 in TRANSITIONS]

DEFAULT_PENALTIES = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    8.00,
    10.00,
    15.00,
    20.00,
    30.00,
    40.00,
    60.00,
    80.00,
]


def decode_model(value: Any) -> str:
    """Decode HDF5 model labels."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return MODEL_MAP[int(value)]


def normalize_dx(dx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize increments per trajectory."""
    mean = dx.mean(axis=1, keepdims=True)
    std = dx.std(axis=1, keepdims=True)
    return ((dx - mean) / np.maximum(std, eps)).astype("float32")


def load_split(file_path: Path, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one HDF5 split and return normalized dx plus labels/metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing {split_name} file: {file_path}")

    with h5py.File(file_path, "r") as file:
        dx = file["dx"][:].astype("float32")
        has_cp = file["has_changepoint"][:].astype("int8")
        cp = file["cp"][:].astype("int16")
        cp_dx = file["cp_dx"][:].astype("int16") if "cp_dx" in file else (cp - 1).astype("int16")
        model1 = file["model1"][:]
        model2 = file["model2"][:]

    if dx.ndim == 2:
        dx = dx[:, :, None]
    if dx.shape[1:] != (DX_LENGTH, 1):
        raise ValueError(f"Expected dx shape (N, {DX_LENGTH}, 1), got {dx.shape}.")

    dx = normalize_dx(dx)
    metadata = pd.DataFrame(
        {
            "split": split_name,
            "has_changepoint": has_cp.astype(int),
            "cp": cp,
            "cp_dx": cp_dx,
            "model1": [decode_model(value) for value in model1],
            "model2": [decode_model(value) for value in model2],
        }
    )
    metadata["transition"] = np.where(
        metadata["has_changepoint"] == 1,
        metadata["model1"] + " -> " + metadata["model2"],
        metadata["model1"] + " (no changepoint)",
    )
    return dx, has_cp, cp_dx, metadata


def balanced_subset(
    x: np.ndarray,
    y_has: np.ndarray,
    y_cp_dx: np.ndarray,
    metadata: pd.DataFrame,
    n_total: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Create a balanced subset preserving transition/model diversity."""
    if n_total <= 0:
        raise ValueError("n_total must be positive.")
    if n_total >= len(x):
        return x, y_has, y_cp_dx, metadata.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_with = n_total // 2
    n_without = n_total - n_with
    with_meta = metadata[metadata["has_changepoint"] == 1]
    without_meta = metadata[metadata["has_changepoint"] == 0]
    selected: list[int] = []

    per_transition = n_with // len(TRANSITION_ORDER)
    extra_transition = n_with % len(TRANSITION_ORDER)
    for i, transition in enumerate(TRANSITION_ORDER):
        group = with_meta[with_meta["transition"] == transition].index.to_numpy()
        k = per_transition + (1 if i < extra_transition else 0)
        if len(group) and k:
            selected.extend(rng.choice(group, size=min(k, len(group)), replace=False).tolist())

    per_model = n_without // len(MODEL_NAMES)
    extra_model = n_without % len(MODEL_NAMES)
    for i, model_name in enumerate(MODEL_NAMES):
        group = without_meta[without_meta["model1"] == model_name].index.to_numpy()
        k = per_model + (1 if i < extra_model else 0)
        if len(group) and k:
            selected.extend(rng.choice(group, size=min(k, len(group)), replace=False).tolist())

    selected_array = np.array(selected, dtype=int)
    rng.shuffle(selected_array)
    return (
        x[selected_array],
        y_has[selected_array],
        y_cp_dx[selected_array],
        metadata.loc[selected_array].reset_index(drop=True),
    )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with edge padding."""
    if window <= 1:
        return values.astype("float32")
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype="float32") / float(window)
    return np.convolve(padded, kernel, mode="valid").astype("float32")


def build_pelt_signal(dx_one: np.ndarray, feature: str, window: int) -> np.ndarray:
    """Build unsupervised PELT features from one normalized dx trajectory."""
    x = np.asarray(dx_one, dtype="float32").reshape(-1)
    abs_x = np.abs(x)
    energy = x**2
    local_abs_mean = rolling_mean(abs_x, window)
    local_energy_mean = rolling_mean(energy, window)
    local_variance = rolling_mean(energy, window) - rolling_mean(x, window) ** 2
    lag1_product = np.concatenate([[0.0], x[1:] * x[:-1]]).astype("float32")

    if feature == "energy":
        signal = energy[:, None]
    elif feature == "abs":
        signal = abs_x[:, None]
    elif feature == "local_variance":
        signal = local_variance[:, None]
    elif feature == "energy_abs":
        signal = np.column_stack([energy, abs_x])
    elif feature == "multichannel":
        signal = np.column_stack(
            [abs_x, energy, local_variance, local_abs_mean, local_energy_mean, lag1_product]
        )
    else:
        raise ValueError(f"Unknown feature: {feature}")

    signal = np.asarray(signal, dtype="float32")
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    return ((signal - mean) / np.maximum(std, 1e-6)).astype("float32")


def representative_changepoint(signal: np.ndarray, candidates: list[int], window: int = 8) -> int | None:
    """
    Select one representative changepoint from PELT candidates using only signal contrast.

    If PELT returns multiple changepoints, the task still has one true boundary.
    The selected candidate is the one with the largest local mean shift.
    """
    if not candidates:
        return None

    n = len(signal)
    best_cp: int | None = None
    best_score = -np.inf
    for cp in candidates:
        left_start = max(0, cp - window)
        left_end = cp + 1
        right_start = cp + 1
        right_end = min(n, cp + 1 + window)
        if left_end <= left_start or right_end <= right_start:
            continue
        left_mean = signal[left_start:left_end].mean(axis=0)
        right_mean = signal[right_start:right_end].mean(axis=0)
        score = float(np.linalg.norm(right_mean - left_mean))
        if score > best_score:
            best_score = score
            best_cp = int(cp)
    return best_cp


def pelt_predict_one(
    dx_one: np.ndarray,
    penalty: float,
    feature: str,
    window: int,
    min_size: int,
    jump: int,
) -> tuple[int, float]:
    """Run PELT on one trajectory and return predicted has_cp and cp_dx."""
    signal = build_pelt_signal(dx_one, feature=feature, window=window)
    bkps = rpt.Pelt(model="l2", min_size=min_size, jump=jump).fit(signal).predict(pen=float(penalty))
    candidates = [
        int(bkp) - 1
        for bkp in bkps
        if int(bkp) < len(signal)
    ]
    candidates = [
        cp for cp in candidates
        if VALID_DX_MIN <= cp <= VALID_DX_MAX
    ]
    cp = representative_changepoint(signal, candidates)
    if cp is None:
        return 0, np.nan
    return 1, float(cp)


def run_pelt_batch(
    x: np.ndarray,
    penalty: float,
    feature: str,
    window: int,
    min_size: int,
    jump: int,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PELT over a batch of trajectories."""
    y_pred = np.zeros(len(x), dtype=int)
    pred_cp_dx = np.full(len(x), np.nan, dtype="float32")
    for i in range(len(x)):
        y_pred[i], pred_cp_dx[i] = pelt_predict_one(
            x[i, :, 0],
            penalty=penalty,
            feature=feature,
            window=window,
            min_size=min_size,
            jump=jump,
        )
        if progress_every and (i + 1) % progress_every == 0:
            print(f"Processed {i + 1}/{len(x)} trajectories for penalty={penalty}", flush=True)
    return y_pred, pred_cp_dx


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    pred_cp_dx: np.ndarray,
    penalty: float,
    split: str,
    missing_localization_penalty: float,
) -> dict[str, Any]:
    """Compute detection and localization metrics."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Convert dx-index predictions to original time-index positions, matching v2 metric convention.
    pred_cp_time = pred_cp_dx + 1.0
    true_cp_time = metadata["cp"].to_numpy(dtype="float32")

    mask_all = y_true == 1
    mask_tp = (y_true == 1) & (y_pred == 1) & np.isfinite(pred_cp_time)

    if int(mask_all.sum()) == 0:
        all_mae = np.nan
        all_rmse = np.nan
    else:
        errors = np.abs(pred_cp_time[mask_all] - true_cp_time[mask_all])
        errors = np.where(np.isfinite(errors), errors, missing_localization_penalty)
        all_mae = float(np.mean(errors))
        all_rmse = float(np.sqrt(np.mean(errors**2)))

    if int(mask_tp.sum()) == 0:
        tp_mae = np.nan
        tp_rmse = np.nan
    else:
        tp_mae = float(mean_absolute_error(true_cp_time[mask_tp], pred_cp_time[mask_tp]))
        tp_rmse = float(np.sqrt(mean_squared_error(true_cp_time[mask_tp], pred_cp_time[mask_tp])))

    return {
        "baseline": "PELT",
        "split": split,
        "penalty": float(penalty),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn + 1e-8)),
        "false_negative_rate": float(fn / (fn + tp + 1e-8)),
        "jaccard_coefficient": float(tp / (tp + fp + fn + 1e-8)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "all_changepoint_n": int(mask_all.sum()),
        "all_changepoint_mae": all_mae,
        "all_changepoint_rmse": all_rmse,
        "true_positive_n": int(mask_tp.sum()),
        "true_positive_mae": tp_mae,
        "true_positive_rmse": tp_rmse,
    }


def selection_score(row: dict[str, Any]) -> float:
    """
    Validation-only score: lower is better.

    The score is detection-oriented to avoid selecting the trivial solution
    that marks nearly every trajectory as containing a changepoint. Localization
    error is still included, but with a smaller weight than FPR/FNR.
    """
    return float(
        10.0 * (1.0 - row["f1_score"])
        + 35.0 * row["false_positive_rate"]
        + 20.0 * row["false_negative_rate"]
        + 0.10 * row["all_changepoint_mae"]
    )


def parse_penalties(value: str) -> list[float]:
    """Parse comma-separated penalty values."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_yaml_like(path: Path, data: dict[str, Any]) -> None:
    """Write a simple YAML-compatible config without requiring PyYAML."""
    lines = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run PELT baseline for L=100.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data_synthetic_with_without_changepoint_dx",
        help="Folder containing val/test L=100 HDF5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "pelt_baseline_L100",
    )
    parser.add_argument(
        "--feature",
        choices=["energy", "abs", "local_variance", "energy_abs", "multichannel"],
        default="energy_abs",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--jump", type=int, default=1)
    parser.add_argument(
        "--penalties",
        default=",".join(str(value) for value in DEFAULT_PENALTIES),
        help="Comma-separated PELT penalties selected on validation.",
    )
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--missing-localization-penalty",
        type=float,
        default=float(LENGTH),
        help="Localization error assigned to true changepoints not detected by PELT.",
    )
    return parser.parse_args()


def main() -> None:
    """Run validation penalty selection and final test evaluation."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    penalties = parse_penalties(args.penalties)

    data_files = {
        "val": args.data_dir / f"val_L{LENGTH}_dim1_with_without_dx.h5",
        "test": args.data_dir / f"test_L{LENGTH}_dim1_with_without_dx.h5",
    }

    print("Loading validation and test splits...", flush=True)
    x_val, y_val, cp_val, val_metadata = load_split(data_files["val"], "validation")
    x_test, y_test, cp_test, test_metadata = load_split(data_files["test"], "test")

    if args.max_val_samples is not None:
        x_val, y_val, cp_val, val_metadata = balanced_subset(
            x_val, y_val, cp_val, val_metadata, args.max_val_samples, args.subset_seed
        )
    if args.max_test_samples is not None:
        x_test, y_test, cp_test, test_metadata = balanced_subset(
            x_test, y_test, cp_test, test_metadata, args.max_test_samples, args.subset_seed + 1
        )

    config_used = {
        "experiment_name": "pelt_baseline_L100",
        "baseline": "PELT",
        "length": LENGTH,
        "dx_length": DX_LENGTH,
        "data_dir": str(args.data_dir),
        "feature": args.feature,
        "window": args.window,
        "min_size": args.min_size,
        "jump": args.jump,
        "penalties": penalties,
        "penalty_selection": "validation_only",
        "test_used_for_penalty_selection": False,
        "validation_samples": len(x_val),
        "test_samples": len(x_test),
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "subset_seed": args.subset_seed,
        "missing_localization_penalty": args.missing_localization_penalty,
    }
    write_yaml_like(args.output_dir / "config_used.yaml", config_used)

    validation_rows = []
    for penalty in penalties:
        print(f"Validation penalty={penalty}", flush=True)
        y_pred_val, pred_cp_val = run_pelt_batch(
            x_val,
            penalty=penalty,
            feature=args.feature,
            window=args.window,
            min_size=args.min_size,
            jump=args.jump,
            progress_every=args.progress_every,
        )
        row = compute_metrics(
            y_true=y_val,
            y_pred=y_pred_val,
            metadata=val_metadata,
            pred_cp_dx=pred_cp_val,
            penalty=penalty,
            split="validation",
            missing_localization_penalty=args.missing_localization_penalty,
        )
        row["selection_score"] = selection_score(row)
        validation_rows.append(row)

    validation_grid = pd.DataFrame(validation_rows)
    validation_grid = validation_grid.sort_values(
        ["selection_score", "false_positive_rate", "false_negative_rate"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    selected_penalty = float(validation_grid.loc[0, "penalty"])
    validation_grid["selected"] = validation_grid["penalty"] == selected_penalty
    validation_grid.to_csv(args.output_dir / "validation_penalty_grid.csv", index=False)
    validation_grid[validation_grid["selected"]].to_csv(
        args.output_dir / "selected_validation_metrics.csv",
        index=False,
    )

    print(f"Selected validation penalty: {selected_penalty}", flush=True)
    print("Evaluating selected penalty on test...", flush=True)
    y_pred_test, pred_cp_test = run_pelt_batch(
        x_test,
        penalty=selected_penalty,
        feature=args.feature,
        window=args.window,
        min_size=args.min_size,
        jump=args.jump,
        progress_every=args.progress_every,
    )
    test_metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        metadata=test_metadata,
        pred_cp_dx=pred_cp_test,
        penalty=selected_penalty,
        split="test",
        missing_localization_penalty=args.missing_localization_penalty,
    )
    pd.DataFrame([test_metrics]).to_csv(args.output_dir / "test_metrics.csv", index=False)

    summary = {
        "selected_penalty": selected_penalty,
        "selected_validation_metrics": validation_grid[validation_grid["selected"]].iloc[0].to_dict(),
        "test_metrics": test_metrics,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("Saved:", args.output_dir / "validation_penalty_grid.csv")
    print("Saved:", args.output_dir / "selected_validation_metrics.csv")
    print("Saved:", args.output_dir / "test_metrics.csv")
    print("Saved:", args.output_dir / "config_used.yaml")


if __name__ == "__main__":
    main()
