#!/usr/bin/env python3
"""Generate the balanced Protocol B dataset described in the revised TFM.

The HDF5 files contain raw positions and increments. Per-trajectory increment
normalization is deliberately left to the training pipeline so that alternative
normalization strategies can be evaluated without regenerating trajectories.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from .build_synthetic_changepoint_dataset import (
        ID_TO_MODEL,
        MODELS,
        MODEL_TO_ID,
        TRANSITIONS,
        load_generator,
        package_version,
        sample_alpha,
        simulate_trajectory,
        standardize_segment,
        simulate_segment,
    )
except ImportError:  # Direct execution: python scripts/<file>.py
    from build_synthetic_changepoint_dataset import (
        ID_TO_MODEL,
        MODELS,
        MODEL_TO_ID,
        TRANSITIONS,
        load_generator,
        package_version,
        sample_alpha,
        simulate_trajectory,
        standardize_segment,
        simulate_segment,
    )


FULL_SPLIT_SIZES = {
    "train": {"with_per_transition": 5_000, "without_per_model": 20_000},
    "val": {"with_per_transition": 500, "without_per_model": 2_000},
    "test": {"with_per_transition": 5_000, "without_per_model": 20_000},
}

DRY_SPLIT_SIZES = {
    "train": {"with_per_transition": 2, "without_per_model": 8},
    "val": {"with_per_transition": 1, "without_per_model": 4},
    "test": {"with_per_transition": 1, "without_per_model": 4},
}


def simulate_no_changepoint_trajectory(generator, model_name, length, dimension, noise_levels, random_state):
    alpha = sample_alpha(model_name, random_state)
    trajectory = standardize_segment(simulate_segment(generator, model_name, length, alpha, dimension))
    noise_sigma = float(random_state.choice(noise_levels)) if noise_levels else -1.0
    if noise_sigma > 0:
        trajectory = trajectory + random_state.normal(0.0, noise_sigma, size=trajectory.shape)
    return trajectory.astype(np.float32), alpha, noise_sigma


def build_record_schedule(with_per_transition, without_per_model, random_state):
    kinds = []
    first_models = []
    second_models = []

    for first_model, second_model in TRANSITIONS:
        kinds.extend([1] * with_per_transition)
        first_models.extend([MODEL_TO_ID[first_model]] * with_per_transition)
        second_models.extend([MODEL_TO_ID[second_model]] * with_per_transition)

    for model in MODELS:
        kinds.extend([0] * without_per_model)
        first_models.extend([MODEL_TO_ID[model]] * without_per_model)
        second_models.extend([MODEL_TO_ID[model]] * without_per_model)

    kinds = np.asarray(kinds, dtype=np.uint8)
    first_models = np.asarray(first_models, dtype=np.int8)
    second_models = np.asarray(second_models, dtype=np.int8)
    permutation = random_state.permutation(len(kinds))
    return kinds[permutation], first_models[permutation], second_models[permutation]


def create_datasets(file, total, length, dimension, compression):
    position_chunks = (min(512, total), length, dimension)
    increment_chunks = (min(512, total), length - 1, dimension)
    file.create_dataset("X", shape=(total, length, dimension), dtype="float32", chunks=position_chunks, compression=compression)
    file.create_dataset("dx", shape=(total, length - 1, dimension), dtype="float32", chunks=increment_chunks, compression=compression)
    file.create_dataset("has_changepoint", shape=(total,), dtype="uint8", compression=compression)
    file.create_dataset("cp", shape=(total,), dtype="int16", compression=compression)
    file.create_dataset("cp_dx", shape=(total,), dtype="int16", compression=compression)
    file.create_dataset("cp_dx_norm", shape=(total,), dtype="float32", compression=compression)
    file.create_dataset("model1", shape=(total,), dtype="int8", compression=compression)
    file.create_dataset("model2", shape=(total,), dtype="int8", compression=compression)
    file.create_dataset("alpha1", shape=(total,), dtype="float32", compression=compression)
    file.create_dataset("alpha2", shape=(total,), dtype="float32", compression=compression)
    file.create_dataset("noise_sigma", shape=(total,), dtype="float32", compression=compression)


def write_pair_counts(output_dir, split_name, first_models, second_models):
    frame = pd.DataFrame({
        "model1": [ID_TO_MODEL[int(value)] for value in first_models],
        "model2": [ID_TO_MODEL[int(value)] for value in second_models],
    })
    counts = frame.groupby(["model1", "model2"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=MODELS, columns=MODELS, fill_value=0)
    counts.to_csv(output_dir / f"{split_name}_pair_counts_with_without.csv")


def generate_split(
    split_name,
    split_sizes,
    generator,
    output_dir,
    length,
    minimum_segment_length,
    dimension,
    noise_levels,
    seed,
    compression,
):
    with_per_transition = split_sizes["with_per_transition"]
    without_per_model = split_sizes["without_per_model"]
    expected_positive = len(TRANSITIONS) * with_per_transition
    expected_negative = len(MODELS) * without_per_model
    if expected_positive != expected_negative:
        raise ValueError("Protocol B requires equal positive and negative class sizes")

    random_state = np.random.default_rng(seed)
    kinds, first_models, second_models = build_record_schedule(with_per_transition, without_per_model, random_state)
    total = len(kinds)
    file_path = output_dir / f"{split_name}_L{length}_dim{dimension}_with_without_dx.h5"

    with h5py.File(file_path, "w") as file:
        file.attrs["protocol"] = "B"
        file.attrs["trajectory_length"] = length
        file.attrs["increment_length"] = length - 1
        file.attrs["minimum_segment_length"] = minimum_segment_length
        file.attrs["dimension"] = dimension
        file.attrs["seed"] = seed
        file.attrs["models"] = json.dumps(MODELS)
        file.attrs["model_to_id"] = json.dumps(MODEL_TO_ID)
        file.attrs["transitions"] = json.dumps(TRANSITIONS)
        file.attrs["noise_levels"] = json.dumps(noise_levels)
        file.attrs["negative_cp_sentinel"] = -1
        file.attrs["dx_normalization"] = "raw; normalize per trajectory during training"
        create_datasets(file, total, length, dimension, compression)

        progress = tqdm(total=total, desc=split_name, unit="trajectory")
        for index, has_changepoint in enumerate(kinds):
            first_model = ID_TO_MODEL[int(first_models[index])]
            second_model = ID_TO_MODEL[int(second_models[index])]

            if has_changepoint:
                trajectory, cp, alpha1, alpha2, noise_sigma = simulate_trajectory(
                    generator,
                    first_model,
                    second_model,
                    length,
                    minimum_segment_length,
                    dimension,
                    noise_levels,
                    random_state,
                )
                cp_dx = cp - 1
                cp_dx_norm = cp_dx / (length - 1)
            else:
                trajectory, alpha1, noise_sigma = simulate_no_changepoint_trajectory(
                    generator,
                    first_model,
                    length,
                    dimension,
                    noise_levels,
                    random_state,
                )
                alpha2 = alpha1
                cp = -1
                cp_dx = -1
                cp_dx_norm = -1.0

            file["X"][index] = trajectory
            file["dx"][index] = np.diff(trajectory, axis=0)
            file["has_changepoint"][index] = has_changepoint
            file["cp"][index] = cp
            file["cp_dx"][index] = cp_dx
            file["cp_dx_norm"][index] = cp_dx_norm
            file["model1"][index] = first_models[index]
            file["model2"][index] = second_models[index]
            file["alpha1"][index] = alpha1
            file["alpha2"][index] = alpha2
            file["noise_sigma"][index] = noise_sigma
            progress.update(1)
        progress.close()

    write_pair_counts(output_dir, split_name, first_models, second_models)
    with h5py.File(file_path, "r") as file:
        unique_noise, noise_counts = np.unique(file["noise_sigma"][:], return_counts=True)
    return file_path, {
        "total": total,
        "with_changepoint": expected_positive,
        "without_changepoint": expected_negative,
        "noise_sigma_counts": {f"{float(level):.6g}": int(count) for level, count in zip(unique_noise, noise_counts)},
    }


def save_summary(output_dir, files, split_statistics, split_sizes, length, minimum_segment_length, dimension, noise_levels, seed):
    summary = {
        "objective": "Balanced binary changepoint detection and temporal localization (TFM Protocol B)",
        "trajectory_length": length,
        "increment_length": length - 1,
        "minimum_segment_length": minimum_segment_length,
        "valid_cp_range": [minimum_segment_length, length - minimum_segment_length],
        "valid_cp_dx_range": [minimum_segment_length - 1, length - minimum_segment_length - 1],
        "dimension": dimension,
        "models": MODELS,
        "model_to_id": MODEL_TO_ID,
        "transitions": TRANSITIONS,
        "split_configuration": split_sizes,
        "split_statistics": split_statistics,
        "noise_levels": noise_levels,
        "seed": seed,
        "files": {name: str(path) for name, path in files.items()},
        "negative_sentinel": {"cp": -1, "cp_dx": -1, "cp_dx_norm": -1.0},
        "training_input": {
            "raw_positions": f"X shape (N, {length}, {dimension})",
            "raw_increments": f"dx shape (N, {length - 1}, {dimension})",
            "normalization": "Per-trajectory z-score of dx with epsilon=1e-6, applied by the notebooks",
            "localization_target": "cp_dx = cp - 1 for positive samples only",
        },
        "software": {
            "andi-datasets": package_version("andi-datasets"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
    }
    with open(output_dir / f"dataset_summary_L{length}.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def parse_noise_levels(value):
    value = value.strip().lower()
    if value in {"none", "false", "no", "0"}:
        return []
    return [float(item) for item in value.split(",") if item.strip()]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("./data_synthetic_with_without_changepoint_dx"))
    parser.add_argument("--length", type=int, choices=(100, 200), default=100)
    parser.add_argument(
        "--minimum-segment-length",
        type=int,
        default=None,
        help="Defaults to 20%% of trajectory length (20 for L=100, 40 for L=200).",
    )
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", default="0.1,0.5,1.0")
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    minimum_segment_length = (
        arguments.minimum_segment_length
        if arguments.minimum_segment_length is not None
        else int(round(0.20 * arguments.length))
    )
    if minimum_segment_length < 1 or 2 * minimum_segment_length > arguments.length:
        raise ValueError("--minimum-segment-length must be positive and at most half of --length")

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_sizes = DRY_SPLIT_SIZES if arguments.dry_run else FULL_SPLIT_SIZES
    noise_levels = parse_noise_levels(arguments.noise)
    compression = None if arguments.compression == "none" else arguments.compression
    generator = load_generator()

    files = {}
    split_statistics = {}
    for offset, (split_name, sizes) in enumerate(split_sizes.items()):
        files[split_name], split_statistics[split_name] = generate_split(
            split_name,
            sizes,
            generator,
            output_dir,
            arguments.length,
            minimum_segment_length,
            arguments.dimension,
            noise_levels,
            arguments.seed + 1000 * offset,
            compression,
        )

    save_summary(
        output_dir,
        files,
        split_statistics,
        split_sizes,
        arguments.length,
        minimum_segment_length,
        arguments.dimension,
        noise_levels,
        arguments.seed,
    )
    print(json.dumps({name: str(path) for name, path in files.items()}, indent=2))


if __name__ == "__main__":
    main()
