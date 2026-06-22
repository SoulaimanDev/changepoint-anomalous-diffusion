#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


MODELS = ["ATTM", "CTRW", "FBM", "LW", "SBM"]
MODEL_TO_ID = {name: idx for idx, name in enumerate(MODELS)}
ID_TO_MODEL = {idx: name for name, idx in MODEL_TO_ID.items()}
TRANSITIONS = [(a, b) for a, b in product(MODELS, MODELS) if a != b]

ALPHA_RANGES = {
    "ATTM": (0.05, 1.00),
    "CTRW": (0.05, 1.00),
    "FBM": (0.05, 1.95),
    "LW": (1.00, 2.00),
    "SBM": (0.05, 2.00),
}

FULL_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 10_000}
DRY_SPLIT_SIZES = {"train": 10, "val": 5, "test": 5}


def load_generator():
    from andi_datasets.models_theory import models_theory
    return models_theory()


def package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def parse_noise_levels(value):
    value = value.strip().lower()
    if value in {"none", "false", "no", "0"}:
        return []
    return [float(item) for item in value.split(",") if item.strip()]


def normalize_output(values, dimension):
    array = np.asarray(values, dtype=np.float64).squeeze()
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim == 2 and array.shape[0] == dimension and array.shape[1] != dimension:
        array = array.T
    if dimension == 1 and array.ndim == 2 and array.shape[1] != 1:
        array = array.reshape(-1, 1)
    if array.shape[1] > dimension:
        array = array[:, :dimension]
    if array.shape[1] != dimension:
        raise ValueError(f"Unexpected output shape: {array.shape}")
    return array


def simulate_segment(generator, model_name, length, alpha, dimension):
    model = getattr(generator, model_name.lower())
    signature = inspect.signature(model)

    keyword_candidates = [
        {"T": length, "alpha": alpha, "dim": dimension},
        {"T": length, "alpha": alpha, "D": dimension},
        {"T": length, "alpha": alpha},
        {"length": length, "alpha": alpha, "dim": dimension},
        {"length": length, "alpha": alpha},
    ]

    for candidate in keyword_candidates:
        arguments = {key: value for key, value in candidate.items() if key in signature.parameters}
        if arguments:
            try:
                output = normalize_output(model(**arguments), dimension)
                if output.shape[0] >= length:
                    return output[:length]
            except Exception:
                pass

    for candidate in [(length, alpha), (alpha, length), (length, alpha, dimension)]:
        try:
            output = normalize_output(model(*candidate), dimension)
            if output.shape[0] >= length:
                return output[:length]
        except Exception:
            pass

    raise RuntimeError(f"Unable to simulate {model_name} with length={length}, alpha={alpha}, dimension={dimension}.")


def standardize_segment(segment):
    segment = np.asarray(segment, dtype=np.float64)
    segment = segment - segment[0:1]
    scale = np.std(np.diff(segment, axis=0))
    return segment / scale if scale > 1e-8 else segment


def sample_alpha(model_name, random_state):
    lower, upper = ALPHA_RANGES[model_name]
    return float(random_state.uniform(lower, upper))


def simulate_trajectory(generator, first_model, second_model, length, minimum_segment_length, dimension, noise_levels, random_state):
    changepoint = int(random_state.integers(minimum_segment_length, length - minimum_segment_length + 1))
    first_alpha = sample_alpha(first_model, random_state)
    second_alpha = sample_alpha(second_model, random_state)

    first_segment = standardize_segment(simulate_segment(generator, first_model, changepoint, first_alpha, dimension))
    second_segment = standardize_segment(simulate_segment(generator, second_model, length - changepoint + 1, second_alpha, dimension))
    trajectory = np.concatenate([first_segment, first_segment[-1:] + second_segment[1:] - second_segment[:1]], axis=0)

    noise_sigma = float(random_state.choice(noise_levels)) if noise_levels else -1.0
    if noise_sigma > 0:
        trajectory = trajectory + random_state.normal(0.0, noise_sigma, size=trajectory.shape)

    return trajectory.astype(np.float32), changepoint, first_alpha, second_alpha, noise_sigma


def write_transition_counts(output_dir, split_name, model1, model2):
    frame = pd.DataFrame({
        "model1": [ID_TO_MODEL[int(value)] for value in model1],
        "model2": [ID_TO_MODEL[int(value)] for value in model2],
    })
    counts = frame.groupby(["model1", "model2"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=MODELS, columns=MODELS, fill_value=0)
    counts.to_csv(output_dir / f"{split_name}_pair_counts.csv")


def generate_split(split_name, trajectories_per_transition, generator, output_dir, length, minimum_segment_length, dimension, noise_levels, seed, compression):
    total = len(TRANSITIONS) * trajectories_per_transition
    random_state = np.random.default_rng(seed)
    file_path = output_dir / f"{split_name}_L{length}_dim{dimension}.h5"

    with h5py.File(file_path, "w") as file:
        file.attrs["trajectory_length"] = length
        file.attrs["minimum_segment_length"] = minimum_segment_length
        file.attrs["dimension"] = dimension
        file.attrs["models"] = json.dumps(MODELS)
        file.attrs["model_to_id"] = json.dumps(MODEL_TO_ID)
        file.attrs["transitions"] = json.dumps(TRANSITIONS)
        file.attrs["trajectories_per_transition"] = trajectories_per_transition
        file.attrs["noise_levels"] = json.dumps(noise_levels)

        file.create_dataset("X", shape=(total, length, dimension), dtype="float32", chunks=(min(1024, total), length, dimension), compression=compression)
        file.create_dataset("cp", shape=(total,), dtype="int16", compression=compression)
        file.create_dataset("model1", shape=(total,), dtype="int8", compression=compression)
        file.create_dataset("model2", shape=(total,), dtype="int8", compression=compression)
        file.create_dataset("alpha1", shape=(total,), dtype="float32", compression=compression)
        file.create_dataset("alpha2", shape=(total,), dtype="float32", compression=compression)
        file.create_dataset("noise_sigma", shape=(total,), dtype="float32", compression=compression)

        records = []
        index = 0
        progress = tqdm(total=total, desc=f"{split_name}", unit="trajectory")

        for pair_id, (first_model, second_model) in enumerate(TRANSITIONS):
            for _ in range(trajectories_per_transition):
                trajectory, changepoint, first_alpha, second_alpha, noise_sigma = simulate_trajectory(
                    generator, first_model, second_model, length, minimum_segment_length, dimension, noise_levels, random_state
                )

                file["X"][index] = trajectory
                file["cp"][index] = changepoint
                file["model1"][index] = MODEL_TO_ID[first_model]
                file["model2"][index] = MODEL_TO_ID[second_model]
                file["alpha1"][index] = first_alpha
                file["alpha2"][index] = second_alpha
                file["noise_sigma"][index] = noise_sigma

                if len(records) < 10_000:
                    records.append([split_name, index, pair_id, changepoint, first_model, second_model, first_alpha, second_alpha, noise_sigma])

                index += 1
                progress.update(1)

        progress.close()
        model1 = file["model1"][:]
        model2 = file["model2"][:]

    pd.DataFrame(records, columns=["split", "index", "pair_id", "changepoint", "model1", "model2", "alpha1", "alpha2", "noise_sigma"]).to_csv(
        output_dir / f"{split_name}_metadata_head.csv", index=False
    )
    write_transition_counts(output_dir, split_name, model1, model2)
    return file_path


def save_summary(output_dir, files, split_sizes, length, minimum_segment_length, dimension, noise_levels, seed):
    summary = {
        "objective": "Synthetic dataset for changepoint localization in anomalous diffusion trajectories",
        "trajectory_length": length,
        "minimum_segment_length": minimum_segment_length,
        "dimension": dimension,
        "models": MODELS,
        "model_to_id": MODEL_TO_ID,
        "alpha_ranges": ALPHA_RANGES,
        "transitions": TRANSITIONS,
        "trajectories_per_transition": split_sizes,
        "total_trajectories": {name: len(TRANSITIONS) * size for name, size in split_sizes.items()},
        "noise_levels": noise_levels,
        "seed": seed,
        "software": {
            "andi-datasets": package_version("andi-datasets"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
        "files": {name: str(path) for name, path in files.items()},
        "training_input": {
            "positions": f"(N, {length}, {dimension})",
            "increments": f"np.diff(X, axis=1), shape (N, {length - 1}, {dimension})",
            "increment_target": "cp - 1",
        },
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("./data_synthetic_changepoint_andi"))
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument(
        "--minimum-segment-length",
        type=int,
        default=None,
        help="Minimum segment length. Defaults to 20%% of --length (20 for L=100, 40 for L=200).",
    )
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=str, default="0.1,0.5,1.0")
    parser.add_argument("--compression", type=str, default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.length < 2:
        raise ValueError("--length must be at least 2")
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
    for offset, (split_name, trajectories_per_transition) in enumerate(split_sizes.items()):
        files[split_name] = generate_split(
            split_name,
            trajectories_per_transition,
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
