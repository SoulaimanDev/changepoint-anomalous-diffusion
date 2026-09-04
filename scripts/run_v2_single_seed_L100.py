#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one v2 architecture and one random seed for L=100.

This script is intended for the multi-seed study of the consolidated v2
architectures:

    - CNN-LSTM-v2
    - ConvTransformer-v2

Protocol:
    1. Train on the training split.
    2. Select the decision threshold on validation only.
    3. Evaluate the selected model/threshold on the test split.
    4. Save one CSV row for validation metrics and one CSV row for test metrics.

The script does not modify the frozen v2 reference folder.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from tensorflow.keras import callbacks, layers, metrics, models, optimizers


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

LEARNING_RATE = 1e-3
MONITOR_METRIC = "val_selection_score"

ARCHITECTURE_CONFIGS: dict[str, dict[str, Any]] = {
    "cnn_lstm": {
        "display_name": "CNN-LSTM-v2",
        "loss_weights": [2.0, 1.8],
        "positive_class_weight": 1.0,
        "negative_class_weight": 1.8,
        "bce_reduction": "none",
        "thresholds": [0.25, 0.28, 0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.42, 0.45, 0.50],
        "selection_rule": "cnn_lstm_v2",
    },
    "convtransformer": {
        "display_name": "ConvTransformer-v2",
        "loss_weights": [2.0, 1.0],
        "positive_class_weight": 1.0,
        "negative_class_weight": 1.8,
        "bce_reduction": "mean",
        "thresholds": [0.30, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.42, 0.45],
        "selection_rule": "convtransformer_v2",
    },
}


def set_global_seed(seed: int) -> None:
    """Fix Python, NumPy and TensorFlow seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass


def decode_model(value: Any) -> str:
    """Decode HDF5 model labels."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return MODEL_MAP[int(value)]


def normalize_dx(dx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize increments per trajectory as in the v2 notebooks."""
    mean = dx.mean(axis=1, keepdims=True)
    std = dx.std(axis=1, keepdims=True)
    return ((dx - mean) / np.maximum(std, eps)).astype("float32")


def load_split(file_path: Path, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one HDF5 split and return normalized dx plus labels/metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing {split_name} file: {file_path}")

    with h5py.File(file_path, "r") as file:
        dx = file["dx"][:].astype("float32")
        has_cp = file["has_changepoint"][:].astype("float32")
        cp = file["cp"][:].astype("int16")
        cp_dx = file["cp_dx"][:].astype("int16") if "cp_dx" in file else (cp - 1).astype("int16")
        model1 = file["model1"][:]
        model2 = file["model2"][:]

    if dx.ndim == 2:
        dx = dx[:, :, None]
    if dx.shape[1:] != (DX_LENGTH, 1):
        raise ValueError(f"Expected dx shape (N, {DX_LENGTH}, 1), got {dx.shape}.")

    dx = normalize_dx(dx)
    cp_class = np.where(has_cp == 1, cp_dx, 0).astype("int32")

    metadata = pd.DataFrame(
        {
            "split": split_name,
            "has_changepoint": has_cp.astype(int),
            "cp": cp,
            "cp_dx": cp_dx,
            "cp_class": cp_class,
            "model1": [decode_model(value) for value in model1],
            "model2": [decode_model(value) for value in model2],
        }
    )
    metadata["transition"] = np.where(
        metadata["has_changepoint"] == 1,
        metadata["model1"] + " -> " + metadata["model2"],
        metadata["model1"] + " (no changepoint)",
    )

    return dx, has_cp.reshape(-1, 1).astype("float32"), cp_class.astype("int32"), metadata


def balanced_subset(
    x: np.ndarray,
    y_has: np.ndarray,
    y_cp_class: np.ndarray,
    metadata: pd.DataFrame,
    n_total: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Balanced subset used only for fast checks."""
    if n_total <= 0:
        raise ValueError("n_total must be positive.")
    if n_total >= len(x):
        return x, y_has, y_cp_class, metadata.reset_index(drop=True)

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
        y_cp_class[selected_array],
        metadata.loc[selected_array].reset_index(drop=True),
    )


def make_sample_weights(y_has: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask localization loss for no-changepoint trajectories."""
    y_flat = y_has.reshape(-1).astype("float32")
    return np.ones_like(y_flat, dtype="float32"), y_flat.astype("float32")


def make_dataset(
    x: np.ndarray,
    y_has: np.ndarray,
    y_cp_class: np.ndarray,
    batch_size: int,
    seed: int,
    training: bool = False,
) -> tf.data.Dataset:
    """Build tf.data dataset with two outputs and sample weights."""
    y = (y_has.astype("float32"), y_cp_class.astype("int32"))
    sample_weight = make_sample_weights(y_has)
    dataset = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))
    if training:
        dataset = dataset.shuffle(min(10_000, len(x)), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@tf.keras.utils.register_keras_serializable(package="Changepoint")
class ValidPositionMask(layers.Layer):
    """Mask invalid changepoint positions before softmax."""

    def __init__(self, sequence_length: int, valid_min: int, valid_max: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.valid_min = int(valid_min)
        self.valid_max = int(valid_max)

    def call(self, logits: tf.Tensor) -> tf.Tensor:
        positions = tf.range(self.sequence_length)
        valid = tf.logical_and(positions >= self.valid_min, positions <= self.valid_max)
        valid = tf.cast(valid, tf.float32)
        mask = (1.0 - valid) * (-1e9)
        return logits + mask[tf.newaxis, :]

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "valid_min": self.valid_min, "valid_max": self.valid_max})
        return config


@tf.keras.utils.register_keras_serializable(package="Changepoint")
class LastStep(layers.Layer):
    """Return last temporal state."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:, -1, :]


@tf.keras.utils.register_keras_serializable(package="Changepoint")
class SqueezeLastAxis(layers.Layer):
    """Remove final singleton axis."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(inputs, axis=-1)


@tf.keras.utils.register_keras_serializable(package="Changepoint")
class PositionalEmbedding(layers.Layer):
    """Learned positional embedding for ConvTransformer-v2."""

    def __init__(self, sequence_length: int, d_model: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.position_embedding = layers.Embedding(input_dim=self.sequence_length, output_dim=self.d_model)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions[tf.newaxis, :, :]

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
        return config


@tf.keras.utils.register_keras_serializable(package="Changepoint")
def sparse_ce_localization(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Sparse cross-entropy for changepoint localization."""
    y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
    n_positions = tf.shape(y_pred)[1]
    y_true = tf.clip_by_value(y_true, 0, n_positions - 1)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    batch_size = tf.shape(y_pred)[0]
    gather_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), y_true], axis=1)
    true_prob = tf.gather_nd(y_pred, gather_indices)
    return -tf.math.log(true_prob)


def make_weighted_bce(positive_weight: float, negative_weight: float, reduction: str):
    """Create weighted binary cross-entropy with architecture-specific weights."""

    @tf.keras.utils.register_keras_serializable(package="Changepoint")
    def weighted_binary_crossentropy_no_cp(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_local = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_local = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        y_pred_local = tf.clip_by_value(y_pred_local, 1e-7, 1.0 - 1e-7)
        positive_loss = -float(positive_weight) * y_true_local * tf.math.log(y_pred_local)
        negative_loss = -float(negative_weight) * (1.0 - y_true_local) * tf.math.log(1.0 - y_pred_local)
        loss = positive_loss + negative_loss
        if reduction == "mean":
            return tf.reduce_mean(loss)
        return loss

    return weighted_binary_crossentropy_no_cp


def conv_lstm_residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 5,
    dilation_rate: int = 1,
    dropout: float = 0.10,
    name: str = "conv_block",
) -> tf.Tensor:
    """Residual dilated convolutional block used by CNN-LSTM-v2."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding="same", name=f"{name}_shortcut")(shortcut)
    y = layers.Conv1D(filters, kernel_size=kernel_size, padding="same", dilation_rate=dilation_rate, name=f"{name}_conv_1")(x)
    y = layers.LayerNormalization(name=f"{name}_norm_1")(y)
    y = layers.Activation("gelu", name=f"{name}_gelu_1")(y)
    y = layers.SpatialDropout1D(dropout, name=f"{name}_spatial_dropout")(y)
    y = layers.Conv1D(filters, kernel_size=kernel_size, padding="same", dilation_rate=dilation_rate, name=f"{name}_conv_2")(y)
    y = layers.LayerNormalization(name=f"{name}_norm_2")(y)
    y = layers.Add(name=f"{name}_residual_add")([shortcut, y])
    return layers.Activation("gelu", name=f"{name}_output_gelu")(y)


def build_cnn_lstm_v2(input_shape: tuple[int, int] = (DX_LENGTH, 1)) -> models.Model:
    """Build CNN-LSTM-v2 as implemented in notebook 09."""
    inputs = layers.Input(shape=input_shape, name="dx_input")
    x = layers.Conv1D(64, kernel_size=5, padding="same", name="conv_stem_1")(inputs)
    x = layers.LayerNormalization(name="conv_stem_norm_1")(x)
    x = layers.Activation("gelu", name="conv_stem_gelu_1")(x)
    x = layers.SpatialDropout1D(0.10, name="conv_stem_dropout_1")(x)
    x = layers.Conv1D(96, kernel_size=5, padding="same", name="conv_stem_2")(x)
    x = layers.LayerNormalization(name="conv_stem_norm_2")(x)
    x = layers.Activation("gelu", name="conv_stem_gelu_2")(x)
    x = conv_lstm_residual_block(x, 96, kernel_size=5, dilation_rate=1, dropout=0.10, name="conv_res_block_d1")
    x = conv_lstm_residual_block(x, 96, kernel_size=5, dilation_rate=2, dropout=0.10, name="conv_res_block_d2")
    x = conv_lstm_residual_block(x, 96, kernel_size=5, dilation_rate=4, dropout=0.10, name="conv_res_block_d4")
    x = layers.LSTM(128, return_sequences=True, dropout=0.15, name="lstm_1")(x)
    x = layers.LayerNormalization(name="lstm_norm_1")(x)
    x = layers.LSTM(96, return_sequences=True, dropout=0.10, name="lstm_2")(x)
    x = layers.LayerNormalization(name="lstm_norm_2")(x)

    avg_pool = layers.GlobalAveragePooling1D(name="average_pooling")(x)
    max_pool = layers.GlobalMaxPooling1D(name="max_pooling")(x)
    last_state = LastStep(name="last_state")(x)
    det = layers.Concatenate(name="detection_features")([avg_pool, max_pool, last_state])
    det = layers.Dense(128, activation="gelu", name="detection_dense_1")(det)
    det = layers.Dropout(0.25, name="detection_dropout_1")(det)
    det = layers.Dense(64, activation="gelu", name="detection_dense_2")(det)
    det = layers.Dropout(0.10, name="detection_dropout_2")(det)
    has_cp = layers.Dense(1, activation="sigmoid", name="has_cp")(det)

    loc = layers.TimeDistributed(layers.Dense(128, activation="gelu"), name="localization_dense_1")(x)
    loc = layers.Dropout(0.15, name="localization_dropout_1")(loc)
    loc = layers.TimeDistributed(layers.Dense(64, activation="gelu"), name="localization_dense_2")(loc)
    loc = layers.Dropout(0.10, name="localization_dropout_2")(loc)
    logits = layers.TimeDistributed(layers.Dense(1), name="localization_logits_td")(loc)
    logits = SqueezeLastAxis(name="localization_logits")(logits)
    logits = ValidPositionMask(DX_LENGTH, VALID_DX_MIN, VALID_DX_MAX, name="valid_position_mask")(logits)
    cp_dist = layers.Activation("softmax", name="cp_dist")(logits)
    return models.Model(inputs=inputs, outputs=[has_cp, cp_dist], name="cnn_lstm_v2_L100")


def convtransformer_stem_block(x: tf.Tensor, d_model: int, dropout_rate: float) -> tf.Tensor:
    """Local feature extractor used by ConvTransformer-v2."""
    x = layers.Conv1D(filters=d_model // 2, kernel_size=5, padding="same", name="conv_stem_1")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="conv_stem_norm_1")(x)
    x = layers.Activation("gelu", name="conv_stem_gelu_1")(x)
    x = layers.Dropout(dropout_rate, name="conv_stem_dropout_1")(x)
    x = layers.Conv1D(filters=d_model, kernel_size=3, padding="same", name="conv_stem_2")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="conv_stem_norm_2")(x)
    x = layers.Activation("gelu", name="conv_stem_gelu_2")(x)
    x = layers.Dropout(dropout_rate, name="conv_stem_dropout_2")(x)
    return x


def convtransformer_residual_block(
    x: tf.Tensor,
    d_model: int,
    kernel_size: int,
    dropout_rate: float,
    block_id: int,
    dilation_rate: int = 1,
) -> tf.Tensor:
    """Residual Conv1D block used by ConvTransformer-v2."""
    residual = x
    y = layers.Conv1D(filters=d_model, kernel_size=kernel_size, padding="same", dilation_rate=dilation_rate, name=f"conv_residual_{block_id}_conv_1")(x)
    y = layers.LayerNormalization(epsilon=1e-6, name=f"conv_residual_{block_id}_norm_1")(y)
    y = layers.Activation("gelu", name=f"conv_residual_{block_id}_gelu_1")(y)
    y = layers.Dropout(dropout_rate, name=f"conv_residual_{block_id}_dropout_1")(y)
    y = layers.Conv1D(filters=d_model, kernel_size=kernel_size, padding="same", dilation_rate=dilation_rate, name=f"conv_residual_{block_id}_conv_2")(y)
    y = layers.Dropout(dropout_rate, name=f"conv_residual_{block_id}_dropout_2")(y)
    x = layers.Add(name=f"conv_residual_{block_id}_add")([residual, y])
    return layers.LayerNormalization(epsilon=1e-6, name=f"conv_residual_{block_id}_norm_2")(x)


def transformer_encoder_block(
    x: tf.Tensor,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    dropout_rate: float,
    block_id: int,
) -> tf.Tensor:
    """Transformer encoder block used by ConvTransformer-v2."""
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name=f"transformer_{block_id}_mha",
    )(x, x)
    attention_output = layers.Dropout(dropout_rate, name=f"transformer_{block_id}_attn_dropout")(attention_output)
    x = layers.Add(name=f"transformer_{block_id}_attn_add")([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"transformer_{block_id}_attn_norm")(x)
    ffn = layers.Dense(ff_dim, activation="gelu", name=f"transformer_{block_id}_ffn_dense_1")(x)
    ffn = layers.Dropout(dropout_rate, name=f"transformer_{block_id}_ffn_dropout_1")(ffn)
    ffn = layers.Dense(d_model, name=f"transformer_{block_id}_ffn_dense_2")(ffn)
    ffn = layers.Dropout(dropout_rate, name=f"transformer_{block_id}_ffn_dropout_2")(ffn)
    x = layers.Add(name=f"transformer_{block_id}_ffn_add")([x, ffn])
    return layers.LayerNormalization(epsilon=1e-6, name=f"transformer_{block_id}_ffn_norm")(x)


def build_convtransformer_v2(input_shape: tuple[int, int] = (DX_LENGTH, 1)) -> models.Model:
    """Build ConvTransformer-v2 as implemented in notebook 11."""
    d_model = 96
    dropout_rate = 0.15
    inputs = layers.Input(shape=input_shape, name="dx_input")
    x = convtransformer_stem_block(inputs, d_model=d_model, dropout_rate=dropout_rate)
    for conv_id in range(1, 3):
        dilation_rate = 1 if conv_id == 1 else 2
        x = convtransformer_residual_block(x, d_model=d_model, kernel_size=5, dropout_rate=dropout_rate, block_id=conv_id, dilation_rate=dilation_rate)
    x = PositionalEmbedding(input_shape[0], d_model, name="positional_embedding")(x)
    x = layers.Dropout(dropout_rate, name="embedding_dropout")(x)
    for block_id in range(1, 4):
        x = transformer_encoder_block(x, d_model=d_model, num_heads=4, ff_dim=192, dropout_rate=dropout_rate, block_id=block_id)

    avg_pool = layers.GlobalAveragePooling1D(name="average_pooling")(x)
    max_pool = layers.GlobalMaxPooling1D(name="max_pooling")(x)
    last_state = LastStep(name="last_state")(x)
    shared = layers.Concatenate(name="shared_features")([avg_pool, max_pool, last_state])
    shared = layers.Dense(128, activation="gelu", name="shared_dense_1")(shared)
    shared = layers.Dropout(0.25, name="shared_dropout_1")(shared)
    shared = layers.Dense(64, activation="gelu", name="shared_dense_2")(shared)

    det = layers.Dense(64, activation="gelu", name="detection_dense_1")(shared)
    det = layers.Dropout(0.20, name="detection_dropout")(det)
    has_cp = layers.Dense(1, activation="sigmoid", name="has_cp")(det)

    loc = layers.Dense(96, activation="gelu", name="localization_dense_1")(x)
    loc = layers.Dropout(0.15, name="localization_dropout")(loc)
    loc = layers.Dense(48, activation="gelu", name="localization_dense_2")(loc)
    logits = layers.Dense(1, name="localization_logits_dense")(loc)
    logits = SqueezeLastAxis(name="localization_logits")(logits)
    logits = ValidPositionMask(DX_LENGTH, VALID_DX_MIN, VALID_DX_MAX, name="valid_position_mask")(logits)
    cp_dist = layers.Activation("softmax", name="cp_dist")(logits)
    return models.Model(inputs=inputs, outputs=[has_cp, cp_dist], name="convtransformer_v2_L100")


def build_model(architecture: str) -> models.Model:
    """Dispatch architecture builders."""
    if architecture == "cnn_lstm":
        return build_cnn_lstm_v2()
    if architecture == "convtransformer":
        return build_convtransformer_v2()
    raise ValueError(f"Unknown architecture: {architecture}")


def compile_model(model: models.Model, architecture: str) -> models.Model:
    """Compile the model with the original v2 losses and weights."""
    cfg = ARCHITECTURE_CONFIGS[architecture]
    bce = make_weighted_bce(cfg["positive_class_weight"], cfg["negative_class_weight"], cfg["bce_reduction"])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=[bce, sparse_ce_localization],
        loss_weights=cfg["loss_weights"],
        metrics=[
            [metrics.BinaryAccuracy(name="accuracy"), metrics.Precision(name="precision"), metrics.Recall(name="recall")],
            [metrics.SparseCategoricalAccuracy(name="sparse_accuracy")],
        ],
    )
    return model


def unpack_predictions(predictions: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract model outputs."""
    if isinstance(predictions, (list, tuple)):
        return predictions[0], predictions[1]
    return predictions["has_cp"], predictions["cp_dist"]


def detection_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute detection metrics."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    probabilities = np.asarray(probabilities).reshape(-1)
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
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
    }


def soft_positions(cp_dist: np.ndarray) -> np.ndarray:
    """Soft expected changepoint position in original trajectory coordinates."""
    positions = np.arange(cp_dist.shape[1], dtype=np.float32)
    return np.sum(cp_dist * positions.reshape(1, -1), axis=1) + 1.0


def localization_metrics(metadata: pd.DataFrame, probabilities: np.ndarray, cp_dist: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute localization metrics."""
    y_true_has = metadata["has_changepoint"].to_numpy(dtype=int)
    y_pred_has = (np.asarray(probabilities).reshape(-1) >= threshold).astype(int)
    pred_cp = soft_positions(cp_dist)
    mask_all = y_true_has == 1
    mask_tp = (y_true_has == 1) & (y_pred_has == 1)

    def compute(mask: np.ndarray, prefix: str) -> dict[str, Any]:
        if int(mask.sum()) == 0:
            return {f"{prefix}_n": 0, f"{prefix}_mae": np.nan, f"{prefix}_rmse": np.nan}
        true_cp = metadata.loc[mask, "cp"].to_numpy(dtype=np.float32)
        return {
            f"{prefix}_n": int(mask.sum()),
            f"{prefix}_mae": float(mean_absolute_error(true_cp, pred_cp[mask])),
            f"{prefix}_rmse": float(np.sqrt(mean_squared_error(true_cp, pred_cp[mask]))),
        }

    values: dict[str, Any] = {}
    values.update(compute(mask_all, "all_changepoint"))
    values.update(compute(mask_tp, "true_positive"))
    return values


def threshold_table(y_true: np.ndarray, probabilities: np.ndarray, cp_dist: np.ndarray, metadata: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Build metrics table across thresholds."""
    rows = []
    for threshold in thresholds:
        row = detection_metrics(y_true, probabilities, threshold)
        row.update(localization_metrics(metadata, probabilities, cp_dist, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame, architecture: str) -> dict[str, Any]:
    """Choose threshold with the original v2 architecture-specific rule."""
    if architecture == "cnn_lstm":
        candidates = table[(table["recall"] >= 0.65) & (table["false_positive_rate"] <= 0.25)].copy()
        if candidates.empty:
            candidates = table[(table["recall"] >= 0.65) & (table["false_positive_rate"] <= 0.30)].copy()
        if candidates.empty:
            candidates = table[table["recall"] >= 0.65].copy()
        if candidates.empty:
            candidates = table.copy()
        candidates = candidates.sort_values(
            ["f1_score", "false_positive_rate", "true_positive_rmse", "precision"],
            ascending=[False, True, True, False],
        )
        return candidates.iloc[0].to_dict()

    candidates = table[(table["false_positive_rate"] <= 0.35) & (table["recall"] >= 0.65)].copy()
    if candidates.empty:
        candidates = table[(table["false_positive_rate"] <= 0.45) & (table["recall"] >= 0.60)].copy()
    if candidates.empty:
        candidates = table.copy()
    candidates = candidates.sort_values(
        ["f1_score", "false_positive_rate", "precision"],
        ascending=[False, True, False],
    )
    return candidates.iloc[0].to_dict()


def selection_score(best: dict[str, Any], architecture: str) -> float:
    """Validation score used for early stopping/checkpoint selection."""
    if architecture == "cnn_lstm":
        return float(
            best["true_positive_rmse"]
            + 15.0 * (1.0 - best["f1_score"])
            + 8.0 * best["false_positive_rate"]
            + 4.0 * best["false_negative_rate"]
        )
    return float(
        best["all_changepoint_mae"]
        + 20.0 * (1.0 - best["f1_score"])
        + 18.0 * best["false_positive_rate"]
        + 10.0 * max(0.0, 0.70 - best["recall"])
    )


class ValidationMetricsCallback(callbacks.Callback):
    """Compute thresholded validation metrics at each epoch."""

    def __init__(
        self,
        x_val: np.ndarray,
        y_has_val: np.ndarray,
        val_metadata: pd.DataFrame,
        architecture: str,
        thresholds: np.ndarray,
        batch_size: int = 2048,
    ):
        super().__init__()
        self.x_val = x_val
        self.y_has_val = y_has_val.reshape(-1).astype(int)
        self.val_metadata = val_metadata.reset_index(drop=True)
        self.architecture = architecture
        self.thresholds = thresholds
        self.batch_size = batch_size
        self.records: list[dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        has_prob, cp_dist = unpack_predictions(self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0))
        has_prob = np.asarray(has_prob).reshape(-1)
        cp_dist = np.asarray(cp_dist)
        table = threshold_table(self.y_has_val, has_prob, cp_dist, self.val_metadata, self.thresholds)
        best = choose_threshold(table, self.architecture)
        score = selection_score(best, self.architecture)
        record = {
            "epoch": epoch + 1,
            "val_best_threshold": float(best["threshold"]),
            "val_detection_accuracy": float(best["accuracy"]),
            "val_detection_precision": float(best["precision"]),
            "val_detection_recall": float(best["recall"]),
            "val_detection_f1": float(best["f1_score"]),
            "val_false_positive_rate": float(best["false_positive_rate"]),
            "val_false_negative_rate": float(best["false_negative_rate"]),
            "val_jaccard_coefficient": float(best["jaccard_coefficient"]),
            "val_localization_mae_all_cp": float(best["all_changepoint_mae"]),
            "val_localization_rmse_all_cp": float(best["all_changepoint_rmse"]),
            "val_localization_mae_true_positive": float(best["true_positive_mae"]),
            "val_localization_rmse_true_positive": float(best["true_positive_rmse"]),
            "val_selection_score": float(score),
        }
        self.records.append(record)
        logs.update({k: v for k, v in record.items() if k != "epoch"})
        print(
            f"\nval_threshold={record['val_best_threshold']:.2f} | "
            f"F1={record['val_detection_f1']:.4f} | "
            f"recall={record['val_detection_recall']:.4f} | "
            f"FPR={record['val_false_positive_rate']:.4f} | "
            f"MAE_all={record['val_localization_mae_all_cp']:.2f}"
        )


def write_yaml_like(path: Path, config: dict[str, Any]) -> None:
    """Write a small YAML file without requiring PyYAML."""

    def scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)

    def write_block(lines: list[str], key: str, value: Any, indent: int = 0) -> None:
        prefix = " " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for child_key, child_value in value.items():
                write_block(lines, str(child_key), child_value, indent + 2)
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                lines.append(f"{prefix}  - {scalar(item)}")
        else:
            lines.append(f"{prefix}{key}: {scalar(value)}")

    lines: list[str] = []
    for config_key, config_value in config.items():
        write_block(lines, config_key, config_value)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_row(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return selected threshold row as one-row DataFrame."""
    row = table.loc[np.isclose(table["threshold"].astype(float), float(threshold))].copy()
    if row.empty:
        raise RuntimeError(f"Selected threshold {threshold} not found in table.")
    return row.iloc[[0]].copy()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run one v2 L=100 architecture/seed.")
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURE_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data_synthetic_with_without_changepoint_dx",
        help="Directory with train/val/test HDF5 files.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--reduce-lr-patience", type=int, default=6)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional balanced subset for fast checks.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional balanced validation subset for fast checks.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional balanced test subset for fast checks.")
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help=(
            "Optional fixed seed for subset selection. Use this when comparing "
            "several model seeds on the same reduced data subset."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run a single architecture/seed experiment."""
    args = parse_args()
    cfg = ARCHITECTURE_CONFIGS[args.architecture]
    thresholds = np.array(cfg["thresholds"], dtype=float)
    set_global_seed(args.seed)

    reduced_run = any(v is not None for v in [args.max_train_samples, args.max_val_samples, args.max_test_samples])
    if args.output_dir is None:
        suffix = "_fast" if reduced_run else ""
        args.output_dir = PROJECT_ROOT / "results" / "multiseed_v2_L100" / f"{args.architecture}_seed{args.seed}{suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {
        "train": args.data_dir / f"train_L{LENGTH}_dim1_with_without_dx.h5",
        "val": args.data_dir / f"val_L{LENGTH}_dim1_with_without_dx.h5",
        "test": args.data_dir / f"test_L{LENGTH}_dim1_with_without_dx.h5",
    }

    print(f"Running {cfg['display_name']} seed={args.seed}")
    x_train, y_has_train, y_cp_train, train_metadata = load_split(data_files["train"], "train")
    x_val, y_has_val, y_cp_val, val_metadata = load_split(data_files["val"], "val")
    x_test, y_has_test, y_cp_test, test_metadata = load_split(data_files["test"], "test")

    subset_seed = args.seed if args.subset_seed is None else args.subset_seed

    if args.max_train_samples is not None:
        x_train, y_has_train, y_cp_train, train_metadata = balanced_subset(
            x_train, y_has_train, y_cp_train, train_metadata, args.max_train_samples, subset_seed
        )
    if args.max_val_samples is not None:
        x_val, y_has_val, y_cp_val, val_metadata = balanced_subset(
            x_val, y_has_val, y_cp_val, val_metadata, args.max_val_samples, subset_seed + 1
        )
    if args.max_test_samples is not None:
        x_test, y_has_test, y_cp_test, test_metadata = balanced_subset(
            x_test, y_has_test, y_cp_test, test_metadata, args.max_test_samples, subset_seed + 2
        )

    train_ds = make_dataset(x_train, y_has_train, y_cp_train, args.batch_size, args.seed, training=True)
    val_ds = make_dataset(x_val, y_has_val, y_cp_val, args.batch_size, args.seed, training=False)

    model = compile_model(build_model(args.architecture), args.architecture)
    val_metrics_callback = ValidationMetricsCallback(
        x_val=x_val,
        y_has_val=y_has_val,
        val_metadata=val_metadata,
        architecture=args.architecture,
        thresholds=thresholds,
    )
    training_callbacks = [
        val_metrics_callback,
        callbacks.EarlyStopping(
            monitor=MONITOR_METRIC,
            patience=args.early_stopping_patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=MONITOR_METRIC,
            factor=0.5,
            patience=args.reduce_lr_patience,
            mode="min",
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    config_used = {
        "experiment_name": "multiseed_v2_L100",
        "architecture": args.architecture,
        "architecture_display": cfg["display_name"],
        "seed": args.seed,
        "length": LENGTH,
        "input_shape": [DX_LENGTH, 1],
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "loss_weights": cfg["loss_weights"],
        "positive_class_weight": cfg["positive_class_weight"],
        "negative_class_weight": cfg["negative_class_weight"],
        "bce_reduction": cfg["bce_reduction"],
        "thresholds": cfg["thresholds"],
        "threshold_selected_on": "validation",
        "test_used_for_threshold_selection": False,
        "reduced_run": reduced_run,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "subset_seed": args.subset_seed,
    }
    write_yaml_like(args.output_dir / "config_used.yaml", config_used)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=training_callbacks,
        verbose=1,
    )
    pd.DataFrame(history.history).to_csv(args.output_dir / "history.csv", index=False)
    pd.DataFrame(val_metrics_callback.records).to_csv(args.output_dir / "validation_epoch_metrics.csv", index=False)

    val_has_prob, val_cp_dist = unpack_predictions(model.predict(x_val, batch_size=args.batch_size, verbose=1))
    test_has_prob, test_cp_dist = unpack_predictions(model.predict(x_test, batch_size=args.batch_size, verbose=1))
    val_has_prob = np.asarray(val_has_prob).reshape(-1)
    test_has_prob = np.asarray(test_has_prob).reshape(-1)
    val_cp_dist = np.asarray(val_cp_dist)
    test_cp_dist = np.asarray(test_cp_dist)

    validation_table = threshold_table(y_has_val, val_has_prob, val_cp_dist, val_metadata, thresholds)
    selected = choose_threshold(validation_table, args.architecture)
    selected_threshold = float(selected["threshold"])
    test_table = threshold_table(y_has_test, test_has_prob, test_cp_dist, test_metadata, thresholds)

    validation_table["selected"] = np.isclose(validation_table["threshold"].astype(float), selected_threshold)
    test_table["selected"] = np.isclose(test_table["threshold"].astype(float), selected_threshold)
    validation_table.to_csv(args.output_dir / "validation_thresholds.csv", index=False)
    test_table.to_csv(args.output_dir / "test_thresholds.csv", index=False)

    validation_metrics = selected_row(validation_table, selected_threshold)
    test_metrics = selected_row(test_table, selected_threshold)
    for frame, split_name in [(validation_metrics, "validation"), (test_metrics, "test")]:
        frame.insert(0, "split", split_name)
        frame.insert(0, "reduced_run", reduced_run)
        frame.insert(0, "test_samples", len(x_test))
        frame.insert(0, "validation_samples", len(x_val))
        frame.insert(0, "train_samples", len(x_train))
        frame.insert(0, "seed", args.seed)
        frame.insert(0, "architecture_display", cfg["display_name"])
        frame.insert(0, "architecture", args.architecture)

    validation_metrics.to_csv(args.output_dir / "validation_metrics.csv", index=False)
    test_metrics.to_csv(args.output_dir / "test_metrics.csv", index=False)

    summary = {
        "architecture": args.architecture,
        "architecture_display": cfg["display_name"],
        "seed": args.seed,
        "selected_threshold": selected_threshold,
        "validation_metrics": validation_metrics.iloc[0].to_dict(),
        "test_metrics": test_metrics.iloc[0].to_dict(),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
