#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature engineering utilities for ConvTransformer-v3a.

This module prepares the multichannel input representation used in the
ConvTransformer-v3a feature pipeline. It does not use labels, changepoint
positions, has_cp, or model identity labels.

Input shape:
    dx.shape == (N, T, 1), where T = L - 1

Output shape:
    features.shape == (N, T, 6)

Channels:
    1. dx_norm
    2. abs_dx
    3. dx_squared
    4. local_variance
    5. local_abs_mean
    6. lag1_product
"""

from __future__ import annotations

import numpy as np


CHANNEL_NAMES = [
    "dx_norm",
    "abs_dx",
    "dx_squared",
    "local_variance",
    "local_abs_mean",
    "lag1_product",
]


def _validate_dx(dx: np.ndarray) -> np.ndarray:
    """
    Validate the input increment array.

    Parameters
    ----------
    dx:
        Input increments expected to have shape (N, T, 1).

    Returns
    -------
    np.ndarray
        Input converted to a NumPy array.

    Raises
    ------
    ValueError
        If the input does not have shape (N, T, 1).
    """
    dx = np.asarray(dx)

    if dx.ndim != 3:
        raise ValueError(
            f"Expected dx with shape (N, T, 1), but got ndim={dx.ndim}, shape={dx.shape}."
        )

    if dx.shape[-1] != 1:
        raise ValueError(
            f"Expected dx with one channel in the last axis, but got shape={dx.shape}."
        )

    if dx.shape[1] < 2:
        raise ValueError(
            f"Expected at least two temporal positions, but got shape={dx.shape}."
        )

    return dx


def _check_window(window: int) -> None:
    """Check that the rolling window is a positive odd integer."""
    if not isinstance(window, int):
        raise ValueError("window must be an integer.")

    if window <= 0:
        raise ValueError("window must be positive.")

    if window % 2 == 0:
        raise ValueError("window must be odd to preserve centered local statistics.")


def _rolling_mean_edge(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute centered rolling mean with edge padding.

    The output keeps the same temporal length as the input.
    """
    _check_window(window)

    pad = window // 2
    x_padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    output = np.empty_like(x, dtype=np.float64)

    for t in range(x.shape[1]):
        output[:, t, :] = x_padded[:, t:t + window, :].mean(axis=1)

    return output


def _rolling_variance_edge(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute centered rolling variance with edge padding.

    The output keeps the same temporal length as the input.
    """
    _check_window(window)

    pad = window // 2
    x_padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    output = np.empty_like(x, dtype=np.float64)

    for t in range(x.shape[1]):
        output[:, t, :] = x_padded[:, t:t + window, :].var(axis=1)

    return output


def build_multichannel_features(
    dx: np.ndarray,
    window: int = 5,
    eps: float = 1e-8,
    dtype=np.float32,
) -> np.ndarray:
    """
    Build multichannel features from trajectory increments.

    This function is used to prepare the ConvTransformer-v3a input pipeline.
    It uses only the observed increment sequence and does not use:
    - the true changepoint position;
    - the binary has_cp label;
    - model identity labels;
    - any target variable.

    Parameters
    ----------
    dx:
        Increment sequence with shape (N, L-1, 1).
    window:
        Odd rolling window size used for local statistics. Default is 5.
    eps:
        Small constant used to avoid division by zero during per-trajectory
        standardization. Default is 1e-8.
    dtype:
        Output dtype. Default is np.float32.

    Returns
    -------
    np.ndarray
        Multichannel representation with shape (N, L-1, 6).

    Notes
    -----
    The lag-1 product channel is defined as:

        lag1_product[t] = dx_norm[t] * dx_norm[t-1]

    For t = 0, there is no previous increment. Therefore, the initial value
    is set to 0.0 by convention.
    """
    dx = _validate_dx(dx).astype(np.float64)
    _check_window(window)

    mean = dx.mean(axis=1, keepdims=True)
    std = dx.std(axis=1, keepdims=True)
    dx_norm = (dx - mean) / (std + eps)

    abs_dx = np.abs(dx_norm)
    dx_squared = dx_norm ** 2
    local_variance = _rolling_variance_edge(dx_norm, window=window)
    local_abs_mean = _rolling_mean_edge(abs_dx, window=window)

    lag1_product = np.zeros_like(dx_norm, dtype=np.float64)
    lag1_product[:, 1:, :] = dx_norm[:, 1:, :] * dx_norm[:, :-1, :]

    features = np.concatenate(
        [
            dx_norm,
            abs_dx,
            dx_squared,
            local_variance,
            local_abs_mean,
            lag1_product,
        ],
        axis=-1,
    ).astype(dtype)

    expected_shape = (dx.shape[0], dx.shape[1], len(CHANNEL_NAMES))
    if features.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape {features.shape}. Expected {expected_shape}."
        )

    if not np.isfinite(features).all():
        raise ValueError("The generated multichannel features contain NaN or inf values.")

    return features


if __name__ == "__main__":
    example_dx = np.random.randn(4, 99, 1).astype(np.float32)
    example_features = build_multichannel_features(example_dx)

    print("Input shape:", example_dx.shape)
    print("Output shape:", example_features.shape)
    print("Channels:", CHANNEL_NAMES)
