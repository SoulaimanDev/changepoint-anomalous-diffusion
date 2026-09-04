#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Label engineering utilities for ConvTransformer-v3b.

ConvTransformer-v3b keeps the multichannel input of v3a and replaces the hard
localization target by a soft distribution centered at the true changepoint
position. These labels are used only for supervised localization training. They
are not input features and therefore do not introduce information leakage into
the observed trajectory representation.
"""

from __future__ import annotations

import numpy as np


def build_soft_cp_labels(
    cp_class: np.ndarray,
    has_cp: np.ndarray,
    length: int = 99,
    sigma: float = 2.0,
    valid_min: int | None = None,
    valid_max: int | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Build soft localization labels for changepoint-position supervision.

    Parameters
    ----------
    cp_class:
        Integer changepoint class/index for each trajectory. For L=100 and
        increment inputs, valid positive positions are usually in [19, 79].
        Values for no-changepoint trajectories are ignored because the
        localization loss is masked by `has_cp`.
    has_cp:
        Binary indicator with shape (N,) or (N, 1). A value of 1 means that a
        real changepoint exists.
    length:
        Number of temporal positions in the localization head. For L=100 and
        dx(t), this is 99.
    sigma:
        Standard deviation of the Gaussian target, in time points. Smaller
        values are closer to hard labels; larger values make the target
        smoother.
    valid_min, valid_max:
        Optional inclusive bounds for valid changepoint positions. When
        provided, probability outside this interval is set to zero and each
        positive distribution is renormalized. This is useful when the model
        masks invalid positions before softmax.
    dtype:
        Output dtype. Default is np.float32.

    Returns
    -------
    np.ndarray
        Array with shape (N, length). Positive examples contain a normalized
        Gaussian-like distribution centered at `cp_class`. No-changepoint
        examples contain a uniform distribution; this choice is neutral because
        their localization loss has zero sample weight in the training script.

    Notes
    -----
    The function does not use the observed signal. It only transforms the
    localization target used by the supervised loss. The binary no-change mask
    must still be passed as sample weights during model training.
    """
    cp_class = np.asarray(cp_class).reshape(-1)
    has_cp = np.asarray(has_cp).reshape(-1).astype(bool)

    if length <= 1:
        raise ValueError("length must be greater than 1.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if valid_min is None:
        valid_min = 0
    if valid_max is None:
        valid_max = length - 1
    if valid_min < 0 or valid_max >= length or valid_min > valid_max:
        raise ValueError(
            f"Invalid valid range [{valid_min}, {valid_max}] for length={length}."
        )
    if cp_class.shape[0] != has_cp.shape[0]:
        raise ValueError(
            f"cp_class and has_cp must have the same number of examples, "
            f"got {cp_class.shape[0]} and {has_cp.shape[0]}."
        )

    n_examples = cp_class.shape[0]
    positions = np.arange(length, dtype=np.float64)
    soft = np.full((n_examples, length), 1.0 / length, dtype=np.float64)
    valid_mask = (positions >= valid_min) & (positions <= valid_max)

    positive_indices = np.where(has_cp)[0]
    if len(positive_indices) > 0:
        centers = np.clip(cp_class[positive_indices].astype(np.float64), valid_min, valid_max)
        distances = positions[None, :] - centers[:, None]
        positive_soft = np.exp(-0.5 * (distances / float(sigma)) ** 2)
        positive_soft[:, ~valid_mask] = 0.0
        positive_soft /= positive_soft.sum(axis=1, keepdims=True)
        soft[positive_indices] = positive_soft

    soft = soft.astype(dtype)
    if not np.isfinite(soft).all():
        raise ValueError("Soft changepoint labels contain NaN or inf values.")

    return soft
