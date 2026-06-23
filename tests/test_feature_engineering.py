import numpy as np
import pytest

from scripts.feature_engineering import CHANNEL_NAMES, build_multichannel_features


def test_l100_shape():
    dx = np.random.randn(4, 99, 1).astype(np.float32)
    features = build_multichannel_features(dx)

    assert features.shape == (4, 99, 6)


def test_output_dtype_float32():
    dx = np.random.randn(4, 99, 1).astype(np.float32)
    features = build_multichannel_features(dx)

    assert features.dtype == np.float32


def test_no_nan():
    dx = np.random.randn(4, 99, 1).astype(np.float32)
    features = build_multichannel_features(dx)

    assert not np.isnan(features).any()


def test_no_inf():
    dx = np.random.randn(4, 99, 1).astype(np.float32)
    features = build_multichannel_features(dx)

    assert not np.isinf(features).any()


def test_abs_and_squared_channels_are_non_negative():
    dx = np.random.randn(4, 99, 1).astype(np.float32)
    features = build_multichannel_features(dx)

    abs_dx = features[:, :, 1]
    dx_squared = features[:, :, 2]

    assert np.all(abs_dx >= 0)
    assert np.all(dx_squared >= 0)


def test_invalid_shape_raises_error():
    dx = np.random.randn(4, 99).astype(np.float32)

    with pytest.raises(ValueError):
        build_multichannel_features(dx)


def test_temporal_length_is_preserved():
    dx = np.random.randn(4, 123, 1).astype(np.float32)
    features = build_multichannel_features(dx)

    assert features.shape[1] == dx.shape[1]


def test_constant_trajectories_do_not_create_nan_or_inf():
    dx = np.ones((4, 99, 1), dtype=np.float32)
    features = build_multichannel_features(dx)

    assert features.shape == (4, 99, 6)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_l200_shape():
    dx = np.random.randn(4, 199, 1).astype(np.float32)
    features = build_multichannel_features(dx)

    assert features.shape == (4, 199, 6)


def test_channel_count():
    assert len(CHANNEL_NAMES) == 6
