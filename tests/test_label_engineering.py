import numpy as np
import pytest

from scripts.label_engineering import build_soft_cp_labels


def test_soft_label_shape():
    cp_class = np.array([19, 40, 79])
    has_cp = np.array([1, 1, 1])

    labels = build_soft_cp_labels(cp_class, has_cp, length=99, sigma=2.0)

    assert labels.shape == (3, 99)


def test_soft_labels_sum_to_one():
    cp_class = np.array([19, 40, 79, -1])
    has_cp = np.array([1, 1, 1, 0])

    labels = build_soft_cp_labels(cp_class, has_cp, length=99, sigma=2.0)

    np.testing.assert_allclose(labels.sum(axis=1), np.ones(4), rtol=1e-6, atol=1e-6)


def test_positive_label_peak_is_near_cp_class():
    cp_class = np.array([19, 40, 79])
    has_cp = np.array([1, 1, 1])

    labels = build_soft_cp_labels(cp_class, has_cp, length=99, sigma=2.0)

    assert np.array_equal(np.argmax(labels, axis=1), cp_class)


def test_no_changepoint_label_is_uniform():
    cp_class = np.array([-1, -1])
    has_cp = np.array([0, 0])

    labels = build_soft_cp_labels(cp_class, has_cp, length=99, sigma=2.0)

    expected = np.full((2, 99), 1.0 / 99, dtype=np.float32)
    np.testing.assert_allclose(labels, expected, rtol=1e-6, atol=1e-6)


def test_accepts_has_cp_column_vector():
    cp_class = np.array([20, 30])
    has_cp = np.array([[1], [1]])

    labels = build_soft_cp_labels(cp_class, has_cp, length=99, sigma=2.0)

    assert labels.shape == (2, 99)


def test_rejects_invalid_sigma():
    with pytest.raises(ValueError):
        build_soft_cp_labels(np.array([20]), np.array([1]), length=99, sigma=0)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_soft_cp_labels(np.array([20, 30]), np.array([1]), length=99, sigma=2.0)


def test_valid_range_sets_outside_probability_to_zero():
    labels = build_soft_cp_labels(
        np.array([19, 40, 79]),
        np.array([1, 1, 1]),
        length=99,
        sigma=2.0,
        valid_min=19,
        valid_max=79,
    )

    assert np.all(labels[:, :19] == 0)
    assert np.all(labels[:, 80:] == 0)
    np.testing.assert_allclose(labels.sum(axis=1), np.ones(3), rtol=1e-6, atol=1e-6)
