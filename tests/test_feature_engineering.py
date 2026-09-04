import unittest

import numpy as np

from scripts.feature_engineering import CHANNEL_NAMES, build_multichannel_features


class FeatureEngineeringTests(unittest.TestCase):
    def test_l100_shape(self):
        dx = np.random.randn(4, 99, 1).astype(np.float32)
        features = build_multichannel_features(dx)

        self.assertEqual(features.shape, (4, 99, 6))

    def test_output_dtype_float32(self):
        dx = np.random.randn(4, 99, 1).astype(np.float32)
        features = build_multichannel_features(dx)

        self.assertEqual(features.dtype, np.float32)

    def test_no_nan(self):
        dx = np.random.randn(4, 99, 1).astype(np.float32)
        features = build_multichannel_features(dx)

        self.assertFalse(np.isnan(features).any())

    def test_no_inf(self):
        dx = np.random.randn(4, 99, 1).astype(np.float32)
        features = build_multichannel_features(dx)

        self.assertFalse(np.isinf(features).any())

    def test_abs_and_squared_channels_are_non_negative(self):
        dx = np.random.randn(4, 99, 1).astype(np.float32)
        features = build_multichannel_features(dx)

        abs_dx = features[:, :, 1]
        dx_squared = features[:, :, 2]

        self.assertTrue(np.all(abs_dx >= 0))
        self.assertTrue(np.all(dx_squared >= 0))

    def test_invalid_shape_raises_error(self):
        dx = np.random.randn(4, 99).astype(np.float32)

        with self.assertRaises(ValueError):
            build_multichannel_features(dx)

    def test_temporal_length_is_preserved(self):
        dx = np.random.randn(4, 123, 1).astype(np.float32)
        features = build_multichannel_features(dx)

        self.assertEqual(features.shape[1], dx.shape[1])

    def test_constant_trajectories_do_not_create_nan_or_inf(self):
        dx = np.ones((4, 99, 1), dtype=np.float32)
        features = build_multichannel_features(dx)

        self.assertEqual(features.shape, (4, 99, 6))
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())

    def test_l200_shape(self):
        dx = np.random.randn(4, 199, 1).astype(np.float32)
        features = build_multichannel_features(dx)

        self.assertEqual(features.shape, (4, 199, 6))

    def test_channel_count(self):
        self.assertEqual(len(CHANNEL_NAMES), 6)


if __name__ == "__main__":
    unittest.main()
