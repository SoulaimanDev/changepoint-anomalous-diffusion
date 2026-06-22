import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.build_synthetic_with_without_changepoint_dx import (
    MODELS,
    TRANSITIONS,
    build_record_schedule,
    generate_split,
)


class FakeGenerator:
    """Small deterministic stand-in for andi-datasets used by structural tests."""

    @staticmethod
    def _trajectory(T, alpha, dim=1):
        steps = np.linspace(0.5, 1.5, T, dtype=np.float64) ** alpha
        trajectory = np.cumsum(steps)
        if dim == 1:
            return trajectory
        return np.repeat(trajectory[:, None], dim, axis=1)

    def attm(self, T, alpha, dim=1):
        return self._trajectory(T, alpha, dim)

    def ctrw(self, T, alpha, dim=1):
        return self._trajectory(T, alpha, dim)

    def fbm(self, T, alpha, dim=1):
        return self._trajectory(T, alpha, dim)

    def lw(self, T, alpha, dim=1):
        return self._trajectory(T, alpha, dim)

    def sbm(self, T, alpha, dim=1):
        return self._trajectory(T, alpha, dim)


class ProtocolBGenerationTests(unittest.TestCase):
    def test_schedule_is_balanced_and_complete(self):
        kinds, model1, model2 = build_record_schedule(
            with_per_transition=2,
            without_per_model=8,
            random_state=np.random.default_rng(42),
        )
        self.assertEqual(int(kinds.sum()), 2 * len(TRANSITIONS))
        self.assertEqual(int((kinds == 0).sum()), 8 * len(MODELS))

        pairs = pd.DataFrame({"model1": model1, "model2": model2, "has_cp": kinds})
        positive_counts = pairs[pairs.has_cp == 1].groupby(["model1", "model2"]).size()
        negative_counts = pairs[pairs.has_cp == 0].groupby("model1").size()
        self.assertTrue((positive_counts == 2).all())
        self.assertTrue((negative_counts == 8).all())

    def test_hdf5_contract_and_changepoint_indices(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            file_path, statistics = generate_split(
                split_name="test",
                split_sizes={"with_per_transition": 1, "without_per_model": 4},
                generator=FakeGenerator(),
                output_dir=output_dir,
                length=100,
                minimum_segment_length=20,
                dimension=1,
                noise_levels=[],
                seed=42,
                compression=None,
            )

            self.assertEqual(statistics["with_changepoint"], 20)
            self.assertEqual(statistics["without_changepoint"], 20)
            with h5py.File(file_path, "r") as file:
                self.assertEqual(file["X"].shape, (40, 100, 1))
                self.assertEqual(file["dx"].shape, (40, 99, 1))
                np.testing.assert_allclose(file["dx"][:], np.diff(file["X"][:], axis=1))

                has_cp = file["has_changepoint"][:].astype(bool)
                cp = file["cp"][:]
                cp_dx = file["cp_dx"][:]
                cp_dx_norm = file["cp_dx_norm"][:]
                self.assertTrue(np.all((cp[has_cp] >= 20) & (cp[has_cp] <= 80)))
                np.testing.assert_array_equal(cp_dx[has_cp], cp[has_cp] - 1)
                np.testing.assert_allclose(cp_dx_norm[has_cp], cp_dx[has_cp] / 99)
                self.assertTrue(np.all(cp[~has_cp] == -1))
                self.assertTrue(np.all(cp_dx[~has_cp] == -1))
                self.assertTrue(np.all(cp_dx_norm[~has_cp] == -1.0))


if __name__ == "__main__":
    unittest.main()
