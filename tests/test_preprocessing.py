from __future__ import annotations

import unittest

import numpy as np

from ml.features import FEATURE_DIMENSION, normalize_landmarks


class TestPreprocessing(unittest.TestCase):
    def _make_vector(self):
        points = []
        for i in range(21):
            points.append((0.25 + 0.02 * i, 0.35 + 0.015 * i))
        vector = []
        for x, y in points:
            vector.extend([x, y])
        return np.asarray(vector, dtype=np.float32)

    def test_output_shape(self):
        vector = self._make_vector()
        features = normalize_landmarks(vector)
        self.assertEqual(features.shape[0], FEATURE_DIMENSION)

    def test_translation_invariance(self):
        vector = self._make_vector()
        shifted = vector.copy()
        shifted[0::2] += 0.13
        shifted[1::2] -= 0.09

        f1 = normalize_landmarks(vector)
        f2 = normalize_landmarks(shifted)
        self.assertTrue(np.allclose(f1, f2, atol=1e-5))

    def test_scale_invariance(self):
        vector = self._make_vector()
        points = vector.reshape(21, 2)
        wrist = points[0]
        scaled = ((points - wrist) * 1.8) + wrist

        f1 = normalize_landmarks(vector)
        f2 = normalize_landmarks(scaled.reshape(-1))
        self.assertTrue(np.allclose(f1, f2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
