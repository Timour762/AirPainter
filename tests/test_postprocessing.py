from __future__ import annotations

import unittest

import numpy as np

from ml.inference import PredictionStabilizer


class TestPostprocessing(unittest.TestCase):
    def test_majority_and_confidence_gate(self):
        stabilizer = PredictionStabilizer(window_size=5, confidence_threshold=0.55, majority_ratio=0.6)
        draw_probs = np.array([0.05, 0.86, 0.03, 0.03, 0.03], dtype=np.float32)

        stable = None
        for _ in range(5):
            stable = stabilizer.update(draw_probs)

        self.assertIsNotNone(stable)
        self.assertEqual(stable.gesture_name, "draw")

    def test_oneshot_debounce(self):
        stabilizer = PredictionStabilizer(
            window_size=5,
            confidence_threshold=0.55,
            majority_ratio=0.6,
            oneshot_cooldown=8,
        )
        clear_probs = np.array([0.02, 0.03, 0.03, 0.89, 0.03], dtype=np.float32)

        emitted = []
        for _ in range(6):
            out = stabilizer.update(clear_probs)
            if out is not None:
                emitted.append(out.gesture_name)

        # For one-shot gesture "clear", we should not emit on every frame.
        self.assertGreaterEqual(len(emitted), 1)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0], "clear")


if __name__ == "__main__":
    unittest.main()
