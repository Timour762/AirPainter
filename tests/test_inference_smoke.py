from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml.inference import GestureInferenceEngine


class TestInferenceSmoke(unittest.TestCase):
    def test_smoke_samples(self):
        checkpoint = Path("models/gesture_mlp.pt")
        sample_file = Path("data/processed/smoke_landmarks.json")

        self.assertTrue(checkpoint.exists(), "Missing model checkpoint for smoke test.")
        self.assertTrue(sample_file.exists(), "Missing smoke landmark sample file.")

        engine = GestureInferenceEngine(str(checkpoint), device="cpu", use_stabilizer=False)
        payload = json.loads(sample_file.read_text(encoding="utf-8"))
        samples = payload.get("samples", [])
        self.assertGreaterEqual(len(samples), 5)

        for sample in samples:
            expected = sample["gesture_name"]
            vector = sample["vector"]
            out = engine.predict(vector)
            self.assertEqual(out["gesture_name"], expected)


if __name__ == "__main__":
    unittest.main()
