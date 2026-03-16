from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .features import normalize_landmarks
from .labels import ONE_SHOT_GESTURES, gesture_name
from .model import GestureMLP


@dataclass
class StablePrediction:
    gesture_id: int
    gesture_name: str
    confidence: float
    probabilities: np.ndarray


class PredictionStabilizer:
    def __init__(
        self,
        window_size: int = 7,
        confidence_threshold: float = 0.55,
        majority_ratio: float = 0.6,
        oneshot_cooldown: int = 18,
    ):
        self.window_size = max(3, window_size)
        self.confidence_threshold = confidence_threshold
        self.majority_ratio = majority_ratio
        self.oneshot_cooldown = max(1, oneshot_cooldown)

        self.prob_history: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.class_history: deque[int] = deque(maxlen=self.window_size)
        self.frame_idx = 0
        self.last_emitted_class = -1
        self.last_emit_frame = -10_000

    def update(self, probabilities: np.ndarray) -> StablePrediction | None:
        probs = np.asarray(probabilities, dtype=np.float32)
        pred = int(probs.argmax())

        self.frame_idx += 1
        self.prob_history.append(probs)
        self.class_history.append(pred)

        avg_probs = np.mean(np.stack(self.prob_history, axis=0), axis=0)
        stable_id = int(avg_probs.argmax())
        stable_conf = float(avg_probs[stable_id])

        majority = sum(c == stable_id for c in self.class_history) / len(self.class_history)
        if stable_conf < self.confidence_threshold or majority < self.majority_ratio:
            return None

        stable_name = gesture_name(stable_id)
        if stable_name in ONE_SHOT_GESTURES:
            if (
                stable_id == self.last_emitted_class
                and self.frame_idx - self.last_emit_frame < self.oneshot_cooldown
            ):
                return None

        self.last_emitted_class = stable_id
        self.last_emit_frame = self.frame_idx
        return StablePrediction(stable_id, stable_name, stable_conf, avg_probs)


class GestureInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        use_stabilizer: bool = True,
        window_size: int = 7,
    ):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        input_dim = int(checkpoint["input_dim"])
        num_classes = int(checkpoint["num_classes"])
        hidden_dims = tuple(checkpoint.get("hidden_dims", (128, 64)))
        dropout = float(checkpoint.get("dropout", 0.25))

        self.model = GestureMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.stabilizer = PredictionStabilizer(window_size=window_size) if use_stabilizer else None

    @torch.no_grad()
    def predict(self, raw_landmark_vector: Iterable[float]) -> dict:
        feature = normalize_landmarks(raw_landmark_vector)
        tensor = torch.from_numpy(feature).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])

        result = {
            "gesture_id": pred,
            "gesture_name": gesture_name(pred),
            "confidence": conf,
            "probabilities": probs,
            "stable_gesture": None,
        }

        if self.stabilizer is not None:
            stable = self.stabilizer.update(probs)
            if stable is not None:
                result["stable_gesture"] = {
                    "gesture_id": stable.gesture_id,
                    "gesture_name": stable.gesture_name,
                    "confidence": stable.confidence,
                }

        return result
