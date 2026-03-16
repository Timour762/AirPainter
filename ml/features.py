from __future__ import annotations

from typing import Iterable

import numpy as np

N_LANDMARKS = 21
RAW_VECTOR_SIZE = N_LANDMARKS * 2
DISTANCE_PAIRS = (
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (0, 4),
    (0, 8),
    (0, 12),
)
FEATURE_DIMENSION = RAW_VECTOR_SIZE + len(DISTANCE_PAIRS)


def _to_array(raw_vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(raw_vector), dtype=np.float32)
    if arr.size != RAW_VECTOR_SIZE:
        raise ValueError(f"Expected {RAW_VECTOR_SIZE} values, got {arr.size}.")
    return arr


def normalize_landmarks(raw_vector: Iterable[float]) -> np.ndarray:
    arr = _to_array(raw_vector)
    points = arr.reshape(N_LANDMARKS, 2)

    wrist = points[0]
    relative = points - wrist

    palm_scale = np.linalg.norm(points[9] - points[0])
    if palm_scale < 1e-6:
        palm_scale = 1.0

    normalized = relative / palm_scale
    distances = np.array(
        [np.linalg.norm(normalized[a] - normalized[b]) for a, b in DISTANCE_PAIRS],
        dtype=np.float32,
    )

    return np.concatenate([normalized.reshape(-1), distances], axis=0).astype(np.float32)


def normalize_batch(raw_vectors: Iterable[Iterable[float]]) -> np.ndarray:
    return np.stack([normalize_landmarks(vector) for vector in raw_vectors], axis=0)
