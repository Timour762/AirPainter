from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import normalize_landmarks

HEADER = ["label"] + [axis + str(i) for i in range(21) for axis in ("x", "y")]


def read_landmark_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    vectors: list[list[float]] = []
    labels: list[int] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        for row in reader:
            label = int(row["label"])
            vector = []
            for i in range(21):
                vector.append(float(row[f"x{i}"]))
                vector.append(float(row[f"y{i}"]))

            labels.append(label)
            vectors.append(vector)

    if not vectors:
        raise ValueError(f"Dataset file is empty: {path}")

    return np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def class_counts(labels: Iterable[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


class GestureDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        raw_vectors, labels = read_landmark_csv(csv_path)
        features = np.stack([normalize_landmarks(v) for v in raw_vectors], axis=0)

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.feature_dim = self.features.shape[1]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]
