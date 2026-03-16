from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.dataset import GestureDataset
from ml.labels import GESTURE_ID_TO_NAME
from ml.model import GestureMLP


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate gesture model and print metrics.")
    parser.add_argument("--checkpoint", default="models/gesture_mlp.pt")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0

    rows = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
        rows.append((cls, precision, recall, f1, int(cm[cls, :].sum())))

    return accuracy, cm, rows


def main():
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    test_ds = GestureDataset(args.test_csv)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = GestureMLP(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        hidden_dims=tuple(checkpoint.get("hidden_dims", (128, 64))),
        dropout=float(checkpoint.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    true_list = []
    pred_list = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            pred_list.extend(preds.tolist())
            true_list.extend(labels.numpy().tolist())

    y_true = np.asarray(true_list, dtype=np.int64)
    y_pred = np.asarray(pred_list, dtype=np.int64)
    num_classes = int(checkpoint["num_classes"])

    accuracy, cm, rows = compute_metrics(y_true, y_pred, num_classes)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("---")
    print("Per-class metrics:")
    for cls, precision, recall, f1, support in rows:
        name = GESTURE_ID_TO_NAME.get(cls, f"class_{cls}")
        print(
            f"{cls}:{name} "
            f"| precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} support={support}"
        )


if __name__ == "__main__":
    main()
