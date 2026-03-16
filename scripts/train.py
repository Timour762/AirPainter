from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.dataset import GestureDataset
from ml.labels import NUM_GESTURES
from ml.model import GestureMLP


def parse_args():
    parser = argparse.ArgumentParser(description="Train GestureMLP on processed landmark dataset.")
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--output", default="models/gesture_mlp.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == labels).sum().item())
            total_count += int(labels.size(0))

    avg_loss = total_loss / max(1, total_count)
    accuracy = total_correct / max(1, total_count)
    return avg_loss, accuracy


def main():
    args = parse_args()

    device = torch.device(args.device)
    train_ds = GestureDataset(args.train_csv)
    val_ds = GestureDataset(args.val_csv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GestureMLP(
        input_dim=train_ds.feature_dim,
        num_classes=NUM_GESTURES,
        hidden_dims=(128, 64),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += int((preds == labels).sum().item())
            running_count += int(labels.size(0))

        train_loss = running_loss / max(1, running_count)
        train_acc = running_correct / max(1, running_count)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": train_ds.feature_dim,
                "num_classes": NUM_GESTURES,
                "hidden_dims": (128, 64),
                "dropout": args.dropout,
                "best_val_acc": best_val_acc,
            }
            torch.save(checkpoint, output_path)
            print(f"Saved new best checkpoint to {output_path} (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
