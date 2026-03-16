from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create stratified train/val/test CSV splits.")
    parser.add_argument("--input", default="data/raw/landmarks_raw.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV has no header.")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no rows.")
    return fieldnames, rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_by_class(rows: list[dict], train_ratio: float, val_ratio: float, rng: random.Random):
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for label, label_rows in by_label.items():
        rng.shuffle(label_rows)
        n = len(label_rows)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if n_test < 1 and n >= 3:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1

        train_rows.extend(label_rows[:n_train])
        val_rows.extend(label_rows[n_train : n_train + n_val])
        test_rows.extend(label_rows[n_train + n_val : n_train + n_val + n_test])

        print(
            f"label={label}: total={n} train={n_train} val={n_val} test={n_test}"
        )

    return train_rows, val_rows, test_rows


def main():
    args = parse_args()
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    fieldnames, rows = read_rows(input_path)

    rng = random.Random(args.seed)
    train_rows, val_rows, test_rows = split_by_class(
        rows=rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        rng=rng,
    )

    write_rows(output_dir / "train.csv", fieldnames, train_rows)
    write_rows(output_dir / "val.csv", fieldnames, val_rows)
    write_rows(output_dir / "test.csv", fieldnames, test_rows)

    print("---")
    print(f"train.csv: {len(train_rows)} samples")
    print(f"val.csv:   {len(val_rows)} samples")
    print(f"test.csv:  {len(test_rows)} samples")


if __name__ == "__main__":
    main()
