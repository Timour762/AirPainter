from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Check class balance and clean noisy landmark rows.")
    parser.add_argument("--input", default="data/raw/landmarks_raw.csv")
    parser.add_argument("--clean-output", default="data/raw/landmarks_clean.csv")
    parser.add_argument("--min-per-class", type=int, default=600)
    parser.add_argument("--skip-clean", action="store_true")
    return parser.parse_args()


def row_to_vector(row: dict) -> np.ndarray:
    values = []
    for i in range(21):
        values.append(float(row[f"x{i}"]))
        values.append(float(row[f"y{i}"]))
    return np.asarray(values, dtype=np.float32)


def is_basic_valid(vector: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(vector)) and np.all((vector >= -0.05) & (vector <= 1.05)))


def palm_scale(vector: np.ndarray) -> float:
    points = vector.reshape(21, 2)
    return float(np.linalg.norm(points[9] - points[0]))


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV has no header.")
        rows = list(reader)
    return fieldnames, rows


def clean_rows(rows: list[dict]) -> tuple[list[dict], int]:
    basic_valid_rows = []
    scales = []

    for row in rows:
        vector = row_to_vector(row)
        if not is_basic_valid(vector):
            continue
        basic_valid_rows.append(row)
        scales.append(palm_scale(vector))

    if not basic_valid_rows:
        return [], len(rows)

    scales_np = np.asarray(scales, dtype=np.float32)
    q1 = float(np.percentile(scales_np, 25))
    q3 = float(np.percentile(scales_np, 75))
    iqr = q3 - q1
    min_scale = q1 - 1.5 * iqr
    max_scale = q3 + 1.5 * iqr

    cleaned = []
    for row in basic_valid_rows:
        scale = palm_scale(row_to_vector(row))
        if min_scale <= scale <= max_scale:
            cleaned.append(row)

    removed = len(rows) - len(cleaned)
    return cleaned, removed


def class_counter(rows: list[dict]) -> Counter:
    counter = Counter()
    for row in rows:
        counter[int(row["label"])] += 1
    return counter


def print_balance(counter: Counter, min_per_class: int):
    print("Class distribution:")
    for label in sorted(counter):
        value = counter[label]
        status = "OK" if value >= min_per_class else "LOW"
        print(f"  label {label}: {value} ({status})")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.input)
    fieldnames, rows = read_rows(input_path)

    print(f"Input rows: {len(rows)}")
    print_balance(class_counter(rows), args.min_per_class)

    if args.skip_clean:
        return

    cleaned_rows, removed = clean_rows(rows)
    write_rows(Path(args.clean_output), fieldnames, cleaned_rows)
    print("---")
    print(f"Cleaned rows: {len(cleaned_rows)}")
    print(f"Removed rows: {removed}")
    print(f"Saved cleaned file: {args.clean_output}")
    print_balance(class_counter(cleaned_rows), args.min_per_class)


if __name__ == "__main__":
    main()
