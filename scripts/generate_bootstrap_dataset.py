from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.labels import GESTURE_ID_TO_NAME

HEADER = ["label"] + [axis + str(i) for i in range(21) for axis in ("x", "y")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic bootstrap landmark dataset (balanced classes)."
    )
    parser.add_argument("--output", default="data/raw/landmarks_raw.csv")
    parser.add_argument("--samples-per-class", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rotate(point: tuple[float, float], angle_rad: float) -> tuple[float, float]:
    x, y = point
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (x * c - y * s, x * s + y * c)


def chain(
    start: tuple[float, float],
    angle_deg: float,
    lengths: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    angle = math.radians(angle_deg)
    points = []
    cur = start
    for seg in lengths:
        cur = (cur[0] + seg * math.cos(angle), cur[1] + seg * math.sin(angle))
        points.append(cur)
    return points


def generate_hand_local(gesture_name: str, rng: random.Random) -> list[tuple[float, float]]:
    wrist = (0.0, 0.0)

    # Base MCP anchors in local hand coordinates.
    thumb_base = (-0.16, 0.02)
    index_base = (-0.07, -0.02)
    middle_base = (0.00, -0.03)
    ring_base = (0.07, -0.02)
    pinky_base = (0.13, 0.00)

    finger_lengths = {
        "thumb": (0.05, 0.04, 0.035, 0.03),
        "index": (0.06, 0.045, 0.035, 0.03),
        "middle": (0.065, 0.05, 0.04, 0.032),
        "ring": (0.06, 0.045, 0.035, 0.03),
        "pinky": (0.05, 0.04, 0.03, 0.026),
    }

    # In image coords Y grows downward, so "up" is negative angle around -90.
    base_angles = {
        "thumb": -160.0,
        "index": -90.0,
        "middle": -88.0,
        "ring": -84.0,
        "pinky": -78.0,
    }
    folded_angles = {
        "thumb": -30.0,
        "index": 25.0,
        "middle": 20.0,
        "ring": 15.0,
        "pinky": 10.0,
    }

    states = {
        "thumb": "fold",
        "index": "fold",
        "middle": "fold",
        "ring": "fold",
        "pinky": "fold",
    }

    if gesture_name == "pause":
        states = {k: "fold" for k in states}
    elif gesture_name == "draw":
        states["index"] = "extend"
    elif gesture_name == "erase":
        states["index"] = "extend"
        states["middle"] = "extend"
    elif gesture_name == "clear":
        states = {k: "extend" for k in states}
    elif gesture_name == "change_color":
        states["thumb"] = "extend"
        states["index"] = "extend"

    def finger_angle(name: str) -> float:
        if states[name] == "extend":
            return base_angles[name] + rng.uniform(-8.0, 8.0)
        return folded_angles[name] + rng.uniform(-12.0, 12.0)

    thumb = chain(thumb_base, finger_angle("thumb"), finger_lengths["thumb"])
    index = chain(index_base, finger_angle("index"), finger_lengths["index"])
    middle = chain(middle_base, finger_angle("middle"), finger_lengths["middle"])
    ring = chain(ring_base, finger_angle("ring"), finger_lengths["ring"])
    pinky = chain(pinky_base, finger_angle("pinky"), finger_lengths["pinky"])

    # Pinch-like behavior for change_color (thumb tip near index tip).
    if gesture_name == "change_color":
        ix_tip = index[-1]
        th_tip = thumb[-1]
        blend = 0.75
        thumb[-1] = (
            th_tip[0] * (1.0 - blend) + ix_tip[0] * blend + rng.uniform(-0.008, 0.008),
            th_tip[1] * (1.0 - blend) + ix_tip[1] * blend + rng.uniform(-0.008, 0.008),
        )

    # MediaPipe order: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
    points = [wrist] + thumb + index + middle + ring + pinky
    return points


def to_global(points_local: list[tuple[float, float]], rng: random.Random) -> list[tuple[float, float]]:
    angle = math.radians(rng.uniform(-22.0, 22.0))
    scale = rng.uniform(2.5, 3.6)
    tx = rng.uniform(0.32, 0.68)
    ty = rng.uniform(0.30, 0.78)

    transformed = []
    for p in points_local:
        x, y = rotate(p, angle)
        x = tx + x * scale + rng.uniform(-0.0045, 0.0045)
        y = ty + y * scale + rng.uniform(-0.0045, 0.0045)
        transformed.append((min(0.99, max(0.01, x)), min(0.99, max(0.01, y))))
    return transformed


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    rows: list[dict] = []
    for gesture_id, gesture_name in GESTURE_ID_TO_NAME.items():
        for _ in range(args.samples_per_class):
            local = generate_hand_local(gesture_name, rng)
            global_points = to_global(local, rng)

            row = {"label": gesture_id}
            for i, (x, y) in enumerate(global_points):
                row[f"x{i}"] = f"{x:.6f}"
                row[f"y{i}"] = f"{y:.6f}"
            rows.append(row)

    rng.shuffle(rows)
    output_path = Path(args.output)
    write_csv(output_path, rows)

    print(
        f"Saved bootstrap dataset to {output_path} "
        f"with {args.samples_per_class} samples/class ({len(rows)} total)."
    )


if __name__ == "__main__":
    main()
