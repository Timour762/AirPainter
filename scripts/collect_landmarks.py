from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional
import sys

import cv2
import mediapipe as mp

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.labels import GESTURE_ID_TO_NAME

HEADER = ["label"] + [axis + str(i) for i in range(21) for axis in ("x", "y")]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect raw hand landmarks for gesture dataset.")
    parser.add_argument("--output", default="data/raw/landmarks_raw.csv")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    parser.add_argument("--mirror", action="store_true", help="Mirror camera frame.")
    return parser.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def count_existing_samples(csv_path: Path) -> dict[int, int]:
    if not csv_path.exists():
        return {gesture_id: 0 for gesture_id in GESTURE_ID_TO_NAME}

    counts = {gesture_id: 0 for gesture_id in GESTURE_ID_TO_NAME}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            counts[label] = counts.get(label, 0) + 1
    return counts


def write_row(csv_path: Path, label: int, vector: list[float]):
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not file_exists:
            writer.writeheader()

        row = {"label": label}
        for i in range(21):
            row[f"x{i}"] = vector[2 * i]
            row[f"y{i}"] = vector[2 * i + 1]
        writer.writerow(row)


def extract_vector(hand_landmarks) -> list[float]:
    vector: list[float] = []
    for lm in hand_landmarks.landmark:
        vector.extend([lm.x, lm.y])
    return vector


def draw_hud(
    frame,
    current_label: int,
    counts: dict[int, int],
    last_saved: Optional[str],
    max_samples_per_class: int,
):
    color = (0, 220, 0)
    cv2.putText(
        frame,
        f"Label: {current_label} ({GESTURE_ID_TO_NAME[current_label]})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        frame,
        "Keys: 0-4 label | S save | Q quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    if max_samples_per_class > 0:
        cv2.putText(
            frame,
            f"Max/class: {max_samples_per_class}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    y = 120
    for gesture_id, name in GESTURE_ID_TO_NAME.items():
        cv2.putText(
            frame,
            f"{gesture_id}:{name} = {counts.get(gesture_id, 0)}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )
        y += 22

    if last_saved:
        cv2.putText(
            frame,
            last_saved,
            (10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def main():
    args = parse_args()
    output_path = Path(args.output)
    ensure_parent(output_path)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    counts = count_existing_samples(output_path)
    current_label = 1
    last_saved_message: Optional[str] = None

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Camera is not available.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            current_vector = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_vector = extract_vector(hand_landmarks)

            draw_hud(
                frame,
                current_label,
                counts,
                last_saved_message,
                args.max_samples_per_class,
            )
            cv2.imshow("Collect Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4")):
                current_label = int(chr(key))
                last_saved_message = f"Switched to {GESTURE_ID_TO_NAME[current_label]}"

            if key == ord("s"):
                if current_vector is None:
                    last_saved_message = "No hand detected"
                    continue

                if args.max_samples_per_class > 0 and counts[current_label] >= args.max_samples_per_class:
                    last_saved_message = "Class limit reached"
                    continue

                write_row(output_path, current_label, current_vector)
                counts[current_label] = counts.get(current_label, 0) + 1
                last_saved_message = (
                    f"Saved {GESTURE_ID_TO_NAME[current_label]} "
                    f"({counts[current_label]})"
                )
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
