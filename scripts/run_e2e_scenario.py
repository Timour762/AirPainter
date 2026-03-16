from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from canvas_manager import CanvasManager
from config import COLOR_PALETTE
from ml.inference import GestureInferenceEngine
from ml.labels import GESTURE_NAME_TO_ID
from ml.runtime import RuntimeState, apply_gesture_action


def parse_args():
    parser = argparse.ArgumentParser(description="Run e2e gesture scenario without camera.")
    parser.add_argument("--checkpoint", default="models/gesture_mlp.pt")
    parser.add_argument("--dataset", default="data/raw/landmarks_clean.csv")
    parser.add_argument("--window-size", type=int, default=7)
    return parser.parse_args()


def load_one_sample_per_label(csv_path: Path) -> dict[int, list[float]]:
    found: dict[int, list[float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            if label in found:
                continue
            vector = []
            for i in range(21):
                vector.extend([float(row[f"x{i}"]), float(row[f"y{i}"])])
            found[label] = vector
            if len(found) == 5:
                break
    if len(found) < 5:
        raise RuntimeError("Could not find samples for all 5 classes.")
    return found


def count_pixels(canvas: CanvasManager) -> int:
    return int((canvas.stroke_mask > 0).sum())


def feed_gesture_frames(
    engine: GestureInferenceEngine,
    vector: list[float],
    canvas: CanvasManager,
    state: RuntimeState,
    gesture_name: str,
    n_frames: int,
    start_point: tuple[int, int],
    dx: int,
) -> RuntimeState:
    x, y = start_point
    for _ in range(n_frames):
        prediction = engine.predict(vector)
        stable = prediction["stable_gesture"]
        active = stable["gesture_name"] if stable else "pause"
        state = apply_gesture_action(
            active,
            (x, y),
            canvas,
            state,
            COLOR_PALETTE,
            draw_enabled=True,
        )
        x += dx
    return state


def main():
    args = parse_args()
    samples = load_one_sample_per_label(Path(args.dataset))

    engine = GestureInferenceEngine(
        checkpoint_path=args.checkpoint,
        device="cpu",
        use_stabilizer=True,
        window_size=args.window_size,
    )

    canvas = CanvasManager(width=640, height=480)
    state = RuntimeState(color_index=0)
    canvas.set_color(COLOR_PALETTE[state.color_index])

    # Warmup so stabilizer history is not empty.
    pause_vector = samples[GESTURE_NAME_TO_ID["pause"]]
    state = feed_gesture_frames(
        engine, pause_vector, canvas, state, "pause", 8, (100, 100), dx=0
    )
    pixels_before_draw = count_pixels(canvas)

    draw_vector = samples[GESTURE_NAME_TO_ID["draw"]]
    state = feed_gesture_frames(
        engine, draw_vector, canvas, state, "draw", 20, (120, 220), dx=6
    )
    pixels_after_draw = count_pixels(canvas)

    erase_vector = samples[GESTURE_NAME_TO_ID["erase"]]
    state = feed_gesture_frames(
        engine, erase_vector, canvas, state, "erase", 20, (120, 220), dx=6
    )
    pixels_after_erase = count_pixels(canvas)

    state = feed_gesture_frames(
        engine, pause_vector, canvas, state, "pause", 8, (200, 200), dx=0
    )
    pixels_after_pause = count_pixels(canvas)

    clear_vector = samples[GESTURE_NAME_TO_ID["clear"]]
    state = feed_gesture_frames(
        engine, clear_vector, canvas, state, "clear", 12, (200, 200), dx=0
    )
    pixels_after_clear = count_pixels(canvas)

    color_before = state.color_index
    change_color_vector = samples[GESTURE_NAME_TO_ID["change_color"]]
    state = feed_gesture_frames(
        engine, change_color_vector, canvas, state, "change_color", 12, (200, 200), dx=0
    )
    color_after = state.color_index

    checks = {
        "draw_adds_pixels": pixels_after_draw > pixels_before_draw,
        "erase_reduces_pixels": pixels_after_erase < pixels_after_draw,
        "pause_keeps_pixels": pixels_after_pause == pixels_after_erase,
        "clear_resets_canvas": pixels_after_clear == 0,
        "change_color_updates_state": color_after != color_before,
    }

    print("E2E scenario checks:")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("---")
    print(
        "pixels: "
        f"before_draw={pixels_before_draw}, after_draw={pixels_after_draw}, "
        f"after_erase={pixels_after_erase}, after_pause={pixels_after_pause}, "
        f"after_clear={pixels_after_clear}"
    )
    print(f"color_index: before={color_before}, after={color_after}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
