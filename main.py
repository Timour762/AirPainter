from pathlib import Path

import cv2

from canvas_manager import CanvasManager
from config import (
    CAMERA_INDEX,
    COLOR_PALETTE,
    DRAW_RADIUS,
    MODEL_CHECKPOINT_PATH,
    WINDOW_NAME,
)
from hand_tracker import HandTracker
from ui import draw_coords, draw_header, draw_pointer

try:
    from ml.inference import GestureInferenceEngine
except Exception:
    GestureInferenceEngine = None


def load_inference_engine():
    if GestureInferenceEngine is None:
        return None, "ML: module unavailable (fallback mode)"

    checkpoint_path = Path(MODEL_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        return None, f"ML: checkpoint not found ({MODEL_CHECKPOINT_PATH})"

    try:
        engine = GestureInferenceEngine(str(checkpoint_path), device="cpu", window_size=7)
        return engine, f"ML: loaded {MODEL_CHECKPOINT_PATH}"
    except Exception as exc:
        return None, f"ML: failed to load ({exc})"


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera is not available. Check CAMERA_INDEX and camera permissions.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")

    height, width = frame.shape[:2]
    tracker = HandTracker()
    canvas = CanvasManager(width, height)
    ml_engine, ml_status = load_inference_engine()

    color_index = 0
    canvas.set_color(COLOR_PALETTE[color_index])
    draw_enabled = True
    tracker_status = tracker.error

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error.")
                break

            frame = cv2.flip(frame, 1)
            results = tracker.process(frame)
            hand_detected = bool(results and results.multi_hand_landmarks)

            mode = "READY" if draw_enabled else "PAUSE"
            status_message = ml_status if ml_engine else (tracker_status or ml_status)
            gesture_message = None

            if hand_detected:
                hand_landmarks = results.multi_hand_landmarks[0]
                tracker.draw_landmarks(frame, hand_landmarks)

                point = tracker.get_index_finger_tip(hand_landmarks, width, height)
                smooth_point = tracker.smooth(point)
                draw_pointer(frame, smooth_point, DRAW_RADIUS, COLOR_PALETTE[color_index])
                draw_coords(frame, smooth_point)

                active_gesture = "pause"
                if ml_engine is not None:
                    raw_vector = tracker.get_landmark_vector(hand_landmarks)
                    prediction = ml_engine.predict(raw_vector)

                    raw_name = prediction["gesture_name"]
                    raw_conf = prediction["confidence"]
                    stable = prediction["stable_gesture"]
                    if stable is not None:
                        active_gesture = stable["gesture_name"]
                        gesture_message = (
                            f"Raw: {raw_name} ({raw_conf:.2f}) | "
                            f"Stable: {stable['gesture_name']} ({stable['confidence']:.2f})"
                        )
                    else:
                        gesture_message = f"Raw: {raw_name} ({raw_conf:.2f}) | Stable: -"
                else:
                    active_gesture = "draw" if tracker.is_draw_gesture(hand_landmarks) else "pause"
                    gesture_message = "ML fallback: rule-based draw gesture"

                if not draw_enabled:
                    active_gesture = "pause"

                if active_gesture == "draw":
                    canvas.draw(smooth_point)
                    mode = "DRAW"
                    status_message = "Action: drawing"
                elif active_gesture == "erase":
                    canvas.erase(smooth_point)
                    mode = "ERASE"
                    status_message = "Action: erasing"
                elif active_gesture == "clear":
                    canvas.clear()
                    canvas.reset_stroke()
                    mode = "CLEAR"
                    status_message = "Action: canvas cleared"
                elif active_gesture == "change_color":
                    color_index = (color_index + 1) % len(COLOR_PALETTE)
                    canvas.set_color(COLOR_PALETTE[color_index])
                    canvas.reset_stroke()
                    mode = "COLOR"
                    status_message = f"Action: color changed ({color_index + 1}/{len(COLOR_PALETTE)})"
                else:
                    canvas.reset_stroke()
                    if draw_enabled:
                        status_message = "Action: pause"
            else:
                tracker.reset()
                canvas.reset_stroke()
                gesture_message = "No hand detected"
                if tracker.available and not tracker_status:
                    status_message = "Tracking: no hand"

            output = canvas.overlay(frame)
            draw_header(output, mode, status_message, gesture_message)
            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                canvas.clear()
            if key == ord("d"):
                draw_enabled = not draw_enabled
                tracker.reset()
                canvas.reset_stroke()

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
