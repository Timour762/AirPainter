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
    from ml.runtime import RuntimeState, apply_gesture_action
except Exception:
    GestureInferenceEngine = None
    RuntimeState = None
    apply_gesture_action = None


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

    runtime_state = RuntimeState(color_index=0) if RuntimeState else None
    color_index = runtime_state.color_index if runtime_state else 0
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
                current_color_index = runtime_state.color_index if runtime_state else color_index
                draw_pointer(frame, smooth_point, DRAW_RADIUS, COLOR_PALETTE[current_color_index])
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

                if runtime_state is not None and apply_gesture_action is not None:
                    runtime_state = apply_gesture_action(
                        active_gesture,
                        smooth_point,
                        canvas,
                        runtime_state,
                        COLOR_PALETTE,
                        draw_enabled=draw_enabled,
                    )
                    mode = runtime_state.mode
                    status_message = runtime_state.status_message
                else:
                    if active_gesture == "draw":
                        canvas.draw(smooth_point)
                        mode = "DRAW"
                        status_message = "Action: drawing"
                    else:
                        canvas.reset_stroke()
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
