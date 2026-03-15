import cv2

from canvas_manager import CanvasManager
from config import BRUSH_COLOR, CAMERA_INDEX, DRAW_RADIUS, WINDOW_NAME
from hand_tracker import HandTracker
from ui import draw_coords, draw_header, draw_pointer


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Камера не открывается. Проверь CAMERA_INDEX и доступ к камере.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось получить кадр с камеры.")

    height, width = frame.shape[:2]

    tracker = HandTracker()
    canvas = CanvasManager(width, height)
    draw_enabled = True
    tracker_status = tracker.error

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка чтения кадра.")
                break

            frame = cv2.flip(frame, 1)

            results = tracker.process(frame)
            hand_detected = bool(results and results.multi_hand_landmarks)
            draw_gesture_active = False
            mode = "READY" if draw_enabled else "PAUSE"
            status_message = tracker_status

            if tracker.available and not tracker_status:
                status_message = "Tracking: hand detected" if hand_detected else "Tracking: no hand"

            if hand_detected:
                hand_landmarks = results.multi_hand_landmarks[0]
                tracker.draw_landmarks(frame, hand_landmarks)

                point = tracker.get_index_finger_tip(hand_landmarks, width, height)
                smooth_point = tracker.smooth(point)
                draw_gesture_active = tracker.is_draw_gesture(hand_landmarks)

                draw_pointer(frame, smooth_point, DRAW_RADIUS, BRUSH_COLOR)
                draw_coords(frame, smooth_point)

                if draw_enabled and draw_gesture_active:
                    canvas.draw(smooth_point)
                    mode = "DRAW"
                    status_message = "Tracking: draw gesture active"
                else:
                    canvas.reset_stroke()
                    if draw_enabled and tracker.available and not tracker_status:
                        status_message = "Tracking: show only index finger to draw"
            else:
                tracker.reset()
                canvas.reset_stroke()

            output = canvas.overlay(frame)
            draw_header(output, mode, status_message)

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
