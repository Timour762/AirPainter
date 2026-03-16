import cv2


def draw_header(frame, mode, status_message=None, gesture_message=None):
    height, width = frame.shape[:2]

    header_height = 120 if (status_message or gesture_message) else 80
    cv2.rectangle(frame, (0, 0), (width, header_height), (40, 40, 40), -1)

    cv2.putText(
        frame,
        "AirPainter - CV + ML",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"MODE: {mode}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    if gesture_message:
        cv2.putText(
            frame,
            gesture_message,
            (300, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 220, 255),
            2,
        )

    controls_y = 85 if (status_message or gesture_message) else 58
    cv2.putText(
        frame,
        "Gestures: draw / erase / clear / change_color | D toggle | C clear | Q quit",
        (300, controls_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )

    if status_message:
        cv2.putText(
            frame,
            status_message,
            (300, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 140, 255),
            2,
        )


def draw_pointer(frame, point, radius, color):
    cv2.circle(frame, point, radius + 3, (255, 255, 255), 2)
    cv2.circle(frame, point, radius, color, -1)


def draw_coords(frame, point):
    cv2.putText(
        frame,
        f"Index finger: {point[0]}, {point[1]}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
