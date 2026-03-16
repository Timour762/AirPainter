import cv2
from collections import deque

try:
    import mediapipe as mp
except ImportError:
    mp = None

from config import (
    FINGER_Y_MARGIN,
    GESTURE_SMOOTHING_FRAMES,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    SMOOTHING_POINTS,
)


class HandTracker:
    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_POINTS)
        self.draw_gesture_history = deque(maxlen=GESTURE_SMOOTHING_FRAMES)
        self.available = False
        self.error = None
        self.mp_hands = None
        self.mp_draw = None
        self.hands = None

        if mp is None:
            self.error = "MediaPipe is not installed."
            return

        if not hasattr(mp, "solutions"):
            self.error = "MediaPipe legacy API is missing; tracking is disabled."
            return

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.available = True

    def process(self, frame):
        if not self.available or self.hands is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw_landmarks(self, frame, hand_landmarks):
        if not self.available or self.mp_draw is None or self.mp_hands is None:
            return

        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
        )

    def get_index_finger_tip(self, hand_landmarks, width, height):
        if not self.available:
            return None

        landmark = hand_landmarks.landmark[8]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return (x, y)

    def get_landmark_vector(self, hand_landmarks):
        vector = []
        for lm in hand_landmarks.landmark:
            vector.extend([lm.x, lm.y])
        return vector

    def is_draw_gesture(self, hand_landmarks):
        if not self.available or self.mp_hands is None:
            return False

        hand_landmark = self.mp_hands.HandLandmark
        raw_gesture = (
            self._is_finger_extended(
                hand_landmarks,
                hand_landmark.INDEX_FINGER_TIP,
                hand_landmark.INDEX_FINGER_PIP,
            )
            and self._is_finger_folded(
                hand_landmarks,
                hand_landmark.MIDDLE_FINGER_TIP,
                hand_landmark.MIDDLE_FINGER_PIP,
            )
            and self._is_finger_folded(
                hand_landmarks,
                hand_landmark.RING_FINGER_TIP,
                hand_landmark.RING_FINGER_PIP,
            )
            and self._is_finger_folded(
                hand_landmarks,
                hand_landmark.PINKY_TIP,
                hand_landmark.PINKY_PIP,
            )
        )

        self.draw_gesture_history.append(raw_gesture)
        return sum(self.draw_gesture_history) >= max(1, len(self.draw_gesture_history) // 2 + 1)

    def _is_finger_extended(self, hand_landmarks, tip_index, pip_index):
        tip = hand_landmarks.landmark[tip_index]
        pip = hand_landmarks.landmark[pip_index]
        return tip.y < pip.y - FINGER_Y_MARGIN

    def _is_finger_folded(self, hand_landmarks, tip_index, pip_index):
        tip = hand_landmarks.landmark[tip_index]
        pip = hand_landmarks.landmark[pip_index]
        return tip.y > pip.y + FINGER_Y_MARGIN

    def smooth(self, point):
        self.history.append(point)
        x = int(sum(current_point[0] for current_point in self.history) / len(self.history))
        y = int(sum(current_point[1] for current_point in self.history) / len(self.history))
        return (x, y)

    def reset(self):
        self.history.clear()
        self.draw_gesture_history.clear()

    def close(self):
        if self.hands is not None:
            self.hands.close()
