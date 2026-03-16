GESTURE_ID_TO_NAME = {
    0: "pause",
    1: "draw",
    2: "erase",
    3: "clear",
    4: "change_color",
}

GESTURE_NAME_TO_ID = {name: idx for idx, name in GESTURE_ID_TO_NAME.items()}
NUM_GESTURES = len(GESTURE_ID_TO_NAME)
ONE_SHOT_GESTURES = {"clear", "change_color"}


def gesture_name(gesture_id: int) -> str:
    return GESTURE_ID_TO_NAME.get(gesture_id, "unknown")
