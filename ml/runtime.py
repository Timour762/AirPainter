from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeState:
    color_index: int = 0
    mode: str = "READY"
    status_message: str = "Action: pause"


def apply_gesture_action(
    gesture_name: str,
    point: tuple[int, int],
    canvas,
    state: RuntimeState,
    color_palette: tuple[tuple[int, int, int], ...],
    draw_enabled: bool = True,
) -> RuntimeState:
    if not draw_enabled:
        gesture_name = "pause"

    if gesture_name == "draw":
        canvas.draw(point)
        state.mode = "DRAW"
        state.status_message = "Action: drawing"
    elif gesture_name == "erase":
        canvas.erase(point)
        state.mode = "ERASE"
        state.status_message = "Action: erasing"
    elif gesture_name == "clear":
        canvas.clear()
        canvas.reset_stroke()
        state.mode = "CLEAR"
        state.status_message = "Action: canvas cleared"
    elif gesture_name == "change_color":
        state.color_index = (state.color_index + 1) % len(color_palette)
        canvas.set_color(color_palette[state.color_index])
        canvas.reset_stroke()
        state.mode = "COLOR"
        state.status_message = (
            f"Action: color changed ({state.color_index + 1}/{len(color_palette)})"
        )
    else:
        canvas.reset_stroke()
        state.mode = "PAUSE" if not draw_enabled else "READY"
        state.status_message = "Action: pause"

    return state
