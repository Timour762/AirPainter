import cv2
import numpy as np

from config import BRUSH_COLOR, BRUSH_THICKNESS


class CanvasManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.stroke_mask = np.zeros((height, width), dtype=np.uint8)
        self.prev_point = None

    def draw(self, point):
        if self.prev_point is None:
            self.prev_point = point
            return

        cv2.line(self.canvas, self.prev_point, point, BRUSH_COLOR, BRUSH_THICKNESS)
        cv2.line(self.stroke_mask, self.prev_point, point, 255, BRUSH_THICKNESS)
        self.prev_point = point

    def reset_stroke(self):
        self.prev_point = None

    def clear(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stroke_mask = np.zeros((self.height, self.width), dtype=np.uint8)

    def overlay(self, frame):
        output = frame.copy()
        mask = self.stroke_mask > 0
        output[mask] = self.canvas[mask]
        return output
