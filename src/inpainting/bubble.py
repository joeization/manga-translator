from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion


class BubbleSegmenter(ABC):
    @abstractmethod
    def segment(self, image: Image.Image, region: TextRegion) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """Return a speech-bubble bounding box and local binary interior mask."""


class OpenCVContourBubbleSegmenter(BubbleSegmenter):
    def __init__(self, white_threshold: int, padding: int) -> None:
        self._white_threshold = white_threshold
        self._padding = padding

    def segment(self, image: Image.Image, region: TextRegion) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        pixels = np.array(image.convert("RGB"))
        left, top, right, bottom = region.bbox
        height, width = pixels.shape[:2]
        outer_left, outer_top = max(0, left - self._padding), max(0, top - self._padding)
        outer_right, outer_bottom = min(width, right + self._padding), min(height, bottom + self._padding)
        roi = pixels[outer_top:outer_bottom, outer_left:outer_right]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        white = np.where(gray >= self._white_threshold, 255, 0).astype(np.uint8)
        closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = (left - outer_left + (right - left) // 2, top - outer_top + (bottom - top) // 2)
        candidates = [contour for contour in contours if cv2.pointPolygonTest(contour, center, False) >= 0]
        if not candidates:
            return None
        contour = min(candidates, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        return (outer_left, outer_top, outer_right, outer_bottom), mask.astype(bool)