from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion


class TextAnchorDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image, regions: list[TextRegion]) -> None:
        """Attach original source-text bounds to regions when they can be detected."""


PUNCTUATION_VERTICAL_MAP: dict[str, str] = {
    "！": "！",
    "？": "？",
    "…": "⋮",
    "―": "︱",
    "—": "︱",
    "～": "︱",
    "~": "︱",
    "「": "﹁",
    "」": "﹂",
    "『": "﹃",
    "』": "﹄",
    "（": "︵",
    "）": "︶",
    "(": "︵",
    ")": "︶",
    "［": "﹇",
    "］": "﹈",
    "[": "﹇",
    "]": "﹈",
    "【": "︻",
    "】": "︼",
    "《": "︽",
    "》": "︾",
    "〈": "︿",
    "〉": "﹀",
    "、": "﹑",
    "。": "◦",
    ",": "﹑",
    ".": "◦",
}


def to_vertical_text(text: str) -> str:
    """Convert horizontal punctuation marks to vertical punctuation equivalents."""
    return "".join(PUNCTUATION_VERTICAL_MAP.get(char, char) for char in text)


class OpenCVInkAnchorDetector(TextAnchorDetector):
    def __init__(self, dark_threshold: int, border_margin: int) -> None:
        self._dark_threshold = dark_threshold
        self._border_margin = border_margin

    def detect(self, image: Image.Image, regions: list[TextRegion]) -> None:
        pixels = np.array(image.convert("RGB"))
        for region in regions:
            if not isinstance(region.layout_mask, np.ndarray) or region.layout_bbox is None:
                region.source_bbox = region.bbox
                continue
            left, top, right, bottom = region.layout_bbox
            gray = cv2.cvtColor(pixels[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
            ink = np.logical_and(gray <= self._dark_threshold, region.layout_mask)
            margin = self._border_margin
            if margin:
                ink[:margin, :] = False
                ink[-margin:, :] = False
                ink[:, :margin] = False
                ink[:, -margin:] = False
            points = np.argwhere(ink)
            if points.size == 0:
                region.source_bbox = region.bbox
                continue
            min_y, min_x = points.min(axis=0)
            max_y, max_x = points.max(axis=0) + 1
            region.source_bbox = (left + int(min_x), top + int(min_y), left + int(max_x), top + int(max_y))