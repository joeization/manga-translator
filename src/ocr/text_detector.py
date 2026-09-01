from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.models import TextRegion


class Manga109YoloTextDetector:
    def __init__(self, model_path: Path, confidence: float = 0.25, text_class: str = "text") -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._text_class = text_class
        self._model: YOLO | None = None

    def detect(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        model = self._get_model()
        class_names = {index: str(name).lower() for index, name in model.names.items()}
        text_class_ids = {index for index, name in class_names.items() if name == self._text_class.lower()}
        if not text_class_ids:
            available = ", ".join(sorted(set(class_names.values())))
            raise RuntimeError(
                f"Manga109 YOLO model has no '{self._text_class}' class. Available classes: {available}"
            )

        result = model.predict(np.asarray(image.convert("RGB")), conf=self._confidence, verbose=False)[0]
        boxes: list[tuple[int, int, int, int]] = []
        for coordinates, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), strict=True):
            if int(class_id) not in text_class_ids:
                continue
            bbox = _clip_bbox(coordinates, image.size)
            if bbox is not None:
                boxes.append(bbox)
        return sorted(boxes, key=lambda bbox: (bbox[1], bbox[0]))

    def _get_model(self) -> YOLO:
        if self._model is None:
            if not self._model_path.is_file():
                raise RuntimeError(
                    "Manga109 YOLO model not found. Download the model to: "
                    f"{self._model_path}"
                )
            self._model = YOLO(self._model_path)
        return self._model


def _clip_bbox(coordinates: list[float], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = (round(value) for value in coordinates)
    width, height = size
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    return (left, top, right, bottom) if right > left and bottom > top else None


class Manga109BubbleSegmenter:
    """Use a Manga109 segmentation model as the direct bubble and OCR-region source."""

    def __init__(self, model_path: Path, confidence: float = 0.25) -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._model: YOLO | None = None

    def detect(self, image: Image.Image) -> list[TextRegion]:
        result = self._get_model().predict(np.asarray(image.convert("RGB")), conf=self._confidence, verbose=False)[0]
        if result.masks is None:
            return []

        regions: list[TextRegion] = []
        for polygon in result.masks.xy:
            bbox = _clip_bbox([polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()], image.size)
            if bbox is None:
                continue
            left, top, right, bottom = bbox
            local_polygon = np.round(polygon - np.array([left, top])).astype(np.int32)
            mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
            cv2.fillPoly(mask, [local_polygon], 1)
            regions.append(TextRegion(bbox=bbox, source_text="", layout_bbox=bbox, layout_mask=mask.astype(bool)))
        return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))

    def _get_model(self) -> YOLO:
        if self._model is None:
            if not self._model_path.is_file():
                raise RuntimeError(f"Manga109 bubble segmentation model not found: {self._model_path}")
            self._model = YOLO(self._model_path)
            if self._model.task != "segment":
                raise RuntimeError(f"Expected a segmentation model, got task: {self._model.task}")
        return self._model

class HybridTextBubbleDetector:
    """Associate YOLO text boxes with bubble-segmentation masks for high-quality region data."""

    def __init__(self, text_detector: Manga109YoloTextDetector, bubble_detector: Manga109BubbleSegmenter) -> None:
        self._text_detector = text_detector
        self._bubble_detector = bubble_detector

    def detect(self, image: Image.Image) -> list[TextRegion]:
        text_regions = [TextRegion(bbox=bbox, source_text="") for bbox in self._text_detector.detect(image)]
        bubbles = self._bubble_detector.detect(image)
        for region in text_regions:
            bubble = _best_matching_bubble(region.bbox, bubbles)
            if bubble is not None:
                region.layout_bbox = bubble.layout_bbox
                region.layout_mask = bubble.layout_mask
        return text_regions

def _best_matching_bubble(text_bbox: tuple[int, int, int, int], bubbles: list[TextRegion]) -> TextRegion | None:
    text_left, text_top, text_right, text_bottom = text_bbox
    center_x = (text_left + text_right) // 2
    center_y = (text_top + text_bottom) // 2
    best_bubble: TextRegion | None = None
    best_overlap = 0
    for bubble in bubbles:
        if not isinstance(bubble.layout_mask, np.ndarray) or bubble.layout_bbox is None:
            continue
        bubble_left, bubble_top, bubble_right, bubble_bottom = bubble.layout_bbox
        if not bubble_left <= center_x < bubble_right or not bubble_top <= center_y < bubble_bottom:
            continue
        local_x, local_y = center_x - bubble_left, center_y - bubble_top
        if not bubble.layout_mask[local_y, local_x]:
            continue
        overlap = _mask_overlap(text_bbox, bubble.layout_bbox, bubble.layout_mask)
        if overlap > best_overlap:
            best_bubble, best_overlap = bubble, overlap
    return best_bubble

def _mask_overlap(text_bbox: tuple[int, int, int, int], bubble_bbox: tuple[int, int, int, int], mask: np.ndarray) -> int:
    text_left, text_top, text_right, text_bottom = text_bbox
    bubble_left, bubble_top, bubble_right, bubble_bottom = bubble_bbox
    left, top = max(text_left, bubble_left), max(text_top, bubble_top)
    right, bottom = min(text_right, bubble_right), min(text_bottom, bubble_bottom)
    if right <= left or bottom <= top:
        return 0
    return int(np.count_nonzero(mask[top - bubble_top : bottom - bubble_top, left - bubble_left : right - bubble_left]))