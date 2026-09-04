from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.models import TextRegion

from .postprocess import postprocess_bubbles


class Manga109YoloTextDetector:
    def __init__(self, model_path: Path, confidence: float = 0.25, text_class: str = "text") -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._text_class = text_class
        self._model: YOLO | None = None

    def _detect_raw(self, image: Image.Image, image_np: np.ndarray | None = None) -> list[TextRegion]:
        model = self._get_model()
        class_names = {index: str(name).lower() for index, name in model.names.items()}
        text_class_ids = {index for index, name in class_names.items() if name == self._text_class.lower()}
        if not text_class_ids:
            available = ", ".join(sorted(set(class_names.values())))
            raise RuntimeError(
                f"Manga109 YOLO model has no '{self._text_class}' class. Available classes: {available}"
            )

        arr = image_np if image_np is not None else np.asarray(image.convert("RGB"))
        quantize = 16 if torch.cuda.is_available() else None
        result = model.predict(arr, conf=self._confidence, quantize=quantize, verbose=False)[0]
        regions: list[TextRegion] = []
        for coordinates, class_id, confidence in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True
        ):
            if int(class_id) not in text_class_ids:
                continue
            bbox = _clip_bbox(coordinates, image.size)
            if bbox is not None:
                regions.append(TextRegion(bbox=bbox, source_text="", detection_confidence=float(confidence)))
        return regions

    def detect(self, image: Image.Image) -> list[TextRegion]:
        return postprocess_bubbles(self._detect_raw(image), [])

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

    def _detect_raw(self, image: Image.Image, image_np: np.ndarray | None = None) -> list[TextRegion]:
        arr = image_np if image_np is not None else np.asarray(image.convert("RGB"))
        quantize = 16 if torch.cuda.is_available() else None
        result = self._get_model().predict(arr, conf=self._confidence, quantize=quantize, verbose=False)[0]
        if result.masks is None:
            return []

        regions: list[TextRegion] = []
        for polygon, confidence in zip(result.masks.xy, result.boxes.conf.cpu().tolist(), strict=True):
            bbox = _clip_bbox([polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()], image.size)
            if bbox is None:
                continue
            left, top, right, bottom = bbox
            local_polygon = np.round(polygon - np.array([left, top])).astype(np.int32)
            mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
            cv2.fillPoly(mask, [local_polygon], 1)
            regions.append(TextRegion(bbox=bbox, source_text="", detection_confidence=float(confidence), layout_bbox=bbox, layout_mask=mask.astype(bool)))
        return regions

    def detect(self, image: Image.Image) -> list[TextRegion]:
        return postprocess_bubbles([], self._detect_raw(image))

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
        image_np = np.asarray(image.convert("RGB"))
        text_candidates = self._text_detector._detect_raw(image, image_np=image_np)
        bubble_candidates = self._bubble_detector._detect_raw(image, image_np=image_np)
        return postprocess_bubbles(text_candidates, bubble_candidates)
