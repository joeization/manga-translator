from __future__ import annotations

import logging
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.inpainting import Inpainter
from src.models import OCRResult
from src.renderer import Renderer, TextAnchorDetector
from src.translator import OllamaCorrector, Translator

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(str(error))
        self.stage = stage
        self.error = error


class MangaTranslationPipeline:
    def __init__(self, ocr: object, corrector: OllamaCorrector, translator: Translator, inpainter: Inpainter, renderer: Renderer, anchor_detector: TextAnchorDetector, minimum_ocr_translation_confidence: float = 0.5) -> None:
        self._ocr = ocr
        self._corrector = corrector
        self._translator = translator
        self._inpainter = inpainter
        self._renderer = renderer
        self._anchor_detector = anchor_detector
        self._minimum_ocr_translation_confidence = minimum_ocr_translation_confidence

    def process_file(self, image_path: Path, cancel_event: Event | None = None) -> tuple[Image.Image, list[OCRResult]]:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return self.process(image, cancel_event)

    def process(self, image: Image.Image, cancel_event: Event | None = None) -> tuple[Image.Image, list[OCRResult]]:
        try:
            _check_cancelled(cancel_event)
            regions = self._ocr.detect(image)
            _check_cancelled(cancel_event)
        except Exception as error:
            raise PipelineError("OCR", error) from error

        try:
            _check_cancelled(cancel_event)
            original_ocr_texts = [region.source_text for region in regions]
            corrections = self._corrector.correct(original_ocr_texts, regions)
            _check_cancelled(cancel_event)
            for region, correction in zip(regions, corrections, strict=True):
                region.source_text = correction
        except Exception as error:
            raise PipelineError("correction", error) from error

        _check_cancelled(cancel_event)
        try:
            translatable_regions = _translation_candidates(regions, self._minimum_ocr_translation_confidence)
            translatable_region_ids = {id(region) for region in translatable_regions}
            translatable_original_texts = [
                original_ocr_texts[index]
                for index, region in enumerate(regions)
                if id(region) in translatable_region_ids
            ]
            translations = self._translator.translate(
                [region.source_text for region in translatable_regions], translatable_original_texts
            )
            for region, translation in zip(translatable_regions, translations, strict=True):
                region.translated_text = translation
        except Exception as error:
            logger.warning("Translation failed; saving the original image instead: %s", error)
            return image, []

        _check_cancelled(cancel_event)

        try:
            _check_cancelled(cancel_event)
            self._anchor_detector.detect(image, translatable_regions)
        except Exception as error:
            raise PipelineError("text anchoring", error) from error

        try:
            _check_cancelled(cancel_event)
            image = self._inpainter.inpaint(image, translatable_regions)
        except Exception as error:
            raise PipelineError("inpainting", error) from error

        try:
            _check_cancelled(cancel_event)
            return self._renderer.render(image, translatable_regions), translatable_regions
        except Exception as error:
            raise PipelineError("rendering", error) from error


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Processing cancelled because the viewer was closed.")


def _translation_candidates(regions: list[OCRResult], minimum_confidence: float = 0.5) -> list[OCRResult]:
    candidates: list[OCRResult] = []
    for region in regions:
        if region.confidence is not None and region.confidence < minimum_confidence:
            logger.warning(
                "Skipping translation for low-confidence OCR entry: %s (sentence confidence: %.4f, minimum: %.4f)",
                region.source_text,
                region.confidence,
                minimum_confidence,
            )
            continue
        candidates.append(region)
    return candidates


def draw_debug_image(image: Image.Image, regions: list[OCRResult]) -> Image.Image:
    debug_image = image.copy()
    draw = ImageDraw.Draw(debug_image)
    for region in regions:
        draw.rectangle(region.bbox, outline="blue", width=3)
        if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
            left, top, _, _ = region.layout_bbox
            contours, _ = cv2.findContours(region.layout_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = [(left + int(point[0][0]), top + int(point[0][1])) for point in contour]
                if len(points) > 1:
                    draw.line(points + [points[0]], fill="red", width=3)
        else:
            left, top, right, bottom = region.bbox
            inset = 4
            draw.rectangle((left + inset, top + inset, right - inset, bottom - inset), outline="red", width=2)
    return debug_image


def save_debug_image(image: Image.Image, regions: list[OCRResult], output_path: Path) -> None:
    debug_image = draw_debug_image(image, regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_image.save(output_path)