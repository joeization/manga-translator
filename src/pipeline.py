from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.inpainting import Inpainter
from src.models import OCRResult
from src.renderer import Renderer, TextAnchorDetector
from src.translator import OllamaCorrector, Translator

logger = logging.getLogger(__name__)


@dataclass
class _PipelinePacket:
    index: int
    item: Any
    original_image: Image.Image | None = None
    regions: list[OCRResult] = field(default_factory=list)
    inpainted_image: Image.Image | None = None
    translated_image: Image.Image | None = None
    error: PipelineError | None = None


def _drain_queue(q: Queue) -> None:
    try:
        while not q.empty():
            q.get_nowait()
            q.task_done()
    except Exception:
        pass


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

    def _run_ocr(self, image: Image.Image, cancel_event: Event | None = None) -> list[OCRResult]:
        try:
            _check_cancelled(cancel_event)
            regions = self._ocr.detect(image)
            _check_cancelled(cancel_event)
            return regions
        except Exception as error:
            raise PipelineError("OCR", error) from error

    def _run_translation(
        self, image: Image.Image, regions: list[OCRResult], cancel_event: Event | None = None
    ) -> tuple[list[OCRResult], bool]:
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
            return translatable_regions, False
        except Exception as error:
            logger.warning("Translation failed; saving the original image instead: %s", error)
            return [], True

    def _run_inpaint(
        self, image: Image.Image, regions: list[OCRResult], cancel_event: Event | None = None
    ) -> Image.Image:
        try:
            _check_cancelled(cancel_event)
            self._anchor_detector.detect(image, regions)
        except Exception as error:
            raise PipelineError("text anchoring", error) from error

        try:
            _check_cancelled(cancel_event)
            return self._inpainter.inpaint(image, regions)
        except Exception as error:
            raise PipelineError("inpainting", error) from error

    def _run_render(
        self, image: Image.Image, regions: list[OCRResult], cancel_event: Event | None = None
    ) -> Image.Image:
        try:
            _check_cancelled(cancel_event)
            return self._renderer.render(image, regions)
        except Exception as error:
            raise PipelineError("rendering", error) from error

    def process(self, image: Image.Image, cancel_event: Event | None = None) -> tuple[Image.Image, list[OCRResult]]:
        regions = self._run_ocr(image, cancel_event)
        translatable_regions, failed = self._run_translation(image, regions, cancel_event)
        if failed:
            return image, []
        _check_cancelled(cancel_event)
        inpainted_image = self._run_inpaint(image, translatable_regions, cancel_event)
        _check_cancelled(cancel_event)
        rendered_image = self._run_render(inpainted_image, translatable_regions, cancel_event)
        return rendered_image, translatable_regions

    def process_pipelined(
        self,
        items: list[Any],
        cancel_event: Event | None = None,
        queue_size: int = 2,
    ) -> Iterator[tuple[Any, Image.Image | None, Image.Image | None, list[OCRResult], PipelineError | None]]:
        if not items:
            return

        q_ocr_to_trans: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)
        q_trans_to_inpaint: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)
        q_inpaint_to_render: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)
        q_render_to_out: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)

        def ocr_worker() -> None:
            for idx, item in enumerate(items):
                if cancel_event is not None and cancel_event.is_set():
                    break
                packet = _PipelinePacket(index=idx, item=item)
                try:
                    img = item.load() if hasattr(item, "load") else item
                    packet.original_image = img
                    packet.regions = self._run_ocr(img, cancel_event)
                except PipelineError as err:
                    packet.error = err
                except Exception as err:
                    packet.error = PipelineError("OCR", err)
                q_ocr_to_trans.put(packet)
            q_ocr_to_trans.put(None)

        def trans_worker() -> None:
            while True:
                packet = q_ocr_to_trans.get()
                if packet is None:
                    q_trans_to_inpaint.put(None)
                    q_ocr_to_trans.task_done()
                    break
                if cancel_event is not None and cancel_event.is_set():
                    q_trans_to_inpaint.put(None)
                    q_ocr_to_trans.task_done()
                    break
                if packet.error is None and packet.original_image is not None:
                    try:
                        translatable, failed = self._run_translation(packet.original_image, packet.regions, cancel_event)
                        if failed:
                            packet.regions = []
                            packet.translated_image = packet.original_image
                        else:
                            packet.regions = translatable
                    except PipelineError as err:
                        packet.error = err
                    except Exception as err:
                        packet.error = PipelineError("translation", err)
                q_trans_to_inpaint.put(packet)
                q_ocr_to_trans.task_done()

        def inpaint_worker() -> None:
            while True:
                packet = q_trans_to_inpaint.get()
                if packet is None:
                    q_inpaint_to_render.put(None)
                    q_trans_to_inpaint.task_done()
                    break
                if cancel_event is not None and cancel_event.is_set():
                    q_inpaint_to_render.put(None)
                    q_trans_to_inpaint.task_done()
                    break
                if packet.error is None and packet.original_image is not None and packet.translated_image is None:
                    try:
                        packet.inpainted_image = self._run_inpaint(packet.original_image.copy(), packet.regions, cancel_event)
                    except PipelineError as err:
                        packet.error = err
                    except Exception as err:
                        packet.error = PipelineError("inpainting", err)
                else:
                    packet.inpainted_image = packet.original_image
                q_inpaint_to_render.put(packet)
                q_trans_to_inpaint.task_done()

        def render_worker() -> None:
            while True:
                packet = q_inpaint_to_render.get()
                if packet is None:
                    q_render_to_out.put(None)
                    q_inpaint_to_render.task_done()
                    break
                if cancel_event is not None and cancel_event.is_set():
                    q_render_to_out.put(None)
                    q_inpaint_to_render.task_done()
                    break
                if packet.error is None and packet.inpainted_image is not None and packet.translated_image is None:
                    try:
                        packet.translated_image = self._run_render(packet.inpainted_image, packet.regions, cancel_event)
                    except PipelineError as err:
                        packet.error = err
                    except Exception as err:
                        packet.error = PipelineError("rendering", err)
                elif packet.translated_image is None:
                    packet.translated_image = packet.original_image
                q_render_to_out.put(packet)
                q_inpaint_to_render.task_done()

        threads = [
            Thread(target=ocr_worker, daemon=True),
            Thread(target=trans_worker, daemon=True),
            Thread(target=inpaint_worker, daemon=True),
            Thread(target=render_worker, daemon=True),
        ]
        for t in threads:
            t.start()

        try:
            while True:
                packet = q_render_to_out.get()
                if packet is None:
                    q_render_to_out.task_done()
                    break
                yield packet.item, packet.original_image, packet.translated_image, packet.regions, packet.error
                q_render_to_out.task_done()
        finally:
            _drain_queue(q_ocr_to_trans)
            _drain_queue(q_trans_to_inpaint)
            _drain_queue(q_inpaint_to_render)
            _drain_queue(q_render_to_out)
            for t in threads:
                t.join(timeout=1.0)


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
            left, top, right, bottom = region.layout_bbox
            segmentation_overlay = Image.new("RGBA", debug_image.size, (0, 0, 0, 0))
            segmentation_pixels = np.asarray(segmentation_overlay).copy()
            segmentation_pixels[top:bottom, left:right][region.layout_mask] = (255, 0, 0, 64)
            debug_image = Image.alpha_composite(debug_image.convert("RGBA"), Image.fromarray(segmentation_pixels, mode="RGBA")).convert("RGB")
            draw = ImageDraw.Draw(debug_image)
            contours, _ = cv2.findContours(region.layout_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = [(left + int(point[0][0]), top + int(point[0][1])) for point in contour]
                if len(points) > 1:
                    draw.line(points + [points[0]], fill="red", width=3)
        else:
            left, top, right, bottom = region.bbox
            inset = 4
            draw.rectangle((left + inset, top + inset, right - inset, bottom - inset), outline="red", width=2)
        if isinstance(region.inpaint_mask, np.ndarray) and region.inpaint_bbox is not None:
            left, top, _, _ = region.inpaint_bbox
            overlay = Image.new("RGBA", debug_image.size, (0, 0, 0, 0))
            overlay_pixels = np.asarray(overlay).copy()
            mask = region.inpaint_mask
            overlay_pixels[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = (255, 255, 0, 120)
            debug_image = Image.alpha_composite(debug_image.convert("RGBA"), Image.fromarray(overlay_pixels, mode="RGBA")).convert("RGB")
            draw = ImageDraw.Draw(debug_image)
    return debug_image


def save_debug_image(image: Image.Image, regions: list[OCRResult], output_path: Path) -> None:
    debug_image = draw_debug_image(image, regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_image.save(output_path)