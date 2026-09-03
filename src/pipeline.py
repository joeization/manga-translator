from __future__ import annotations

import logging
import unicodedata
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
from src.ocr.postprocess import sort_manga_reading_order
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
    def __init__(
        self,
        ocr: object,
        corrector: OllamaCorrector,
        translator: Translator,
        inpainter: Inpainter,
        renderer: Renderer,
        anchor_detector: TextAnchorDetector,
        minimum_ocr_translation_confidence: float = 0.75,
        ocr_weight_sentence: float = 0.5,
        ocr_weight_mean: float = 0.5,
        ocr_weight_std: float = 0.80,
    ) -> None:
        self._ocr = ocr
        self._corrector = corrector
        self._translator = translator
        self._inpainter = inpainter
        self._renderer = renderer
        self._anchor_detector = anchor_detector
        self._minimum_ocr_translation_confidence = minimum_ocr_translation_confidence
        self._ocr_weight_sentence = ocr_weight_sentence
        self._ocr_weight_mean = ocr_weight_mean
        self._ocr_weight_std = ocr_weight_std

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
        _check_cancelled(cancel_event)
        valid_regions = _translation_candidates(
            regions,
            minimum_confidence=self._minimum_ocr_translation_confidence,
            weight_sentence=self._ocr_weight_sentence,
            weight_mean=self._ocr_weight_mean,
            weight_std=self._ocr_weight_std,
        )
        if not valid_regions:
            return [], False

        try:
            _check_cancelled(cancel_event)
            original_ocr_texts = [region.source_text for region in valid_regions]
            corrections = self._corrector.correct(original_ocr_texts, valid_regions)
            _check_cancelled(cancel_event)
            for region, correction in zip(valid_regions, corrections, strict=True):
                region.source_text = correction
        except Exception as error:
            raise PipelineError("correction", error) from error

        _check_cancelled(cancel_event)
        try:
            translations = self._translator.translate(
                [region.source_text for region in valid_regions], original_ocr_texts
            )
            for region, translation in zip(valid_regions, translations, strict=True):
                region.translated_text = translation
            return valid_regions, False
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


_PUNCTUATION_CHARS = frozenset("…‥・:：、。!?！？-〜ー—~\"'()（）[]「」『』〈〉《》")


def _is_punctuation(ch: str) -> bool:
    return ch in _PUNCTUATION_CHARS or unicodedata.category(ch).startswith("P")


def compute_ocr_quality_score(
    sentence_confidence: float | None,
    character_confidences: list[float] | None = None,
    source_text: str | None = None,
    weight_sentence: float = 0.5,
    weight_mean: float = 0.5,
    weight_std: float = 0.80,
) -> float | None:
    """Compute combined OCR quality score from sentence confidence, mean char confidence, and char std.

    Score formula:
        score = weight_sentence * S + weight_mean * mean(chars) - weight_std * std(chars)

    Punctuation marks (e.g. ellipsis '…', colons '：', commas '、') are excluded from
    character distribution metrics when content characters exist, preventing ambiguous
    punctuation dots from dragging down the quality score of genuine sentences.
    """
    if sentence_confidence is None:
        return None

    if not character_confidences:
        return float(sentence_confidence)

    eval_confidences = character_confidences
    if source_text is not None and len(source_text) == len(character_confidences):
        content_confs = [c for c, ch in zip(character_confidences, source_text) if not _is_punctuation(ch)]
        if content_confs:
            eval_confidences = content_confs

    mean_char = float(sum(eval_confidences) / len(eval_confidences))
    if len(eval_confidences) > 1:
        variance = sum((c - mean_char) ** 2 for c in eval_confidences) / len(eval_confidences)
        std_char = float(np.sqrt(variance))
    else:
        std_char = 0.0

    return float(weight_sentence * sentence_confidence + weight_mean * mean_char - weight_std * std_char)


def _translation_candidates(
    regions: list[OCRResult],
    minimum_confidence: float = 0.75,
    weight_sentence: float = 0.5,
    weight_mean: float = 0.5,
    weight_std: float = 0.80,
) -> list[OCRResult]:
    # Sort into manga reading order (right-to-left, top-to-bottom) before passing context to the LLM.
    # This ensures the corrector and translator see dialogue in the correct narrative sequence.
    regions = sort_manga_reading_order(regions)  # type: ignore[arg-type]
    candidates: list[OCRResult] = []
    for region in regions:
        score = compute_ocr_quality_score(
            region.confidence,
            region.character_confidences,
            source_text=region.source_text,
            weight_sentence=weight_sentence,
            weight_mean=weight_mean,
            weight_std=weight_std,
        )
        if score is not None and score < minimum_confidence:
            logger.warning(
                "Skipping translation for low-confidence OCR entry: %s (quality score: %.4f, sentence confidence: %s, minimum: %.4f)",
                region.source_text,
                score,
                f"{region.confidence:.4f}" if region.confidence is not None else "unavailable",
                minimum_confidence,
            )
            continue
        candidates.append(region)
    return candidates


def draw_debug_image(image: Image.Image, regions: list[OCRResult]) -> Image.Image:
    debug_image = image.convert("RGB")
    w, h = debug_image.size
    overlay_pixels = np.zeros((h, w, 4), dtype=np.uint8)

    # 1. Accumulate semi-transparent mask overlays
    for region in regions:
        # Red semi-transparent fill for segmentation bubble
        if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
            left, top, right, bottom = region.layout_bbox
            cl, ct = max(0, left), max(0, top)
            cr, cb = min(w, right), min(h, bottom)
            if cr > cl and cb > ct:
                src = region.layout_mask[ct - top : cb - top, cl - left : cr - left]
                overlay_pixels[ct:cb, cl:cr][src] = (255, 0, 0, 64)

        # Yellow semi-transparent fill for inpaint mask
        if isinstance(region.inpaint_mask, np.ndarray) and region.inpaint_bbox is not None:
            left, top, right, bottom = region.inpaint_bbox
            cl, ct = max(0, left), max(0, top)
            cr, cb = min(w, right), min(h, bottom)
            if cr > cl and cb > ct:
                src = region.inpaint_mask[ct - top : cb - top, cl - left : cr - left]
                overlay_pixels[ct:cb, cl:cr][src] = (255, 255, 0, 120)

    # 2. Single alpha-composite for the entire page
    if np.any(overlay_pixels[:, :, 3] > 0):
        overlay_img = Image.fromarray(overlay_pixels, mode="RGBA")
        debug_image = Image.alpha_composite(debug_image.convert("RGBA"), overlay_img).convert("RGB")

    # 3. Draw sharp contour outlines and bounding boxes on top
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