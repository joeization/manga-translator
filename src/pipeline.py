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
from src.translator import Translator
from src.translator.ollama import format_response

logger = logging.getLogger(__name__)


@dataclass
class _PipelinePacket:
    index: int
    item: Any
    original_image: Image.Image | None = None
    regions: list[OCRResult] = field(default_factory=list)
    skipped_regions: list[OCRResult] = field(default_factory=list)
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


def _bboxes_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def _restore_skipped_regions(
    rendered_image: Image.Image,
    original_image: Image.Image,
    skipped_regions: list[OCRResult],
    translatable_regions: list[OCRResult],
) -> Image.Image:
    """Restore original source text pixels for any OCR region skipped during translation.

    Ensures that untranslated or low-confidence text is never left as a blank erased hole,
    preserving the original Japanese text and artwork from the source manga image.
    preserving the original source text and artwork from the input image.
    """
    if not skipped_regions or original_image is None:
        return rendered_image

    result = rendered_image.copy()
    w, h = result.size

    for region in skipped_regions:
        bbox = region.source_bbox or region.bbox
        l, t, r, b = bbox
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)
        if r <= l or b <= t:
            continue

        orig_patch = original_image.crop((l, t, r, b))

        has_overlap = any(
            _bboxes_overlap((l, t, r, b), tr.source_bbox or tr.bbox)
            for tr in translatable_regions
        )

        if not has_overlap:
            result.paste(orig_patch, (l, t))
        else:
            if isinstance(region.ocr_mask, np.ndarray) and region.ocr_mask.shape[:2] == (b - t, r - l):
                mask = Image.fromarray((region.ocr_mask.astype(np.uint8) * 255), mode="L")
                result.paste(orig_patch, (l, t), mask)
            else:
                crop_arr = np.array(orig_patch.convert("RGB"))
                gray = cv2.cvtColor(crop_arr, cv2.COLOR_RGB2GRAY)
                p75 = float(np.percentile(gray, 75))
                if p75 >= 128:
                    ink = gray <= min(180, p75 - 25)
                else:
                    p25 = float(np.percentile(gray, 25))
                    ink = gray >= max(100, p25 + 25)
                mask = Image.fromarray((ink.astype(np.uint8) * 255), mode="L")
                result.paste(orig_patch, (l, t), mask)

    return result


class MangaTranslationPipeline:
    def __init__(
        self,
        ocr: object,
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
        self._translator = translator
        self._inpainter = inpainter
        self._renderer = renderer
        self._anchor_detector = anchor_detector
        self._minimum_ocr_translation_confidence = minimum_ocr_translation_confidence
        self._ocr_weight_sentence = ocr_weight_sentence
        self._ocr_weight_mean = ocr_weight_mean
        self._ocr_weight_std = ocr_weight_std
        self._last_page_context: str = ""

    def reset_context(self) -> None:
        self._last_page_context = ""

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
        self,
        image: Image.Image,
        regions: list[OCRResult],
        cancel_event: Event | None = None,
        context: str | None = None,
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

        _check_cancelled(cancel_event)
        try:
            untranslated = [r for r in valid_regions if not r.translated_text]
            if untranslated:
                texts = [region.source_text for region in untranslated]
                translations = self._translator.translate(texts, context=context)
                for region, translation in zip(untranslated, translations, strict=True):
                    region.translated_text = format_response(translation)
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

    def process(
        self,
        image: Image.Image,
        cancel_event: Event | None = None,
        context: str | None = None,
    ) -> tuple[Image.Image, list[OCRResult]]:
        regions = self._run_ocr(image, cancel_event)
        ctx = context if context is not None else self._last_page_context
        translatable_regions, failed = self._run_translation(image, regions, cancel_event, context=ctx)
        if failed:
            return image, []
        page_translations = [r.translated_text for r in translatable_regions if r.translated_text]
        if page_translations:
            self._last_page_context = "\n".join(page_translations)
        _check_cancelled(cancel_event)
        inpainted_image = self._run_inpaint(image, translatable_regions, cancel_event)
        _check_cancelled(cancel_event)
        rendered_image = self._run_render(inpainted_image, translatable_regions, cancel_event)
        trans_ids = set(id(r) for r in translatable_regions)
        skipped_regions = [r for r in regions if id(r) not in trans_ids]
        final_image = _restore_skipped_regions(rendered_image, image, skipped_regions, translatable_regions)
        return final_image, translatable_regions

    def process_pipelined(
        self,
        items: list[Any],
        cancel_event: Event | None = None,
        queue_size: int = 2,
    ) -> Iterator[tuple[Any, Image.Image | None, Image.Image | None, list[OCRResult], PipelineError | None]]:
        if not items:
            return

        q_ocr_to_trans: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)
        q_ocr_to_inpaint: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)
        q_trans_done: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)
        q_inpaint_done: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)
        q_join_to_render: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)
        q_render_to_out: Queue[_PipelinePacket | None] = Queue(maxsize=queue_size)

        def ocr_worker() -> None:
            for idx, item in enumerate(items):
                if cancel_event is not None and cancel_event.is_set():
                    break
                packet = _PipelinePacket(index=idx, item=item)
                try:
                    img = item.load() if hasattr(item, "load") else item
                    packet.original_image = img
                    all_regions = self._run_ocr(img, cancel_event)
                    translatable = _translation_candidates(
                        all_regions,
                        minimum_confidence=self._minimum_ocr_translation_confidence,
                        weight_sentence=self._ocr_weight_sentence,
                        weight_mean=self._ocr_weight_mean,
                        weight_std=self._ocr_weight_std,
                    )
                    packet.regions = translatable
                    trans_ids = set(id(r) for r in translatable)
                    packet.skipped_regions = [r for r in all_regions if id(r) not in trans_ids]
                except PipelineError as err:
                    packet.error = err
                except Exception as err:
                    packet.error = PipelineError("OCR", err)

                inpaint_packet = _PipelinePacket(
                    index=packet.index,
                    item=packet.item,
                    original_image=packet.original_image,
                    regions=packet.regions,
                    skipped_regions=packet.skipped_regions,
                    error=packet.error,
                )
                q_ocr_to_trans.put(packet)
                q_ocr_to_inpaint.put(inpaint_packet)
            q_ocr_to_trans.put(None)
            q_ocr_to_inpaint.put(None)

        def trans_worker() -> None:
            prev_context: str = ""
            while True:
                packet = q_ocr_to_trans.get()
                if packet is None:
                    q_trans_done.put(None)
                    q_ocr_to_trans.task_done()
                    break
                if cancel_event is not None and cancel_event.is_set():
                    q_trans_done.put(None)
                    q_ocr_to_trans.task_done()
                    break
                if packet.error is None and packet.original_image is not None and packet.regions:
                    try:
                        untranslated = [r for r in packet.regions if not r.translated_text]
                        if untranslated:
                            texts = [region.source_text for region in untranslated]
                            translations = self._translator.translate(texts, context=prev_context)
                            for region, translation in zip(untranslated, translations, strict=True):
                                region.translated_text = format_response(translation)
                        page_translations = [r.translated_text for r in packet.regions if r.translated_text]
                        if page_translations:
                            prev_context = "\n".join(page_translations)
                    except Exception as err:
                        logger.warning("Translation failed; saving the original image instead: %s", err)
                        packet.skipped_regions = packet.skipped_regions + packet.regions
                        packet.regions = []
                        packet.translated_image = packet.original_image
                elif packet.error is None and not packet.regions and packet.original_image is not None:
                    packet.translated_image = packet.original_image

                q_trans_done.put(packet)
                q_ocr_to_trans.task_done()

        def inpaint_worker() -> None:
            while True:
                packet = q_ocr_to_inpaint.get()
                if packet is None:
                    q_inpaint_done.put(None)
                    q_ocr_to_inpaint.task_done()
                    break
                if cancel_event is not None and cancel_event.is_set():
                    q_inpaint_done.put(None)
                    q_ocr_to_inpaint.task_done()
                    break
                if packet.error is None and packet.original_image is not None and packet.regions:
                    try:
                        packet.inpainted_image = self._run_inpaint(packet.original_image.copy(), packet.regions, cancel_event)
                    except PipelineError as err:
                        packet.error = err
                    except Exception as err:
                        packet.error = PipelineError("inpainting", err)
                else:
                    packet.inpainted_image = packet.original_image
                q_inpaint_done.put(packet)
                q_ocr_to_inpaint.task_done()

        def join_worker() -> None:
            while True:
                p_trans = q_trans_done.get()
                p_inpaint = q_inpaint_done.get()
                if p_trans is None or p_inpaint is None:
                    q_join_to_render.put(None)
                    if p_trans is not None:
                        q_trans_done.task_done()
                    if p_inpaint is not None:
                        q_inpaint_done.task_done()
                    break
                if cancel_event is not None and cancel_event.is_set():
                    q_join_to_render.put(None)
                    q_trans_done.task_done()
                    q_inpaint_done.task_done()
                    break

                p_trans.inpainted_image = p_inpaint.inpainted_image
                if p_inpaint.error is not None and p_trans.error is None:
                    p_trans.error = p_inpaint.error

                q_join_to_render.put(p_trans)
                q_trans_done.task_done()
                q_inpaint_done.task_done()

        def render_worker() -> None:
            while True:
                packet = q_join_to_render.get()
                if packet is None:
                    q_render_to_out.put(None)
                    q_join_to_render.task_done()
                    break
                if cancel_event is not None and cancel_event.is_set():
                    q_render_to_out.put(None)
                    q_join_to_render.task_done()
                    break
                if packet.error is None and packet.inpainted_image is not None and packet.translated_image is None:
                    try:
                        rendered = self._run_render(packet.inpainted_image, packet.regions, cancel_event)
                        if packet.original_image is not None:
                            packet.translated_image = _restore_skipped_regions(
                                rendered, packet.original_image, packet.skipped_regions, packet.regions
                            )
                        else:
                            packet.translated_image = rendered
                    except PipelineError as err:
                        packet.error = err
                    except Exception as err:
                        packet.error = PipelineError("rendering", err)
                elif packet.translated_image is None:
                    packet.translated_image = packet.original_image
                q_render_to_out.put(packet)
                q_join_to_render.task_done()

        threads = [
            Thread(target=ocr_worker, daemon=True),
            Thread(target=trans_worker, daemon=True),
            Thread(target=inpaint_worker, daemon=True),
            Thread(target=join_worker, daemon=True),
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
            _drain_queue(q_ocr_to_inpaint)
            _drain_queue(q_trans_done)
            _drain_queue(q_inpaint_done)
            _drain_queue(q_join_to_render)
            _drain_queue(q_render_to_out)
            for t in threads:
                t.join(timeout=1.0)


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Processing cancelled because the viewer was closed.")


_PUNCTUATION_CHARS = frozenset("…‥・:：、。!?！？-〜ー—~\"'()（）[]「」『』〈〉《》.．")


def _is_punctuation(ch: str) -> bool:
    return ch in _PUNCTUATION_CHARS or unicodedata.category(ch).startswith("P")


def compute_ocr_quality_score(
    sentence_confidence: float | None,
    character_confidences: list[float] | None = None,
    source_text: str | None = None,
    weight_sentence: float = 0.5,
    weight_mean: float = 0.5,
    weight_std: float = 0.80,
    weight_low_conf: float = 0.25,
    low_conf_threshold: float = 0.70,
) -> float | None:
    """Compute combined OCR quality score.

    Score is based on:
        - sentence-level confidence
        - mean character confidence
        - proportion of low-confidence characters
        - standard deviation of character confidences

    A single low-confidence character should not strongly penalize the whole
    sentence. The low-confidence penalty increases as the proportion of
    unreliable characters increases.

    Punctuation marks are excluded from character-level metrics when
    source_text is available and lengths match.
    """
    if sentence_confidence is None:
        return None

    if not character_confidences:
        return float(sentence_confidence)

    eval_confidences = character_confidences

    # Exclude punctuation from character-level evaluation.
    if source_text is not None and len(source_text) == len(character_confidences):
        content_confs = [
            c
            for c, ch in zip(character_confidences, source_text)
            if not _is_punctuation(ch)
        ]
        if content_confs:
            eval_confidences = content_confs
        else:
            # If all characters are punctuation (e.g. "．．．．" or "！？"), evaluate by sentence confidence directly
            return float(sentence_confidence)

    if not eval_confidences:
        return float(sentence_confidence)

    mean_char = float(sum(eval_confidences) / len(eval_confidences))
    std_char = float((sum((c - mean_char) ** 2 for c in eval_confidences) / len(eval_confidences)) ** 0.5)

    # Penalize the proportion of characters with suspiciously low confidence.
    #
    # Example:
    #   [0.26, 0.95, 1.00, 1.00, 1.00]
    #   low_conf_ratio = 0.2
    #
    #   [0.26, 0.31, 0.42, 0.45, 0.48]
    #   low_conf_ratio = 1.0
    low_conf_ratio = sum(
        c < low_conf_threshold for c in eval_confidences
    ) / len(eval_confidences)

    score = (
        weight_sentence * sentence_confidence
        + weight_mean * mean_char
        - weight_low_conf * low_conf_ratio
        - weight_std * std_char * 0.1
    )

    return float(score)


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

    from src.renderer.pillow_renderer import compute_non_overlapping_render_bounds
    render_bounds_list = compute_non_overlapping_render_bounds(regions)

    # 1. Accumulate semi-transparent mask overlays
    for region, bounds in zip(regions, render_bounds_list):
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
    for region, bounds in zip(regions, render_bounds_list):
        # Blue: YOLO text detection box
        draw.rectangle(region.bbox, outline="blue", width=3)

        # Red: Segmentation bubble contour
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

        # Green: Render region outline & bounding box
        rl, rt, rr, rb = bounds
        if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
            layout_left, layout_top, _, _ = region.layout_bbox
            ml = max(0, rl - layout_left)
            mt = max(0, rt - layout_top)
            mr = min(region.layout_mask.shape[1], rr - layout_left)
            mb = min(region.layout_mask.shape[0], rb - layout_top)
            if mr > ml and mb > mt:
                sub_mask = np.zeros_like(region.layout_mask, dtype=np.uint8)
                sub_mask[mt:mb, ml:mr] = region.layout_mask[mt:mb, ml:mr].astype(np.uint8)
                r_contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in r_contours:
                    points = [(layout_left + int(point[0][0]), layout_top + int(point[0][1])) for point in contour]
                    if len(points) > 1:
                        draw.line(points + [points[0]], fill="#00e600", width=3)
        draw.rectangle((rl, rt, rr, rb), outline="#00e600", width=2)

    return debug_image


def save_debug_image(image: Image.Image, regions: list[OCRResult], output_path: Path) -> None:
    debug_image = draw_debug_image(image, regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_image.save(output_path)