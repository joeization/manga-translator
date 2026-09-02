from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion

from .base import Inpainter
from .bubble import OpenCVContourBubbleSegmenter


class OpenCVInpainter(Inpainter):
    def __init__(
        self,
        dark_threshold: int,
        white_threshold: int,
        white_ratio: float,
        inpaint_radius: int,
        mask_dilation: int,
        ocr_clear_padding: int,
        bubble_padding: int,
        bubble_close_kernel: int,
        bubble_clear_mode: str,
        bubble_min_overlap: float,
        bubble_border_width: int,
    ) -> None:
        self._dark_threshold = dark_threshold
        self._white_threshold = white_threshold
        self._white_ratio = white_ratio
        self._inpaint_radius = inpaint_radius
        self._mask_dilation = mask_dilation
        self._ocr_clear_padding = ocr_clear_padding
        self._bubble_padding = bubble_padding
        self._bubble_close_kernel = bubble_close_kernel
        self._bubble_clear_mode = bubble_clear_mode
        self._bubble_min_overlap = bubble_min_overlap
        self._bubble_border_width = bubble_border_width
        self._bubble_segmenter = OpenCVContourBubbleSegmenter(white_threshold, bubble_padding)

    def inpaint(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        pixels = np.array(image.convert("RGB"))
        for region in regions:
            self._inpaint_region(pixels, region)
        return Image.fromarray(pixels, mode="RGB")

    def _inpaint_region(self, pixels: np.ndarray, region: TextRegion) -> None:
        padded_region = replace(region, bbox=_expand_bbox(region.bbox, pixels.shape[1], pixels.shape[0], self._ocr_clear_padding))
        bubble = _existing_bubble(region) or self._bubble_segmenter.segment(Image.fromarray(pixels), padded_region) or self._find_bubble(pixels, padded_region)
        if bubble is not None:
            bubble_bbox, bubble_mask = bubble
            region.layout_bbox = bubble_bbox
            region.layout_mask = bubble_mask
            inpaint_bbox = _clip_bbox(bubble_bbox, pixels.shape[1], pixels.shape[0])
        else:
            inpaint_bbox = padded_region.bbox

        left, top, right, bottom = inpaint_bbox
        roi = pixels[top:bottom, left:right]
        if roi.size == 0:
            return
        bubble_interior = (
            _bubble_mask_in_region(inpaint_bbox, bubble_bbox, bubble_mask)
            if bubble is not None
            else np.ones(roi.shape[:2], dtype=bool)
        )

        final_mask = _foreground_mask(roi, bubble_interior, region.bbox, inpaint_bbox, self._dark_threshold, self._mask_dilation)
        region.inpaint_bbox = inpaint_bbox
        region.inpaint_mask = final_mask
        if np.any(final_mask):
            pixels[top:bottom, left:right] = cv2.inpaint(roi, final_mask.astype(np.uint8) * 255, self._inpaint_radius, cv2.INPAINT_TELEA)

    def _find_bubble(self, pixels: np.ndarray, region: TextRegion) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        left, top, right, bottom = region.bbox
        image_height, image_width = pixels.shape[:2]
        outer_left = max(0, left - self._bubble_padding)
        outer_top = max(0, top - self._bubble_padding)
        outer_right = min(image_width, right + self._bubble_padding)
        outer_bottom = min(image_height, bottom + self._bubble_padding)
        context = pixels[outer_top:outer_bottom, outer_left:outer_right]
        context_gray = cv2.cvtColor(context, cv2.COLOR_RGB2GRAY)
        white = np.where(context_gray >= self._white_threshold, 255, 0).astype(np.uint8)
        kernel = np.ones((self._bubble_close_kernel, self._bubble_close_kernel), dtype=np.uint8)
        closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

        text_left, text_top = left - outer_left, top - outer_top
        text_right, text_bottom = right - outer_left, bottom - outer_top
        text_area = (right - left) * (bottom - top)
        best_label = 0
        best_overlap = 0
        for label in range(1, component_count):
            component_left, component_top, component_width, component_height, _ = stats[label]
            touches_context_edge = (
                component_left == 0
                or component_top == 0
                or component_left + component_width == context.shape[1]
                or component_top + component_height == context.shape[0]
            )
            component_area = component_width * component_height
            if touches_context_edge and component_area >= context.shape[0] * context.shape[1] * 0.9:
                continue
            overlap = int(np.count_nonzero(labels[text_top:text_bottom, text_left:text_right] == label))
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap

        if best_label == 0 or best_overlap < text_area * self._bubble_min_overlap:
            return None
        component = np.where(labels == best_label, 255, 0).astype(np.uint8)
        filled_component = _fill_enclosed_holes(component)
        return (outer_left, outer_top, outer_right, outer_bottom), filled_component.astype(bool)

def _fill_enclosed_holes(component: np.ndarray) -> np.ndarray:
    outside = component.copy()
    flood_mask = np.zeros((component.shape[0] + 2, component.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(outside, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(outside)
    return cv2.bitwise_or(component, holes)


def _existing_bubble(region: TextRegion) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
        return region.layout_bbox, region.layout_mask
    return None


def _expand_bbox(bbox: tuple[int, int, int, int], image_width: int, image_height: int, padding: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(image_width, right + padding),
        min(image_height, bottom + padding),
    )


def _clip_bbox(bbox: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return max(0, left), max(0, top), min(image_width, right), min(image_height, bottom)


def _foreground_mask(
    roi: np.ndarray,
    allowed_mask: np.ndarray,
    detected_bbox: tuple[int, int, int, int],
    roi_bbox: tuple[int, int, int, int],
    dark_threshold: int,
    dilation: int,
) -> np.ndarray:
    background_pixels = _local_background_pixels(roi, allowed_mask, detected_bbox, roi_bbox)
    background = np.median(background_pixels, axis=0)
    color_distance = np.linalg.norm(roi.astype(np.float32) - background, axis=2)
    reference_distance = np.linalg.norm(background_pixels.astype(np.float32) - background, axis=1)
    contrast_threshold = max(20.0, float(np.median(reference_distance) + 3 * np.median(np.abs(reference_distance - np.median(reference_distance)))))
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    mask = np.logical_and(allowed_mask, np.logical_or(color_distance >= contrast_threshold, gray <= dark_threshold))
    if dilation:
        kernel_size = dilation * 2 + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1).astype(bool)
        mask = np.logical_and(mask, allowed_mask)
    return mask


def _local_background_pixels(
    roi: np.ndarray, allowed_mask: np.ndarray, detected_bbox: tuple[int, int, int, int], roi_bbox: tuple[int, int, int, int]
) -> np.ndarray:
    detected_mask = np.zeros(allowed_mask.shape, dtype=bool)
    left = max(0, detected_bbox[0] - roi_bbox[0])
    top = max(0, detected_bbox[1] - roi_bbox[1])
    right = min(roi.shape[1], detected_bbox[2] - roi_bbox[0])
    bottom = min(roi.shape[0], detected_bbox[3] - roi_bbox[1])
    detected_mask[top:bottom, left:right] = True
    background_pixels = roi[np.logical_and(allowed_mask, ~detected_mask)]
    if len(background_pixels) < 16:
        background_pixels = roi[allowed_mask]
    return background_pixels if len(background_pixels) else roi.reshape(-1, 3)


def _bubble_mask_in_region(
    region_bbox: tuple[int, int, int, int], bubble_bbox: tuple[int, int, int, int], bubble_mask: np.ndarray
) -> np.ndarray:
    left, top, right, bottom = region_bbox
    bubble_left, bubble_top, bubble_right, bubble_bottom = bubble_bbox
    result = np.zeros((bottom - top, right - left), dtype=bool)
    overlap_left, overlap_top = max(left, bubble_left), max(top, bubble_top)
    overlap_right, overlap_bottom = min(right, bubble_right), min(bottom, bubble_bottom)
    if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
        return result
    source_top, source_left = overlap_top - bubble_top, overlap_left - bubble_left
    source_bottom, source_right = overlap_bottom - bubble_top, overlap_right - bubble_left
    target_top, target_left = overlap_top - top, overlap_left - left
    target_bottom, target_right = overlap_bottom - top, overlap_right - left
    source = bubble_mask[source_top:source_bottom, source_left:source_right]
    height, width = min(source.shape[0], target_bottom - target_top), min(source.shape[1], target_right - target_left)
    result[target_top : target_top + height, target_left : target_left + width] = source[:height, :width]
    return result