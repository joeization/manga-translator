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
        expanded_region = replace(region, bbox=_expand_bbox(region.bbox, pixels.shape[1], pixels.shape[0], self._ocr_clear_padding))
        left, top, right, bottom = expanded_region.bbox
        roi = pixels[top:bottom, left:right]
        if roi.size == 0:
            return

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        text_mask = np.where(gray <= self._dark_threshold, 255, 0).astype(np.uint8)
        if self._mask_dilation:
            kernel_size = self._mask_dilation * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        if not np.any(text_mask):
            return

        bubble = _existing_bubble(expanded_region) or self._bubble_segmenter.segment(Image.fromarray(pixels), expanded_region) or self._find_bubble(pixels, expanded_region)
        if bubble is not None:
            bubble_bbox, bubble_mask = bubble
            expanded_region.layout_bbox = bubble_bbox
            expanded_region.layout_mask = bubble_mask
            bubble_interior = _bubble_mask_in_region(expanded_region.bbox, bubble_bbox, bubble_mask)
            if self._bubble_clear_mode == "interior" and np.any(bubble_interior):
                roi[_inset_bubble_mask(bubble_interior, self._bubble_border_width)] = 255
                return
            bubble_text_mask = np.where(bubble_interior, text_mask, 0).astype(np.uint8)
            if np.any(bubble_text_mask):
                roi[bubble_text_mask > 0] = 255
                return

        self._clear_bbox_edge(text_mask)
        pixels[top:bottom, left:right] = cv2.inpaint(roi, text_mask, self._inpaint_radius, cv2.INPAINT_TELEA)

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

    @staticmethod
    def _clear_bbox_edge(mask: np.ndarray) -> None:
        if min(mask.shape) > 2:
            mask[0, :] = 0
            mask[-1, :] = 0
            mask[:, 0] = 0
            mask[:, -1] = 0


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


def _inset_bubble_mask(mask: np.ndarray, border_width: int) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    if border_width <= 0:
        return mask.astype(bool)
    kernel_size = border_width * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def _expand_bbox(bbox: tuple[int, int, int, int], image_width: int, image_height: int, padding: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(image_width, right + padding),
        min(image_height, bottom + padding),
    )


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