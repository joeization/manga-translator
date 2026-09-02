from __future__ import annotations

from dataclasses import replace
import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion


def crop_for_ocr(image: Image.Image, region: TextRegion) -> Image.Image:
    crop = image.crop(region.source_bbox or region.bbox).convert("RGB")
    if not isinstance(region.ocr_mask, np.ndarray) or region.ocr_mask.shape != (crop.height, crop.width):
        return crop

    cropped_pixels = np.array(crop)
    background = np.full_like(cropped_pixels, fill_value=255)
    return Image.fromarray(np.where(region.ocr_mask[:, :, None], cropped_pixels, background))


def _mask_in_region(region: TextRegion) -> np.ndarray:
    left, top, right, bottom = region.bbox
    h, w = bottom - top, right - left
    if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
        mask_left, mask_top, mask_right, mask_bottom = region.layout_bbox
        mask = np.zeros((h, w), dtype=bool)
        overlap_left, overlap_top = max(left, mask_left), max(top, mask_top)
        overlap_right, overlap_bottom = min(right, mask_right), min(bottom, mask_bottom)
        if overlap_left < overlap_right and overlap_top < overlap_bottom:
            source = region.layout_mask[
                overlap_top - mask_top : overlap_bottom - mask_top,
                overlap_left - mask_left : overlap_right - mask_left,
            ]
            mask[
                overlap_top - top : overlap_bottom - top,
                overlap_left - left : overlap_right - left,
            ] = source
            return mask
    return np.ones((h, w), dtype=bool)


def _find_blank_bands(projection: np.ndarray, min_gap_length: int, low_threshold: int = 2) -> list[tuple[int, int]]:
    """Find contiguous interior blank bands in a 1D projection profile."""
    bands: list[tuple[int, int]] = []
    start: int | None = None
    n = len(projection)
    for idx, count in enumerate(projection):
        is_blank = (count <= low_threshold)
        if is_blank and start is None:
            start = idx
        elif not is_blank and start is not None:
            if start > 0 and idx < n and (idx - start) >= min_gap_length:
                bands.append((start, idx))
            start = None
    return bands


def split_text_regions(image: Image.Image, region: TextRegion, image_array: np.ndarray | None = None) -> list[TextRegion]:
    """Projection-profile based region splitter.
    Splits ONLY when clear large interior blank gaps exist between distinct speech bubbles or paragraphs.
    """
    left, top, right, bottom = region.bbox
    h_region, w_region = bottom - top, right - left
    if w_region <= 0 or h_region <= 0:
        return [region]

    bubble_mask = _mask_in_region(region)
    if image_array is None:
        image_array = np.asarray(image.convert("RGB"))
    pixels = image_array[top:bottom, left:right]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = np.where(bubble_mask, ink, 0).astype(np.uint8)

    # Estimate median glyph dimensions via connected components
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    components = [(l, t, l + w, t + h) for l, t, w, h, area in stats[1:count] if area >= 4 and w >= 3 and h >= 3]
    if len(components) < 2:
        return [region]

    widths = [r - l for l, t, r, b in components if (r - l) >= 6 and (b - t) >= 6]
    heights = [b - t for l, t, r, b in components if (r - l) >= 6 and (b - t) >= 6]
    median_w = float(np.median(widths)) if widths else 30.0
    median_h = float(np.median(heights)) if heights else 30.0

    # Single vertical speech bubble protection:
    # Speech bubbles with w <= 150px (or w <= 2.5 * median_w) contain vertical lines of text side-by-side.
    # Splitting along Y chops staggered vertical lines apart; thus Y-splitting is bypassed.
    if w_region <= max(150, round(median_w * 2.5)):
        return [region]

    min_y_gap = max(30, round(median_h * 1.0))
    min_x_gap = max(30, round(median_w * 1.8))

    low_thresh_y = max(5, round(w_region * 0.05))
    low_thresh_x = max(5, round(h_region * 0.05))

    row_sums = (ink > 0).sum(axis=1)
    col_sums = (ink > 0).sum(axis=0)

    y_blank_bands = _find_blank_bands(row_sums, min_gap_length=min_y_gap, low_threshold=low_thresh_y)
    x_blank_bands = _find_blank_bands(col_sums, min_gap_length=min_x_gap, low_threshold=low_thresh_x)

    # Filter Y blank bands: only split if all sub-regions are large enough (>= 35px or 1.6 * median_h)
    min_sub_h = max(35, round(median_h * 1.6))
    valid_y_bands = []
    if y_blank_bands:
        y_pts = [(s + e) // 2 for s, e in y_blank_bands]
        bounds = [0] + y_pts + [h_region]
        all_valid = True
        for i in range(len(bounds) - 1):
            if (bounds[i + 1] - bounds[i]) < min_sub_h:
                all_valid = False
                break
        if all_valid:
            valid_y_bands = y_blank_bands

    if valid_y_bands:
        y_split_points = [top + (s + e) // 2 for s, e in valid_y_bands]
        y_bounds = [top] + y_split_points + [bottom]
        sub_regions = []
        for i in range(len(y_bounds) - 1):
            sub_box = (left, y_bounds[i], right, y_bounds[i + 1])
            sub_mask = ink[y_bounds[i] - top : y_bounds[i + 1] - top, :].astype(bool)
            sub_regions.append(replace(region, bbox=sub_box, source_bbox=sub_box, ocr_mask=sub_mask))
        return sub_regions

    elif x_blank_bands:
        x_split_points = [left + (s + e) // 2 for s, e in x_blank_bands]
        x_bounds = [left] + x_split_points + [right]
        sub_regions = []
        for i in range(len(x_bounds) - 1):
            sub_box = (x_bounds[i], top, x_bounds[i + 1], bottom)
            sub_mask = ink[:, x_bounds[i] - left : x_bounds[i + 1] - left].astype(bool)
            sub_regions.append(replace(region, bbox=sub_box, source_bbox=sub_box, ocr_mask=sub_mask))
        return sub_regions

    return [region]


def merge_contained_regions(regions: list[TextRegion], *args, **kwargs) -> list[TextRegion]:
    """Filter out smaller regions fully contained inside larger regions."""
    if len(regions) <= 1:
        return regions
    res: list[TextRegion] = []
    for r in regions:
        b1 = r.bbox
        is_inside = False
        for other in regions:
            b2 = other.bbox
            if b1 != b2 and b1[0] >= b2[0] and b1[1] >= b2[1] and b1[2] <= b2[2] and b1[3] <= b2[3]:
                is_inside = True
                break
        if not is_inside:
            res.append(r)
    return res


def resolve_overlapping_regions(regions: list[TextRegion], *args, **kwargs) -> list[TextRegion]:
    """Resolve contained regions."""
    return merge_contained_regions(regions, *args, **kwargs)