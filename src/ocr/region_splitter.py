from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion


def split_text_regions(image: Image.Image, region: TextRegion) -> list[TextRegion]:
    """Split only clearly separated ink groups; retain the original region when uncertain."""
    if not isinstance(region.layout_mask, np.ndarray) or region.layout_bbox is None:
        return [region]

    left, top, right, bottom = region.bbox
    if right <= left or bottom <= top:
        return [region]
    bubble_mask = _mask_in_region(region)
    pixels = np.asarray(image.convert("RGB"))[top:bottom, left:right]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = np.where(bubble_mask, ink, 0).astype(np.uint8)
    components = _text_components(ink)
    groups = _group_components(components, ink)
    if len(groups) < 2:
        return [region]

    candidates = [
        replace(
            region,
            bbox=(left + group_left, top + group_top, left + group_right, top + group_bottom),
            source_bbox=(left + group_left, top + group_top, left + group_right, top + group_bottom),
            ocr_mask=ink[group_top:group_bottom, group_left:group_right].astype(bool),
        )
        for group_left, group_top, group_right, group_bottom in groups
    ]
    return _deduplicate_crop_candidates(_merge_overlapping_crop_candidates(candidates))


def crop_for_ocr(image: Image.Image, region: TextRegion) -> Image.Image:
    crop = image.crop(region.source_bbox or region.bbox).convert("RGB")
    if not isinstance(region.ocr_mask, np.ndarray) or region.ocr_mask.shape != (crop.height, crop.width):
        return crop
    pixels = np.asarray(crop).copy()
    pixels[~region.ocr_mask] = 255
    return Image.fromarray(pixels, mode="RGB")


def sort_manga_reading_order(regions: list[TextRegion]) -> list[TextRegion]:
    """Sort regions in Japanese manga reading order: Top-to-Bottom, Right-to-Left."""
    if not regions:
        return []
    sorted_by_top = sorted(regions, key=lambda region: region.bbox[1])
    rows: list[list[TextRegion]] = []
    for region in sorted_by_top:
        r_left, r_top, r_right, r_bottom = region.bbox
        r_height = max(1, r_bottom - r_top)
        placed = False
        for row in rows:
            row_top = min(member.bbox[1] for member in row)
            row_bottom = max(member.bbox[3] for member in row)
            row_height = max(1, row_bottom - row_top)
            overlap_top = max(r_top, row_top)
            overlap_bottom = min(r_bottom, row_bottom)
            overlap = max(0, overlap_bottom - overlap_top)
            if overlap >= min(r_height, row_height) * 0.4:
                row.append(region)
                placed = True
                break
        if not placed:
            rows.append([region])
    result: list[TextRegion] = []
    for row in rows:
        result.extend(sorted(row, key=lambda region: (-region.bbox[2], -region.bbox[0])))
    return result


def _deduplicate_crop_candidates(candidates: list[TextRegion]) -> list[TextRegion]:
    """Suppress duplicate crop geometry using actual source-ink overlap when available."""
    kept: list[TextRegion] = []
    for candidate in sorted(candidates, key=_valid_ink_area, reverse=True):
        if any(_is_redundant_crop(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return sort_manga_reading_order(kept)


def _merge_overlapping_crop_candidates(candidates: list[TextRegion]) -> list[TextRegion]:
    """Merge split crops when their bounding boxes or ink masks overlap significantly."""
    remaining = list(candidates)
    merged: list[TextRegion] = []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in remaining[:]:
                if any(_should_merge_crop_candidates(candidate, member) for member in group):
                    remaining.remove(candidate)
                    group.append(candidate)
                    changed = True
        merged.append(_merge_crop_group(group))
    return merged


def _should_merge_crop_candidates(first: TextRegion, second: TextRegion) -> bool:
    if _bbox_overlap_ratio(first.bbox, second.bbox) >= 0.5:
        return True
    if _bbox_iou(first.bbox, second.bbox) >= 0.4:
        return True
    overlap = _ink_overlap(first, second)
    if overlap is not None:
        shared, first_ink, second_ink = overlap
        min_ink = min(first_ink, second_ink)
        if min_ink > 0 and shared / min_ink >= 0.5:
            return True
    return False


def _bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _merge_crop_group(candidates: list[TextRegion]) -> TextRegion:
    if len(candidates) == 1:
        return candidates[0]
    left = min(candidate.bbox[0] for candidate in candidates)
    top = min(candidate.bbox[1] for candidate in candidates)
    right = max(candidate.bbox[2] for candidate in candidates)
    bottom = max(candidate.bbox[3] for candidate in candidates)
    merged_bbox = (left, top, right, bottom)
    merged_mask = np.zeros((bottom - top, right - left), dtype=bool)
    for candidate in candidates:
        if not isinstance(candidate.ocr_mask, np.ndarray):
            continue
        candidate_left, candidate_top, candidate_right, candidate_bottom = candidate.bbox
        if candidate.ocr_mask.shape != (candidate_bottom - candidate_top, candidate_right - candidate_left):
            continue
        merged_mask[
            candidate_top - top : candidate_bottom - top,
            candidate_left - left : candidate_right - left,
        ] |= candidate.ocr_mask
    template = max(candidates, key=_valid_ink_area)
    return replace(template, bbox=merged_bbox, source_bbox=merged_bbox, ocr_mask=merged_mask)


def _is_redundant_crop(candidate: TextRegion, existing: TextRegion) -> bool:
    if candidate.bbox == existing.bbox:
        return True
    overlap = _ink_overlap(candidate, existing)
    if overlap is not None:
        shared, candidate_area, existing_area = overlap
        if candidate_area and shared / candidate_area >= 0.9:
            return True
        return min(candidate_area, existing_area) > 0 and shared / min(candidate_area, existing_area) >= 0.8
    return _bbox_overlap_ratio(candidate.bbox, existing.bbox) >= 0.9


def _valid_ink_area(region: TextRegion) -> int:
    return int(np.count_nonzero(region.ocr_mask)) if isinstance(region.ocr_mask, np.ndarray) else 0


def _ink_overlap(first: TextRegion, second: TextRegion) -> tuple[int, int, int] | None:
    if not isinstance(first.ocr_mask, np.ndarray) or not isinstance(second.ocr_mask, np.ndarray):
        return None
    first_left, first_top, first_right, first_bottom = first.bbox
    second_left, second_top, second_right, second_bottom = second.bbox
    if first.ocr_mask.shape != (first_bottom - first_top, first_right - first_left):
        return None
    if second.ocr_mask.shape != (second_bottom - second_top, second_right - second_left):
        return None
    left, top = max(first_left, second_left), max(first_top, second_top)
    right, bottom = min(first_right, second_right), min(first_bottom, second_bottom)
    shared = 0
    if left < right and top < bottom:
        first_mask = first.ocr_mask[top - first_top : bottom - first_top, left - first_left : right - first_left]
        second_mask = second.ocr_mask[top - second_top : bottom - second_top, left - second_left : right - second_left]
        shared = int(np.count_nonzero(np.logical_and(first_mask, second_mask)))
    return shared, _valid_ink_area(first), _valid_ink_area(second)


def _bbox_overlap_ratio(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / min(first_area, second_area) if first_area and second_area else 0.0


def _mask_in_region(region: TextRegion) -> np.ndarray:
    assert isinstance(region.layout_mask, np.ndarray)
    assert region.layout_bbox is not None
    left, top, right, bottom = region.bbox
    mask_left, mask_top, mask_right, mask_bottom = region.layout_bbox
    mask = np.zeros((bottom - top, right - left), dtype=bool)
    overlap_left, overlap_top = max(left, mask_left), max(top, mask_top)
    overlap_right, overlap_bottom = min(right, mask_right), min(bottom, mask_bottom)
    if overlap_left >= overlap_right or overlap_top >= overlap_bottom:
        return mask
    source = region.layout_mask[
        overlap_top - mask_top : overlap_bottom - mask_top,
        overlap_left - mask_left : overlap_right - mask_left,
    ]
    mask[
        overlap_top - top : overlap_bottom - top,
        overlap_left - left : overlap_right - left,
    ] = source
    return mask


def _text_components(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    minimum_area = max(4, ink.size // 50_000)
    maximum_width, maximum_height = ink.shape[1] * 0.8, ink.shape[0] * 0.8
    return [
        (left, top, left + width, top + height)
        for left, top, width, height, area in stats[1:count]
        if area >= minimum_area and width < maximum_width and height < maximum_height
    ]


def _group_components(components: list[tuple[int, int, int, int]], ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    if len(components) < 2:
        return []
    widths = [right - left for left, _, right, _ in components]
    heights = [bottom - top for _, top, _, bottom in components]
    # Link ordinary loose glyph spacing and punctuation before looking for a
    # substantially wider blank band that indicates a separate text block.
    kernel_width = max(1, round(float(np.median(widths)) * 1.3))
    kernel_height = max(1, round(float(np.median(heights)) * 1.3))
    component_mask = np.zeros(ink.shape, dtype=np.uint8)
    for left, top, right, bottom in components:
        component_mask[top:bottom, left:right] = 255
    grouped = cv2.dilate(component_mask, np.ones((kernel_height, kernel_width), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped)
    groups: list[tuple[int, int, int, int]] = []
    for label, (left, top, width, height, area) in enumerate(stats[1:count], start=1):
        if area <= kernel_width * kernel_height:
            continue
        group_ink = np.where(labels == label, ink, 0).astype(np.uint8)
        trimmed = _trim_ink_bounds(group_ink, 0, 0)
        if not trimmed:
            continue
        group = trimmed[0]
        groups.extend(_split_group_at_blank_band(group, group_ink, widths, heights))
    return _sort_groups_manga_reading_order(groups)


def _sort_groups_manga_reading_order(groups: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not groups:
        return []
    sorted_by_top = sorted(groups, key=lambda g: g[1])
    rows: list[list[tuple[int, int, int, int]]] = []
    for g in sorted_by_top:
        g_left, g_top, g_right, g_bottom = g
        g_height = max(1, g_bottom - g_top)
        placed = False
        for row in rows:
            row_top = min(m[1] for m in row)
            row_bottom = max(m[3] for m in row)
            row_height = max(1, row_bottom - row_top)
            overlap = max(0, min(g_bottom, row_bottom) - max(g_top, row_top))
            if overlap >= min(g_height, row_height) * 0.4:
                row.append(g)
                placed = True
                break
        if not placed:
            rows.append([g])
    result: list[tuple[int, int, int, int]] = []
    for row in rows:
        result.extend(sorted(row, key=lambda g: (-g[2], -g[0])))
    return result


def _split_group_at_blank_band(
    group: tuple[int, int, int, int], ink: np.ndarray, widths: list[int], heights: list[int]
) -> list[tuple[int, int, int, int]]:
    left, top, right, bottom = group
    cropped_ink = ink[top:bottom, left:right] > 0
    candidates = [
        (width / max(1, float(np.median(widths))), "vertical", start, end)
        for start, end, width in _interior_blank_bands(cropped_ink.any(axis=0))
    ] + [
        (height / max(1, float(np.median(heights))), "horizontal", start, end)
        for start, end, height in _interior_blank_bands(cropped_ink.any(axis=1))
    ]
    if not candidates:
        return [group]
    blank_band_ratio, direction, start, end = max(candidates)
    if blank_band_ratio < 1.3:
        return [group]
    if direction == "vertical":
        parts = _trim_ink_bounds(cropped_ink[:, :start], left, top) + _trim_ink_bounds(cropped_ink[:, end:], left + end, top)
    else:
        parts = _trim_ink_bounds(cropped_ink[:start, :], left, top) + _trim_ink_bounds(cropped_ink[end:, :], left, top + end)

    sub_groups: list[tuple[int, int, int, int]] = []
    for part in parts:
        part_ink = ink[part[1] : part[3], part[0] : part[2]]
        sub_groups.extend(_split_group_at_blank_band(part, part_ink, widths, heights))
    return sub_groups


def _interior_blank_bands(projection: np.ndarray) -> list[tuple[int, int, int]]:
    bands: list[tuple[int, int, int]] = []
    start: int | None = None
    for index, occupied in enumerate(projection):
        if not occupied and start is None:
            start = index
        elif occupied and start is not None:
            if start > 0 and index < len(projection):
                bands.append((start, index, index - start))
            start = None
    return bands


def _trim_ink_bounds(ink: np.ndarray, offset_x: int, offset_y: int) -> list[tuple[int, int, int, int]]:
    points = cv2.findNonZero(ink.astype(np.uint8))
    if points is None:
        return []
    left, top, width, height = cv2.boundingRect(points)
    return [(offset_x + left, offset_y + top, offset_x + left + width, offset_y + top + height)]