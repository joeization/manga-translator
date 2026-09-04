from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.models import TextRegion

from .region_splitter import resolve_overlapping_regions

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class BubblePostprocessConfig:
    yolo_nms_iou: float = 0.5
    segmentation_duplicate_iou: float = 0.6
    source_match_iou: float = 0.1
    long_strip_aspect_ratio: float = 3.0
    long_strip_cross_axis_overlap: float = 0.9
    long_strip_gap_ratio: float = 0.0
    mask_erosion_kernel_size: int = 3
    mask_erosion_iterations: int = 1


DEFAULT_CONFIG = BubblePostprocessConfig()


def postprocess_segmentation_mask(
    mask: np.ndarray,
    layout_bbox: tuple[int, int, int, int],
    detection_box: tuple[int, int, int, int],
    kernel_size: int = 3,
    iterations: int = 1,
    protect_margin: int = 4,
) -> np.ndarray:
    """Post-process segmentation mask by eroding outside the detection box while preserving the inside exactly as-is.

    The resulting behavior is:
        final_mask = eroded_mask outside (box + protect_margin)
                     original_mask inside (box + protect_margin)
    """
    if not isinstance(mask, np.ndarray) or not np.any(mask) or kernel_size <= 1 or iterations <= 0:
        return mask

    # 1. Save original segmentation mask
    original_mask = mask.astype(bool)

    # 2. Apply a gentle morphological cross erosion to the entire mask once
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    eroded = cv2.erode(original_mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    eroded_mask = eroded if np.any(eroded) else original_mask

    # 3. Restore original mask pixels inside the detection box plus outward margin
    # This ensures original text outward strokes / outlines (原字的外擴) are never eroded.
    sl, st, sr, sb = layout_bbox
    dl, dt, dr, db = detection_box

    ol, ot = max(sl, dl - protect_margin), max(st, dt - protect_margin)
    or_, ob = min(sr, dr + protect_margin), min(sb, db + protect_margin)

    # 4. Use the eroded result only outside the protected zone
    final_mask = eroded_mask.copy()
    if or_ > ol and ob > ot:
        box_l, box_t = ol - sl, ot - st
        box_r, box_b = or_ - sl, ob - st
        final_mask[box_t:box_b, box_l:box_r] = original_mask[box_t:box_b, box_l:box_r]

    return final_mask


def postprocess_bubbles(
    yolo_candidates: list[TextRegion], segmentation_candidates: list[TextRegion], config: BubblePostprocessConfig = DEFAULT_CONFIG
) -> list[TextRegion]:
    yolo_bubbles = _non_maximum_suppression(yolo_candidates, config.yolo_nms_iou)
    segmentation_bubbles = _merge_mask_candidates(segmentation_candidates, config)
    results: list[TextRegion] = []
    matched_segmentation: set[int] = set()
    visited_yolo: set[int] = set()
    matches_by_yolo = [
        _best_matching_segmentation(yolo.bbox, segmentation_bubbles, config.source_match_iou) for yolo in yolo_bubbles
    ]

    for yolo_index, yolo in enumerate(yolo_bubbles):
        if yolo_index in visited_yolo:
            continue
        segmentation_index = matches_by_yolo[yolo_index]
        if segmentation_index is None:
            results.append(TextRegion(bbox=yolo.bbox, source_text="", detection_confidence=yolo.detection_confidence))
            continue
        yolo_indices = _long_strip_component(yolo_index, yolo_bubbles, matches_by_yolo, config)
        visited_yolo.update(yolo_indices)
        matched_segmentation.add(segmentation_index)
        segmentation = segmentation_bubbles[segmentation_index]
        yolo = _merge_bboxes([yolo_bubbles[index] for index in yolo_indices])

        layout_mask = segmentation.layout_mask
        if (
            isinstance(segmentation.layout_mask, np.ndarray)
            and segmentation.layout_bbox is not None
            and config.mask_erosion_kernel_size > 1
            and config.mask_erosion_iterations > 0
        ):
            layout_mask = postprocess_segmentation_mask(
                segmentation.layout_mask,
                segmentation.layout_bbox,
                yolo.bbox,
                kernel_size=config.mask_erosion_kernel_size,
                iterations=config.mask_erosion_iterations,
            )

        results.append(
            TextRegion(
                bbox=yolo.bbox,
                source_text="",
                detection_confidence=max(yolo.detection_confidence, segmentation.detection_confidence),
                source_bbox=yolo.bbox,
                layout_bbox=segmentation.layout_bbox,
                layout_mask=layout_mask,
            )
        )

    results.extend(segmentation for index, segmentation in enumerate(segmentation_bubbles) if index not in matched_segmentation)
    results = resolve_overlapping_regions(results, threshold=0.35)
    return _suppress_furigana(results)


def _suppress_furigana(regions: list[TextRegion]) -> list[TextRegion]:
    """Suppress furigana / ruby text annotations.

    A region is classified as furigana if:
      - Its height is less than 40% of a vertically-adjacent neighbour's height, AND
      - It horizontally overlaps that neighbour by at least 50% of its own width, AND
      - Its vertical gap to the neighbour is small (< 0.5× its own height).

    Ruby text / furigana annotations (small phonetic guides printed adjacent to base ideographs)
    are often detected as separate YOLO boxes that would OCR as noise.
    """
    if len(regions) < 2:
        return regions

    furigana_indices: set[int] = set()
    for i, cand in enumerate(regions):
        cl, ct, cr, cb = cand.bbox
        ch = cb - ct
        cw = cr - cl
        if ch <= 0 or cw <= 0:
            continue
        for j, neighbour in enumerate(regions):
            if i == j:
                continue
            nl, nt, nr, nb = neighbour.bbox
            nh = nb - nt
            if nh <= 0:
                continue
            # Must be significantly smaller in height
            if ch >= nh * 0.40:
                continue
            # Must be positioned directly above the neighbour (or slightly below)
            vertical_gap = nt - cb  # positive = cand is above neighbour
            if not (-ch * 0.5 <= vertical_gap <= ch * 0.5):
                continue
            # Must overlap horizontally by >= 50% of candidate width
            horiz_overlap = max(0, min(cr, nr) - max(cl, nl))
            if horiz_overlap < cw * 0.5:
                continue
            furigana_indices.add(i)
            break

    if furigana_indices:
        kept = [r for i, r in enumerate(regions) if i not in furigana_indices]
        logger.debug("Furigana suppression removed %d small annotation regions", len(furigana_indices))
        return kept
    return regions


def sort_manga_reading_order(regions: list[TextRegion]) -> list[TextRegion]:
    """Sort regions in vertical-RTL reading order: Top-to-Bottom, Right-to-Left."""
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


def _best_matching_segmentation(
    yolo_bbox: tuple[int, int, int, int], segmentation_bubbles: list[TextRegion], minimum_iou: float
) -> int | None:
    matches = [
        index
        for index, segmentation in enumerate(segmentation_bubbles)
        if _matches_yolo(yolo_bbox, segmentation, minimum_iou)
    ]
    return max(matches, key=lambda index: _match_score(yolo_bbox, segmentation_bubbles[index])) if matches else None


def _long_strip_component(
    start: int, yolo_bubbles: list[TextRegion], matches_by_yolo: list[int | None], config: BubblePostprocessConfig
) -> set[int]:
    matched_segmentation = matches_by_yolo[start]
    component = {start}
    changed = True
    while changed:
        changed = False
        for index, candidate in enumerate(yolo_bubbles):
            if index in component or matches_by_yolo[index] != matched_segmentation:
                continue
            if any(_is_same_long_strip(candidate.bbox, yolo_bubbles[member].bbox, config) for member in component):
                component.add(index)
                changed = True
    return component


def _non_maximum_suppression(candidates: list[TextRegion], threshold: float) -> list[TextRegion]:
    kept: list[TextRegion] = []
    for candidate in sorted(candidates, key=lambda region: region.detection_confidence, reverse=True):
        if all(_bbox_iou(candidate.bbox, existing.bbox) < threshold for existing in kept):
            kept.append(candidate)
    return kept


def _merge_mask_candidates(candidates: list[TextRegion], config: BubblePostprocessConfig) -> list[TextRegion]:
    return _merge_connected(candidates, lambda first, second: _should_merge_masks(first, second, config), _merge_masks)


def _merge_connected(candidates: list[TextRegion], matches: object, merge: object) -> list[TextRegion]:
    remaining = list(candidates)
    merged: list[TextRegion] = []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in remaining[:]:
                if any(matches(candidate, member) for member in group):
                    remaining.remove(candidate)
                    group.append(candidate)
                    changed = True
        merged.append(merge(group))
    return merged


def _should_merge_masks(first: TextRegion, second: TextRegion, config: BubblePostprocessConfig) -> bool:
    return _mask_iou(first, second) >= config.segmentation_duplicate_iou


def _matches_yolo(yolo_bbox: tuple[int, int, int, int], segmentation: TextRegion, minimum_iou: float) -> bool:
    if segmentation.layout_bbox is None or not isinstance(segmentation.layout_mask, np.ndarray):
        return False
    center_x = (yolo_bbox[0] + yolo_bbox[2]) // 2
    center_y = (yolo_bbox[1] + yolo_bbox[3]) // 2
    left, top, right, bottom = segmentation.layout_bbox
    if left <= center_x < right and top <= center_y < bottom and segmentation.layout_mask[center_y - top, center_x - left]:
        return True
    return _bbox_iou(yolo_bbox, segmentation.bbox) >= minimum_iou


def _match_score(yolo_bbox: tuple[int, int, int, int], segmentation: TextRegion) -> float:
    return _bbox_iou(yolo_bbox, segmentation.bbox)


def _is_same_long_strip(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int], config: BubblePostprocessConfig
) -> bool:
    first_width, first_height = first[2] - first[0], first[3] - first[1]
    second_width, second_height = second[2] - second[0], second[3] - second[1]
    if first_height / first_width >= config.long_strip_aspect_ratio and second_height / second_width >= config.long_strip_aspect_ratio:
        return _interval_overlap_ratio(first[0], first[2], second[0], second[2]) >= config.long_strip_cross_axis_overlap and _interval_gap(first[1], first[3], second[1], second[3]) <= min(first_height, second_height) * config.long_strip_gap_ratio
    if first_width / first_height >= config.long_strip_aspect_ratio and second_width / second_height >= config.long_strip_aspect_ratio:
        return _interval_overlap_ratio(first[1], first[3], second[1], second[3]) >= config.long_strip_cross_axis_overlap and _interval_gap(first[0], first[2], second[0], second[2]) <= min(first_width, second_width) * config.long_strip_gap_ratio
    return False


def _merge_bboxes(candidates: list[TextRegion]) -> TextRegion:
    left = min(candidate.bbox[0] for candidate in candidates)
    top = min(candidate.bbox[1] for candidate in candidates)
    right = max(candidate.bbox[2] for candidate in candidates)
    bottom = max(candidate.bbox[3] for candidate in candidates)
    return TextRegion(bbox=(left, top, right, bottom), source_text="", detection_confidence=max(candidate.detection_confidence for candidate in candidates))


def _merge_masks(candidates: list[TextRegion]) -> TextRegion:
    if len(candidates) == 1:
        return candidates[0]

    left = min(c.bbox[0] for c in candidates)
    top = min(c.bbox[1] for c in candidates)
    right = max(c.bbox[2] for c in candidates)
    bottom = max(c.bbox[3] for c in candidates)

    layout_boxes = [c.layout_bbox for c in candidates if isinstance(c.layout_mask, np.ndarray) and c.layout_bbox is not None]
    if layout_boxes:
        l_left = min(b[0] for b in layout_boxes)
        l_top = min(b[1] for b in layout_boxes)
        l_right = max(b[2] for b in layout_boxes)
        l_bottom = max(b[3] for b in layout_boxes)
        mask_bbox = (min(left, l_left), min(top, l_top), max(right, l_right), max(bottom, l_bottom))
    else:
        mask_bbox = (left, top, right, bottom)

    m_left, m_top, m_right, m_bottom = mask_bbox
    mask = np.zeros((m_bottom - m_top, m_right - m_left), dtype=bool)

    for candidate in candidates:
        if not isinstance(candidate.layout_mask, np.ndarray) or candidate.layout_bbox is None:
            continue
        c_left, c_top, c_right, c_bottom = candidate.layout_bbox
        mask_h, mask_w = candidate.layout_mask.shape[:2]
        h = min(c_bottom - c_top, mask_h)
        w = min(c_right - c_left, mask_w)
        d_top = max(0, c_top - m_top)
        d_left = max(0, c_left - m_left)
        mask[d_top : d_top + h, d_left : d_left + w] |= candidate.layout_mask[:h, :w]

    return TextRegion(
        bbox=(left, top, right, bottom),
        source_text="",
        detection_confidence=max(c.detection_confidence for c in candidates),
        layout_bbox=mask_bbox,
        layout_mask=mask,
    )


def _mask_iou(first: TextRegion, second: TextRegion) -> float:
    if not isinstance(first.layout_mask, np.ndarray) or first.layout_bbox is None:
        return 0.0
    if not isinstance(second.layout_mask, np.ndarray) or second.layout_bbox is None:
        return 0.0
    left = max(first.layout_bbox[0], second.layout_bbox[0])
    top = max(first.layout_bbox[1], second.layout_bbox[1])
    right = min(first.layout_bbox[2], second.layout_bbox[2])
    bottom = min(first.layout_bbox[3], second.layout_bbox[3])
    if right <= left or bottom <= top:
        return 0.0
    first_mask = first.layout_mask[top - first.layout_bbox[1] : bottom - first.layout_bbox[1], left - first.layout_bbox[0] : right - first.layout_bbox[0]]
    second_mask = second.layout_mask[top - second.layout_bbox[1] : bottom - second.layout_bbox[1], left - second.layout_bbox[0] : right - second.layout_bbox[0]]
    intersection = int(np.count_nonzero(np.logical_and(first_mask, second_mask)))
    union = _mask_area(first) + _mask_area(second) - intersection
    return intersection / union if union else 0.0


def _mask_area(candidate: TextRegion) -> int:
    return int(np.count_nonzero(candidate.layout_mask)) if isinstance(candidate.layout_mask, np.ndarray) else 0


def _bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union else 0.0


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _interval_gap(first_start: int, first_end: int, second_start: int, second_end: int) -> int:
    return max(0, max(first_start, second_start) - min(first_end, second_end))


def _interval_overlap_ratio(first_start: int, first_end: int, second_start: int, second_end: int) -> float:
    overlap = max(0, min(first_end, second_end) - max(first_start, second_start))
    return overlap / min(first_end - first_start, second_end - second_start)


