from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models import TextRegion


@dataclass(frozen=True)
class BubblePostprocessConfig:
    yolo_nms_iou: float = 0.5
    segmentation_duplicate_iou: float = 0.6
    source_match_iou: float = 0.1
    long_strip_aspect_ratio: float = 3.0
    long_strip_cross_axis_overlap: float = 0.9
    long_strip_gap_ratio: float = 0.0


DEFAULT_CONFIG = BubblePostprocessConfig()


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
        results.append(
            TextRegion(
                bbox=yolo.bbox,
                source_text="",
                detection_confidence=max(yolo.detection_confidence, segmentation.detection_confidence),
                source_bbox=yolo.bbox,
                layout_bbox=segmentation.layout_bbox,
                layout_mask=segmentation.layout_mask,
            )
        )

    results.extend(segmentation for index, segmentation in enumerate(segmentation_bubbles) if index not in matched_segmentation)
    return sort_manga_reading_order(results)


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
    left = min(candidate.bbox[0] for candidate in candidates)
    top = min(candidate.bbox[1] for candidate in candidates)
    right = max(candidate.bbox[2] for candidate in candidates)
    bottom = max(candidate.bbox[3] for candidate in candidates)
    mask = np.zeros((bottom - top, right - left), dtype=bool)
    for candidate in candidates:
        if not isinstance(candidate.layout_mask, np.ndarray) or candidate.layout_bbox is None:
            continue
        candidate_left, candidate_top, candidate_right, candidate_bottom = candidate.layout_bbox
        mask[candidate_top - top : candidate_bottom - top, candidate_left - left : candidate_right - left] |= candidate.layout_mask
    bbox = (left, top, right, bottom)
    return TextRegion(bbox=bbox, source_text="", detection_confidence=max(candidate.detection_confidence for candidate in candidates), layout_bbox=bbox, layout_mask=mask)


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


