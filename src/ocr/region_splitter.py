from __future__ import annotations

from dataclasses import replace
import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion
from src.inpainting.utils import slice_mask_to_roi


def crop_for_ocr(image: Image.Image, region: TextRegion) -> Image.Image:
    crop = image.crop(region.source_bbox or region.bbox).convert("RGB")
    if not isinstance(region.ocr_mask, np.ndarray) or region.ocr_mask.shape != (crop.height, crop.width):
        return crop

    cropped_pixels = np.array(crop)
    background = np.full_like(cropped_pixels, fill_value=255)
    return Image.fromarray(np.where(region.ocr_mask[:, :, None], cropped_pixels, background))


def _mask_in_region(region: TextRegion) -> np.ndarray:
    if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
        return slice_mask_to_roi(region.bbox, region.layout_bbox, region.layout_mask)
    left, top, right, bottom = region.bbox
    return np.ones((bottom - top, right - left), dtype=bool)


def _find_blank_bands(projection: np.ndarray, min_gap_length: int, low_threshold: int = 2) -> list[tuple[int, int]]:
    """Find contiguous *interior* blank bands in a 1D projection profile.

    A band is interior when it does not start at index 0 (i.e. there is ink on
    both sides), ensuring we never split at the very edge of a region.
    Trailing blank runs at the end of the projection are also ignored because
    the loop exits without a flush step.
    """
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, count in enumerate(projection):
        is_blank = count <= low_threshold
        if is_blank and start is None:
            start = idx
        elif not is_blank and start is not None:
            if start > 0 and (idx - start) >= min_gap_length:
                bands.append((start, idx))
            start = None
    return bands


def _valid_split_points(bands: list[tuple[int, int]], total: int, min_sub_size: int) -> list[int]:
    """Return midpoints of *bands* such that every resulting sub-segment is at
    least *min_sub_size* pixels wide.

    Bands are processed left-to-right; a candidate midpoint is kept only when
    the segment before it (since the last accepted point) is large enough.
    A final pass drops the last accepted point if it would leave a trailing
    segment that is too short.
    """
    valid = [
        (start, end)
        for start, end in bands
        if start >= min_sub_size and total - end >= min_sub_size
    ]

    if not valid:
        return []

    start, end = max(valid, key=lambda band: band[1] - band[0])
    return [(start + end) // 2]


def split_text_regions(image: Image.Image, region: TextRegion, image_array: np.ndarray | None = None) -> list[TextRegion]:
    """Projection-profile based region splitter.

    Splits only when clear large interior blank gaps exist between distinct
    speech bubbles or paragraphs.

    Strategy
    --------
    1. Attempt a Y-axis (horizontal) split using row-ink projection.
    2. For every Y sub-region, independently attempt an X-axis (vertical)
       split using column-ink projection.

    This replaces the old mutual-exclusion (``elif``) logic, allowing a wide
    region that contains e.g. two bubble rows – each with side-by-side bubbles –
    to be correctly segmented on both axes.
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

    # Estimate median glyph dimensions via connected components.
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    components = [
        (l, t, l + w, t + h)
        for l, t, w, h, area in stats[1:n_labels]
        if area >= 4 and w >= 3 and h >= 3
    ]
    if len(components) < 2:
        return [region]

    widths = [r - l for l, t, r, b in components if (r - l) >= 6 and (b - t) >= 6]
    heights = [b - t for l, t, r, b in components if (r - l) >= 6 and (b - t) >= 6]
    median_w = float(np.median(widths)) if widths else 30.0
    median_h = float(np.median(heights)) if heights else 30.0

    # Vertical-bubble protection: a narrow region contains vertical text
    # columns placed side-by-side.  Splitting along Y would chop staggered
    # columns apart.  The threshold scales with actual median glyph width,
    # replacing the old resolution-dependent 150 px hard-code.
    if w_region <= round(median_w * 2.5):
        return [region]

    min_y_gap = max(30, round(median_h * 0.7))
    min_x_gap = max(30, round(median_w * 1.2))
    min_sub_h = max(35, round(median_h * 1.6))
    min_sub_w = max(35, round(median_w * 1.6))

    low_thresh_y = max(5, round(w_region * 0.05))

    row_sums = (ink > 0).sum(axis=1)
    y_pts_local = _valid_split_points(
        _find_blank_bands(row_sums, min_gap_length=min_y_gap, low_threshold=low_thresh_y),
        h_region,
        min_sub_h,
    )

    # Build Y sub-regions together with their region-local ink slices.
    if y_pts_local:
        y_abs_bounds = [top] + [top + pt for pt in y_pts_local] + [bottom]
        y_parts: list[tuple[TextRegion, np.ndarray]] = []
        for i in range(len(y_abs_bounds) - 1):
            sub_box = (left, y_abs_bounds[i], right, y_abs_bounds[i + 1])
            lt, lb = y_abs_bounds[i] - top, y_abs_bounds[i + 1] - top
            sub_ink = ink[lt:lb, :]
            y_parts.append((
                replace(region, bbox=sub_box, source_bbox=sub_box, ocr_mask=sub_ink.astype(bool)),
                sub_ink,
            ))
    else:
        y_parts = [(region, ink)]

    # For each Y part, independently attempt an X-axis split.
    result: list[TextRegion] = []
    for y_sub, y_ink in y_parts:
        ys_left, ys_top, ys_right, ys_bottom = y_sub.bbox
        ys_h = ys_bottom - ys_top

        low_thresh_x = max(5, round(ys_h * 0.05))
        col_sums = (y_ink > 0).sum(axis=0)
        x_pts_local = _valid_split_points(
            _find_blank_bands(col_sums, min_gap_length=min_x_gap, low_threshold=low_thresh_x),
            ys_right - ys_left,
            min_sub_w,
        )

        if x_pts_local:
            x_abs_bounds = [ys_left] + [ys_left + pt for pt in x_pts_local] + [ys_right]
            for i in range(len(x_abs_bounds) - 1):
                sub_box = (x_abs_bounds[i], ys_top, x_abs_bounds[i + 1], ys_bottom)
                ll, lr = x_abs_bounds[i] - ys_left, x_abs_bounds[i + 1] - ys_left
                result.append(replace(y_sub, bbox=sub_box, source_bbox=sub_box, ocr_mask=y_ink[:, ll:lr].astype(bool)))
        else:
            result.append(y_sub)

    return result


def merge_contained_regions(regions: list[TextRegion], threshold: float = 1.0) -> list[TextRegion]:
    """Filter out regions whose bounding box is fully contained inside another."""
    return resolve_overlapping_regions(regions, threshold=threshold)


def resolve_overlapping_regions(regions: list[TextRegion], threshold: float = 1.0) -> list[TextRegion]:
    """Remove or merge regions with substantial spatial overlap.

    For each pair (A, B):
    - If A is >= *threshold* fraction inside B and B is strictly larger: A is
      *dominated* and removed.
    - If A is >= *threshold* inside B and B is >= *threshold* inside A (symmetric
      overlap): A and B are *merged* into their union bounding box.

    - ``threshold=1.0`` only removes perfectly contained regions (original
      strict-containment behaviour); no merging occurs.
    - ``threshold=0.35`` removes any region where >= 35 % of its area is
      covered by a strictly larger region, and merges equal-size regions whose
      mutual overlap exceeds 35 %.
    """
    if len(regions) <= 1:
        return regions

    n = len(regions)
    areas = [max(1, (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])) for r in regions]

    def _overlap_frac(i: int, j: int) -> float:
        il, it, ir, ib = regions[i].bbox
        jl, jt, jr, jb = regions[j].bbox
        inter = max(0, min(ir, jr) - max(il, jl)) * max(0, min(ib, jb) - max(it, jt))
        return inter / areas[i]

    # Union-find (merge groups).
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        px, py = _find(x), _find(y)
        if px != py:
            parent[py] = px  # merge y's root into x's root

    dominated: set[int] = set()

    for i in range(n):
        if i in dominated:
            continue
        for j in range(n):
            if i == j or j in dominated:
                continue
            frac_i = _overlap_frac(i, j)
            if frac_i < threshold:
                continue
            if areas[j] > areas[i]:
                # j is strictly larger and covers >= threshold of i → i is dominated.
                dominated.add(i)
                break
            # areas[j] <= areas[i]: check symmetric overlap.
            if _overlap_frac(j, i) >= threshold:
                _union(i, j)

    # Group surviving regions by union-find root.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        if i in dominated:
            continue
        root = _find(i)
        groups.setdefault(root, []).append(i)

    result: list[TextRegion] = []
    for members in groups.values():
        if len(members) == 1:
            result.append(regions[members[0]])
        else:
            # Merge into union bbox; carry over metadata from the highest-confidence member.
            ml = min(regions[m].bbox[0] for m in members)
            mt = min(regions[m].bbox[1] for m in members)
            mr = max(regions[m].bbox[2] for m in members)
            mb = max(regions[m].bbox[3] for m in members)
            base = max(members, key=lambda m: regions[m].detection_confidence)
            result.append(replace(regions[base], bbox=(ml, mt, mr, mb)))

    return result


def region_has_text(image_array: np.ndarray, region: TextRegion, min_char_area: int = 14) -> bool:
    """Pre-OCR verification: determine whether a region contains actual text strokes.

    Discards empty speech bubbles, screentones, and circular background objects
    before OCR, preventing autoregressive OCR models from hallucinating text.
    """
    l, t, r, b = region.source_bbox or region.bbox
    h, w = b - t, r - l
    if w < 6 or h < 6:
        return False

    roi = image_array[t:b, l:r]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
        mask = slice_mask_to_roi((l, t, r, b), region.layout_bbox, region.layout_mask)
        # Erode mask to safely ignore the black bubble outline
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        interior = cv2.erode(mask.astype(np.uint8), k).astype(bool)
        if not np.any(interior):
            interior = mask
    else:
        interior = np.zeros((h, w), dtype=bool)
        pad_y = min(4, h // 4)
        pad_x = min(4, w // 4)
        interior[pad_y : h - pad_y, pad_x : w - pad_x] = True

    pixels = gray[interior]
    if pixels.size < 16:
        return False

    p80 = float(np.percentile(pixels, 80))
    if p80 >= 128:
        ink = interior & (gray <= min(180, p80 - 32))
    else:
        p20 = float(np.percentile(pixels, 20))
        ink = interior & (gray >= max(100, p20 + 32))

    if np.count_nonzero(ink) < min_char_area:
        return False

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    for i in range(1, num_labels):
        cw, ch, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area >= min_char_area and (cw >= 4 or ch >= 4):
            return True

    return False