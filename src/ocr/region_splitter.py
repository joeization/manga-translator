from __future__ import annotations

from dataclasses import replace
import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion
from src.inpainting.utils import (
    UnionFind,
    bbox_gap_2d,
    partition_mask_by_centers,
    reading_order_sort_key,
    slice_mask_to_roi,
)


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


def _cluster_components(
    components: list[tuple[int, int, int, int, int]],
    max_gap_x: int,
    max_gap_y: int,
) -> list[list[int]]:
    """Union-Find clustering of text components based on bounding box proximity.

    Two components belong to the same text group if their horizontal gap is
    <= max_gap_x and their vertical gap is <= max_gap_y. This groups characters
    in the same column/line and adjacent columns/lines within a single paragraph.
    """
    n = len(components)
    uf = UnionFind(n)
    for i in range(n):
        li, ti, ri, bi, _ = components[i]
        for j in range(i + 1, n):
            lj, tj, rj, bj, _ = components[j]
            gap_x = max(0, max(li, lj) - min(ri, rj))
            gap_y = max(0, max(ti, tj) - min(bi, bj))
            if gap_x <= max_gap_x and gap_y <= max_gap_y:
                uf.union(i, j)

    return list(uf.groups().values())


def _split_2d_clusters(
    region: TextRegion,
    clusters: list[tuple[int, int, int, int]],
    ink: np.ndarray,
    median_w: float,
    median_h: float,
) -> list[TextRegion]:
    """Split text region based on 2D Euclidean distance clustering and Voronoi mask partitioning.

    Handles compound / stepped / diagonal speech bubbles where 1D orthogonal projections
    overlap along both X and Y axes.
    """
    k = len(clusters)
    if k <= 1:
        return [region]

    # Minimum required Euclidean gap to consider two clusters as physically distinct chambers/lobes
    min_2d_gap = max(20.0, min(median_w, median_h) * 1.0)

    uf = UnionFind(k)
    for i in range(k):
        for j in range(i + 1, k):
            _, _, dist_2d = bbox_gap_2d(clusters[i], clusters[j])
            if dist_2d < min_2d_gap:
                uf.union(i, j)

    groups_map = uf.groups()
    if len(groups_map) <= 1:
        return [region]

    group_indices = list(groups_map.values())
    left, top, right, bottom = region.bbox
    w_region, h_region = right - left, bottom - top

    # Group composite bounding boxes (relative to region.bbox)
    group_boxes: list[tuple[int, int, int, int]] = []
    for g_idxs in group_indices:
        gl = min(clusters[i][0] for i in g_idxs)
        gt = min(clusters[i][1] for i in g_idxs)
        gr = max(clusters[i][2] for i in g_idxs)
        gb = max(clusters[i][3] for i in g_idxs)
        group_boxes.append((gl, gt, gr, gb))

    # Determine layout mask baseline
    if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
        base_box = region.layout_bbox
        base_mask = region.layout_mask
    else:
        base_box = region.bbox
        base_mask = _mask_in_region(region)

    # Group centers relative to base_box
    group_centers = [
        (
            float((left + (gl + gr) / 2.0) - base_box[0]),
            float((top + (gt + gb) / 2.0) - base_box[1]),
        )
        for gl, gt, gr, gb in group_boxes
    ]

    partitions = partition_mask_by_centers(base_mask, group_centers)

    pad_x = max(2, int(median_w * 0.25))
    pad_y = max(2, int(median_h * 0.25))

    sub_regions: list[TextRegion] = []
    for idx, (gl, gt, gr, gb) in enumerate(group_boxes):
        gl_p = max(0, gl - pad_x)
        gt_p = max(0, gt - pad_y)
        gr_p = min(w_region, gr + pad_x)
        gb_p = min(h_region, gb + pad_y)

        sub_box = (left + gl_p, top + gt_p, left + gr_p, top + gb_p)
        sub_ocr_mask = ink[gt_p:gb_p, gl_p:gr_p].astype(bool)

        part = partitions[idx]
        if part is not None:
            (lx, ly, rx, by), cropped_chamber = part
            sub_layout_bbox = (base_box[0] + lx, base_box[1] + ly, base_box[0] + rx, base_box[1] + by)
            sub_layout_mask = cropped_chamber
        else:
            sub_layout_bbox = sub_box
            sub_layout_mask = np.ones((sub_box[3] - sub_box[1], sub_box[2] - sub_box[0]), dtype=np.uint8)

        sub_regions.append(
            replace(
                region,
                bbox=sub_box,
                source_bbox=sub_box,
                ocr_mask=sub_ocr_mask,
                layout_bbox=sub_layout_bbox,
                layout_mask=sub_layout_mask,
            )
        )

    # Sort sub-regions in reading order: top-to-bottom bands, then right-to-left
    band_h = max(25.0, median_h * 1.5)
    sub_regions.sort(
        key=lambda r: reading_order_sort_key(
            (r.bbox[0] + r.bbox[2]) / 2.0,
            (r.bbox[1] + r.bbox[3]) / 2.0,
            band_h,
        )
    )

    return sub_regions


def split_text_regions(
    image: Image.Image,
    region: TextRegion,
    image_array: np.ndarray | None = None,
    gray_image: np.ndarray | None = None,
) -> list[TextRegion]:
    """Segment a compound speech bubble into discrete semantic text chunks.

    Splits only when clear large interior blank gaps exist between distinct
    speech bubbles or paragraphs, while strictly preserving multi-column vertical
    and multi-line horizontal paragraphs belonging to the same text block.
    """
    left, top, right, bottom = region.bbox
    h_region, w_region = bottom - top, right - left
    if w_region <= 0 or h_region <= 0:
        return [region]

    bubble_mask = _mask_in_region(region)
    if gray_image is not None:
        gray = gray_image[top:bottom, left:right]
    else:
        if image_array is None:
            image_array = np.asarray(image.convert("RGB"))
        pixels = image_array[top:bottom, left:right]
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = np.where(bubble_mask, ink, 0).astype(np.uint8)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    components = [
        (int(stats[i, cv2.CC_STAT_LEFT]),
         int(stats[i, cv2.CC_STAT_TOP]),
         int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
         int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
         int(stats[i, cv2.CC_STAT_AREA]))
        for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= 4
        and stats[i, cv2.CC_STAT_WIDTH] >= 3
        and stats[i, cv2.CC_STAT_HEIGHT] >= 3
    ]
    if len(components) < 2:
        return [region]

    widths = [r - l for l, t, r, b, _ in components if (r - l) >= 6 and (b - t) >= 6]
    heights = [b - t for l, t, r, b, _ in components if (r - l) >= 6 and (b - t) >= 6]
    median_w = float(np.median(widths)) if widths else 30.0
    median_h = float(np.median(heights)) if heights else 30.0

    # Conservative grouping: connect glyphs within normal paragraph / bubble spacing
    max_gap_x = max(28, round(median_w * 2.2))
    max_gap_y = max(28, round(median_h * 2.0))

    cluster_indices = _cluster_components(components, max_gap_x, max_gap_y)

    min_cluster_area = max(14, round(median_w * median_h * 0.25))
    valid_clusters: list[tuple[int, int, int, int]] = []
    for c_idxs in cluster_indices:
        total_area = sum(components[i][4] for i in c_idxs)
        has_substantial_glyph = any(
            (components[i][2] - components[i][0]) >= 6 and (components[i][3] - components[i][1]) >= 6
            for i in c_idxs
        )
        if len(c_idxs) >= 2 or (total_area >= min_cluster_area and has_substantial_glyph):
            cl = min(components[i][0] for i in c_idxs)
            ct = min(components[i][1] for i in c_idxs)
            cr = max(components[i][2] for i in c_idxs)
            cb = max(components[i][3] for i in c_idxs)
            valid_clusters.append((cl, ct, cr, cb))

    if len(valid_clusters) <= 1:
        # All ink belongs to a single connected paragraph / speech bubble
        return [region]

    min_y_gap = max(30, round(median_h * 1.5))
    min_x_gap = max(30, round(median_w * 1.5))
    min_sub_h = max(35, round(median_h * 1.6))
    min_sub_w = max(35, round(median_w * 1.6))

    def split_axis(
        clusters: list[tuple[int, int, int, int]],
        axis: int,  # 1 for Y, 0 for X
        total_size: int,
        min_gap: int,
        min_sub_size: int,
    ) -> list[int]:
        if axis == 1:
            sorted_c = sorted(clusters, key=lambda c: (c[1], c[3]))
            start_coord, end_coord = 1, 3
        else:
            sorted_c = sorted(clusters, key=lambda c: (c[0], c[2]))
            start_coord, end_coord = 0, 2

        split_pts: list[int] = []
        max_prev_end = sorted_c[0][end_coord]
        last_pt = 0

        for i in range(len(sorted_c) - 1):
            max_prev_end = max(max_prev_end, sorted_c[i][end_coord])
            next_start = min(c[start_coord] for c in sorted_c[i + 1:])
            gap = next_start - max_prev_end
            if gap >= min_gap:
                pt = (max_prev_end + next_start) // 2
                if pt - last_pt >= min_sub_size and total_size - pt >= min_sub_size:
                    split_pts.append(pt)
                    last_pt = pt

        return split_pts

    # 1. Attempt orthogonal 1D Y-axis split
    y_pts = split_axis(valid_clusters, axis=1, total_size=h_region, min_gap=min_y_gap, min_sub_size=min_sub_h)
    if y_pts:
        y_bounds = [0] + y_pts + [h_region]
        y_parts: list[tuple[TextRegion, np.ndarray, list[tuple[int, int, int, int]]]] = []
        for i in range(len(y_bounds) - 1):
            y0, y1 = y_bounds[i], y_bounds[i + 1]
            sub_box = (left, top + y0, right, top + y1)
            sub_ink = ink[y0:y1, :]
            sub_clusters = [
                (cl, ct - y0, cr, cb - y0)
                for cl, ct, cr, cb in valid_clusters
                if ct < y1 and cb > y0
            ]
            y_parts.append((
                replace(region, bbox=sub_box, source_bbox=sub_box, ocr_mask=sub_ink.astype(bool)),
                sub_ink,
                sub_clusters,
            ))
    else:
        y_parts = [(region, ink, valid_clusters)]

    # 2. For each Y sub-region, attempt orthogonal 1D X-axis split, or 2D cluster split for non-orthogonal clusters
    result: list[TextRegion] = []
    for y_sub, y_ink, sub_c in y_parts:
        if len(sub_c) <= 1:
            result.append(y_sub)
            continue
        ys_left, ys_top, ys_right, ys_bottom = y_sub.bbox
        ys_w = ys_right - ys_left
        x_pts = split_axis(sub_c, axis=0, total_size=ys_w, min_gap=min_x_gap, min_sub_size=min_sub_w)
        if x_pts:
            x_bounds = [0] + x_pts + [ys_w]
            for i in range(len(x_bounds) - 1):
                x0, x1 = x_bounds[i], x_bounds[i + 1]
                sub_box = (ys_left + x0, ys_top, ys_left + x1, ys_bottom)
                sub_c_in_x = [
                    (cl - x0, ct, cr - x0, cb)
                    for cl, ct, cr, cb in sub_c
                    if cl < x1 and cr > x0
                ]
                sub_ink_x = y_ink[:, x0:x1]
                sub_part = replace(y_sub, bbox=sub_box, source_bbox=sub_box, ocr_mask=sub_ink_x.astype(bool))
                if len(sub_c_in_x) >= 2:
                    result.extend(_split_2d_clusters(sub_part, sub_c_in_x, sub_ink_x, median_w, median_h))
                else:
                    result.append(sub_part)
        else:
            # Orthogonal 1D splitting cannot separate remaining clusters (e.g. diagonal/stepped bubbles)
            result.extend(_split_2d_clusters(y_sub, sub_c, y_ink, median_w, median_h))

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
    uf = UnionFind(n)
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
                uf.union(i, j)

    # Group surviving regions by union-find root.
    groups = uf.groups(active_indices={i for i in range(n) if i not in dominated})

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

    # Final pass: strictly guarantee zero spatial overlap among all returned regions
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            l1, t1, r1, b1 = result[i].bbox
            l2, t2, r2, b2 = result[j].bbox
            inter_w = max(0, min(r1, r2) - max(l1, l2))
            inter_h = max(0, min(b1, b2) - max(t1, t2))
            if inter_w > 0 and inter_h > 0:
                if inter_h >= inter_w:
                    x_sep = (max(l1, l2) + min(r1, r2)) // 2
                    if (l1 + r1) <= (l2 + r2):
                        result[i] = replace(result[i], bbox=(l1, t1, min(r1, x_sep), b1))
                        result[j] = replace(result[j], bbox=(max(l2, x_sep), t2, r2, b2))
                    else:
                        result[i] = replace(result[i], bbox=(max(l1, x_sep), t1, r1, b1))
                        result[j] = replace(result[j], bbox=(l2, t2, min(r2, x_sep), b2))
                else:
                    y_sep = (max(t1, t2) + min(b1, b2)) // 2
                    if (t1 + b1) <= (t2 + b2):
                        result[i] = replace(result[i], bbox=(l1, t1, r1, min(b1, y_sep)))
                        result[j] = replace(result[j], bbox=(l2, max(t2, y_sep), r2, b2))
                    else:
                        result[i] = replace(result[i], bbox=(l1, max(t1, y_sep), r1, b1))
                        result[j] = replace(result[j], bbox=(l2, t2, r2, min(b2, y_sep)))

    return result


def region_has_text(
    image_array: np.ndarray,
    region: TextRegion,
    min_char_area: int = 14,
    gray_image: np.ndarray | None = None,
) -> bool:
    """Pre-OCR verification: determine whether a region contains actual text strokes.

    Discards empty speech bubbles, screentones, and circular background objects
    before OCR, preventing autoregressive OCR models from hallucinating text.
    """
    l, t, r, b = region.source_bbox or region.bbox
    h, w = b - t, r - l
    if w < 6 or h < 6:
        return False

    if gray_image is not None:
        gray = gray_image[t:b, l:r]
    else:
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

def prepare_ocr_regions(
    image: Image.Image,
    detector: object,
    image_array: np.ndarray | None = None,
) -> tuple[list[TextRegion], list[Image.Image]]:
    """Detect, split, resolve overlaps, filter empty bubbles, and generate crops for OCR."""
    if image_array is None:
        image_array = np.asarray(image.convert("RGB"))
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    candidates = detector.detect(image)
    all_split_regions: list[TextRegion] = []
    for candidate in candidates:
        region = candidate if isinstance(candidate, TextRegion) else TextRegion(bbox=candidate, source_text="")
        all_split_regions.extend(split_text_regions(image, region, image_array=image_array, gray_image=gray_image))

    all_split_regions = resolve_overlapping_regions(all_split_regions, threshold=0.35)
    all_split_regions = [r for r in all_split_regions if region_has_text(image_array, r, gray_image=gray_image)]
    if not all_split_regions:
        return [], []

    crops = [crop_for_ocr(image, text_region) for text_region in all_split_regions]
    return all_split_regions, crops
