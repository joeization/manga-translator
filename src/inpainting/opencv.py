"""
Processing flow per region
--------------------------
1. Locate the surrounding speech bubble via three fallback strategies:
   existing segmentation mask → contour-based segmenter → connected-component search.
2. Sample background colour from the bubble periphery (corners-first).
3. Build a fill mask, choosing one of two modes:

   ``interior``  Erode the bubble mask by *bubble_border_width* and clear the
                 entire eroded interior.  Skips text detection entirely – best
                 for standard white speech bubbles.

   ``glyph``     Detect individual glyphs via adaptive colour-distance plus
                 dark/bright anchor heuristics.  Handles coloured bubbles,
                 inverted text, and regions with no containing bubble.

4. Apply the fill:
   background std < *solid_fill_std_threshold* → flat colour fill (fast, clean)
   otherwise                                   → cv2.INPAINT_TELEA
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion

from .base import Inpainter
from .bubble import OpenCVContourBubbleSegmenter
from .utils import estimate_adaptive_dilation
# Minimum background sample size before we trust the colour estimate.
_MIN_BG_SAMPLES: int = 16
# Hard floor on the colour-distance contrast threshold.
_MIN_CONTRAST: float = 20.0


class OpenCVInpainter(Inpainter):
    """OpenCV-based text-removal inpainter."""

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
        text_bright_threshold: int = 220,
        solid_fill_std_threshold: float = 5.0,
        inpaint_algorithm: str = "telea",
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
        self._text_bright_threshold = text_bright_threshold
        self._solid_fill_std_threshold = solid_fill_std_threshold
        self._inpaint_algorithm = inpaint_algorithm.lower()
        self._bubble_segmenter = OpenCVContourBubbleSegmenter(white_threshold, bubble_padding)

    def inpaint(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        pixels = np.array(image.convert("RGB"))
        for region in regions:
            self._process_region(pixels, region)
        return Image.fromarray(pixels, mode="RGB")

    # ------------------------------------------------------------------
    # Region-level dispatch
    # ------------------------------------------------------------------

    def _process_region(self, pixels: np.ndarray, region: TextRegion) -> None:
        padded = replace(
            region,
            bbox=_expand_bbox(region.bbox, pixels.shape[1], pixels.shape[0], self._ocr_clear_padding),
        )
        bubble = _existing_bubble(region)
        bubble_source = "existing"
        if bubble is None:
            bubble = self._bubble_segmenter.segment(pixels, padded)
            bubble_source = "contour"
        if bubble is None:
            bubble = self._find_bubble(pixels, padded)
            bubble_source = "component"

        if bubble is not None:
            bubble_bbox, bubble_mask = bubble
            region.layout_bbox = bubble_bbox
            region.layout_mask = bubble_mask
            roi_bbox = _clip_bbox(bubble_bbox, pixels.shape[1], pixels.shape[0])
        else:
            roi_bbox = padded.bbox

        left, top, right, bottom = roi_bbox
        roi = pixels[top:bottom, left:right]
        if roi.size == 0:
            return

        # Detected text bbox in ROI-local coordinates (exact un-dilated detect rect).
        detect_offset = _detect_offset_in_roi(region.bbox, roi_bbox)
        detect_mask = _text_area_mask(roi.shape[:2], detect_offset)

        # Bubble segmentation mask in ROI coordinates, eroded to protect the bubble border.
        bubble_mask_roi = (
            _bubble_mask_in_region(roi_bbox, bubble_bbox, bubble_mask)
            if bubble is not None
            else np.ones(roi.shape[:2], dtype=bool)
        )
        if bubble is not None and self._bubble_border_width > 0:
            ks = self._bubble_border_width * 2 + 1
            k_border = np.ones((ks, ks), dtype=np.uint8)
            eroded_bubble = cv2.erode(bubble_mask_roi.astype(np.uint8), k_border, iterations=1).astype(bool)
            if np.any(eroded_bubble):
                bubble_mask_roi = eroded_bubble

        # Inpainting target is strictly the INTERSECTION of un-dilated detect rect and segmentation interior!
        allowed = detect_mask & bubble_mask_roi

        bg_samples = _sample_background(roi, bubble_mask_roi, detect_offset)
        bg_color = self._background_color(roi, bubble_mask_roi, bg_samples)
        bg_std = np.std(bg_samples.astype(float), axis=0)
        use_interior_clear = (
            self._bubble_clear_mode == "interior"
            and bubble is not None
            and self._is_plain_white_region(roi, bubble_mask_roi)
        )

        if use_interior_clear:
            fill_mask, bg_color = self._interior_clear_mask(roi, allowed, detect_offset)
            target_pixels = roi[allowed]
            target_std = np.std(target_pixels.astype(float), axis=0) if len(target_pixels) > 0 else bg_std
            is_uniform = float(np.max(target_std)) < min(2.0, self._solid_fill_std_threshold)
        else:
            fill_mask = self._glyph_fill_mask(
                region,
                roi_bbox,
                roi,
                allowed,
                bg_color,
                bg_std,
                img_w=pixels.shape[1],
                img_h=pixels.shape[0],
            )
            # Evaluate std on non-text background pixels within allowed target area
            bg_pixels = roi[allowed & ~fill_mask]
            if len(bg_pixels) >= _MIN_BG_SAMPLES:
                bg_color = np.median(bg_pixels, axis=0)
                target_std = np.std(bg_pixels.astype(float), axis=0)
            else:
                target_std = bg_std
            is_uniform = float(np.max(target_std)) < min(2.0, self._solid_fill_std_threshold)

        region.inpaint_bbox = roi_bbox
        region.inpaint_mask = fill_mask

        bg_luma = float(0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
        ink_allowed = _ink_allowed_mask(region, roi_bbox, roi.shape[:2], allowed)
        region.metadata = {**(region.metadata or {}), "ink_color": _estimate_ink_color(
            roi,
            detect_offset,
            ink_allowed,
            bg_color,
            self._dark_threshold,
            self._text_bright_threshold,
            bg_luma,
        )}

        if not np.any(fill_mask):
            return

        fill_color = np.clip(np.round(bg_color), 0, 255).astype(np.uint8)
        # If background is uniform, fill glyphs/interior directly with the true background color.
        # Otherwise, use spatial inpainting to interpolate gradient or textured background.
        if is_uniform:
            result = roi.copy()
            result[fill_mask] = fill_color
            pixels[top:bottom, left:right] = result
        else:
            inpaint_flag = cv2.INPAINT_NS if self._inpaint_algorithm == "ns" else cv2.INPAINT_TELEA
            pixels[top:bottom, left:right] = cv2.inpaint(
                roi, fill_mask.astype(np.uint8) * 255, self._inpaint_radius, inpaint_flag
            )

    # ------------------------------------------------------------------

    def _interior_clear_mask(
        self,
        roi: np.ndarray,
        allowed: np.ndarray,
        detect_offset: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Erode the bubble interior by border_width; estimate background from border ring.

        The eroded-away ring (``allowed & ~interior``) is guaranteed to be bubble
        background pixels — no text can be there — so it is the most reliable source
        for the fill-colour estimate.
        """
        if self._bubble_border_width > 0:
            ks = self._bubble_border_width * 2 + 1
            kernel = np.ones((ks, ks), dtype=np.uint8)
            interior = cv2.erode(allowed.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            interior = allowed.copy()

        ring = allowed & ~interior  # guaranteed background pixels
        if np.count_nonzero(ring) >= _MIN_BG_SAMPLES:
            bg_samples = roi[ring]
        else:
            # Fallback: corners-first sampling (very large border_width or tiny bubble).
            bg_samples = _sample_background(roi, allowed, detect_offset)

        return interior, self._background_color(roi, allowed, bg_samples)

    def _background_color(
        self,
        roi: np.ndarray,
        allowed: np.ndarray,
        samples: np.ndarray,
    ) -> np.ndarray:
        """Estimate fill colour, preserving plain white bubbles as white."""
        if not len(samples):
            samples = roi[allowed]
        if not len(samples):
            samples = roi.reshape(-1, 3)

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        white_pixels = roi[allowed & (gray >= self._white_threshold)]
        allowed_count = max(1, int(np.count_nonzero(allowed)))
        if len(white_pixels) >= _MIN_BG_SAMPLES and len(white_pixels) / allowed_count >= self._white_ratio:
            return np.median(white_pixels, axis=0)

        sample_gray = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
        white_samples = samples[sample_gray >= self._white_threshold]
        if len(white_samples) >= _MIN_BG_SAMPLES and len(white_samples) / len(samples) >= self._white_ratio:
            return np.median(white_samples, axis=0)

        return np.median(samples, axis=0)

    def _is_plain_white_region(
        self,
        roi: np.ndarray,
        allowed: np.ndarray,
    ) -> bool:
        if not np.any(allowed) or not _looks_like_enclosed_bubble_mask(allowed):
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        allowed_gray = gray[allowed]

        # Evaluate non-text background pixels within the bubble region
        bg_gray = allowed_gray[allowed_gray >= self._dark_threshold]
        if len(bg_gray) < _MIN_BG_SAMPLES:
            return False

        # Must be overwhelmingly clean white (at least 90% of non-text background >= white_threshold)
        white_count = int(np.count_nonzero(bg_gray >= self._white_threshold))
        if white_count / len(bg_gray) < max(0.90, self._white_ratio):
            return False

        # Background variation must be uniform (std < 10 to reject textures/screentones/lines)
        return float(np.std(bg_gray.astype(float))) < min(10.0, self._solid_fill_std_threshold)

    def _glyph_fill_mask(
        self,
        region: TextRegion,
        roi_bbox: tuple[int, int, int, int],
        roi: np.ndarray,
        allowed: np.ndarray,
        bg_color: np.ndarray,
        bg_std: np.ndarray,
        img_w: int = 1600,
        img_h: int = 1600,
    ) -> np.ndarray:
        ocr_mask = _ocr_mask_in_roi(region, roi_bbox, roi.shape[:2])
        if ocr_mask is not None and np.any(ocr_mask):
            adaptive_dilation = estimate_adaptive_dilation(
                region.bbox,
                region.source_text,
                img_w,
                img_h,
                base_dilation=self._mask_dilation,
                glyph_mask=ocr_mask,
            )
            mask = ocr_mask & allowed
            return self._dilate_glyph_mask(mask, allowed, dilation=adaptive_dilation, close=True)
        detect_offset = _detect_offset_in_roi(region.bbox, roi_bbox)
        return self._glyph_mask(
            roi,
            allowed,
            detect_offset,
            bg_color,
            bg_std,
            region=region,
            img_w=img_w,
            img_h=img_h,
        )

    def _dilate_glyph_mask(
        self,
        mask: np.ndarray,
        allowed: np.ndarray,
        dilation: int = 2,
        close: bool = False,
    ) -> np.ndarray:
        ks = max(1, dilation) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        prepared = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        if close:
            prepared = cv2.morphologyEx(prepared.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        return prepared & allowed

    def _glyph_mask(
        self,
        roi: np.ndarray,
        allowed: np.ndarray,
        detect_offset: tuple[int, int, int, int],
        bg_color: np.ndarray,
        bg_std: np.ndarray,
        region: TextRegion | None = None,
        img_w: int = 1600,
        img_h: int = 1600,
        dilation: int = 2,
    ) -> np.ndarray:
        """Detect text glyphs as pixels that deviate from the estimated background."""
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Adaptive colour-distance threshold: grows with background variance.
        color_dist = np.linalg.norm(roi.astype(float) - bg_color, axis=2)
        threshold = max(_MIN_CONTRAST, 3.0 * float(np.max(bg_std)))
        color_text = allowed & (color_dist >= threshold)

        bg_luma = float(0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
        if bg_luma < self._dark_threshold:
            dark_text = np.zeros(roi.shape[:2], dtype=bool)
            # Anchor: unambiguously bright pixels when the background is dark.
            bright_text: np.ndarray = allowed & (gray >= self._text_bright_threshold)
        else:
            # Anchor: unambiguously dark pixels when the background is light.
            dark_text = allowed & (gray <= self._dark_threshold)
            bright_text = np.zeros(roi.shape[:2], dtype=bool)

        raw_candidate = color_text | dark_text | bright_text
        if not np.any(raw_candidate):
            return np.zeros(roi.shape[:2], dtype=bool)

        # Anchor to detect rect: keep connected components that overlap with detect_rect
        # (allowing stylized/artistic text to extend outward, while ignoring unrelated bubble art)
        detect_rect = _text_area_mask(roi.shape[:2], detect_offset)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_candidate.astype(np.uint8))
        candidate = np.zeros_like(raw_candidate)
        for i in range(1, num_labels):
            comp_mask = labels == i
            if np.any(comp_mask & detect_rect):
                candidate |= comp_mask

        if not np.any(candidate):
            candidate = raw_candidate & detect_rect

        if region is not None:
            actual_dilation = estimate_adaptive_dilation(
                region.bbox,
                region.source_text,
                img_w,
                img_h,
                base_dilation=self._mask_dilation,
                glyph_mask=candidate,
            )
        else:
            actual_dilation = dilation

        return self._dilate_glyph_mask(candidate, allowed, dilation=actual_dilation)

    def _find_bubble(
        self, pixels: np.ndarray, region: TextRegion
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        left, top, right, bottom = region.bbox
        ih, iw = pixels.shape[:2]
        ol = max(0, left - self._bubble_padding)
        ot = max(0, top - self._bubble_padding)
        or_ = min(iw, right + self._bubble_padding)
        ob = min(ih, bottom + self._bubble_padding)

        context = pixels[ot:ob, ol:or_]
        gray = cv2.cvtColor(context, cv2.COLOR_RGB2GRAY)
        white = np.where(gray >= self._white_threshold, 255, 0).astype(np.uint8)
        kernel = np.ones((self._bubble_close_kernel, self._bubble_close_kernel), dtype=np.uint8)
        closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

        tl, tt = left - ol, top - ot
        tr, tb = right - ol, bottom - ot
        text_area = (right - left) * (bottom - top)

        best_label, best_overlap = 0, 0
        for label in range(1, n_labels):
            cl, ct, cw, ch, _ = stats[label]
            touches_edge = (
                cl == 0
                or ct == 0
                or cl + cw == context.shape[1]
                or ct + ch == context.shape[0]
            )
            if touches_edge and cw * ch >= context.shape[0] * context.shape[1] * 0.9:
                continue
            overlap = int(np.count_nonzero(labels[tt:tb, tl:tr] == label))
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap

        if best_label == 0 or best_overlap < text_area * self._bubble_min_overlap:
            return None

        component = np.where(labels == best_label, 255, 0).astype(np.uint8)
        filled = _fill_enclosed_holes(component)
        return (ol, ot, or_, ob), filled.astype(bool)


# ---------------------------------------------------------------------------
# Background sampling
# ---------------------------------------------------------------------------

def _sample_background(
    roi: np.ndarray,
    allowed: np.ndarray,
    detect_offset: tuple[int, int, int, int],
    corner_size: int = 12,
) -> np.ndarray:
    """
    Sample background pixels using a corners-first strategy.

    Priority:
    1. Four corners of the allowed mask – avoids the text that is usually centred.
    2. Allowed pixels outside the detected-text bounding box.
    3. All allowed pixels (last resort; may include some text pixels).
    """
    h, w = roi.shape[:2]
    cs = max(4, min(corner_size, h // 4, w // 4))

    corner_mask = np.zeros((h, w), dtype=bool)
    corner_mask[:cs, :cs] = True
    corner_mask[:cs, w - cs:] = True
    corner_mask[h - cs:, :cs] = True
    corner_mask[h - cs:, w - cs:] = True

    corner_bg = allowed & corner_mask
    if np.count_nonzero(corner_bg) >= _MIN_BG_SAMPLES:
        return roi[corner_bg]

    dl, dt, dr, db = detect_offset
    detect_mask = np.zeros((h, w), dtype=bool)
    detect_mask[dt:db, dl:dr] = True
    ring_bg = allowed & ~detect_mask
    if np.count_nonzero(ring_bg) >= _MIN_BG_SAMPLES:
        return roi[ring_bg]

    all_bg = roi[allowed]
    return all_bg if len(all_bg) > 0 else roi.reshape(-1, 3)


def _estimate_ink_color(
    roi: np.ndarray,
    detect_offset: tuple[int, int, int, int],
    allowed: np.ndarray,
    bg_color: np.ndarray,
    dark_threshold: int,
    bright_threshold: int,
    bg_luma: float,
) -> tuple[int, int, int]:
    """Return the dominant ink colour from the original (pre-fill) text patch.

    Uses the same dark/bright polarity logic as ``_glyph_mask``: when the
    background is dark (bg_luma < dark_threshold), the ink is likely bright, and
    vice versa.  Falls back to a legible default when no clear ink pixels exist.
    """
    dl, dt, dr, db = detect_offset
    h, w = roi.shape[:2]
    dl, dt = max(0, dl), max(0, dt)
    dr, db = min(w, dr), min(h, db)
    if dr <= dl or db <= dt:
        return (255, 255, 255) if bg_luma < dark_threshold else (0, 0, 0)
    patch = roi[dt:db, dl:dr]
    if patch.size == 0:
        return (255, 255, 255) if bg_luma < dark_threshold else (0, 0, 0)
    patch_allowed = allowed[dt:db, dl:dr]
    if not np.any(patch_allowed):
        return (255, 255, 255) if bg_luma < dark_threshold else (0, 0, 0)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    color_distance = np.linalg.norm(patch.astype(float) - bg_color, axis=2)
    color_mask = patch_allowed & (color_distance >= _MIN_CONTRAST)
    if np.count_nonzero(color_mask) >= 4:
        strong_distance = np.percentile(color_distance[color_mask], 70)
        strong_mask = color_mask & (color_distance >= strong_distance)
        if np.count_nonzero(strong_mask) >= 4:
            ink = np.median(patch[strong_mask], axis=0)
        else:
            ink = np.median(patch[color_mask], axis=0)
        return _normalize_neutral_ink(ink, bg_luma, dark_threshold)
    if bg_luma < dark_threshold:
        # Dark background: look for bright ink (white text on dark).
        mask = patch_allowed & (gray >= bright_threshold)
        if np.count_nonzero(mask) >= 4:
            return tuple(int(c) for c in np.median(patch[mask], axis=0))  # type: ignore[return-value]
        return (255, 255, 255)
    else:
        # Light background: look for dark ink (black text on white).
        mask = patch_allowed & (gray <= dark_threshold)
        if np.count_nonzero(mask) >= 4:
            return tuple(int(c) for c in np.median(patch[mask], axis=0))  # type: ignore[return-value]
        return (0, 0, 0)




def _normalize_neutral_ink(
    ink: np.ndarray,
    bg_luma: float,
    dark_threshold: int,
) -> tuple[int, int, int]:
    luma = float(0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2])
    saturation = float(np.max(ink) - np.min(ink))
    if saturation < 25:
        if bg_luma < dark_threshold and luma > bg_luma:
            return (255, 255, 255)
        if bg_luma >= dark_threshold and luma < bg_luma:
            return (0, 0, 0)
    return tuple(int(c) for c in np.clip(np.round(ink), 0, 255))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _detect_offset_in_roi(
    detect_bbox: tuple[int, int, int, int],
    roi_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the detect bbox clipped to ROI-local pixel coordinates."""
    rl, rt, rr, rb = roi_bbox
    dl, dt, dr, db = detect_bbox
    rh, rw = rb - rt, rr - rl
    return (
        max(0, dl - rl),
        max(0, dt - rt),
        min(rw, dr - rl),
        min(rh, db - rt),
    )


def _ocr_mask_in_roi(
    region: TextRegion,
    roi_bbox: tuple[int, int, int, int],
    roi_shape: tuple[int, int],
) -> np.ndarray | None:
    if not isinstance(region.ocr_mask, np.ndarray):
        return None

    source_mask = region.ocr_mask.astype(bool)
    if source_mask.ndim == 3:
        source_mask = np.any(source_mask, axis=2)
    if source_mask.ndim != 2:
        return None

    source_bbox = region.source_bbox or region.bbox
    sl, st, sr, sb = source_bbox
    rl, rt, rr, rb = roi_bbox
    ol, ot = max(sl, rl), max(st, rt)
    or_, ob = min(sr, rr), min(sb, rb)
    if or_ <= ol or ob <= ot:
        return None

    src_top, src_left = ot - st, ol - sl
    dst_top, dst_left = ot - rt, ol - rl
    height = min(ob - ot, source_mask.shape[0] - src_top, roi_shape[0] - dst_top)
    width = min(or_ - ol, source_mask.shape[1] - src_left, roi_shape[1] - dst_left)
    if height <= 0 or width <= 0:
        return None

    result = np.zeros(roi_shape, dtype=bool)
    result[dst_top : dst_top + height, dst_left : dst_left + width] = source_mask[
        src_top : src_top + height,
        src_left : src_left + width,
    ]
    return result


def _ink_allowed_mask(
    region: TextRegion,
    roi_bbox: tuple[int, int, int, int],
    roi_shape: tuple[int, int],
    allowed: np.ndarray,
) -> np.ndarray:
    ocr_mask = _ocr_mask_in_roi(region, roi_bbox, roi_shape)
    if ocr_mask is not None and np.any(ocr_mask & allowed):
        return ocr_mask & allowed

    detect_offset = _detect_offset_in_roi(region.bbox, roi_bbox)
    return _text_area_mask(roi_shape, detect_offset) & allowed


def _text_area_mask(
    shape: tuple[int, int],
    detect_offset: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = shape
    left, top, right, bottom = detect_offset
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    mask = np.zeros(shape, dtype=bool)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = True
    return mask


def _looks_like_enclosed_bubble_mask(mask: np.ndarray) -> bool:
    if mask.size == 0 or not np.any(mask):
        return False

    height, width = mask.shape[:2]
    fill_ratio = float(np.count_nonzero(mask) / mask.size)
    if fill_ratio > 0.92:
        return False

    edge_ratio = max(1, min(height, width) // 24)
    edge_ratio = min(edge_ratio, 6)
    top_band = mask[:edge_ratio, :]
    bottom_band = mask[height - edge_ratio :, :]
    left_band = mask[:, :edge_ratio]
    right_band = mask[:, width - edge_ratio :]
    touched_edges = sum(
        float(np.count_nonzero(edge)) / edge.size > 0.35
        for edge in (top_band, bottom_band, left_band, right_band)
    )
    if touched_edges >= 3:
        return False

    x, y, box_width, box_height = cv2.boundingRect(mask.astype(np.uint8))
    if box_width >= width * 0.96 and box_height >= height * 0.96:
        return False

    return True


def _expand_bbox(
    bbox: tuple[int, int, int, int], iw: int, ih: int, padding: int
) -> tuple[int, int, int, int]:
    l, t, r, b = bbox
    return max(0, l - padding), max(0, t - padding), min(iw, r + padding), min(ih, b + padding)


def _clip_bbox(
    bbox: tuple[int, int, int, int], iw: int, ih: int
) -> tuple[int, int, int, int]:
    l, t, r, b = bbox
    return max(0, l), max(0, t), min(iw, r), min(ih, b)


def _bubble_mask_in_region(
    roi_bbox: tuple[int, int, int, int],
    bubble_bbox: tuple[int, int, int, int],
    bubble_mask: np.ndarray,
) -> np.ndarray:
    rl, rt, rr, rb = roi_bbox
    bl, bt, br, bb = bubble_bbox
    result = np.zeros((rb - rt, rr - rl), dtype=bool)

    ol, ot = max(rl, bl), max(rt, bt)
    or_, ob = min(rr, br), min(rb, bb)
    if or_ <= ol or ob <= ot:
        return result

    st, sl = ot - bt, ol - bl
    sb, sr = ob - bt, or_ - bl
    tt, tl = ot - rt, ol - rl
    tb, tr = ob - rt, or_ - rl

    src = bubble_mask[st:sb, sl:sr]
    h = min(src.shape[0], tb - tt)
    w = min(src.shape[1], tr - tl)
    result[tt : tt + h, tl : tl + w] = src[:h, :w]
    return result


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _fill_enclosed_holes(component: np.ndarray) -> np.ndarray:
    outside = component.copy()
    flood_mask = np.zeros((component.shape[0] + 2, component.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(outside, flood_mask, (0, 0), 255)
    return cv2.bitwise_or(component, cv2.bitwise_not(outside))


def _existing_bubble(
    region: TextRegion,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
        return region.layout_bbox, region.layout_mask
    return None
