"""
Inpainting utility functions, including original text ink color estimation.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.models import TextRegion


def estimate_adaptive_dilation(
    bbox: tuple[int, int, int, int],
    source_text: str | None,
    img_w: int,
    img_h: int,
    base_dilation: int = 1,
    glyph_mask: np.ndarray | None = None,
) -> int:
    """Dynamically estimate optimal mask dilation from actual text glyphs, font size, and image resolution.

    - Prioritizes directly measuring connected components and stroke thickness on the actual glyph mask
      to accurately determine font size, completely immune to bubbles not being filled by text.
    - Falls back to bounding box shorter dimension (column width / line height) if no glyph mask is available.
    - Scales with total page resolution (relative to 1600px standard manga page).
    """
    page_dimension = max(img_w, img_h)
    resolution_scale = max(0.6, min(2.5, page_dimension / 1600.0))
    user_bias = max(0, base_dilation - 1)

    # 1. Directly measure physical character size and stroke width from detected glyph mask
    if isinstance(glyph_mask, np.ndarray) and np.any(glyph_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glyph_mask.astype(np.uint8))
        raw_sizes: list[int] = []
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # Exclude tiny single-pixel noise specs
            if area >= 8 and (w >= 4 or h >= 4):
                raw_sizes.append(max(w, h))

        if raw_sizes:
            # Filter out punctuation marks (dots, commas, ellipsis periods < 0.40 * max character dimension)
            # so punctuation never drags down the measured font size
            max_char_dim = max(raw_sizes)
            main_chars = [s for s in raw_sizes if s >= 0.40 * max_char_dim]
            actual_font_size = float(np.median(main_chars)) if main_chars else float(max_char_dim)

            dist = cv2.distanceTransform(glyph_mask.astype(np.uint8), cv2.DIST_L2, 3)
            valid_pts = dist[(dist >= 0.8) & (dist < 500.0) & np.isfinite(dist)]
            stroke_radius = float(np.median(valid_pts)) if valid_pts.size > 0 else 1.5

            calc_px = (actual_font_size * 0.08 + stroke_radius * 0.6) * resolution_scale
            adaptive_px = round(calc_px) if math.isfinite(calc_px) else 3
            return int(max(2, min(24, adaptive_px + user_bias)))

    # 2. Geometry fallback when no glyph mask is available:
    # Filter out punctuation characters (dots, ellipsis, brackets) so punctuation never inflates character count
    punctuation_chars = set("。、，．・…‥⋮―—～~「」『』()（）[]［］【】《》〈〉!?！？·-─:：;；\"'“”‘’ 　\n\r\t")
    clean_text = "".join(c for c in (source_text or "") if c not in punctuation_chars)
    letter_count = len(clean_text)

    l, t, r, b = bbox
    w, h = max(1, r - l), max(1, b - t)
    min_dim = min(w, h)

    if letter_count > 0:
        # Aspect-ratio aware line width / character estimation
        est_size = max(12.0, min(float(min_dim), 64.0))
    else:
        est_size = max(12.0, min(float(min_dim), 64.0))

    adaptive_px = round(est_size * 0.10 * resolution_scale)
    return int(max(2, min(24, adaptive_px + user_bias)))


def estimate_ink_color(
    img_rgb: np.ndarray,
    region: TextRegion,
    dark_threshold: int = 100,
    bright_threshold: int = 200,
) -> tuple[int, int, int]:
    """Estimate original text body ink color (RGB tuple) from region before inpainting.

    Uses patch 80th-percentile luminance and local background sampling to reliably
    distinguish light speech bubbles from dark backgrounds without misclassification.
    """
    h, w = img_rgb.shape[:2]
    bbox = region.source_bbox or region.bbox
    l, t, r, b = bbox
    l, t = max(0, l), max(0, t)
    r, b = min(w, r), min(h, b)
    if r <= l or b <= t:
        return (0, 0, 0)

    text_patch = img_rgb[t:b, l:r]
    if text_patch.size == 0:
        return (0, 0, 0)

    gray = cv2.cvtColor(text_patch, cv2.COLOR_RGB2GRAY)

    # 80th percentile luminance inside text patch accurately identifies speech bubble background
    patch_p80_luma = float(np.percentile(gray, 80))

    margin = 6
    bl, bt = max(0, l - margin), max(0, t - margin)
    br, bb = min(w, r + margin), min(h, b + margin)
    roi = img_rgb[bt:bb, bl:br]
    bg_mask = np.ones(roi.shape[:2], dtype=bool)
    bg_mask[t - bt : b - bt, l - bl : r - bl] = False

    if np.any(bg_mask):
        bg_samples = roi[bg_mask]
        bg_lumas = 0.299 * bg_samples[:, 0] + 0.587 * bg_samples[:, 1] + 0.114 * bg_samples[:, 2]
        bg_luma = float(np.median(bg_lumas))
    else:
        bg_luma = patch_p80_luma

    is_light_bg = (patch_p80_luma >= 140) or (bg_luma >= 130)

    if is_light_bg:
        # Light speech bubble background: extract dark text pixels
        dark_pixels = text_patch[gray <= 130]
        if len(dark_pixels) >= 4:
            ink = np.median(dark_pixels, axis=0)
            sat = float(np.max(ink) - np.min(ink))
            if sat < 30:
                return (0, 0, 0)
            return tuple(int(c) for c in np.clip(np.round(ink), 0, 255))
        return (0, 0, 0)
    else:
        # Dark panel background: extract bright text pixels
        bright_pixels = text_patch[gray >= 150]
        if len(bright_pixels) >= 4:
            ink = np.median(bright_pixels, axis=0)
            sat = float(np.max(ink) - np.min(ink))
            if sat < 30:
                return (255, 255, 255)
            return tuple(int(c) for c in np.clip(np.round(ink), 0, 255))
        return (255, 255, 255)


def slice_mask_to_roi(
    roi_bbox: tuple[int, int, int, int],
    mask_bbox: tuple[int, int, int, int],
    mask: np.ndarray,
) -> np.ndarray:
    """Project a mask positioned at mask_bbox into a local boolean array covering roi_bbox."""
    rl, rt, rr, rb = roi_bbox
    bl, bt, br, bb = mask_bbox
    result = np.zeros((rb - rt, rr - rl), dtype=bool)

    ol, ot = max(rl, bl), max(rt, bt)
    or_, ob = min(rr, br), min(rb, bb)
    if or_ <= ol or ob <= ot:
        return result

    src = mask[ot - bt : ob - bt, ol - bl : or_ - bl]
    h = min(src.shape[0], ob - ot)
    w = min(src.shape[1], or_ - ol)
    result[ot - rt : ot - rt + h, ol - rl : ol - rl + w] = src[:h, :w]
    return result
