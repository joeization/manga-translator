"""
Inpainting utility functions, including original text ink color estimation.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.models import TextRegion


def estimate_ink_color(
    img_rgb: np.ndarray,
    region: TextRegion,
    dark_threshold: int = 100,
    bright_threshold: int = 200,
) -> tuple[int, int, int]:
    """Estimate original text body ink color (RGB tuple) from region before inpainting.

    Strips away outer outline strokes (e.g. white outlines around black text) by
    eroding candidate text pixels to sample the inner core text color.
    """
    h, w = img_rgb.shape[:2]
    bbox = region.source_bbox or region.bbox
    l, t, r, b = bbox
    l, t = max(0, l), max(0, t)
    r, b = min(w, r), min(h, b)
    if r <= l or b <= t:
        return (0, 0, 0)

    margin = 8
    bl, bt = max(0, l - margin), max(0, t - margin)
    br, bb = min(w, r + margin), min(h, b + margin)

    roi = img_rgb[bt:bb, bl:br]
    if roi.size == 0:
        return (0, 0, 0)

    tl, tt = l - bl, t - bt
    tr, tb = r - bl, b - bt

    text_patch = roi[tt:tb, tl:tr]
    if text_patch.size == 0:
        return (0, 0, 0)

    # Sample surrounding border pixels for background color estimate
    bg_mask = np.ones(roi.shape[:2], dtype=bool)
    bg_mask[tt:tb, tl:tr] = False
    if np.any(bg_mask):
        bg_color = np.median(roi[bg_mask], axis=0)
    else:
        bg_color = np.median(roi.reshape(-1, 3), axis=0)

    bg_luma = float(0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])

    gray = cv2.cvtColor(text_patch, cv2.COLOR_RGB2GRAY)
    color_dist = np.linalg.norm(text_patch.astype(float) - bg_color, axis=2)

    # High-contrast candidate text pixels
    contrast_mask = (color_dist >= 25.0).astype(np.uint8)

    # Erode outer outline border to isolate inner core text pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_mask = cv2.erode(contrast_mask, kernel, iterations=1) > 0

    if not np.any(eroded_mask):
        eroded_mask = contrast_mask > 0

    # If background is light (e.g. white speech bubble), check for dark core text
    if bg_luma >= 128:
        dark_mask = eroded_mask & (gray <= 120)
        if np.count_nonzero(dark_mask) >= 4:
            ink = np.median(text_patch[dark_mask], axis=0)
            sat = float(np.max(ink) - np.min(ink))
            if sat < 25:
                return (0, 0, 0)
            return tuple(int(c) for c in np.clip(np.round(ink), 0, 255))
        if np.count_nonzero(eroded_mask) >= 4:
            ink = np.median(text_patch[eroded_mask], axis=0)
            return tuple(int(c) for c in np.clip(np.round(ink), 0, 255))
        return (0, 0, 0)
    else:
        # Dark background: check for bright core text
        bright_mask = eroded_mask & (gray >= 150)
        if np.count_nonzero(bright_mask) >= 4:
            ink = np.median(text_patch[bright_mask], axis=0)
            sat = float(np.max(ink) - np.min(ink))
            if sat < 25:
                return (255, 255, 255)
            return tuple(int(c) for c in np.clip(np.round(ink), 0, 255))
        if np.count_nonzero(eroded_mask) >= 4:
            ink = np.median(text_patch[eroded_mask], axis=0)
            return tuple(int(c) for c in np.clip(np.round(ink), 0, 255))
        return (255, 255, 255)
