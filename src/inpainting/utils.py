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
