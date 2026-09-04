"""
Pillow-based renderer for placing translated text onto images.
Supports horizontal and vertical-rtl text directions, mask-aware bubble layout,
and two-pass outward-only white/dark stroke outline rendering.
"""
from __future__ import annotations

import logging
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models import TextRegion
from src.translator.ollama import format_response

from .base import Renderer
from .layout import (
    TextAutoLayout,
    _largest_vertical_bbox_font_size,
    _score_vertical_layout,
    _vertical_mask_layout,
    find_maximum_inscribed_rectangle,
    segment_text,
)

logger = logging.getLogger(__name__)

VERTICAL_PUNCTUATION_MAP = {
    "\u300c": "\ufe41",  # 「 -> ﹁
    "\u300d": "\ufe42",  # 」 -> ﹂
    "\u300e": "\ufe43",  # 『 -> ﹃
    "\u300f": "\ufe44",  # 』 -> ﹄
    "\uff08": "\ufe35",  # （ -> ︵
    "\uff09": "\ufe36",  # ） -> ︶
    "\u3010": "\ufe3b",  # 【 -> ︻
    "\u3011": "\ufe3c",  # 】 -> ︼
    "\u300a": "\ufe3d",  # 《 -> ︽
    "\u300b": "\ufe3e",  # 》 -> ︾
    "\u3014": "\ufe39",  # 〔 -> ︹
    "\u3015": "\ufe3a",  # 〕 -> ︺
    "(": "\ufe35",       # ( -> ︵
    ")": "\ufe36",       # ) -> ︶
    "[": "\ufe39",       # [ -> ︹
    "]": "\ufe3a",       # ] -> ︺
    "~": "\ufe31",       # ~ -> ︱
    "\uff5e": "\ufe31",  # ～ -> ︱
    "\u301c": "\ufe31",  # 〜 -> ︱
    "\u2015": "\ufe31",  # ― -> ︱
    "\u2014": "\ufe31",  # — -> ︱
    "-": "\ufe31",       # - -> ︱
    "ー": "︱",          # Prolonged sound mark -> ︱
    "ｰ": "︱",          # Halfwidth prolonged sound mark -> ︱
    "|": "︱",          # ASCII pipe / vertical bar -> ︱
    "｜": "︱",         # Fullwidth vertical line -> ︱
    "–": "︱",          # En dash -> ︱
    "\u2026": "\ufe19",  # … -> ︙
    "\u22ef": "\ufe19",  # ⋯ -> ︙
    "\u2025": "\ufe19",  # ‥ -> ︙
    "\u22ee": "\ufe19",  # ⋮ -> ︙
}


def to_vertical_text(text: str) -> str:
    """Convert horizontal punctuation and brackets to vertical orientation symbols."""
    text = format_response(text)
    text = re.sub(r"\.{2,}", "…", text)
    text = re.sub(r"．{2,}", "…", text)
    return "".join(VERTICAL_PUNCTUATION_MAP.get(char, char) for char in text)


def _expand_bbox_towards_layout(
    bbox: tuple[int, int, int, int],
    layout_bbox: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    """Expand text bbox towards surrounding bubble/layout bbox."""
    if layout_bbox is None:
        return bbox
    return layout_bbox


def compute_non_overlapping_render_bounds(
    regions: list[TextRegion],
) -> list[tuple[int, int, int, int]]:
    """Compute adaptively expanded render bounding boxes that are strictly non-overlapping."""
    n = len(regions)
    if n == 0:
        return []
    if n == 1:
        return [_expand_bbox_towards_layout(regions[0].bbox, regions[0].layout_bbox)]

    expanded_bounds: list[tuple[int, int, int, int]] = []
    for i in range(n):
        ri = regions[i]
        li, ti, ri_x, bi = ri.bbox
        if ri.layout_bbox is not None:
            lim_l, lim_t, lim_r, lim_b = ri.layout_bbox
        else:
            lim_l, lim_t, lim_r, lim_b = li, ti, ri_x, bi

        lim_l = min(lim_l, li)
        lim_t = min(lim_t, ti)
        lim_r = max(lim_r, ri_x)
        lim_b = max(lim_b, bi)

        for j in range(n):
            if i == j:
                continue
            rj = regions[j]
            lj, tj, rj_x, bj = rj.bbox

            v_overlap = max(0, min(bi, bj) - max(ti, tj))
            h_overlap = max(0, min(ri_x, rj_x) - max(li, lj))

            if v_overlap > 0 and h_overlap > 0:
                if v_overlap >= h_overlap:
                    x_sep = (max(li, lj) + min(ri_x, rj_x)) // 2
                    if (li + ri_x) < (lj + rj_x):
                        lim_r = min(lim_r, x_sep)
                    else:
                        lim_l = max(lim_l, x_sep)
                else:
                    y_sep = (max(ti, tj) + min(bi, bj)) // 2
                    if (ti + bi) < (tj + bj):
                        lim_b = min(lim_b, y_sep)
                    else:
                        lim_t = max(lim_t, y_sep)
            else:
                gap_x = max(0, max(li, lj) - min(ri_x, rj_x))
                gap_y = max(0, max(ti, tj) - min(bi, bj))
                separate_horizontal = (v_overlap > 0) or (h_overlap == 0 and gap_x <= gap_y)
                if separate_horizontal:
                    if ri_x <= lj:
                        lim_r = min(lim_r, (ri_x + lj) // 2)
                    elif rj_x <= li:
                        lim_l = max(lim_l, (rj_x + li) // 2)
                else:
                    if bi <= tj:
                        lim_b = min(lim_b, (bi + tj) // 2)
                    elif bj <= ti:
                        lim_t = max(lim_t, (bj + ti) // 2)

        lim_l = min(lim_l, li)
        lim_r = max(lim_r, ri_x)
        lim_t = min(lim_t, ti)
        lim_b = max(lim_b, bi)

        if ri.layout_bbox is not None:
            expanded_bounds.append((lim_l, lim_t, lim_r, lim_b))
        else:
            text_area = max(1, (ri_x - li) * (bi - ti))
            lim_area = max(1, (lim_r - lim_l) * (lim_b - lim_t))
            cov = min(1.0, text_area / lim_area)
            exp = min(0.85, max(0.0, (0.75 - cov) * 1.5))

            new_l = max(lim_l, li - int((li - lim_l) * exp))
            new_r = min(lim_r, ri_x + int((lim_r - ri_x) * exp))
            new_t = max(lim_t, ti - int((ti - lim_t) * exp))
            new_b = min(lim_b, bi + int((lim_b - bi) * exp))
            expanded_bounds.append((new_l, new_t, new_r, new_b))

    return expanded_bounds


def _stroke_info(fill: tuple[int, int, int], font_size: int) -> tuple[int, tuple[int, int, int]]:
    """Determine stroke width and stroke color for text outline expansion."""
    luma = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
    stroke_fill = (255, 255, 255) if luma < 128 else (0, 0, 0)
    stroke_width = max(2, min(5, font_size // 14))
    return stroke_width, stroke_fill


def _draw_text_with_stroke(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    is_multiline: bool = False,
    spacing: int = 0,
    align: str = "center",
) -> None:
    """Two-pass text rendering: draws outer stroke first, then draws inner font body on top."""
    stroke_width, stroke_fill = _stroke_info(fill, font.size)
    if stroke_width > 0:
        if is_multiline:
            draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing, align=align, stroke_width=stroke_width, stroke_fill=stroke_fill)
        else:
            draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    if is_multiline:
        draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing, align=align, stroke_width=0)
    else:
        draw.text(xy, text, font=font, fill=fill, stroke_width=0)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> str:
    lines: list[str] = []
    current_line = ""
    for character in text:
        if character == "\n":
            lines.append(current_line)
            current_line = ""
        elif not current_line or draw.textlength(current_line + character, font=font) <= width:
            current_line += character
        else:
            lines.append(current_line)
            current_line = character
    lines.append(current_line)
    return "\n".join(lines)


def _draw_vertical_columns(
    draw: ImageDraw.ImageDraw,
    columns: list[str],
    font: ImageFont.FreeTypeFont,
    left: int,
    top: int,
    width: int,
    height: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    size = font.size
    num_cols = len(columns)
    if num_cols == 0:
        return

    total_glyph_width = num_cols * size
    spare_width = max(0, width - total_glyph_width)

    if num_cols == 1:
        col_xs = [left + (width - size) / 2]
    else:
        gap = min(spare_width / (num_cols + 1), size * 1.5)
        total_span = total_glyph_width + (num_cols - 1) * gap
        start_right = left + (width + total_span) / 2 - size
        col_xs = [start_right - i * (size + gap) for i in range(num_cols)]

    for col_idx, column in enumerate(columns):
        x = col_xs[col_idx]
        col_height = len(column) * size
        y = top + (height - col_height) / 2
        for character in column:
            _draw_text_with_stroke(draw, (x, y), character, font, fill, is_multiline=False)
            y += size


class PillowRenderer(Renderer):
    """Pillow renderer with intelligent layout and outline stroke support."""

    def __init__(
        self,
        font_path: Path | str,
        font_size: int = 56,
        max_font_size: int = 72,
        min_font_size: int = 12,
        padding: int = 4,
        text_direction: str = "vertical-rtl",
    ) -> None:
        self._font_path = Path(font_path)
        self._font_size = font_size
        self._max_font_size = max_font_size
        self._min_font_size = min_font_size
        self._padding = padding
        self._text_direction = text_direction
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}

    def render(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        active_regions = [r for r in regions if r.translated_text]
        if not active_regions:
            return image

        bounds_list = compute_non_overlapping_render_bounds(active_regions)
        for region, bounds in zip(active_regions, bounds_list):
            self._render_region(draw, region, render_bounds=bounds)
        return image

    def _text_color(self, region: TextRegion) -> tuple[int, int, int]:
        ink = (region.metadata or {}).get("ink_color")
        if isinstance(ink, (tuple, list)) and len(ink) == 3:
            return (int(ink[0]), int(ink[1]), int(ink[2]))
        return (0, 0, 0)

    def _is_horizontal_region(
        self,
        region: TextRegion,
        width: int | None = None,
        height: int | None = None,
        render_bounds: tuple[int, int, int, int] | None = None,
        aspect_ratio_threshold: float = 1.3,
    ) -> bool:
        """Check if region is a horizontal rectangle that should be rendered left-to-right, top-to-bottom."""
        if self._text_direction == "horizontal":
            return True

        text = region.translated_text or ""
        has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
        has_alpha = bool(re.search(r"[a-zA-Z\u00C0-\u024F\u0400-\u04FF]", text))
        if has_alpha and not has_cjk:
            return True

        # In vertical-rtl mode, speech bubbles (with layout_mask) are dialogue and must remain vertical.
        if region.layout_mask is not None:
            return False

        # If CJK text is present without a layout_mask, only allow horizontal for wide banners (aspect ratio >= 2.0)
        effective_threshold = max(2.0, aspect_ratio_threshold) if has_cjk else aspect_ratio_threshold

        bl, bt, br, bb = region.bbox
        bw = max(1, br - bl)
        bh = max(1, bb - bt)

        if bw >= bh * effective_threshold:
            return True

        if width is not None and height is not None and height > 0:
            if width >= height * effective_threshold and bw >= bh * 1.05:
                return True

        if render_bounds is not None:
            rl, rt, rr, rb = render_bounds
            rw = max(1, rr - rl)
            rh = max(1, rb - rt)
            if rw >= rh * effective_threshold and bw >= bh * 1.05:
                return True

        return False

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(str(self._font_path), size)
        return self._fonts[size]

    def _render_region(
        self,
        draw: ImageDraw.ImageDraw,
        region: TextRegion,
        render_bounds: tuple[int, int, int, int] | None = None,
    ) -> None:
        if not region.translated_text:
            return

        left, top, right, bottom = (
            render_bounds
            if render_bounds is not None
            else _expand_bbox_towards_layout(region.bbox, region.layout_bbox)
        )
        width = max(1, right - left)
        height = max(1, bottom - top)
        fill = self._text_color(region)

        is_horizontal = self._is_horizontal_region(region, width, height, render_bounds=render_bounds)

        # 1. Horizontal rendering (LTR)
        if is_horizontal:
            available_width = max(1, width - self._padding * 2)
            available_height = max(1, height - self._padding * 2)
            auto_layout = TextAutoLayout(
                region.translated_text,
                orientation="horizontal",
                font_resolver=self._font,
            )
            layout_res = auto_layout.find_optimal_layout(
                available_width=available_width,
                available_height=available_height,
                max_font_size=min(self._max_font_size, max(self._font_size, available_height)),
                preferred_font_size=self._font_size,
                min_font_size=self._min_font_size,
            )
            if layout_res is not None and layout_res.score > 0:
                font = self._font(layout_res.font_size)
                cur_y = top + self._padding + (available_height - layout_res.total_height) / 2
                for line in layout_res.line_details:
                    bounds = draw.textbbox((0, 0), line.text, font=font)
                    lw = bounds[2] - bounds[0]
                    lx = left + self._padding + (available_width - lw) / 2 - bounds[0]
                    _draw_text_with_stroke(draw, (lx, cur_y - bounds[1]), line.text, font, fill, is_multiline=False)
                    cur_y += layout_res.font_size + layout_res.line_spacing
                    if line.is_paragraph_end:
                        cur_y += layout_res.paragraph_spacing
                return

            font = self._font(self._min_font_size)
            text = _wrap_text(draw, region.translated_text, font, available_width)
            _draw_text_with_stroke(draw, (left + self._padding, top + self._padding), text, font, fill, is_multiline=True)
            return

        # 2. Vertical rendering (Vertical-RTL)
        vertical_text = to_vertical_text(region.translated_text)
        segments = segment_text(vertical_text)
        sentences = [s.text.strip() for s in segments if s.text.strip()]
        if not sentences:
            return

        # Prepare layout mask
        mask: np.ndarray | None = None
        mask_origin = (left, top)
        if region.layout_mask is not None and region.layout_bbox is not None:
            raw_mask = np.asarray(region.layout_mask, dtype=np.uint8)
            lx, ly, rx, by = region.layout_bbox
            crop_l = max(0, left - lx)
            crop_t = max(0, top - ly)
            crop_r = min(raw_mask.shape[1], right - lx)
            crop_b = min(raw_mask.shape[0], bottom - ly)
            if crop_r > crop_l and crop_b > crop_t:
                mask = raw_mask[crop_t:crop_b, crop_l:crop_r]
                mask_origin = (left, top)

        if mask is None or mask.size == 0 or np.count_nonzero(mask) == 0:
            mask = np.ones((height, width), dtype=np.uint8)
            mask_origin = (left, top)

        # 2a. Check if regular oval / rectangle can use TextAutoLayout via Inscribed Rectangle
        inscribed = find_maximum_inscribed_rectangle(mask)
        mask_area = float(np.count_nonzero(mask))
        if inscribed is not None and mask_area > 0:
            rx, ry, rw, rh = inscribed
            inscribed_ratio = (rw * rh) / mask_area
            if inscribed_ratio >= 0.70 and rw >= self._min_font_size and rh >= self._min_font_size * 2:
                auto_layout = TextAutoLayout(
                    vertical_text,
                    orientation="vertical-rtl",
                    font_resolver=self._font,
                )
                layout_res = auto_layout.find_optimal_layout(
                    available_width=max(1, rw - self._padding * 2),
                    available_height=max(1, rh - self._padding * 2),
                    max_font_size=min(self._max_font_size, max(self._font_size, rh)),
                    preferred_font_size=self._font_size,
                    min_font_size=self._min_font_size,
                )
                comfortable_floor = max(self._min_font_size, int(self._font_size * 0.55))
                if layout_res is not None and layout_res.score > 0 and layout_res.font_size >= comfortable_floor:
                    font = self._font(layout_res.font_size)
                    _draw_vertical_columns(
                        draw,
                        layout_res.lines,
                        font=font,
                        left=mask_origin[0] + rx,
                        top=mask_origin[1] + ry,
                        width=rw,
                        height=rh,
                        fill=fill,
                    )
                    return

        # 2b. Irregular, compound, or multi-sentence bubble: use _vertical_mask_layout
        low = self._min_font_size
        high = min(self._max_font_size, max(self._font_size, height))
        best_vertical_layout: tuple[list[tuple[str, int, int]], int] | None = None

        while low <= high:
            size = (low + high) // 2
            placements = _vertical_mask_layout(mask, sentences, size, self._padding)
            if placements is not None:
                best_vertical_layout = (placements, size)
                low = size + 1
            else:
                high = size - 1

        if best_vertical_layout is not None:
            max_fitting_size = best_vertical_layout[1]
            best_layout = best_vertical_layout
            best_score = _score_vertical_layout(max_fitting_size, max_fitting_size, best_vertical_layout[0])

            # Search nearby font sizes to optimize column balance
            min_candidate = max(self._min_font_size, int(max_fitting_size * 0.70))
            for cand_size in range(max_fitting_size - 1, min_candidate - 1, -1):
                placements = _vertical_mask_layout(mask, sentences, cand_size, self._padding)
                if placements is not None:
                    score = _score_vertical_layout(cand_size, max_fitting_size, placements)
                    if score > best_score:
                        best_score = score
                        best_layout = (placements, cand_size)

            placements, size = best_layout
            font = self._font(size)
            ox, oy = mask_origin
            for column, column_x, column_y in placements:
                cy = oy + column_y
                for character in column:
                    char_bounds = font.getbbox(character)
                    char_w = char_bounds[2] - char_bounds[0] if char_bounds else size
                    char_x = ox + column_x + max(0, (size - char_w) // 2)
                    _draw_text_with_stroke(draw, (char_x, cy), character, font, fill, is_multiline=False)
                    cy += size
            return

        # 2c. Fallback to simple TextAutoLayout
        auto_layout = TextAutoLayout(
            vertical_text,
            orientation="vertical-rtl",
            font_resolver=self._font,
        )
        layout_res = auto_layout.find_optimal_layout(
            available_width=max(1, width - self._padding * 2),
            available_height=max(1, height - self._padding * 2),
            min_font_size=self._min_font_size,
        )
        if layout_res is not None:
            _draw_vertical_columns(
                draw,
                layout_res.lines,
                font=self._font(layout_res.font_size),
                left=left,
                top=top,
                width=width,
                height=height,
                fill=fill,
            )


# Backward-compatibility alias
MaskAwarePillowRenderer = PillowRenderer
