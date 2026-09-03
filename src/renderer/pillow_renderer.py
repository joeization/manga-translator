"""
Pillow-based renderer for placing translated text onto images.
Supports horizontal and vertical-rtl text directions, mask-aware bubble layout,
and two-pass outward-only white/dark stroke outline rendering.
"""
from __future__ import annotations
from src.translator.ollama import format_response

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models import TextRegion

from .base import Renderer
from .layout import TextAutoLayout

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
    "ー": "︱",          # U+30FC Katakana-Hiragana prolonged sound mark -> ︱
    "ｰ": "︱",          # U+FF70 Halfwidth prolonged sound mark -> ︱
    "|": "︱",          # U+007C ASCII pipe / vertical bar -> ︱
    "｜": "︱",         # U+FF5C Fullwidth vertical line -> ︱
    "–": "︱",          # U+2013 En dash -> ︱
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
    """Compute adaptively expanded render bounding boxes that are strictly non-overlapping.

    Expands each region towards its layout_bbox / surrounding bubble space, but
    strictly bounds expansion against all other regions on the page to guarantee
    that no two rendered regions collide or overlap.
    """
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

            if v_overlap > 0 and h_overlap == 0:
                if ri_x <= lj:
                    x_sep = (ri_x + lj) // 2
                    lim_r = min(lim_r, x_sep)
                elif rj_x <= li:
                    x_sep = (rj_x + li) // 2
                    lim_l = max(lim_l, x_sep)
            elif h_overlap > 0 and v_overlap == 0:
                if bi <= tj:
                    y_sep = (bi + tj) // 2
                    lim_b = min(lim_b, y_sep)
                elif bj <= ti:
                    y_sep = (bj + ti) // 2
                    lim_t = max(lim_t, y_sep)
            elif v_overlap > 0 and h_overlap > 0:
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
                if gap_x <= gap_y:
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


class PillowRenderer(Renderer):
    """Pillow renderer with text wrapping and outline stroke support."""

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
        """Return fill colour to use for region's translated text.

        Reads ``ink_color`` from ``region.metadata``. Defaults to black (0,0,0).
        """
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

        bl, bt, br, bb = region.bbox
        bw = max(1, br - bl)
        bh = max(1, bb - bt)

        # 1. Detected text box itself is distinctly horizontal (e.g. wide banner, narration bar)
        if bw >= bh * aspect_ratio_threshold:
            return True

        # 2. Bubble layout mask is distinctly horizontal and text is not strictly vertical
        if width is not None and height is not None and height > 0:
            if width >= height * aspect_ratio_threshold and bw >= bh * 1.05:
                return True

        # 3. Non-overlapping render bounds are distinctly horizontal and text is not strictly vertical
        if render_bounds is not None:
            rl, rt, rr, rb = render_bounds
            rw = max(1, rr - rl)
            rh = max(1, rb - rt)
            if rw >= rh * aspect_ratio_threshold and bw >= bh * 1.05:
                return True

        return False

    def _render_region(
        self,
        draw: ImageDraw.ImageDraw,
        region: TextRegion,
        render_bounds: tuple[int, int, int, int] | None = None,
    ) -> None:
        if self._text_direction == "vertical-rtl" and not self._is_horizontal_region(region, render_bounds=render_bounds):
            self._render_vertical_region(draw, region, render_bounds=render_bounds)
            return
        left, top, right, bottom = render_bounds if render_bounds is not None else _expand_bbox_towards_layout(region.bbox, region.layout_bbox)
        available_width = max(1, right - left - self._padding * 2)
        available_height = max(1, bottom - top - self._padding * 2)
        fill = self._text_color(region)

        # 1. Intelligent auto-layout with sentence & paragraph formatting
        auto_layout = TextAutoLayout(
            region.translated_text or "",
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

        # Fallback to binary search
        low = self._min_font_size
        high = min(self._max_font_size, max(self._font_size, available_height))
        best_layout: tuple[str, tuple[float, float], ImageFont.FreeTypeFont] | None = None

        while low <= high:
            size = (low + high) // 2
            font = self._font(size)
            text = _wrap_text(draw, region.translated_text or "", font, available_width)
            bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=0, align="center")
            text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            if text_width <= available_width and text_height <= available_height:
                x = left + (right - left - text_width) / 2 - bounds[0]
                y = top + (bottom - top - text_height) / 2 - bounds[1]
                best_layout = (text, (x, y), font)
                low = size + 1
            else:
                high = size - 1

        if best_layout is not None:
            text, pos, font = best_layout
            _draw_text_with_stroke(draw, pos, text, font, fill, is_multiline=True)
            return

    def _render_vertical_region(
        self,
        draw: ImageDraw.ImageDraw,
        region: TextRegion,
        render_bounds: tuple[int, int, int, int] | None = None,
    ) -> None:
        left, top, right, bottom = render_bounds if render_bounds is not None else _expand_bbox_towards_layout(region.bbox, region.layout_bbox)
        available_width = max(1, right - left - self._padding * 2)
        available_height = max(1, bottom - top - self._padding * 2)
        vertical_text = to_vertical_text(region.translated_text or "")

        # 1. Intelligent auto-layout with sentence & paragraph formatting
        auto_layout = TextAutoLayout(
            vertical_text,
            orientation="vertical-rtl",
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
            _draw_vertical_columns(
                draw,
                layout_res.lines,
                font=font,
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
                fill=self._text_color(region),
            )
            return

        # Fallback to aspect-ratio vertical columns
        size = _largest_vertical_bbox_font_size(
            vertical_text,
            min(self._max_font_size, max(self._font_size, available_height)),
            self._min_font_size,
            available_width,
            available_height,
        )
        if size is not None:
            columns = _vertical_columns_for_size(vertical_text, size, available_height, width=available_width)
            _draw_vertical_columns(draw, columns, font=self._font(size), left=left, top=top, width=right - left, height=bottom - top, fill=self._text_color(region))

    def _font_sizes(self, available_height: int) -> range:
        return range(min(self._max_font_size, max(self._font_size, available_height)), self._min_font_size - 1, -1)

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(str(self._font_path), size)
        return self._fonts[size]


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


class MaskAwarePillowRenderer(PillowRenderer):
    def _render_region(
        self,
        draw: ImageDraw.ImageDraw,
        region: TextRegion,
        render_bounds: tuple[int, int, int, int] | None = None,
    ) -> None:
        if not isinstance(region.layout_mask, np.ndarray) or region.layout_bbox is None:
            super()._render_region(draw, region, render_bounds=render_bounds)
            return

        mask = region.layout_mask.astype(np.uint8)
        layout_left, layout_top, _, _ = region.layout_bbox
        fill = self._text_color(region)

        if render_bounds is not None:
            target_left, target_top, target_right, target_bottom = render_bounds
        else:
            target_left, target_top, target_right, target_bottom = _expand_bbox_towards_layout(region.bbox, region.layout_bbox)

        left = max(0, target_left - layout_left)
        top = max(0, target_top - layout_top)
        right = min(mask.shape[1], target_right - layout_left)
        bottom = min(mask.shape[0], target_bottom - layout_top)
        if right <= left or bottom <= top:
            super()._render_region(draw, region, render_bounds=render_bounds)
            return

        # Find the mask pixels strictly within render_bounds
        allowed = np.zeros_like(mask)
        allowed[top:bottom, left:right] = 1
        clipped = np.logical_and(mask, allowed).astype(np.uint8)
        if not np.any(clipped):
            super()._render_region(draw, region, render_bounds=render_bounds)
            return

        x, y, width, height = cv2.boundingRect(clipped)
        mask = clipped[y : y + height, x : x + width]
        left, top = layout_left, layout_top
        fill = self._text_color(region)

        is_horiz = self._is_horizontal_region(region, width=width, height=height, render_bounds=render_bounds)
        if self._text_direction == "vertical-rtl" and not is_horiz:
            self._render_vertical_mask_region(draw, region, mask, left + x, top + y, width, height)
            return


        # Option A for horizontal text: Try Maximum Inscribed Rectangle first
        inscribed = find_maximum_inscribed_rectangle(mask)
        mask_area = float(np.count_nonzero(mask))
        if inscribed is not None and mask_area > 0:
            rx, ry, rw, rh = inscribed
            inscribed_ratio = (rw * rh) / mask_area
            if inscribed_ratio >= 0.45 and rw >= self._min_font_size * 2 and rh >= self._min_font_size:
                auto_layout = TextAutoLayout(
                    region.translated_text or "",
                    orientation="horizontal",
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
                    cur_y = top + y + ry + self._padding + (rh - self._padding * 2 - layout_res.total_height) / 2
                    for line in layout_res.line_details:
                        bounds = draw.textbbox((0, 0), line.text, font=font)
                        lw = bounds[2] - bounds[0]
                        lx = left + x + rx + self._padding + (rw - self._padding * 2 - lw) / 2 - bounds[0]
                        _draw_text_with_stroke(draw, (lx, cur_y - bounds[1]), line.text, font, fill, is_multiline=False)
                        cur_y += layout_res.font_size + layout_res.line_spacing
                        if line.is_paragraph_end:
                            cur_y += layout_res.paragraph_spacing
                    return

        # Fallback to mask placement
        low = self._min_font_size
        high = min(self._max_font_size, max(self._font_size, height))
        best_mask_layout: tuple[str, tuple[float, float], ImageFont.FreeTypeFont] | None = None

        while low <= high:
            size = (low + high) // 2
            font = self._font(size)
            text = _wrap_text(draw, region.translated_text or "", font, max(1, width - self._padding * 2))
            bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=0, align="center")
            text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            placement = _find_mask_placement(
                mask,
                text_width + self._padding * 2,
                text_height + self._padding * 2,
                _region_center(region.bbox, left + x, top + y),
            )
            if placement is not None:
                placement_x, placement_y = placement
                pos = (
                    left + x + placement_x + self._padding - bounds[0],
                    top + y + placement_y + self._padding - bounds[1],
                )
                best_mask_layout = (text, pos, font)
                low = size + 1
            else:
                high = size - 1

        if best_mask_layout is not None:
            text, pos, font = best_mask_layout
            _draw_text_with_stroke(draw, pos, text, font, fill, is_multiline=True)
            return

        super()._render_region(draw, region, render_bounds=render_bounds)

    def _render_vertical_mask_region(
        self,
        draw: ImageDraw.ImageDraw,
        region: TextRegion,
        mask: np.ndarray,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> None:
        anchor = _region_center(region.bbox, left, top)
        vertical_text = to_vertical_text(region.translated_text or "")
        fill = self._text_color(region)

        # Pre-segment sentences once before the layout search
        from .layout import segment_text
        segments = segment_text(vertical_text)
        sentences = [s.text.strip() for s in segments if s.text.strip()]
        if not sentences:
            super()._render_region(draw, region)
            return

        # 1. Standard rectangular bubble or clean oval: try Maximum Inscribed Rectangle with TextAutoLayout
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
                        left=left + rx,
                        top=top + ry,
                        width=rw,
                        height=rh,
                        fill=fill,
                    )
                    return

        # 2. Multi-sentence / irregular / connected bubble: use Column-wise Vertical Spans inside the mask
        low = self._min_font_size
        high = min(self._max_font_size, max(self._font_size, height))
        best_vertical_layout: tuple[list[tuple[str, int, int]], int] | None = None

        while low <= high:
            size = (low + high) // 2
            placements = _vertical_mask_layout(mask, sentences, size, self._padding, anchor)
            if placements is not None:
                best_vertical_layout = (placements, size)
                low = size + 1
            else:
                high = size - 1

        if best_vertical_layout is not None:
            max_fitting_size = best_vertical_layout[1]
            best_layout = best_vertical_layout
            best_score = _score_vertical_layout(max_fitting_size, max_fitting_size, best_vertical_layout[0])

            # Search nearby font sizes to avoid orphan last columns (e.g. 2nd column having only 1 character)
            min_candidate = max(self._min_font_size, int(max_fitting_size * 0.70))
            for cand_size in range(max_fitting_size - 1, min_candidate - 1, -1):
                placements = _vertical_mask_layout(mask, sentences, cand_size, self._padding, anchor)
                if placements is not None:
                    score = _score_vertical_layout(cand_size, max_fitting_size, placements)
                    if score > best_score:
                        best_score = score
                        best_layout = (placements, cand_size)

            placements, size = best_layout
            font = self._font(size)
            for column, column_x, column_y in placements:
                for character in column:
                    _draw_text_with_stroke(draw, (left + column_x, top + column_y), character, font, fill, is_multiline=False)
                    column_y += size
            return

        super()._render_region(draw, region)


def _score_vertical_layout(size: int, max_size: int, placements: list[tuple[str, int, int]]) -> float:
    """Score a vertical column layout based on font size and column balance.

    Penalizes orphan trailing columns (e.g. only 1 or 2 characters in the last column
    while other columns have 5+ characters) and rewards balanced column distribution
    or fitting cleanly into a single column.
    """
    if not placements:
        return -1.0
    col_lengths = [len(p[0]) for p in placements]
    num_cols = len(col_lengths)
    score = 1.0 * (size / max(1, max_size))

    if num_cols == 1:
        score += 0.25
    elif num_cols >= 2:
        # In vertical text, never split short phrases (<= 3 chars, e.g. "当然") across multiple columns
        if max(col_lengths) <= 1 or sum(col_lengths) <= 3:
            return -1000.0

        last_len = col_lengths[-1]
        max_len = max(col_lengths)
        min_len = min(col_lengths)

        if last_len == 1 and max_len >= 3:
            score -= 0.45
        elif last_len == 2 and max_len >= 5:
            score -= 0.25

        balance_ratio = min_len / max(1, max_len)
        score += 0.20 * balance_ratio

    return score


def find_maximum_inscribed_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the largest axis-aligned rectangle (x, y, w, h) inside a binary mask using histogram heights."""
    if mask.ndim != 2:
        return None
    h, w = mask.shape
    if h == 0 or w == 0:
        return None

    heights = [0] * w
    max_area = 0
    best_rect: tuple[int, int, int, int] | None = None

    for r in range(h):
        row = mask[r]
        for c in range(w):
            if row[c]:
                heights[c] += 1
            else:
                heights[c] = 0

        stack: list[int] = []
        for c in range(w + 1):
            cur_h = heights[c] if c < w else 0
            while stack and heights[stack[-1]] >= cur_h:
                bar_h = heights[stack.pop()]
                bar_w = c if not stack else c - stack[-1] - 1
                area = bar_h * bar_w
                if area > max_area:
                    max_area = area
                    rect_x = stack[-1] + 1 if stack else 0
                    rect_y = r - bar_h + 1
                    best_rect = (rect_x, rect_y, bar_w, bar_h)
            stack.append(c)

    return best_rect


def _find_mask_placement(mask: np.ndarray, width: int, height: int, preferred_center: tuple[int, int]) -> tuple[int, int] | None:
    center_x, center_y = preferred_center
    candidate_x = max(0, min(mask.shape[1] - width, center_x - width // 2))
    candidate_y = max(0, min(mask.shape[0] - height, center_y - height // 2))
    if width <= mask.shape[1] and height <= mask.shape[0] and np.all(mask[candidate_y : candidate_y + height, candidate_x : candidate_x + width]):
        return candidate_x, candidate_y
    return None


def _region_center(bbox: tuple[int, int, int, int], origin_x: int, origin_y: int) -> tuple[int, int]:
    left, top, right, bottom = bbox
    return (left + right) // 2 - origin_x, (top + bottom) // 2 - origin_y


def _vertical_column_options(text: str, size: int, height: int) -> list[list[str]]:
    characters = [character for character in text if character != "\n"]
    maximum_rows = max(1, height // size)
    return [_vertical_columns(characters, rows) for rows in range(maximum_rows, 0, -1)]


def _vertical_columns(characters: list[str], rows: int) -> list[str]:
    return ["".join(characters[index : index + rows]) for index in range(0, len(characters), rows)] or [""]


def _vertical_columns_for_size(text: str, size: int, height: int, width: int = 0) -> list[str]:
    characters = [character for character in text if character != "\n"]
    if not characters:
        return [""]
    max_rows = max(1, height // size)
    # When the region is wide (width > height), find the row count that best fills the width
    # so text does not collapse into a single thin strip in the center of a wide bubble.
    if width > 0 and width > height and max_rows > 1:
        target_aspect = width / max(1, height)
        best_rows = max_rows
        best_diff = float("inf")
        for r in range(1, max_rows + 1):
            cols = (len(characters) + r - 1) // r
            if cols * size <= width:
                aspect = (cols * size) / (r * size)
                diff = abs(aspect - target_aspect)
                if diff < best_diff:
                    best_diff = diff
                    best_rows = r
        return _vertical_columns(characters, best_rows)
    return _vertical_columns(characters, max_rows)


def _largest_vertical_bbox_font_size(text: str, maximum: int, minimum: int, width: int, height: int) -> int | None:
    character_count = len([character for character in text if character != "\n"])
    if character_count == 0:
        return maximum
    best: int | None = None
    low, high = minimum, maximum
    while low <= high:
        size = (low + high) // 2
        max_rows = max(1, height // size)
        fits = any(
            ((character_count + r - 1) // r) * size <= width
            for r in range(max_rows, 0, -1)
        )
        if fits:
            best = size
            low = size + 1
        else:
            high = size - 1
    return best


def _split_chars_by_capacity(chars: list[str], caps: list[int]) -> list[str] | None:
    n = len(chars)
    total_cap = sum(caps)
    if total_cap < n:
        return None

    chunks: list[str] = []
    rem_chars = n
    rem_cap = total_cap
    cur = 0

    for j, cap in enumerate(caps):
        if j == len(caps) - 1:
            take = rem_chars
        else:
            take = max(1, round(rem_chars * cap / rem_cap))
            take = min(take, cap)
            rem_after_cap = sum(caps[j + 1 :])
            if rem_chars - take > rem_after_cap:
                take = rem_chars - rem_after_cap

        if take > cap or take <= 0:
            return None

        chunks.append("".join(chars[cur : cur + take]))
        cur += take
        rem_chars -= take
        rem_cap -= cap

    return chunks


def _vertical_mask_layout(
    mask: np.ndarray,
    text_or_sentences: str | list[str],
    font_or_size: ImageFont.FreeTypeFont | int,
    padding: int,
    anchor: tuple[int, int],
) -> list[tuple[str, int, int]] | None:
    size = font_or_size.size if hasattr(font_or_size, "size") else int(font_or_size)
    if isinstance(text_or_sentences, list):
        sentences = text_or_sentences
    else:
        if not text_or_sentences.strip():
            return []
        from .layout import segment_text
        segments = segment_text(text_or_sentences)
        sentences = [s.text.strip() for s in segments if s.text.strip()]

    if not sentences:
        return []

    column_ranges = _mask_column_ranges(mask, size, padding)
    if not column_ranges:
        return None

    # Sort available columns from right to left (descending X for manga vertical-rtl)
    anchor_x, anchor_y = anchor
    # Keep the best span for each unique column X
    best_spans: dict[int, tuple[int, int]] = {}
    for x, top, bottom in sorted(column_ranges, key=lambda c: -c[0]):
        if (bottom - top) >= size:
            if x not in best_spans or (bottom - top) > (best_spans[x][1] - best_spans[x][0]):
                best_spans[x] = (top, bottom)

    ordered_cols = [(x, top, bottom, (bottom - top) // size) for x, (top, bottom) in sorted(best_spans.items(), key=lambda item: -item[0])]
    if not ordered_cols:
        return None

    # Distribute the available columns among the sentences:
    # 1. No two sentences ever share a column (每句話不共用同一條 column)
    # 2. Sentences are distributed proportionally across all available columns,
    #    so columns aren't bunched at the beginning leaving the remaining bubbles empty.
    # 3. Long sentences occupy multiple balanced columns.
    # 4. Each column is vertically centered in its own span.
    m_sents = len(sentences)
    total_cols = len(ordered_cols)
    if total_cols < m_sents:
        return None

    # Detect natural bubble lobes/clusters by vertical overlap between adjacent columns
    clusters: list[list[tuple[int, int, int, int]]] = [[ordered_cols[0]]]
    for col in ordered_cols[1:]:
        prev = clusters[-1][-1]
        overlap = min(prev[2], col[2]) - max(prev[1], col[1])
        if overlap >= size * 0.75:
            clusters[-1].append(col)
        else:
            clusters.append([col])

    # If the number of detected bubble lobes matches the number of sentences,
    # assign each sentence strictly to its own bubble lobe!
    if len(clusters) == m_sents:
        groups = clusters
    else:
        lens = [max(1, len([c for c in s if c != "\n"])) for s in sentences]
        total_chars = sum(lens)
        alloc_cols: list[int] = []
        rem_c = total_cols
        rem_l = total_chars
        for i, l in enumerate(lens):
            if i == m_sents - 1:
                c = rem_c
            else:
                c = max(1, round(rem_c * l / rem_l))
                c = min(c, rem_c - (m_sents - 1 - i))
            alloc_cols.append(c)
            rem_c -= c
            rem_l -= l

        groups = []
        cur = 0
        for c in alloc_cols:
            groups.append(ordered_cols[cur : cur + c])
            cur += c

    placements: list[tuple[str, int, int]] = []

    for i, sent in enumerate(sentences):
        chars = [c for c in sent if c != "\n"]
        n_chars = len(chars)
        if n_chars == 0:
            continue

        group = groups[i]
        target_center_x = (group[0][0] + group[-1][0]) / 2
        best_cand: tuple[list[tuple[int, int, int, int]], list[str]] | None = None
        best_score = float("inf")
        # In vertical layout, short phrases (<= 3 chars, e.g. "当然") must stay in 1 column
        max_k = 1 if n_chars <= 3 else len(group)
        for k in range(1, max_k + 1):
            for s_idx in range(len(group) - k + 1):
                cand = group[s_idx : s_idx + k]
                caps = [c[3] for c in cand]
                if sum(caps) < n_chars:
                    continue
                chunks = _split_chars_by_capacity(chars, caps)
                if chunks is not None:
                    center_x = (cand[0][0] + cand[-1][0]) / 2
                    score = abs(center_x - target_center_x)
                    if score < best_score:
                        best_score = score
                        best_cand = (cand, chunks)
            if best_cand is not None:
                break

        if best_cand is None:
            return None

        cand, chunks = best_cand

        # Horizontal centering: shift columns to balance left and right margins (留白對稱)
        cand_center_x = (cand[0][0] + cand[-1][0]) / 2
        group_center_x = (group[0][0] + group[-1][0]) / 2
        target_shift = int(round(group_center_x - cand_center_x))
        best_shift = 0
        if target_shift != 0:
            step = 1 if target_shift > 0 else -1
            glyph_w = max(1, size - padding * 2)
            for s in range(target_shift, 0, -step):
                valid = True
                for col_data, chunk in zip(cand, chunks):
                    cx, top, bottom, _ = col_data
                    nx = cx + s
                    col_h = len(chunk) * size
                    cy = top + (bottom - top - col_h) // 2
                    if nx < 0 or nx + padding + glyph_w > mask.shape[1] or cy < 0 or cy + col_h > mask.shape[0]:
                        valid = False
                        break
                    if not np.all(mask[cy : cy + col_h, nx + padding : nx + padding + glyph_w]):
                        valid = False
                        break
                if valid:
                    best_shift = s
                    break

        for col_data, chunk in zip(cand, chunks):
            x, top, bottom, _ = col_data
            col_h = len(chunk) * size
            # Center vertically within THIS column's own [top, bottom] span
            start_y = top + (bottom - top - col_h) // 2
            placements.append((chunk, x + best_shift, start_y))

    return placements


def _mask_column_ranges(mask: np.ndarray, size: int, padding: int) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    glyph_width = max(1, size - padding * 2)
    h, w = mask.shape
    for x in range(padding, w - size - padding + 1, size):
        strip = np.all(mask[:, x + padding : x + padding + glyph_width], axis=1)
        diff = np.diff(np.pad(strip.astype(np.int8), (1, 1), "constant"))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= size:
                ranges.append((x, int(s), int(e)))
    return ranges


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

