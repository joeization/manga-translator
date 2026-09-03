"""
Pillow-based renderer for placing translated text onto images.
Supports horizontal and vertical-rtl text directions, mask-aware bubble layout,
and two-pass outward-only white/dark stroke outline rendering.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models import TextRegion

from .base import Renderer

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
    "\u2026": "\ufe19",  # … -> ︙
    "\u22ee": "\ufe19",  # ⋮ -> ︙
}


def to_vertical_text(text: str) -> str:
    """Convert horizontal punctuation and brackets to vertical orientation symbols."""
    return "".join(VERTICAL_PUNCTUATION_MAP.get(char, char) for char in text)


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
        for region in regions:
            if region.translated_text:
                self._render_region(draw, region)
        return image

    def _text_color(self, region: TextRegion) -> tuple[int, int, int]:
        """Return fill colour to use for region's translated text.

        Reads ``ink_color`` from ``region.metadata``. Defaults to black (0,0,0).
        """
        ink = (region.metadata or {}).get("ink_color")
        if isinstance(ink, (tuple, list)) and len(ink) == 3:
            return (int(ink[0]), int(ink[1]), int(ink[2]))
        return (0, 0, 0)

    def _render_region(self, draw: ImageDraw.ImageDraw, region: TextRegion) -> None:
        if self._text_direction == "vertical-rtl":
            self._render_vertical_region(draw, region)
            return
        left, top, right, bottom = region.bbox
        available_width = max(1, right - left - self._padding * 2)
        available_height = max(1, bottom - top - self._padding * 2)

        fill = self._text_color(region)
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

    def _render_vertical_region(self, draw: ImageDraw.ImageDraw, region: TextRegion) -> None:
        left, top, right, bottom = region.bbox
        available_width = max(1, right - left - self._padding * 2)
        available_height = max(1, bottom - top - self._padding * 2)
        vertical_text = to_vertical_text(region.translated_text or "")
        size = _largest_vertical_bbox_font_size(
            vertical_text,
            min(self._font_size, available_height),
            self._min_font_size,
            available_width,
            available_height,
        )
        if size is not None:
            columns = _vertical_columns_for_size(vertical_text, size, available_height)
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
    spacing: float = 0,
    align: str = "center",
) -> None:
    """Two-pass text rendering: draws outer stroke first, then draws inner font body on top."""
    sw, sf = _stroke_info(fill, font.size)
    if sw > 0:
        if is_multiline:
            draw.multiline_text(xy, text, font=font, fill=sf, spacing=spacing, align=align, stroke_width=sw, stroke_fill=sf)
        else:
            draw.text(xy, text, font=font, fill=sf, stroke_width=sw, stroke_fill=sf)
    # Pass 2: Crisp inner body fill drawn on top with stroke_width=0
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
    def _render_region(self, draw: ImageDraw.ImageDraw, region: TextRegion) -> None:
        if not isinstance(region.layout_mask, np.ndarray) or region.layout_bbox is None:
            super()._render_region(draw, region)
            return

        mask = region.layout_mask.astype(np.uint8)
        layout_left, layout_top, _, _ = region.layout_bbox
        text_left, text_top, text_right, text_bottom = region.bbox
        allowed = np.zeros_like(mask)
        left = max(0, text_left - layout_left)
        top = max(0, text_top - layout_top)
        right = min(mask.shape[1], text_right - layout_left)
        bottom = min(mask.shape[0], text_bottom - layout_top)
        if right <= left or bottom <= top:
            super()._render_region(draw, region)
            return
        allowed[top:bottom, left:right] = 1
        mask = np.logical_and(mask, allowed).astype(np.uint8)
        if not np.any(mask):
            super()._render_region(draw, region)
            return
        x, y, width, height = cv2.boundingRect(mask)
        mask = mask[y : y + height, x : x + width]
        left, top = layout_left, layout_top
        fill = self._text_color(region)
        if self._text_direction == "vertical-rtl":
            self._render_vertical_mask_region(draw, region, mask, left + x, top + y, width, height)
            return
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

        super()._render_region(draw, region)

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
        low = self._min_font_size
        high = min(self._max_font_size, max(self._font_size, height))
        best_vertical_layout: tuple[list[tuple[str, int, int]], int] | None = None

        while low <= high:
            size = (low + high) // 2
            placements = _vertical_mask_layout(mask, vertical_text, self._font(size), self._padding, anchor)
            if placements is not None:
                best_vertical_layout = (placements, size)
                low = size + 1
            else:
                high = size - 1

        if best_vertical_layout is not None:
            placements, size = best_vertical_layout
            font = self._font(size)
            fill = self._text_color(region)
            for column, column_x, column_y in placements:
                for character in column:
                    _draw_text_with_stroke(draw, (left + column_x, top + column_y), character, font, fill, is_multiline=False)
                    column_y += size
            return

        super()._render_region(draw, region)


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


def _vertical_columns_for_size(text: str, size: int, height: int) -> list[str]:
    characters = [character for character in text if character != "\n"]
    return _vertical_columns(characters, max(1, height // size))


def _largest_vertical_bbox_font_size(text: str, maximum: int, minimum: int, width: int, height: int) -> int | None:
    character_count = len([character for character in text if character != "\n"])
    best: int | None = None
    low, high = minimum, maximum
    while low <= high:
        size = (low + high) // 2
        rows = max(1, height // size)
        columns = (character_count + rows - 1) // rows
        if columns * size <= width:
            best = size
            low = size + 1
        else:
            high = size - 1
    return best


def _vertical_mask_layout(
    mask: np.ndarray, text: str, font: ImageFont.FreeTypeFont, padding: int, anchor: tuple[int, int]
) -> list[tuple[str, int, int]] | None:
    size = font.size
    characters = [character for character in text if character != "\n"]
    if not characters:
        return []
    column_ranges = _mask_column_ranges(mask, size, padding)
    if not column_ranges:
        return None

    anchor_x, anchor_y = anchor
    selected_ranges = _centered_column_ranges(column_ranges, len(characters), size, anchor_x)
    if selected_ranges is None:
        return None
    selected_ranges = _center_column_baselines(selected_ranges, font, anchor_x)
    verified_ranges: list[tuple[int, int, int]] = []
    for x, _, _ in selected_ranges:
        span = _mask_vertical_span(mask, x, size, padding, anchor_y)
        if span is None:
            return None
        verified_ranges.append((x, *span))

    placements: list[tuple[str, int, int]] = []
    character_index = 0
    ordered_ranges = list(reversed(verified_ranges))
    for index, (x, top, bottom) in enumerate(ordered_ranges):
        capacity = (bottom - top) // size
        if capacity <= 0:
            continue
        remaining = len(characters) - character_index
        later_capacity = sum((range_bottom - range_top) // size for _, range_top, range_bottom in ordered_ranges[index + 1 :])
        count = min(capacity, max((remaining + len(ordered_ranges) - index - 1) // (len(ordered_ranges) - index), remaining - later_capacity))
        column_height = count * size
        start = min(max(anchor_y - column_height // 2, top), bottom - column_height)
        column = "".join(characters[character_index : character_index + count])
        if not column:
            break
        placements.append((column, x, start))
        character_index += len(column)
        if character_index == len(characters):
            return placements
    return None


def _centered_column_ranges(
    column_ranges: list[tuple[int, int, int]], character_count: int, size: int, anchor_x: int
) -> list[tuple[int, int, int]] | None:
    ranges = sorted(column_ranges)
    for count in range(1, len(ranges) + 1):
        candidates = [ranges[index : index + count] for index in range(len(ranges) - count + 1)]
        fitting = [candidate for candidate in candidates if sum((bottom - top) // size for _, top, bottom in candidate) >= character_count]
        if fitting:
            return min(fitting, key=lambda candidate: abs((candidate[0][0] + candidate[-1][0] + size) / 2 - anchor_x))
    return None


def _center_column_baselines(
    column_ranges: list[tuple[int, int, int]], font: ImageFont.FreeTypeFont, anchor_x: int
) -> list[tuple[int, int, int]]:
    size = font.size
    count = len(column_ranges)
    center = (column_ranges[0][0] + column_ranges[-1][0] + size) / 2
    offset = int(round(anchor_x - center))
    return [(x + offset, top, bottom) for x, top, bottom in column_ranges]


def _mask_column_ranges(mask: np.ndarray, size: int, padding: int) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    glyph_width = max(1, size - padding * 2)
    for x in range(padding, mask.shape[1] - size - padding + 1, size):
        strip = np.all(mask[:, x + padding : x + padding + glyph_width], axis=1)
        start = 0
        while start < len(strip):
            while start < len(strip) and not strip[start]:
                start += 1
            end = start
            while end < len(strip) and strip[end]:
                end += 1
            if end - start >= size:
                ranges.append((x, start, end))
            start = end + 1
    return ranges


def _mask_vertical_span(mask: np.ndarray, x: int, size: int, padding: int, anchor_y: int) -> tuple[int, int] | None:
    glyph_width = max(1, size - padding * 2)
    left, right = x + padding, x + padding + glyph_width
    if left < 0 or right > mask.shape[1]:
        return None
    strip = np.all(mask[:, left:right], axis=1)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(strip):
        while start < len(strip) and not strip[start]:
            start += 1
        end = start
        while end < len(strip) and strip[end]:
            end += 1
        if end - start >= size:
            spans.append((start, end))
        start = end + 1
    return min(spans, key=lambda span: abs((span[0] + span[1]) / 2 - anchor_y)) if spans else None


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
    x = left + width - size
    y = top + (height - max(len(column) for column in columns) * size) / 2
    for column in columns:
        for character in column:
            _draw_text_with_stroke(draw, (x, y), character, font, fill, is_multiline=False)
            y += size
        x -= size
        y = top + (height - max(len(column) for column in columns) * size) / 2
