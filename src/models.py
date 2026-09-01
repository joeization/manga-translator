from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextRegion:
    bbox: tuple[int, int, int, int]
    source_text: str
    translated_text: str | None = None
    source_bbox: tuple[int, int, int, int] | None = None
    layout_bbox: tuple[int, int, int, int] | None = None
    layout_mask: object | None = None


OCRResult = TextRegion