from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextRegion:
    bbox: tuple[int, int, int, int]
    source_text: str
    detection_confidence: float = 1.0
    # None means the OCR backend does not provide this information; it does not mean zero confidence.
    confidence: float | None = None
    character_confidences: list[float] | None = None
    metadata: dict[str, object] | None = None
    ocr_mask: object | None = None
    inpaint_bbox: tuple[int, int, int, int] | None = None
    inpaint_mask: object | None = None
    translated_text: str | None = None
    source_bbox: tuple[int, int, int, int] | None = None
    layout_bbox: tuple[int, int, int, int] | None = None
    layout_mask: object | None = None


OCRResult = TextRegion