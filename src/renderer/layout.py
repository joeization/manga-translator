from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import re
from typing import Callable

import cv2
import numpy as np
from PIL import ImageFont

from src.inpainting.utils import partition_mask_by_centers, reading_order_sort_key


class BreakPriority(IntEnum):
    """Break opportunity priority. Lower value means stronger, more natural break."""
    PARAGRAPH = 1   # Explicit newline / paragraph separation (0 penalty)
    SENTENCE = 2    # Sentence terminal punctuation: 。！？!?…
    CLAUSE = 3      # Clause punctuation: ，、；;：:—～~
    WORD = 4        # Space between words
    CHAR = 5        # Character-level break (fallback when line overflows)


SENTENCE_TERMINALS = set("。！？!?…︙⋯‥⋮")
CLAUSE_TERMINALS = set("，、；;：:—～~︱︲︴")
CLOSING_PUNCTUATION = set("」』”’）)]}〕》〉﹂﹄︶︸︺︼︾﹀﹈")
OPENING_PUNCTUATION = set("「『“‘（([{〔《〈﹁﹃︵︷︹︻︽︿﹇")

# Universal Kinsoku Shori line-breaking rules (W3C / international standards for East Asian typography)
LINE_START_PROHIBITED = CLOSING_PUNCTUATION | SENTENCE_TERMINALS | set(
    "，、；;：:,.…︙︱︲︴—―-~～ーｰぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ"
)
LINE_END_PROHIBITED = OPENING_PUNCTUATION


def is_latin_word_char(c: str) -> bool:
    """Return True if character is a Latin/alphanumeric word character that should not be split across lines."""
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9") or c in "'-_"



@dataclass(frozen=True)
class TextSegment:
    """A natural chunk of text with its succeeding break opportunity."""
    text: str
    break_after: BreakPriority = BreakPriority.WORD


def segment_text(text: str) -> list[TextSegment]:
    """Segment text based on paragraphs, punctuation, and word boundaries.

    Keeps punctuation attached to the preceding text segment (Kinsoku rule).
    Opening brackets stay attached to following text if possible.
    """
    if not text:
        return []

    paragraphs = text.splitlines()
    segments: list[TextSegment] = []

    for p_idx, paragraph in enumerate(paragraphs):
        p_text = paragraph.strip()
        if not p_text:
            continue

        is_last_paragraph = (p_idx == len(paragraphs) - 1)
        p_segments = _segment_paragraph(p_text)

        if p_segments:
            # The last segment of a paragraph gets PARAGRAPH break priority
            last_seg = p_segments[-1]
            p_segments[-1] = TextSegment(
                text=last_seg.text,
                break_after=BreakPriority.PARAGRAPH if not is_last_paragraph else BreakPriority.SENTENCE,
            )
            segments.extend(p_segments)

    return segments


def _segment_paragraph(text: str) -> list[TextSegment]:
    """Segment a single paragraph by punctuation and spaces."""
    segments: list[TextSegment] = []
    n = len(text)
    idx = 0
    current_chars: list[str] = []

    while idx < n:
        char = text[idx]
        current_chars.append(char)
        idx += 1

        # Check for multiple consecutive dots (e.g. "..." or "..")
        if char == "." and idx < n and text[idx] == ".":
            while idx < n and (text[idx] in SENTENCE_TERMINALS or text[idx] in CLOSING_PUNCTUATION or text[idx] == "."):
                current_chars.append(text[idx])
                idx += 1
            while idx < n and text[idx] == " ":
                current_chars.append(text[idx])
                idx += 1
            segments.append(TextSegment(text="".join(current_chars), break_after=BreakPriority.SENTENCE))
            current_chars = []
            continue

        # Consume any consecutive punctuation or closing quotes attached to this segment
        if char in SENTENCE_TERMINALS:
            while idx < n and (text[idx] in SENTENCE_TERMINALS or text[idx] in CLOSING_PUNCTUATION or text[idx] == "."):
                current_chars.append(text[idx])
                idx += 1
            # Consume following space if any
            while idx < n and text[idx] == " ":
                current_chars.append(text[idx])
                idx += 1
            segments.append(TextSegment(text="".join(current_chars), break_after=BreakPriority.SENTENCE))
            current_chars = []

        elif char in CLAUSE_TERMINALS:
            while idx < n and (text[idx] in CLAUSE_TERMINALS or text[idx] in CLOSING_PUNCTUATION):
                current_chars.append(text[idx])
                idx += 1
            while idx < n and text[idx] == " ":
                current_chars.append(text[idx])
                idx += 1
            segments.append(TextSegment(text="".join(current_chars), break_after=BreakPriority.CLAUSE))
            current_chars = []

        elif char == " ":
            # Space between words
            segments.append(TextSegment(text="".join(current_chars), break_after=BreakPriority.WORD))
            current_chars = []

    if current_chars:
        segments.append(TextSegment(text="".join(current_chars), break_after=BreakPriority.WORD))

    return segments


@dataclass
class LayoutLine:
    text: str
    is_paragraph_end: bool = False
    is_sentence_end: bool = False


@dataclass
class LayoutResult:
    lines: list[str]
    font_size: int
    total_width: float
    total_height: float
    line_spacing: float
    paragraph_spacing: float
    score: float
    line_details: list[LayoutLine]


class TextAutoLayout:
    """Deterministic auto-layout engine for manga text rendering.

    Finds the optimal line breaks, column counts, spacing, and font size using
    natural text segmentation and deterministic penalty scoring.
    """

    def __init__(
        self,
        text: str,
        orientation: str = "vertical-rtl",
        font_resolver: Callable[[int], ImageFont.FreeTypeFont] | None = None,
    ) -> None:
        # Normalize whitespace in short numbers/titles (e.g. "第 3 話" -> "第3話", "第 1 卷" -> "第1卷")
        clean = [c for c in text if not c.isspace()]
        if len(clean) <= 4 and bool(re.search(r"[\u4e00-\u9fff\d]", text)):
            text = "".join(clean)
        self.raw_text = text
        self.orientation = orientation
        self.font_resolver = font_resolver
        self.segments = segment_text(text)

    def find_optimal_layout(
        self,
        available_width: int,
        available_height: int,
        max_font_size: int = 72,
        preferred_font_size: int = 56,
        min_font_size: int = 12,
    ) -> LayoutResult | None:
        """Search font sizes and layout configurations to find the best layout."""
        if not self.segments:
            return None

        # Start search from min(max_font_size, max(preferred_font_size, box_limit))
        if self.orientation == "vertical-rtl":
            size_upper = min(max_font_size, max(preferred_font_size, available_height))
        else:
            size_upper = min(max_font_size, max(preferred_font_size, available_width))

        if size_upper < min_font_size:
            size_upper = min_font_size

        # Binary search for the optimal (largest fitting) font size
        low = min_font_size
        high = size_upper
        best_fitting_size: int | None = None

        while low <= high:
            mid = (low + high) // 2
            font = self.font_resolver(mid) if self.font_resolver else None
            res = self._evaluate_layout_at_size(
                font_size=mid,
                font=font,
                available_width=available_width,
                available_height=available_height,
                para_factor=0.0,
            )
            if res is not None and res.score > -50000.0:
                best_fitting_size = mid
                low = mid + 1
            else:
                high = mid - 1

        if best_fitting_size is None:
            best_fitting_size = min_font_size

        # At the optimal font size, try generous paragraph spacing if space allows
        font = self.font_resolver(best_fitting_size) if self.font_resolver else None
        res_para = self._evaluate_layout_at_size(
            font_size=best_fitting_size,
            font=font,
            available_width=available_width,
            available_height=available_height,
            para_factor=0.4,
        )
        if res_para is not None and res_para.score > -50000.0:
            return res_para

        return self._evaluate_layout_at_size(
            font_size=best_fitting_size,
            font=font,
            available_width=available_width,
            available_height=available_height,
            para_factor=0.0,
        )

    def _evaluate_layout_at_size(
        self,
        font_size: int,
        font: ImageFont.FreeTypeFont | None,
        available_width: int,
        available_height: int,
        para_factor: float,
        break_on_sentence: bool | None = None,
    ) -> LayoutResult | None:
        fn = self._layout_vertical if self.orientation == "vertical-rtl" else self._layout_horizontal
        if break_on_sentence is not None:
            return fn(font_size, font, available_width, available_height, para_factor, break_on_sentence=break_on_sentence)

        # 1. Prioritize natural sentence separation (second sentence starts on its own line)
        res_sent = fn(font_size, font, available_width, available_height, para_factor, break_on_sentence=True)
        if res_sent is not None and res_sent.score > -50000.0:
            return res_sent

        # 2. Fallback to compact packing if the speech bubble is narrow/short
        return fn(font_size, font, available_width, available_height, para_factor, break_on_sentence=False)

    def _measure_text_length(self, text: str, font: ImageFont.FreeTypeFont | None, font_size: int) -> float:
        """Measure horizontal advance of text."""
        if font is not None:
            try:
                return float(font.getlength(text))
            except AttributeError:
                pass
        # Fallback estimation based on character types
        total = 0.0
        for ch in text:
            if ord(ch) < 128:
                total += font_size * 0.55
            else:
                total += font_size
        return total

    def _measure_vertical_length(self, text: str, font_size: int) -> float:
        """Measure vertical length of column. Matches rendering advance where each character advances by font_size."""
        chars = [c for c in text if c != "\n"]
        return float(len(chars) * font_size)

    def _layout_horizontal(
        self,
        font_size: int,
        font: ImageFont.FreeTypeFont | None,
        available_width: int,
        available_height: int,
        para_factor: float,
        break_on_sentence: bool = True,
    ) -> LayoutResult | None:
        line_height = font_size * 1.25
        para_gap = font_size * para_factor

        # Partition segments into lines using dynamic break search
        lines = self._break_segments_into_lines(
            segments=self.segments,
            line_limit=available_width,
            measure_func=lambda t: self._measure_text_length(t, font, font_size),
            font_size=font_size,
            break_on_sentence=break_on_sentence,
        )
        if not lines:
            return None

        # Compute dimensions
        line_widths = [self._measure_text_length(line.text, font, font_size) for line in lines]
        total_width = max(line_widths) if line_widths else 0.0

        total_height = 0.0
        for idx, line in enumerate(lines):
            total_height += line_height
            if line.is_paragraph_end and idx < len(lines) - 1:
                total_height += para_gap

        # Scoring
        score = self._score_horizontal(
            font_size=font_size,
            lines=lines,
            line_widths=line_widths,
            total_width=total_width,
            total_height=total_height,
            available_width=available_width,
            available_height=available_height,
        )

        return LayoutResult(
            lines=[line.text for line in lines],
            font_size=font_size,
            total_width=total_width,
            total_height=total_height,
            line_spacing=line_height - font_size,
            paragraph_spacing=para_gap,
            score=score,
            line_details=lines,
        )

    def _layout_vertical(
        self,
        font_size: int,
        font: ImageFont.FreeTypeFont | None,
        available_width: int,
        available_height: int,
        para_factor: float,
        break_on_sentence: bool = True,
    ) -> LayoutResult | None:
        col_width = font_size
        para_gap = font_size * para_factor

        # In vertical layout, column length limit is available_height
        columns = self._break_segments_into_lines(
            segments=self.segments,
            line_limit=available_height,
            measure_func=lambda t: self._measure_vertical_length(t, font_size),
            font_size=font_size,
            break_on_sentence=break_on_sentence,
        )
        if not columns:
            return None

        col_heights = [self._measure_vertical_length(col.text, font_size) for col in columns]
        total_height = max(col_heights) if col_heights else 0.0

        # Number of columns along X
        num_cols = len(columns)
        col_gap = min(font_size * 0.4, max(0.0, available_width - num_cols * col_width) / max(1, num_cols + 1))
        total_width = num_cols * col_width + max(0, num_cols - 1) * col_gap

        score = self._score_vertical(
            font_size=font_size,
            columns=columns,
            col_heights=col_heights,
            total_width=total_width,
            total_height=total_height,
            available_width=available_width,
            available_height=available_height,
        )

        return LayoutResult(
            lines=[col.text for col in columns],
            font_size=font_size,
            total_width=total_width,
            total_height=total_height,
            line_spacing=col_gap,
            paragraph_spacing=para_gap,
            score=score,
            line_details=columns,
        )

    def _break_segments_into_lines(
        self,
        segments: list[TextSegment],
        line_limit: float,
        measure_func: Callable[[str], float],
        font_size: int,
        break_on_sentence: bool = True,
    ) -> list[LayoutLine]:
        """Dynamic line breaking prioritizing sentence boundaries and balanced multi-line splits."""
        effective_segments: list[TextSegment] = []
        for seg in segments:
            seg_len = measure_func(seg.text)
            if seg_len <= line_limit:
                effective_segments.append(seg)
            else:
                # Need intra-segment balanced character breaking
                sub_chunks = self._split_segment_chars(seg.text, line_limit, measure_func)
                for c_idx, chunk in enumerate(sub_chunks):
                    is_last_chunk = (c_idx == len(sub_chunks) - 1)
                    effective_segments.append(
                        TextSegment(
                            text=chunk,
                            break_after=seg.break_after if is_last_chunk else BreakPriority.CHAR,
                        )
                    )

        if not effective_segments:
            return []

        lines: list[LayoutLine] = []
        current_chunk: list[str] = []
        current_len = 0.0
        current_is_para = False
        current_is_sent = False

        for idx, seg in enumerate(effective_segments):
            seg_len = measure_func(seg.text)
            if self.orientation == "vertical-rtl":
                test_len = current_len + seg_len
            else:
                test_line = "".join(current_chunk) + seg.text
                test_len = measure_func(test_line)

            # Check if this segment must start a new line
            if current_chunk and test_len > line_limit:
                lines.append(
                    LayoutLine(
                        text="".join(current_chunk),
                        is_paragraph_end=current_is_para,
                        is_sentence_end=current_is_sent,
                    )
                )
                current_chunk = [seg.text]
                current_len = seg_len
                current_is_para = (seg.break_after == BreakPriority.PARAGRAPH)
                current_is_sent = (seg.break_after == BreakPriority.SENTENCE)
            else:
                current_chunk.append(seg.text)
                current_len = test_len
                current_is_para = (seg.break_after == BreakPriority.PARAGRAPH)
                current_is_sent = (seg.break_after == BreakPriority.SENTENCE)

            # If segment ends a paragraph or sentence (when break_on_sentence is active), start new line
            is_natural_end = (
                seg.break_after == BreakPriority.PARAGRAPH
                or (break_on_sentence and seg.break_after == BreakPriority.SENTENCE)
            )
            if is_natural_end and current_chunk:
                lines.append(
                    LayoutLine(
                        text="".join(current_chunk),
                        is_paragraph_end=(seg.break_after == BreakPriority.PARAGRAPH),
                        is_sentence_end=True,
                    )
                )
                current_chunk = []
                current_len = 0.0
                current_is_para = False
                current_is_sent = False

        if current_chunk:
            lines.append(
                LayoutLine(
                    text="".join(current_chunk),
                    is_paragraph_end=current_is_para,
                    is_sentence_end=current_is_sent,
                )
            )

        return lines

    def _split_segment_chars(
        self,
        text: str,
        line_limit: float,
        measure_func: Callable[[str], float],
    ) -> list[str]:
        """Split a long segment at character boundaries into balanced chunks."""
        total_len = measure_func(text)
        if total_len <= line_limit or not text:
            return [text]

        num_chunks = max(2, math.ceil(total_len / line_limit))
        chars = list(text)
        total_chars = len(chars)

        # In vertical layout, never split short phrases (<= 4 chars) across multiple columns
        if self.orientation == "vertical-rtl":
            if total_chars <= 4 or (total_chars / num_chunks) < 1.5:
                return [text]

        target_chunk_len = total_chars / num_chunks

        chunks: list[str] = []
        cur_idx = 0
        for i in range(num_chunks):
            next_idx = round((i + 1) * target_chunk_len) if i < num_chunks - 1 else total_chars
            # Apply Kinsoku Shori (JIS X 4051 / W3C text layout):
            # 1. Line-start prohibition: next line should not start with a prohibited character
            while next_idx < total_chars and chars[next_idx] in LINE_START_PROHIBITED:
                next_idx += 1
            # 2. Line-end prohibition: current line should not end with an opening bracket
            while next_idx > cur_idx + 1 and chars[next_idx - 1] in LINE_END_PROHIBITED:
                next_idx -= 1

            # 3. Whole-word boundary protection: never break inside an alphanumeric Latin/English word (e.g. "Windows")
            if next_idx < total_chars and cur_idx < next_idx:
                if is_latin_word_char(chars[next_idx - 1]) and is_latin_word_char(chars[next_idx]):
                    word_start = next_idx - 1
                    while word_start > cur_idx and is_latin_word_char(chars[word_start - 1]):
                        word_start -= 1
                    if word_start > cur_idx:
                        next_idx = word_start

            chunk = "".join(chars[cur_idx:next_idx])
            while len(chunk) > 1 and measure_func(chunk) > line_limit:
                next_idx -= 1
                if next_idx < total_chars and cur_idx < next_idx:
                    if is_latin_word_char(chars[next_idx - 1]) and is_latin_word_char(chars[next_idx]):
                        word_start = next_idx - 1
                        while word_start > cur_idx and is_latin_word_char(chars[word_start - 1]):
                            word_start -= 1
                        if word_start > cur_idx:
                            next_idx = word_start
                while next_idx > cur_idx + 1 and chars[next_idx - 1] in LINE_END_PROHIBITED:
                    next_idx -= 1
                chunk = "".join(chars[cur_idx:next_idx])

            if next_idx < total_chars and chars[next_idx - 1] in LINE_END_PROHIBITED:
                next_idx += 1
                chunk = "".join(chars[cur_idx:next_idx])

            chunks.append(chunk)
            cur_idx = next_idx
            if cur_idx >= total_chars:
                break

        if cur_idx < total_chars:
            chunks.append("".join(chars[cur_idx:]))

        return [c for c in chunks if c]

    def _score_horizontal(
        self,
        font_size: int,
        lines: list[LayoutLine],
        line_widths: list[float],
        total_width: float,
        total_height: float,
        available_width: int,
        available_height: int,
    ) -> float:
        # 1. Overflow penalty
        if total_width > available_width or total_height > available_height:
            overflow_w = max(0.0, total_width - available_width)
            overflow_h = max(0.0, total_height - available_height)
            return -100000.0 - (overflow_w + overflow_h) * 100.0

        score = 0.0

        # 2. Font size preference: larger font has a strong reward
        score += font_size * 25.0

        # 3. Line balance / raggedness penalty
        if len(line_widths) > 1:
            avg_width = sum(line_widths) / len(line_widths)
            variance = sum((w - avg_width) ** 2 for w in line_widths) / len(line_widths)
            std_dev = math.sqrt(variance)
            score -= (std_dev / max(1.0, avg_width)) * 30.0

            # Widow penalty: avoid very short last line (e.g. < 25% of average width)
            last_w = line_widths[-1]
            if last_w < avg_width * 0.35 and not lines[-1].is_paragraph_end:
                score -= 40.0

        # 4. Box fill ratio reward (healthy 65% ~ 90% coverage of bubble area)
        area_used = total_width * total_height
        box_area = max(1, available_width * available_height)
        fill_ratio = min(1.0, area_used / box_area)
        score += fill_ratio * 40.0

        # 5. Sentence integrity: penalize lines that broke mid-sentence
        unbroken_sentences = sum(1 for line in lines if line.is_sentence_end or line.is_paragraph_end)
        broken_mid_sentence = len(lines) - unbroken_sentences
        score -= broken_mid_sentence * 35.0

        # 6. Latin word integrity: reject layouts where an alphanumeric word was split across lines
        for idx in range(len(lines) - 1):
            line_a = lines[idx].text
            line_b = lines[idx + 1].text
            if line_a and line_b and is_latin_word_char(line_a[-1]) and is_latin_word_char(line_b[0]):
                return -100000.0

        return score

    def _score_vertical(
        self,
        font_size: int,
        columns: list[LayoutLine],
        col_heights: list[float],
        total_width: float,
        total_height: float,
        available_width: int,
        available_height: int,
    ) -> float:
        if total_width > available_width or total_height > available_height:
            overflow_w = max(0.0, total_width - available_width)
            overflow_h = max(0.0, total_height - available_height)
            return -100000.0 - (overflow_w + overflow_h) * 100.0

        score = 0.0

        # Latin word integrity: reject layouts where an alphanumeric word was split across columns
        for idx in range(len(columns) - 1):
            col_a = columns[idx].text
            col_b = columns[idx + 1].text
            if col_a and col_b and is_latin_word_char(col_a[-1]) and is_latin_word_char(col_b[0]):
                return -100000.0

        # In vertical typography, short phrases (<= 3 chars, e.g. "当然", "好的") must NEVER be split
        # into multiple 1-character columns (which turns vertical text into backward horizontal RTL "然 当").
        col_lens = [len(c.text) for c in columns]
        if len(columns) >= 2:
            if max(col_lens) <= 1 or sum(col_lens) <= 3:
                return -100000.0
            if min(col_lens) == 1:
                score -= 300.0

        # Font size preference
        score += font_size * 25.0

        # Column balance penalty
        if len(col_heights) > 1:
            avg_h = sum(col_heights) / len(col_heights)
            variance = sum((h - avg_h) ** 2 for h in col_heights) / len(col_heights)
            std_dev = math.sqrt(variance)
            score -= (std_dev / max(1.0, avg_h)) * 30.0

            # Last column orphan penalty
            last_h = col_heights[-1]
            if last_h < avg_h * 0.3 and not columns[-1].is_paragraph_end:
                score -= 40.0

        # Aspect ratio match reward: in vertical manga, bubbles are often taller than wide
        target_aspect = available_width / max(1, available_height)
        current_aspect = total_width / max(1, total_height)
        aspect_diff = abs(current_aspect - target_aspect)
        score -= aspect_diff * 15.0

        # Box fill reward
        area_used = total_width * total_height
        box_area = max(1, available_width * available_height)
        fill_ratio = min(1.0, area_used / box_area)
        score += fill_ratio * 40.0

        # Sentence integrity: penalize columns that broke mid-sentence
        unbroken_sentences = sum(1 for col in columns if col.is_sentence_end or col.is_paragraph_end)
        broken_mid_sentence = len(columns) - unbroken_sentences
        score -= broken_mid_sentence * 35.0

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


def _mask_column_ranges(mask: np.ndarray, size: int, padding: int) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    glyph_width = max(1, size - padding * 2)
    h, w = mask.shape
    for x in range(padding, w - size - padding + 1, size):
        strip = np.all(mask[:, x + padding : x + padding + glyph_width], axis=1)
        diff = np.diff(np.pad(strip.astype(np.int8), (1, 1), "constant"))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            if end - start >= size:
                ranges.append((x, int(start), int(end)))
    return ranges


def _split_chars_by_capacity(chars: list[str], caps: list[int]) -> list[str] | None:
    n = len(chars)
    k = len(caps)
    if k == 0 or sum(caps) < n:
        return None
    if k == 1:
        return ["".join(chars)] if caps[0] >= n else None

    # In vertical layout, never split short phrases (<= 4 chars) across multiple columns
    if n <= 4 and k > 1:
        return None

    def can_break_at(pos: int) -> bool:
        if pos <= 0 or pos >= n:
            return False
        # Cannot break inside Latin word
        if is_latin_word_char(chars[pos - 1]) and is_latin_word_char(chars[pos]):
            return False
        # Next line cannot start with line-start prohibited char
        if chars[pos] in LINE_START_PROHIBITED:
            return False
        # Current line cannot end with line-end prohibited char
        if chars[pos - 1] in LINE_END_PROHIBITED:
            return False
        return True

    suffix_caps = [0] * (k + 1)
    for i in range(k - 1, -1, -1):
        suffix_caps[i] = suffix_caps[i + 1] + caps[i]

    memo: dict[tuple[int, int], tuple[float, list[int]] | None] = {}

    def solve(char_idx: int, col_idx: int) -> tuple[float, list[int]] | None:
        state = (char_idx, col_idx)
        if state in memo:
            return memo[state]

        rem_chars = n - char_idx
        rem_cap = suffix_caps[col_idx]
        if rem_chars > rem_cap:
            memo[state] = None
            return None

        if col_idx == k - 1:
            if 0 < rem_chars <= caps[col_idx]:
                ideal = n / k
                cost = (rem_chars - ideal) ** 2
                res = (cost, [n])
                memo[state] = res
                return res
            memo[state] = None
            return None

        rem_cols = k - col_idx
        ideal = rem_chars / rem_cols
        best: tuple[float, list[int]] | None = None

        min_take = max(1, rem_chars - suffix_caps[col_idx + 1])
        max_take = min(caps[col_idx], rem_chars - (rem_cols - 1))
        candidates = sorted(range(min_take, max_take + 1), key=lambda t: abs(t - ideal))

        # Pass 1: strictly valid breakpoints (avoid Latin word break & Kinsoku rules)
        for take in candidates:
            next_pos = char_idx + take
            if not can_break_at(next_pos):
                continue
            sub = solve(next_pos, col_idx + 1)
            if sub is not None:
                cost = (take - ideal) ** 2 + sub[0]
                if best is None or cost < best[0]:
                    best = (cost, [next_pos] + sub[1])

        # Pass 2: soft breaks (still strictly avoiding Latin word breaks)
        if best is None:
            for take in candidates:
                next_pos = char_idx + take
                if is_latin_word_char(chars[next_pos - 1]) and is_latin_word_char(chars[next_pos]):
                    continue
                sub = solve(next_pos, col_idx + 1)
                if sub is not None:
                    cost = 1000.0 + (take - ideal) ** 2 + sub[0]
                    if best is None or cost < best[0]:
                        best = (cost, [next_pos] + sub[1])

        # Pass 3: fallback if impossible
        if best is None:
            for take in candidates:
                next_pos = char_idx + take
                sub = solve(next_pos, col_idx + 1)
                if sub is not None:
                    cost = 10000.0 + (take - ideal) ** 2 + sub[0]
                    if best is None or cost < best[0]:
                        best = (cost, [next_pos] + sub[1])

        memo[state] = best
        return best

    res = solve(0, 0)
    if res is None:
        return None

    breaks = res[1]
    chunks: list[str] = []
    prev = 0
    for b in breaks:
        chunks.append("".join(chars[prev:b]))
        prev = b
    return chunks


def _score_vertical_layout(size: int, max_size: int, placements: list[tuple[str, int, int]]) -> float:
    if not placements:
        return -1.0
    col_lengths = [len(p[0]) for p in placements]
    num_cols = len(col_lengths)
    score = 1.0 * (size / max(1, max_size))

    if num_cols == 1:
        score += 0.25
    elif num_cols >= 2:
        # In vertical text, never split short phrases (<= 3 chars) across multiple columns
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


def detect_bubble_lobes(mask: np.ndarray, min_radius: float = 12.0) -> list[dict[str, object]]:
    """Decompose an arbitrary binary bubble mask into N natural constituent lobes/chambers.

    Uses Euclidean Distance Transform (EDT) and local peak clustering to discover
    the natural geometric centers and inscribed radii of all constituent bubble lobes.
    """
    if mask.ndim != 2 or np.count_nonzero(mask) == 0:
        return []

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_d = float(np.max(dist))
    if max_d < min_radius:
        ys, xs = np.where(mask > 0)
        return [{
            "center": (int(np.mean(xs)), int(np.mean(ys))),
            "radius": max_d,
            "bbox": (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)) + 1, int(np.max(ys)) + 1),
            "mask": mask,
        }]

    ksize = max(15, int(min_radius * 1.5))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(dist, kernel)

    peak_mask = (dist == dilated) & (dist >= min_radius)
    ys, xs = np.where(peak_mask)
    if len(xs) == 0:
        max_y, max_x = np.unravel_index(np.argmax(dist), dist.shape)
        xs, ys = np.array([max_x]), np.array([max_y])

    candidates = sorted([(int(x), int(y), float(dist[y, x])) for x, y in zip(xs, ys)], key=lambda p: -p[2])

    suppressed_peaks: list[tuple[int, int, float]] = []
    for x, y, r in candidates:
        is_distinct = True
        for ex, ey, er in suppressed_peaks:
            d = np.hypot(x - ex, y - ey)
            if d < 0.70 * min(r, er):
                is_distinct = False
                break
            # Geometric constriction (neck) check:
            # Two peaks are distinct lobes IF AND ONLY IF there is a constriction (isthmus/neck)
            # along the path between them where the distance to the boundary drops significantly.
            # If the path between them remains wide (no constriction), they are part of the SAME
            # continuous elongated chamber and must not be sliced into separate lobes.
            num_samples = max(10, int(d))
            s_xs = np.clip(np.linspace(x, ex, num_samples).round().astype(int), 0, mask.shape[1] - 1)
            s_ys = np.clip(np.linspace(y, ey, num_samples).round().astype(int), 0, mask.shape[0] - 1)
            min_path_d = float(np.min(dist[s_ys, s_xs]))
            if min_path_d >= 0.72 * min(r, er):
                is_distinct = False
                break
        if is_distinct:
            suppressed_peaks.append((x, y, r))

    if len(suppressed_peaks) <= 1:
        ys, xs = np.where(mask > 0)
        cx, cy, r = suppressed_peaks[0] if suppressed_peaks else (int(np.mean(xs)), int(np.mean(ys)), max_d)
        return [{
            "center": (cx, cy),
            "radius": r,
            "bbox": (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)) + 1, int(np.max(ys)) + 1),
            "mask": mask,
        }]

    # Multi-lobe Voronoi chamber partition
    partitions = partition_mask_by_centers(mask, [(float(px), float(py)) for px, py, _ in suppressed_peaks])
    lobes: list[dict[str, object]] = []
    for idx, (px, py, pr) in enumerate(suppressed_peaks):
        part = partitions[idx]
        if part is None:
            continue
        (lx, ly, rx, by), cropped_lobe = part
        lobe_mask = np.zeros_like(mask, dtype=np.uint8)
        lobe_mask[ly:by, lx:rx] = cropped_lobe
        lobes.append({
            "center": (px, py),
            "radius": pr,
            "bbox": (lx, ly, rx, by),
            "mask": lobe_mask,
        })

    # Sort lobes in vertical-RTL reading order: top-to-bottom bands, then right-to-left
    avg_r = float(np.mean([float(l["radius"]) for l in lobes]))
    band_h = max(20.0, avg_r * 1.5)
    lobes.sort(key=lambda l: reading_order_sort_key(l["center"][0], l["center"][1], band_h))

    return lobes


def _vertical_mask_layout(
    mask: np.ndarray,
    text_or_sentences: str | list[str],
    font_or_size: ImageFont.FreeTypeFont | int,
    padding: int,
    anchor: tuple[int, int] = (0, 0),
    allow_lobe_decomposition: bool = True,
) -> list[tuple[str, int, int]] | None:
    size = font_or_size.size if hasattr(font_or_size, "size") else int(font_or_size)
    if isinstance(text_or_sentences, list):
        sentences = text_or_sentences
    else:
        if not text_or_sentences.strip():
            return []
        segments = segment_text(text_or_sentences)
        sentences = [s.text.strip() for s in segments if s.text.strip()]

    if not sentences:
        return []

    # 1. Multi-lobe compound bubble decomposition (Distance Transform Local Maxima & Watershed)
    if allow_lobe_decomposition and len(sentences) >= 2:
        lobes = detect_bubble_lobes(mask, min_radius=max(12.0, float(size) * 0.6))
        if len(lobes) >= 2:
            n_lobes = len(lobes)
            m_sents = len(sentences)
            if m_sents <= n_lobes:
                sents_per_lobe = [[sentences[i]] if i < m_sents else [] for i in range(n_lobes)]
            else:
                lobe_areas = [max(1.0, float(np.count_nonzero(l["mask"]))) for l in lobes]
                total_area = sum(lobe_areas)
                sent_lens = [max(1, len([c for c in s if not c.isspace()])) for s in sentences]
                total_chars = sum(sent_lens)

                sents_per_lobe = [[] for _ in range(n_lobes)]
                curr_l = 0
                curr_chars = 0
                target_chars = total_chars * (lobe_areas[0] / total_area)

                for s_idx, s in enumerate(sentences):
                    s_len = sent_lens[s_idx]
                    rem_sents = len(sentences) - s_idx
                    rem_lobes = n_lobes - curr_l
                    if curr_l < n_lobes - 1:
                        if curr_chars > 0 and (curr_chars + s_len * 0.5 > target_chars) and rem_sents >= rem_lobes:
                            curr_l += 1
                            curr_chars = 0
                            target_chars = total_chars * (lobe_areas[curr_l] / total_area)
                    sents_per_lobe[curr_l].append(s)
                    curr_chars += s_len

            placements: list[tuple[str, int, int]] = []
            all_succeeded = True
            for lobe_idx, lobe in enumerate(lobes):
                lobe_sents = sents_per_lobe[lobe_idx]
                if not lobe_sents:
                    continue
                lobe_mask = lobe["mask"]
                lx, ly, rx, by = lobe["bbox"]
                cropped_mask = lobe_mask[ly:by, lx:rx]
                sub_placements = _vertical_mask_layout(
                    cropped_mask,
                    lobe_sents,
                    size,
                    padding=padding,
                    allow_lobe_decomposition=False,
                )
                if sub_placements is None:
                    all_succeeded = False
                    break
                for chunk, cx, cy in sub_placements:
                    placements.append((chunk, lx + cx, ly + cy))

            if all_succeeded and placements:
                return placements

    # 2. Single-lobe or unified bubble layout
    column_ranges = _mask_column_ranges(mask, size, padding)
    if not column_ranges:
        return None

    # Keep the best span for each unique column X
    best_spans: dict[int, tuple[int, int]] = {}
    for x, top, bottom in sorted(column_ranges, key=lambda c: -c[0]):
        if (bottom - top) >= size:
            if x not in best_spans or (bottom - top) > (best_spans[x][1] - best_spans[x][0]):
                best_spans[x] = (top, bottom)

    raw_cols = [(x, top, bottom, (bottom - top) // size) for x, (top, bottom) in best_spans.items()]
    if not raw_cols:
        return None

    ordered_cols = sorted(raw_cols, key=lambda c: -c[0])

    m_sents = len(sentences)
    total_cols = len(ordered_cols)
    if total_cols < m_sents:
        return None

    # Proportional column allocation
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

    groups: list[list[tuple[int, int, int, int]]] = []
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
        # Short phrases (<= 4 chars) must always stay in 1 column
        clean_non_space = [c for c in chars if not c.isspace()]
        if len(clean_non_space) <= 4 and any(c.isalnum() for c in clean_non_space):
            chars = clean_non_space
            n_chars = len(chars)
        max_k = 1 if n_chars <= 4 else len(group)
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

        for col_data, chunk in zip(cand, chunks):
            x, top, bottom, _ = col_data
            col_h = len(chunk) * size
            start_y = top + (bottom - top - col_h) // 2
            placements.append((chunk, x, start_y))

    return placements


def _largest_vertical_bbox_font_size(text: str, maximum: int, minimum: int, width: int, height: int) -> int | None:
    """Find largest vertical font size that fits characters into a rectangular bounding box."""
    characters = [character for character in text if character != "\n"]
    if not characters:
        return None
    low = minimum
    high = maximum
    best: int | None = None
    while low <= high:
        size = (low + high) // 2
        rows = max(1, height // size)
        cols = (len(characters) + rows - 1) // rows
        if cols * size <= width and size <= height:
            best = size
            low = size + 1
        else:
            high = size - 1
    return best



