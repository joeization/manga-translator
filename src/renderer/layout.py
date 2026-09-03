from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import re
from typing import Callable

from PIL import ImageFont


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

# Universal Kinsoku Shori rules (JIS X 4051 / W3C Requirements for Chinese & Japanese Text Layout)
# Universal Kinsoku Shori line-breaking rules (W3C / international standards for East Asian typography)
LINE_START_PROHIBITED = CLOSING_PUNCTUATION | SENTENCE_TERMINALS | set(
    "，、；;：:,.…︙︱︲︴—―-~～ーｰぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ"
)
LINE_END_PROHIBITED = OPENING_PUNCTUATION



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

        # In vertical layout, never split short phrases (<= 3 chars) across multiple columns
        if self.orientation == "vertical-rtl":
            if total_chars <= 3 or (total_chars / num_chunks) < 1.5:
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

            chunk = "".join(chars[cur_idx:next_idx])
            while len(chunk) > 1 and measure_func(chunk) > line_limit:
                next_idx -= 1
                while next_idx > cur_idx + 1 and chars[next_idx - 1] in LINE_END_PROHIBITED:
                    next_idx -= 1
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

