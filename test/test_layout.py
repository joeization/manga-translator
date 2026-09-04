from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import ImageFont

from src.renderer.layout import (
    BreakPriority,
    TextAutoLayout,
    _score_vertical_layout,
    _vertical_mask_layout,
    segment_text,
    split_sentences,
)


class TextLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.font_path = Path("C:/Windows/Fonts/NotoSansCJKtc-Bold.otf")
        if not self.font_path.exists():
            self.font_path = Path("C:/Windows/Fonts/msjh.ttc")

    def _font_resolver(self, size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(str(self.font_path), size)
        except Exception:
            return ImageFont.load_default()

    def test_segment_text_sentence_and_clauses(self) -> None:
        text = "AAA！BBB，CCC。\nDDD。"
        segments = segment_text(text)

        self.assertTrue(len(segments) >= 4)
        # Check first segment: "AAA！" with SENTENCE break
        self.assertEqual(segments[0].text, "AAA！")
        self.assertEqual(segments[0].break_after, BreakPriority.SENTENCE)

        # Check clause segment: "BBB，" with CLAUSE break
        self.assertEqual(segments[1].text, "BBB，")
        self.assertEqual(segments[1].break_after, BreakPriority.CLAUSE)

        # Check paragraph break after "CCC。"
        self.assertEqual(segments[2].text, "CCC。")
        self.assertEqual(segments[2].break_after, BreakPriority.PARAGRAPH)

        # Check second paragraph
        self.assertEqual(segments[3].text, "DDD。")

    def test_segment_text_closing_quote_attachment(self) -> None:
        text = "「AAA！」BBB。"
        segments = segment_text(text)

        # The closing quote must stay attached to the exclamation mark
        self.assertEqual(segments[0].text, "「AAA！」")
        self.assertEqual(segments[0].break_after, BreakPriority.SENTENCE)

    def test_segment_text_english_words_and_spaces(self) -> None:
        text = "AAA BBB! CCC DDD, EEE."
        segments = segment_text(text)

        texts = [s.text for s in segments]
        self.assertIn("AAA ", texts)
        self.assertIn("BBB! ", texts)
        self.assertIn("DDD, ", texts)

    def test_layout_keeps_complete_sentence_together_when_space_permits(self) -> None:
        text = "AAAA！BBBB。"
        layout_engine = TextAutoLayout(text, orientation="horizontal", font_resolver=self._font_resolver)
        result = layout_engine.find_optimal_layout(
            available_width=600,
            available_height=200,
            max_font_size=40,
            preferred_font_size=32,
            min_font_size=16,
        )

        self.assertIsNotNone(result)
        assert result is not None
        # Should fit on 1 or 2 lines without breaking inside words
        self.assertLessEqual(len(result.lines), 2)
        # First sentence should not be cut in half
        self.assertTrue(any("AAAA！" in line for line in result.lines))

    def test_layout_prefers_natural_sentence_break_over_arbitrary_split(self) -> None:
        text = "AAAAAAA！BBBBBBB。"
        layout_engine = TextAutoLayout(text, orientation="horizontal", font_resolver=self._font_resolver)
        result = layout_engine.find_optimal_layout(
            available_width=250,
            available_height=300,
            max_font_size=24,
            preferred_font_size=24,
            min_font_size=12,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.lines), 2)
        self.assertEqual(result.lines[0], "AAAAAAA！")
        self.assertEqual(result.lines[1], "BBBBBBB。")

    def test_layout_vertical_rtl_orientation(self) -> None:
        text = "AAA！BBB。"
        layout_engine = TextAutoLayout(text, orientation="vertical-rtl", font_resolver=self._font_resolver)
        result = layout_engine.find_optimal_layout(
            available_width=150,
            available_height=200,
            max_font_size=36,
            preferred_font_size=30,
            min_font_size=14,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(len(result.lines) >= 1)
        self.assertLessEqual(result.total_width, 150)
        self.assertLessEqual(result.total_height, 200)

    def test_layout_falls_back_to_character_split_and_smaller_font_when_narrow(self) -> None:
        text = "AAAAAAAAAAAAAAAAAAAAAAA"
        layout_engine = TextAutoLayout(text, orientation="horizontal", font_resolver=self._font_resolver)
        result = layout_engine.find_optimal_layout(
            available_width=80,
            available_height=300,
            max_font_size=32,
            preferred_font_size=24,
            min_font_size=12,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(len(result.lines) > 2)
        self.assertLessEqual(result.total_width, 80)
        self.assertLessEqual(result.total_height, 300)

    def test_detect_bubble_lobes_single_oval(self) -> None:
        import cv2
        import numpy as np
        from src.renderer.layout import detect_bubble_lobes

        mask = np.zeros((100, 80), dtype=np.uint8)
        cv2.ellipse(mask, (40, 50), (30, 40), 0, 0, 360, 1, -1)

        lobes = detect_bubble_lobes(mask, min_radius=15.0)
        self.assertEqual(len(lobes), 1)
        self.assertAlmostEqual(lobes[0]["center"][0], 40, delta=3)
        self.assertAlmostEqual(lobes[0]["center"][1], 50, delta=3)

    def test_detect_bubble_lobes_two_diagonal_lobes(self) -> None:
        import cv2
        import numpy as np
        from src.renderer.layout import detect_bubble_lobes

        # Compound bubble: Top-Right lobe and Bottom-Left lobe merged together
        mask = np.zeros((200, 160), dtype=np.uint8)
        cv2.circle(mask, (110, 50), 35, 1, -1)  # Top-Right
        cv2.circle(mask, (50, 140), 45, 1, -1)  # Bottom-Left
        # Overlapping neck connecting them
        cv2.line(mask, (110, 50), (50, 140), 1, thickness=20)

        lobes = detect_bubble_lobes(mask, min_radius=20.0)
        self.assertEqual(len(lobes), 2)
        # Reading order: Top-Right lobe first, Bottom-Left lobe second
        self.assertGreater(lobes[0]["center"][0], lobes[1]["center"][0])
        self.assertLess(lobes[0]["center"][1], lobes[1]["center"][1])

    def test_detect_bubble_lobes_three_chambers_l_shape(self) -> None:
        import cv2
        import numpy as np
        from src.renderer.layout import detect_bubble_lobes

        # 3 lobes connected: Lobe 1 (Top-Right), Lobe 2 (Middle-Left), Lobe 3 (Bottom-Right)
        mask = np.zeros((300, 200), dtype=np.uint8)
        cv2.circle(mask, (140, 50), 35, 1, -1)
        cv2.circle(mask, (60, 150), 35, 1, -1)
        cv2.circle(mask, (140, 240), 35, 1, -1)
        # Connect with necks
        cv2.line(mask, (140, 50), (60, 150), 1, thickness=20)
        cv2.line(mask, (60, 150), (140, 240), 1, thickness=20)

        lobes = detect_bubble_lobes(mask, min_radius=20.0)
        self.assertEqual(len(lobes), 3)

    def test_vertical_mask_layout_three_lobes(self) -> None:
        import cv2
        import numpy as np
        from src.renderer.layout import _vertical_mask_layout

        # 3 lobes connected in a compound bubble
        mask = np.zeros((300, 200), dtype=np.uint8)
        cv2.circle(mask, (140, 50), 35, 1, -1)
        cv2.circle(mask, (60, 150), 35, 1, -1)
        cv2.circle(mask, (140, 240), 35, 1, -1)
        cv2.line(mask, (140, 50), (60, 150), 1, thickness=20)
        cv2.line(mask, (60, 150), (140, 240), 1, thickness=20)

        sentences = ["A!?", "BBBB？", "CCCC！"]
        placements = _vertical_mask_layout(mask, sentences, 18, padding=4)
        self.assertIsNotNone(placements)
        assert placements is not None
        self.assertTrue(len(placements) >= 3)
        # Ensure text is distributed across the different vertical chambers
        ys = [cy for _, _, cy in placements]
        self.assertLess(min(ys), 90)
        self.assertGreater(max(ys), 180)


    def test_mixed_script_vertical_layout_preserves_latin_words(self) -> None:
        """CJK text containing Latin words must not break inside the Latin word under vertical layout."""
        word = "WORD"
        text = f"甲甲甲甲甲甲甲甲甲甲甲甲甲甲，乙乙乙乙乙{word}丙丙丙丙"
        layout = TextAutoLayout(text, orientation="vertical-rtl", font_resolver=self._font_resolver)
        res = layout.find_optimal_layout(available_width=440, available_height=300, min_font_size=12, preferred_font_size=28, max_font_size=72)
        self.assertIsNotNone(res)
        assert res is not None
        # Check that Latin word is never split across lines
        for line in res.lines:
            if word in line:
                continue
            for prefix_len in range(1, len(word)):
                prefix = word[:prefix_len]
                self.assertFalse(
                    line.endswith(prefix),
                    f"Line '{line}' split '{word}' at '{prefix}'",
                )

    def test_split_chars_by_capacity_preserves_latin_words(self) -> None:
        """_split_chars_by_capacity must not slice across Latin word boundaries."""
        from src.renderer.layout import _split_chars_by_capacity
        word = "WORD"
        text = f"甲甲甲甲甲甲甲甲甲甲甲甲甲甲，乙乙乙乙乙{word}丙丙丙丙"
        chars = list(text)
        caps = [8, 8, 8, 8, 8, 8]
        chunks = _split_chars_by_capacity(chars, caps)
        self.assertIsNotNone(chunks)
        assert chunks is not None
        for chunk in chunks:
            if word in chunk:
                continue
            for prefix_len in range(1, len(word)):
                prefix = word[:prefix_len]
                self.assertFalse(
                    chunk.endswith(prefix),
                    f"Chunk '{chunk}' split '{word}' at '{prefix}'",
                )

    def test_segment_text_bracket_ignoring_punctuation(self) -> None:
        """Punctuation inside brackets/quotes must be ignored for segmentation."""
        # 1. Punctuation inside quotes is not split
        text1 = "「AAA！BBB」CCC。"
        segs1 = segment_text(text1)
        self.assertEqual(len(segs1), 1)
        self.assertEqual(segs1[0].text, "「AAA！BBB」CCC。")
        self.assertEqual(segs1[0].break_after, BreakPriority.SENTENCE)

        # 2. Punctuation inside parentheses is not split
        text2 = "（AAA，BBB？CCC）DDD。"
        segs2 = segment_text(text2)
        self.assertEqual(len(segs2), 1)
        self.assertEqual(segs2[0].text, "（AAA，BBB？CCC）DDD。")
        self.assertEqual(segs2[0].break_after, BreakPriority.SENTENCE)

        # 3. Nested brackets ending in punctuation
        text3 = "（「AAA！」）BBB。"
        segs3 = segment_text(text3)
        self.assertEqual(len(segs3), 2)
        self.assertEqual(segs3[0].text, "（「AAA！」）")
        self.assertEqual(segs3[0].break_after, BreakPriority.SENTENCE)
        self.assertEqual(segs3[1].text, "BBB。")
        self.assertEqual(segs3[1].break_after, BreakPriority.SENTENCE)

    def test_segment_text_unicode_symbols_weak_priority(self) -> None:
        """Unicode symbols and ambiguous punctuation outside brackets get WEAK priority."""
        text = "AAA~ BBB♥ CCC★ DDD"
        segs = segment_text(text)
        # ~ is WEAK
        self.assertEqual(segs[0].text, "AAA~ ")
        self.assertEqual(segs[0].break_after, BreakPriority.WEAK)
        # ♥ is WEAK
        self.assertEqual(segs[1].text, "BBB♥ ")
        self.assertEqual(segs[1].break_after, BreakPriority.WEAK)
        # ★ is WEAK
        self.assertEqual(segs[2].text, "CCC★ ")
        self.assertEqual(segs[2].break_after, BreakPriority.WEAK)

    def test_split_sentences_prefers_strong_punctuation(self) -> None:
        """split_sentences prefers strong punctuation over weak symbols."""
        # Strong punctuation present: weak symbols inside sentence do not split
        sents1 = split_sentences("AAA~ BBB。")
        self.assertEqual(sents1, ["AAA~ BBB。"])

        # Punctuation inside quotes ignored: forms single sentence
        sents2 = split_sentences("「AAA！BBB」CCC。")
        self.assertEqual(sents2, ["「AAA！BBB」CCC。"])

        # Quote ends sentence: splits into 2 sentences
        sents3 = split_sentences("「AAA！」BBB。")
        self.assertEqual(sents3, ["「AAA！」", "BBB。"])

        # Only weak symbols present: splits at weak symbols
        sents4 = split_sentences("AAA~ BBB~ CCC~")
        self.assertEqual(sents4, ["AAA~", "BBB~", "CCC~"])

        sents5 = split_sentences("AAA♥ BBB★ CCC")
        self.assertEqual(sents5, ["AAA♥", "BBB★", "CCC"])

        # Clause punctuation without strong punctuation
        sents6 = split_sentences("AAA，BBB")
        self.assertEqual(sents6, ["AAA，", "BBB"])

        # Decimal numbers and apostrophes not broken
        sents7 = split_sentences("AAA 3.14 BBB.")
        self.assertEqual(sents7, ["AAA 3.14 BBB."])

        sents8 = split_sentences("WORD1's AAA! WORD2 don't BBB.")
        self.assertEqual(sents8, ["WORD1's AAA!", "WORD2 don't BBB."])

    def test_score_vertical_layout_prefers_unbroken_sentences(self) -> None:
        """Unbroken layout (e.g. 2 sentences in 2 columns) scores higher than 3 columns even if 3 columns has larger font."""
        score_unbroken = _score_vertical_layout(22, 30, [("AAAAAA", 90, 10), ("BBBBBB", 60, 10)], num_sentences=2)
        score_broken = _score_vertical_layout(26, 30, [("AAAAAA", 90, 10), ("BBB", 60, 10), ("BBB", 30, 10)], num_sentences=2)
        self.assertGreater(score_unbroken, score_broken)

    def test_vertical_mask_layout_maximizes_font_size_unbroken(self) -> None:
        """_vertical_mask_layout keeps sentences unbroken when they fit in 1 column each."""
        mask = np.ones((180, 120), dtype=np.uint8)
        sentences = ["AAAAAA", "BBBBBB"]
        placements = _vertical_mask_layout(mask, sentences, font_or_size=22, padding=4, allow_sentence_split=False)
        self.assertIsNotNone(placements)
        self.assertEqual(len(placements), 2)
        self.assertEqual(placements[0][0], "AAAAAA")
        self.assertEqual(placements[1][0], "BBBBBB")

    def test_text_auto_layout_maximizes_font_size_without_intra_sentence_breaks(self) -> None:
        """TextAutoLayout prioritizes keeping each sentence in a single vertical column without line breaks."""
        text = "AAAAAA！BBBBBB！"
        layout_engine = TextAutoLayout(text, orientation="vertical-rtl", font_resolver=self._font_resolver)
        result = layout_engine.find_optimal_layout(
            available_width=120,
            available_height=180,
            max_font_size=40,
            preferred_font_size=32,
            min_font_size=12,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result.lines), 2)
        self.assertEqual(result.lines[0], "AAAAAA！")
        self.assertEqual(result.lines[1], "BBBBBB！")


if __name__ == "__main__":
    unittest.main()


