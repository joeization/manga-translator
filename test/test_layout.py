from __future__ import annotations

import unittest
from pathlib import Path

from PIL import ImageFont

from src.renderer.layout import (
    BreakPriority,
    TextAutoLayout,
    segment_text,
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


if __name__ == "__main__":
    unittest.main()

