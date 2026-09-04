from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.models import TextRegion
from src.renderer.pillow_renderer import MaskAwarePillowRenderer, PillowRenderer


class RendererBinarySearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.font_path = Path("C:/Windows/Fonts/arial.ttf")
        if not self.font_path.exists():
            self.skipTest("System font arial.ttf not found")

    def test_horizontal_renderer_binary_search_fits_text(self) -> None:
        renderer = PillowRenderer(
            font_path=self.font_path,
            font_size=40,
            max_font_size=72,
            min_font_size=12,
            padding=4,
            text_direction="horizontal",
        )
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        region = TextRegion(
            bbox=(20, 20, 180, 80),
            source_text="AAA BBB",
            translated_text="AAA BBB",
        )
        rendered = renderer.render(img, [region])
        rendered_np = np.array(rendered)
        # Should not be pure white
        self.assertFalse(np.all(rendered_np == 255))

    def test_mask_aware_vertical_renderer_binary_search(self) -> None:
        renderer = MaskAwarePillowRenderer(
            font_path=self.font_path,
            font_size=40,
            max_font_size=60,
            min_font_size=12,
            padding=4,
            text_direction="vertical-rtl",
        )
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        mask = np.ones((100, 60), dtype=bool)
        region = TextRegion(
            bbox=(50, 50, 110, 150),
            source_text="ABC",
            translated_text="ABC",
            layout_bbox=(50, 50, 110, 150),
            layout_mask=mask,
        )
        rendered = renderer.render(img, [region])
        rendered_np = np.array(rendered)
        # Should have text drawn
    def test_wide_region_vertical_columns_distributed(self) -> None:
        from src.renderer.pillow_renderer import _draw_vertical_columns
        from PIL import ImageDraw, ImageFont

        font = ImageFont.truetype(str(self.font_path), 30)
        img = Image.new("RGB", (400, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # 3 columns in a 400px wide box
        columns = ["A", "B", "C"]
        _draw_vertical_columns(draw, columns, font, left=0, top=0, width=400, height=100)

        arr = np.array(img)
        # Check that text ink exists in both left half (x < 200) and right half (x > 200)
        has_ink_left = np.any(arr[:, :200] < 200)
        has_ink_right = np.any(arr[:, 200:] < 200)
        self.assertTrue(has_ink_left, "Left side should contain distributed column")
        self.assertTrue(has_ink_right, "Right side should contain distributed column")

    def test_single_column_in_wide_region_centered(self) -> None:
        from src.renderer.pillow_renderer import _draw_vertical_columns
        from PIL import ImageDraw, ImageFont

        font = ImageFont.truetype(str(self.font_path), 30)
        img = Image.new("RGB", (400, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # 1 column in a 400px wide box
        columns = ["A"]
        _draw_vertical_columns(draw, columns, font, left=0, top=0, width=400, height=100)

        arr = np.array(img)
        # Ink should be near the center (150 < x < 250), not glued to the right edge (x > 350)
        has_ink_center = np.any(arr[:, 170:230] < 200)
        has_ink_far_right = np.any(arr[:, 360:] < 200)
        self.assertTrue(has_ink_center, "Single column should be centered horizontally")
        self.assertFalse(has_ink_far_right, "Single column should not be glued to the far right edge")

    def test_wide_region_font_size_scales_up_to_max(self) -> None:
        from src.renderer.pillow_renderer import _largest_vertical_bbox_font_size

        # In a wide box (500x120), a 3-character text should be able to scale up to 72
        size = _largest_vertical_bbox_font_size("ABC", maximum=72, minimum=12, width=500, height=120)
        self.assertIsNotNone(size)
        assert size is not None
        self.assertGreaterEqual(size, 60, f"Font size {size} should scale up in a wide box")

    def test_expand_bbox_towards_layout(self) -> None:
        from src.renderer.pillow_renderer import _expand_bbox_towards_layout

        # Small text box (60x100) inside large bubble (200x300)
        text_bbox = (70, 100, 130, 200)
        layout_bbox = (0, 0, 200, 300)
        expanded = _expand_bbox_towards_layout(text_bbox, layout_bbox)

        # Expanded box should be larger than original text_bbox
        self.assertLess(expanded[0], text_bbox[0])
        self.assertLess(expanded[1], text_bbox[1])
        self.assertGreater(expanded[2], text_bbox[2])
        self.assertGreater(expanded[3], text_bbox[3])
        # Expanded box must stay within layout_bbox
        self.assertGreaterEqual(expanded[0], layout_bbox[0])
        self.assertGreaterEqual(expanded[1], layout_bbox[1])
        self.assertLessEqual(expanded[2], layout_bbox[2])
        self.assertLessEqual(expanded[3], layout_bbox[3])

    def test_mask_aware_renderer_adapts_to_large_bubble(self) -> None:
        # Verify that MaskAwarePillowRenderer successfully renders text
        # in a small bbox inside a large segmentation bubble with larger font
        renderer = MaskAwarePillowRenderer(
            font_path=self.font_path,
            font_size=48,
            max_font_size=72,
            min_font_size=12,
            padding=4,
            text_direction="vertical-rtl",
        )
        img = Image.new("RGB", (300, 400), (255, 255, 255))
        # 200x300 elliptical bubble
        mask = np.zeros((300, 200), dtype=bool)
        import cv2
        cv2.ellipse(mask.view(np.uint8), (100, 150), (90, 140), 0, 0, 360, 1, -1)

        # Small text box (50x80) inside the 200x300 bubble
        reg = TextRegion(
            bbox=(125, 160, 175, 240),
            source_text="A",
            translated_text="AAAAAAAAAAAAAA",
            layout_bbox=(50, 50, 250, 350),
            layout_mask=mask,
        )
        rendered = renderer.render(img, [reg])
        rendered_np = np.array(rendered)
        # Should have rendered text
        self.assertFalse(np.all(rendered_np == 255))

    def test_multi_region_expansion_never_overlaps(self) -> None:
        from src.renderer.pillow_renderer import compute_non_overlapping_render_bounds

        # Two side-by-side text regions inside the same speech bubble
        # Original: reg1 is from x=50 to 120, reg2 is from x=120 to 180
        # Both share bubble layout_bbox (0, 0, 250, 300)
        bubble = (0, 0, 250, 300)
        reg1 = TextRegion(bbox=(50, 50, 120, 250), source_text="AAA", translated_text="AAA", layout_bbox=bubble)
        reg2 = TextRegion(bbox=(120, 50, 180, 250), source_text="BBB", translated_text="BBB", layout_bbox=bubble)

        bounds = compute_non_overlapping_render_bounds([reg1, reg2])
        self.assertEqual(len(bounds), 2)
        b1, b2 = bounds[0], bounds[1]

        # Verify that b1 and b2 do NOT overlap in 2D space
        inter_w = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        inter_h = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        self.assertEqual(inter_w * inter_h, 0, f"Render bounds {b1} and {b2} must not overlap")

    def test_maximum_inscribed_rectangle(self) -> None:
        from src.renderer.pillow_renderer import find_maximum_inscribed_rectangle

        # Create a 200x200 mask with a 100x120 rectangle of 1s in the center
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[40:160, 50:150] = 1

        rect = find_maximum_inscribed_rectangle(mask)
        self.assertIsNotNone(rect)
        assert rect is not None
        rx, ry, rw, rh = rect
        self.assertEqual((rx, ry, rw, rh), (50, 40, 100, 120))

    def test_column_strip_layout_figure_eight(self) -> None:
        import cv2
        from PIL import ImageFont
        from src.renderer.pillow_renderer import _vertical_mask_layout

        # 300x300 canvas with a 45-degree figure-8
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (200, 100), 60, 1, -1)
        cv2.circle(mask, (100, 200), 60, 1, -1)
        cv2.line(mask, (200, 100), (100, 200), 1, 45)

        # 2 sentences
        text = "AAAAAAA！BBBBBBB。"
        font = ImageFont.truetype(str(self.font_path), 20)
        placements = _vertical_mask_layout(mask, text, font, padding=4, anchor=(150, 150))
        self.assertIsNotNone(placements)
        assert placements is not None
        self.assertGreaterEqual(len(placements), 2)
        # Sentence 1 should be placed in columns with higher X (on the right)
        # Sentence 2 should be placed in columns with lower X (on the left)
        x_coords = [p[1] for p in placements]
        self.assertGreater(x_coords[0], x_coords[-1], "Sentence 1 should be to the right of Sentence 2")

    def test_column_strip_layout_three_bubbles(self) -> None:
        import cv2
        from PIL import ImageFont
        from src.renderer.pillow_renderer import _vertical_mask_layout

        # 400x400 canvas with a 3-bubble chain (N=3)
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(mask, (300, 100), 50, 1, -1)
        cv2.circle(mask, (200, 200), 50, 1, -1)
        cv2.circle(mask, (100, 300), 50, 1, -1)
        cv2.line(mask, (300, 100), (200, 200), 1, 35)
        cv2.line(mask, (200, 200), (100, 300), 1, 35)

        # 3 sentences
        text = "AAA。BBB。CCC。"
        font = ImageFont.truetype(str(self.font_path), 18)
        placements = _vertical_mask_layout(mask, text, font, padding=4, anchor=(200, 200))
        self.assertIsNotNone(placements)
        assert placements is not None
        self.assertGreaterEqual(len(placements), 3)
        x_coords = [p[1] for p in placements]
        self.assertGreater(x_coords[0], x_coords[-1], "Sentence 1 should be to the right of Sentence 3")

    def test_column_strip_layout_spreads_across_all_columns(self) -> None:
        from PIL import ImageFont
        from src.renderer.pillow_renderer import _vertical_mask_layout

        # A mask that fits 6 columns of font size 20 (width = 130, height = 100)
        mask = np.ones((100, 130), dtype=np.uint8)

        # 3 sentences
        text = "AAA。BBB。CCC。"
        font = ImageFont.truetype(str(self.font_path), 20)
        placements = _vertical_mask_layout(mask, text, font, padding=4, anchor=(65, 50))
        self.assertIsNotNone(placements)
        assert placements is not None
        x_coords = [p[1] for p in placements]
        # Check that columns spread across the space, with the last column near the left boundary
        self.assertLess(min(x_coords), 50, "Columns should spread across to the left side")
        self.assertGreater(max(x_coords), 80, "Columns should start on the right side")

    def test_diagonal_bubble_uses_top_right_and_bottom_left(self) -> None:
        import cv2
        from PIL import ImageFont
        from src.renderer.pillow_renderer import _vertical_mask_layout

        # 300x300 canvas with a 45-degree figure-8
        # Upper lobe at (200, 100), Lower lobe at (100, 200)
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (200, 100), 60, 1, -1)
        cv2.circle(mask, (100, 200), 60, 1, -1)
        cv2.line(mask, (200, 100), (100, 200), 1, 45)

        # Single long sentence across the diagonal bubble
        text = "AAAAAAAAAAAAAAAAAAAAAA！！"
        font = ImageFont.truetype(str(self.font_path), 20)
        placements = _vertical_mask_layout(mask, text, font, padding=4, anchor=(150, 150))
        self.assertIsNotNone(placements)
        assert placements is not None

        # The rightmost placement should be high up in the top-right lobe (Y < 120)
        rightmost = max(placements, key=lambda p: p[1])
        self.assertLess(rightmost[2], 120, "Rightmost column should be high up in the top-right lobe")

        # The leftmost placement should be down in the bottom-left lobe (Y > 150)
        leftmost = min(placements, key=lambda p: p[1])
        self.assertGreaterEqual(leftmost[2], 140, "Leftmost column should be down in the bottom-left lobe")

    def test_score_vertical_layout_penalizes_orphan_column(self) -> None:
        from src.renderer.pillow_renderer import _score_vertical_layout

        # Layout A: size 30 with [6 chars, 1 char] (orphan in second column)
        score_orphan = _score_vertical_layout(30, 30, [("AAAAAA", 50, 10), ("B", 20, 10)])

        # Layout B: size 26 with [7 chars] in a single clean column
        score_single = _score_vertical_layout(26, 30, [("AAAAAAA", 50, 10)])

        # Layout B should score higher than Layout A despite slightly smaller font size
        self.assertGreater(score_single, score_orphan)

    def test_format_response_in_renderer(self) -> None:
        from src.renderer.pillow_renderer import format_response, to_vertical_text

        self.assertEqual(format_response("...AAA"), "AAA")
        self.assertEqual(format_response("……AAA"), "AAA")
        self.assertEqual(format_response("⋯⋯AAA？"), "AAA？")
        self.assertEqual(to_vertical_text("...「AAA」"), "﹁AAA﹂")
        self.assertEqual(to_vertical_text("AーB"), "A︱B")
        self.assertEqual(to_vertical_text("A|B"), "A︱B")
        self.assertEqual(to_vertical_text("A｜B"), "A︱B")
        self.assertEqual(to_vertical_text("A—B"), "A︱B")
        self.assertEqual(to_vertical_text("A―！"), "A︱！")

    def test_lobe_aware_clustering_assigns_sentences_per_lobe(self) -> None:
        """Two disconnected lobes: each sentence should be in its own lobe."""
        from PIL import ImageFont
        from src.renderer.pillow_renderer import _vertical_mask_layout

        # Top-right bubble: X=[80..160], Y=[0..120]
        # Bottom-left bubble: X=[20..80], Y=[140..260]
        mask = np.zeros((280, 180), dtype=np.uint8)
        mask[0:120, 80:160] = 1   # upper-right lobe
        mask[140:260, 20:80] = 1  # lower-left lobe

        sentences = ["AAA", "BBB"]
        font = ImageFont.truetype(str(self.font_path), 20)
        placements = _vertical_mask_layout(mask, sentences, font, padding=4, anchor=(90, 140))
        self.assertIsNotNone(placements)
        assert placements is not None

        # All placements for the 1st sentence (first 2 chars) should be in the upper-right lobe (X >= 80)
        placed_xs = sorted([p[1] for p in placements], reverse=True)
        # First sentence must land in upper-right lobe (X >= 80)
        self.assertGreaterEqual(placed_xs[0], 80, "First sentence should be in upper-right lobe")
        # Last sentence must land in lower-left lobe (X < 80)
        self.assertLess(placed_xs[-1], 80, "Second sentence should be in lower-left lobe")

    def test_horizontal_rect_auto_detection(self) -> None:
        renderer = PillowRenderer(
            font_path=self.font_path,
            text_direction="vertical-rtl",
        )

        # Wide horizontal banner (aspect 5.0)
        wide_region = TextRegion(bbox=(10, 10, 310, 70), source_text="A")
        self.assertTrue(renderer._is_horizontal_region(wide_region))

        # Vertical bubble (aspect 0.33)
        vert_region = TextRegion(bbox=(10, 10, 60, 160), source_text="A")
        self.assertFalse(renderer._is_horizontal_region(vert_region))

        # Circular bubble (aspect 1.0)
        round_region = TextRegion(bbox=(10, 10, 110, 110), source_text="A")
        self.assertFalse(renderer._is_horizontal_region(round_region))

    def test_horizontal_rect_renders_horizontally_under_vertical_mode(self) -> None:
        renderer = PillowRenderer(
            font_path=self.font_path,
            font_size=24,
            max_font_size=36,
            min_font_size=12,
            padding=4,
            text_direction="vertical-rtl",
        )
        img = Image.new("RGB", (400, 100), (255, 255, 255))
        region = TextRegion(
            bbox=(20, 20, 380, 80),
            source_text="AAAAAAA，BB",
            translated_text="AAAAAAA，BB",
        )
        rendered = renderer.render(img, [region])
        arr = np.array(rendered)
        # Should render ink across the wide horizontal range
        self.assertFalse(np.all(arr == 255))

    def test_ellipsis_breaks_sentences_and_lines(self) -> None:
        from src.renderer.layout import BreakPriority, segment_text, TextAutoLayout

        # ASCII dots
        segs_dots = segment_text("AAAAA...BBBBBB")
        self.assertEqual(len(segs_dots), 2)
        self.assertEqual(segs_dots[0].text, "AAAAA...")
        self.assertEqual(segs_dots[0].break_after, BreakPriority.SENTENCE)

        # Unicode ellipsis
        segs_uni = segment_text("AAAAA…BBBBBB")
        self.assertEqual(len(segs_uni), 2)
        self.assertEqual(segs_uni[0].text, "AAAAA…")
        self.assertEqual(segs_uni[0].break_after, BreakPriority.SENTENCE)

        # Vertical presentation form
        segs_vert = segment_text("AAAAA︙BBBBBB")
        self.assertEqual(len(segs_vert), 2)
        self.assertEqual(segs_vert[0].text, "AAAAA︙")
        self.assertEqual(segs_vert[0].break_after, BreakPriority.SENTENCE)

        # AutoLayout line breaking
        layout = TextAutoLayout("AAAAA...BBBBBB", orientation="vertical-rtl")
        res = layout.find_optimal_layout(available_width=200, available_height=300)
        self.assertIsNotNone(res)
        assert res is not None
        # Should break into at least 2 columns, not pack together in one
        self.assertGreaterEqual(len(res.lines), 2)

    def test_ellipsis_in_connected_bubble_lobes(self) -> None:
        from PIL import ImageFont
        from src.renderer.pillow_renderer import _vertical_mask_layout, to_vertical_text

        # 2 distinct bubble lobes:
        # Lobe 1 (upper-right): X=[80..160], Y=[0..120]
        # Lobe 2 (lower-left): X=[20..80], Y=[140..260]
        mask = np.zeros((280, 180), dtype=np.uint8)
        mask[0:120, 80:160] = 1
        mask[140:260, 20:80] = 1

        text = "AAAAA...BBBBBB"
        vert_text = to_vertical_text(text)
        font = ImageFont.truetype(str(self.font_path), 20)

        # Pre-segment sentences
        from src.renderer.layout import segment_text
        sentences = [s.text.strip() for s in segment_text(vert_text) if s.text.strip()]
        self.assertEqual(len(sentences), 2)

        placements = _vertical_mask_layout(mask, sentences, font, padding=4, anchor=(80, 120))
        self.assertIsNotNone(placements)
        assert placements is not None

        # Placements should span both upper-right lobe (X >= 80) and lower-left lobe (X < 80)
        placed_xs = sorted([p[1] for p in placements], reverse=True)
        self.assertGreaterEqual(placed_xs[0], 80, "Upper-right lobe should contain placements")
        self.assertLess(placed_xs[-1], 80, "Lower-left lobe should contain placements")

    def test_kinsoku_shori_line_start_and_end_prohibition(self) -> None:
        """Closing punctuation cannot start a line; opening brackets cannot end a line."""
        from src.renderer.layout import TextAutoLayout

        # Long text where naively splitting would land "！" or "。" or "」" at line start
        text = "「AAAAAAAAAAAAAAAAAAAAAAA！"
        layout = TextAutoLayout(text, orientation="vertical-rtl")
        # Split into small chunks
        chunks = layout._split_segment_chars(text, line_limit=60, measure_func=lambda s: len(s) * 20)
        # None of the chunks (except possibly the very first char of whole text) should start with closing punctuation
        from src.renderer.layout import LINE_START_PROHIBITED, LINE_END_PROHIBITED
        for chunk in chunks[1:]:
            self.assertNotIn(chunk[0], LINE_START_PROHIBITED, f"Chunk '{chunk}' started with prohibited character {chunk[0]}")
        for chunk in chunks[:-1]:
            self.assertNotIn(chunk[-1], LINE_END_PROHIBITED, f"Chunk '{chunk}' ended with prohibited character {chunk[-1]}")

    def test_script_aware_horizontal_detection(self) -> None:
        """Pure English/Latin text should automatically render horizontally even if global direction is vertical."""
        renderer = PillowRenderer(font_path=self.font_path, text_direction="vertical-rtl")
        region_en = TextRegion(bbox=(10, 10, 50, 150), source_text="A", translated_text="ABC DEF GHI JKL MNO?")
        self.assertTrue(renderer._is_horizontal_region(region_en), "English text should force horizontal orientation")

        region_cjk = TextRegion(bbox=(10, 10, 50, 150), source_text="A", translated_text="甲乙，丙丁戊己庚？")
        self.assertFalse(renderer._is_horizontal_region(region_cjk), "CJK vertical text box should remain vertical")

    def test_short_phrase_vertical_single_column(self) -> None:
        """Short phrases (<= 3 chars) must remain in a single vertical column."""
        from src.renderer.layout import TextAutoLayout
        from PIL import ImageFont

        font_resolver = lambda size: ImageFont.truetype(str(self.font_path), size)
        for phrase in ["AA", "BB", "CCC"]:
            layout = TextAutoLayout(phrase, orientation="vertical-rtl", font_resolver=font_resolver)
            res = layout.find_optimal_layout(available_width=140, available_height=120, max_font_size=72, preferred_font_size=56, min_font_size=12)
            self.assertIsNotNone(res)
            assert res is not None
            self.assertEqual(len(res.lines), 1, f"Phrase '{phrase}' was split into {len(res.lines)} columns: {res.lines}")
            self.assertEqual(res.lines[0], phrase)


if __name__ == "__main__":
    unittest.main()
