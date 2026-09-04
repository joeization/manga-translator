from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from src.models import TextRegion
from src.ocr.region_splitter import crop_for_ocr, merge_contained_regions, split_text_regions


class TextRegionSplitterTests(unittest.TestCase):
    def test_merge_contained_regions(self) -> None:
        large = _crop((0, 0, 200, 300), np.ones((300, 200), dtype=bool))
        small = _crop((30, 40, 100, 120), np.ones((80, 70), dtype=bool))

        merged = merge_contained_regions([small, large])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, (0, 0, 200, 300))

    def test_splits_separated_ink_groups_inside_one_bubble(self) -> None:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 24, 24), fill="black")
        draw.rectangle((140, 140, 154, 154), fill="black")
        region = TextRegion(
            (0, 0, 200, 200),
            "",
            layout_bbox=(0, 0, 200, 200),
            layout_mask=np.ones((200, 200), dtype=bool),
        )

        regions = split_text_regions(image, region)

        self.assertEqual(len(regions), 2)
        self.assertTrue(all(region.source_bbox == region.bbox for region in regions))
        self.assertFalse(_bboxes_overlap(*(region.bbox for region in regions)))
        self.assertTrue(all(isinstance(region.ocr_mask, np.ndarray) for region in regions))

    def test_ocr_crop_masks_ink_from_other_split_regions(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 24, 24), fill="black")
        draw.rectangle((46, 10, 60, 24), fill="black")
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:25, 10:25] = True
        region = TextRegion(
            (0, 0, 100, 100),
            "",
            ocr_mask=mask,
        )

        crop = np.asarray(crop_for_ocr(image, region))

        self.assertTrue(np.any(crop[10:25, 10:25] < 255))
        self.assertTrue(np.all(crop[10:25, 46:61] == 255))

    def test_keeps_groups_separated_by_one_glyph_width_together(self) -> None:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 24, 24), fill="black")
        draw.rectangle((40, 10, 54, 24), fill="black")
        region = TextRegion(
            (0, 0, 200, 200),
            "",
            layout_bbox=(0, 0, 200, 200),
            layout_mask=np.ones((200, 200), dtype=bool),
        )

        self.assertEqual(split_text_regions(image, region), [region])

    def test_splits_groups_separated_by_more_than_two_glyph_widths(self) -> None:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 24, 24), fill="black")
        draw.rectangle((120, 10, 134, 24), fill="black")
        region = TextRegion(
            (0, 0, 200, 200),
            "",
            layout_bbox=(0, 0, 200, 200),
            layout_mask=np.ones((200, 200), dtype=bool),
        )

        self.assertEqual(len(split_text_regions(image, region)), 2)

    def test_retains_region_when_ink_forms_one_group(self) -> None:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 24, 24), fill="black")
        draw.rectangle((28, 10, 42, 24), fill="black")
        region = TextRegion(
            (0, 0, 200, 200),
            "",
            layout_bbox=(0, 0, 200, 200),
            layout_mask=np.ones((200, 200), dtype=bool),
        )

        self.assertEqual(split_text_regions(image, region), [region])

    def test_splits_multi_paragraph_region(self) -> None:
        image = Image.new("RGB", (200, 300), "white")
        draw = ImageDraw.Draw(image)
        # Paragraph 1 (top)
        draw.rectangle((10, 10, 24, 24), fill="black")
        draw.rectangle((28, 10, 42, 24), fill="black")
        # Paragraph 2 (middle)
        draw.rectangle((10, 100, 24, 114), fill="black")
        draw.rectangle((28, 100, 42, 114), fill="black")
        # Paragraph 3 (bottom)
        draw.rectangle((10, 200, 24, 214), fill="black")
        draw.rectangle((28, 200, 42, 214), fill="black")

        region = TextRegion(
            (0, 0, 200, 300),
            "",
            layout_bbox=(0, 0, 200, 300),
            layout_mask=np.ones((300, 200), dtype=bool),
        )

        split_regions = split_text_regions(image, region)
        self.assertEqual(len(split_regions), 3)

    def test_merges_small_region_contained_in_large_region(self) -> None:
        from src.ocr.region_splitter import merge_contained_regions

        large = _crop((0, 0, 200, 300), np.ones((300, 200), dtype=bool))
        small = _crop((30, 40, 100, 120), np.ones((80, 70), dtype=bool))

        merged = merge_contained_regions([small, large], threshold=0.50)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, (0, 0, 200, 300))

    def test_splits_staggered_sentences_in_irregular_bubble(self) -> None:
        image = Image.new("RGB", (300, 300), "white")
        draw = ImageDraw.Draw(image)
        # Sentence 1 (top-right)
        draw.rectangle((200, 10, 280, 50), fill="black")
        # Sentence 2 (bottom-left)
        draw.rectangle((10, 150, 90, 190), fill="black")

        region = TextRegion(
            (0, 0, 300, 300),
            "",
            layout_bbox=(0, 0, 300, 300),
            layout_mask=np.ones((300, 300), dtype=bool),
        )

        split_regions = split_text_regions(image, region)
        self.assertEqual(len(split_regions), 2)

    def test_region_has_text_rejects_empty_circular_bubble(self) -> None:
        import cv2
        from src.ocr.region_splitter import region_has_text

        img = np.full((120, 120, 3), 200, dtype=np.uint8)
        mask = np.zeros((120, 120), dtype=bool)
        cv2.circle(mask.view(np.uint8), (60, 60), 40, 1, -1)
        img[mask] = 255
        cv2.circle(img, (60, 60), 40, (0, 0, 0), 2)

        reg = TextRegion((20, 20, 100, 100), "", layout_bbox=(0, 0, 120, 120), layout_mask=mask)
        self.assertFalse(region_has_text(img, reg))

    def test_region_has_text_accepts_bubble_with_text(self) -> None:
        import cv2
        from src.ocr.region_splitter import region_has_text

        img = np.full((120, 120, 3), 200, dtype=np.uint8)
        mask = np.zeros((120, 120), dtype=bool)
        cv2.circle(mask.view(np.uint8), (60, 60), 40, 1, -1)
        img[mask] = 255
        cv2.circle(img, (60, 60), 40, (0, 0, 0), 2)
        # Add character 'A' in the center
        cv2.putText(img, "A", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

        reg = TextRegion((20, 20, 100, 100), "", layout_bbox=(0, 0, 120, 120), layout_mask=mask)
        self.assertTrue(region_has_text(img, reg))

    def test_region_has_text_rejects_empty_screentone_bubble(self) -> None:
        import cv2
        from src.ocr.region_splitter import region_has_text

        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        mask = np.zeros((120, 120), dtype=bool)
        cv2.circle(mask.view(np.uint8), (60, 60), 40, 1, -1)
        for y in range(20, 100, 4):
            for x in range(20, 100, 4):
                if mask[y, x]:
                    img[y, x] = (160, 160, 160)
        cv2.circle(img, (60, 60), 40, (0, 0, 0), 2)

        reg = TextRegion((20, 20, 100, 100), "", layout_bbox=(0, 0, 120, 120), layout_mask=mask)
        self.assertFalse(region_has_text(img, reg))

    def test_region_has_text_accepts_inverted_text(self) -> None:
        import cv2
        from src.ocr.region_splitter import region_has_text

        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        mask = np.zeros((120, 120), dtype=bool)
        cv2.circle(mask.view(np.uint8), (60, 60), 40, 1, -1)
        img[mask] = 20
        cv2.putText(img, "A", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        reg = TextRegion((20, 20, 100, 100), "", layout_bbox=(0, 0, 120, 120), layout_mask=mask)
        self.assertTrue(region_has_text(img, reg))

    def test_multi_column_vertical_paragraph_not_split(self) -> None:
        # Realistic 2-column manga bubble (28px font, 42px gap)
        img = Image.new("RGB", (300, 400), "white")
        d = ImageDraw.Draw(img)
        for y in range(50, 250, 40):
            d.rectangle((180, y, 208, y + 28), fill="black")
            d.rectangle((110, y, 138, y + 28), fill="black")
        r = TextRegion((0, 0, 300, 400), "", layout_bbox=(0, 0, 300, 400), layout_mask=np.ones((400, 300), dtype=bool))
        res = split_text_regions(img, r)
        self.assertEqual(len(res), 1)

    def test_multi_column_vertical_3_columns_not_split(self) -> None:
        img = Image.new("RGB", (300, 400), "white")
        d = ImageDraw.Draw(img)
        for y in range(50, 250, 40):
            d.rectangle((220, y, 248, y + 28), fill="black")
            d.rectangle((150, y, 178, y + 28), fill="black")
            d.rectangle((80, y, 108, y + 28), fill="black")
        r = TextRegion((0, 0, 300, 400), "", layout_bbox=(0, 0, 300, 400), layout_mask=np.ones((400, 300), dtype=bool))
        res = split_text_regions(img, r)
        self.assertEqual(len(res), 1)

    def test_multi_line_horizontal_paragraph_not_split(self) -> None:
        img = Image.new("RGB", (300, 300), "white")
        d = ImageDraw.Draw(img)
        for x in range(50, 250, 40):
            d.rectangle((x, 50, x + 28, 78), fill="black")
            d.rectangle((x, 120, x + 28, 148), fill="black")
        r = TextRegion((0, 0, 300, 300), "", layout_bbox=(0, 0, 300, 300), layout_mask=np.ones((300, 300), dtype=bool))
        res = split_text_regions(img, r)
        self.assertEqual(len(res), 1)

    def test_two_distinct_bubbles_are_split(self) -> None:
        img = Image.new("RGB", (300, 500), "white")
        d = ImageDraw.Draw(img)
        for y in range(30, 150, 40):
            d.rectangle((140, y, 168, y + 28), fill="black")
        for y in range(280, 400, 40):
            d.rectangle((140, y, 168, y + 28), fill="black")
        r = TextRegion((0, 0, 300, 500), "", layout_bbox=(0, 0, 300, 500), layout_mask=np.ones((500, 300), dtype=bool))
        res = split_text_regions(img, r)
        self.assertEqual(len(res), 2)

    def test_staggered_vertical_columns_not_split(self) -> None:
        # Col 1 is short (top only), Col 2 is long (extends further down)
        img = Image.new("RGB", (250, 350), "white")
        d = ImageDraw.Draw(img)
        # Col 1 (right): 2 chars at top
        d.rectangle((160, 40, 188, 68), fill="black")
        d.rectangle((160, 80, 188, 108), fill="black")
        # Col 2 (left): 5 chars extending down
        for y in range(40, 240, 40):
            d.rectangle((100, y, 128, y + 28), fill="black")
        r = TextRegion((0, 0, 250, 350), "", layout_bbox=(0, 0, 250, 350), layout_mask=np.ones((350, 250), dtype=bool))
        res = split_text_regions(img, r)
        self.assertEqual(len(res), 1)

    def test_two_side_by_side_distant_bubbles_split(self) -> None:
        # Two bubbles far apart horizontally (gap = 132px)
        img = Image.new("RGB", (400, 300), "white")
        d = ImageDraw.Draw(img)
        for y in range(50, 250, 40):
            d.rectangle((50, y, 78, y + 28), fill="black")
            d.rectangle((240, y, 268, y + 28), fill="black")
        r = TextRegion((0, 0, 400, 300), "", layout_bbox=(0, 0, 400, 300), layout_mask=np.ones((300, 400), dtype=bool))
        res = split_text_regions(img, r)
        self.assertEqual(len(res), 2)


if __name__ == "__main__":
    unittest.main()


def _bboxes_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def _crop(bbox: tuple[int, int, int, int], ocr_mask: np.ndarray) -> TextRegion:
    return TextRegion(bbox, "", source_bbox=bbox, ocr_mask=ocr_mask)