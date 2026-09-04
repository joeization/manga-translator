from __future__ import annotations

import unittest

import numpy as np

from src.models import TextRegion
from src.ocr.postprocess import postprocess_bubbles


class BubblePostprocessTests(unittest.TestCase):
    def test_hybrid_keeps_nearby_text_boxes_as_independent_ocr_regions(self) -> None:
        mask = np.ones((205, 10), dtype=bool)
        segmentation = TextRegion(
            (0, 0, 10, 205),
            "",
            detection_confidence=0.9,
            layout_bbox=(0, 0, 10, 205),
            layout_mask=mask,
        )
        first_box = TextRegion((0, 0, 10, 100), "", detection_confidence=0.9)
        second_box = TextRegion((0, 105, 10, 205), "", detection_confidence=0.9)

        regions = postprocess_bubbles([first_box, second_box], [segmentation])

        self.assertEqual([region.bbox for region in regions], [first_box.bbox, second_box.bbox])
        self.assertEqual([region.source_bbox for region in regions], [first_box.bbox, second_box.bbox])


    def test_sort_manga_reading_order_right_to_left(self) -> None:
        right_bubble = TextRegion((200, 10, 300, 100), "", detection_confidence=0.9)
        left_bubble = TextRegion((50, 15, 150, 105), "", detection_confidence=0.9)
        bottom_bubble = TextRegion((100, 300, 250, 400), "", detection_confidence=0.9)

        from src.ocr.postprocess import sort_manga_reading_order
        ordered = sort_manga_reading_order([left_bubble, bottom_bubble, right_bubble])

        self.assertEqual(ordered, [right_bubble, left_bubble, bottom_bubble])


    def test_postprocess_segmentation_mask_preserves_inside_box_and_erodes_outside(self) -> None:
        import cv2
        from src.ocr.postprocess import postprocess_segmentation_mask

        # 100x100 circle mask with layout_bbox (50, 50, 150, 150)
        mask = np.zeros((100, 100), dtype=bool)
        cv2.circle(mask.view(np.uint8), (50, 50), 40, 1, -1)

        layout_bbox = (50, 50, 150, 150)
        # Detection box at (70, 70, 130, 130), in mask coordinates (20, 20, 80, 80)
        detection_box = (70, 70, 130, 130)

        protect_margin = 4
        final_mask = postprocess_segmentation_mask(
            mask, layout_bbox, detection_box, kernel_size=3, iterations=1, protect_margin=protect_margin
        )

        # 1. Mask inside detection box + outward protect margin must be preserved EXACTLY as-is
        box_l = max(0, 20 - protect_margin)
        box_t = max(0, 20 - protect_margin)
        box_r = min(100, 80 + protect_margin)
        box_b = min(100, 80 + protect_margin)
        np.testing.assert_array_equal(final_mask[box_t:box_b, box_l:box_r], mask[box_t:box_b, box_l:box_r])

        # 2. Mask outside protected zone must match eroded mask
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        expected_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        outside_mask = np.ones((100, 100), dtype=bool)
        outside_mask[box_t:box_b, box_l:box_r] = False
        np.testing.assert_array_equal(final_mask[outside_mask], expected_eroded[outside_mask])

        # 3. Outer boundary was successfully trimmed
        self.assertLess(np.count_nonzero(final_mask), np.count_nonzero(mask))
        self.assertGreater(np.count_nonzero(final_mask), np.count_nonzero(expected_eroded))


class FuriganaSuppresionTests(unittest.TestCase):
    def test_small_region_above_large_region_is_suppressed(self) -> None:
        from src.ocr.postprocess import _suppress_furigana

        # Furigana: small box (h=20) directly above big box (h=80) with full horizontal overlap
        furigana = TextRegion((100, 50, 200, 70), "", detection_confidence=0.6)   # h=20
        main_text = TextRegion((100, 72, 200, 152), "", detection_confidence=0.9)  # h=80

        result = _suppress_furigana([furigana, main_text])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, main_text.bbox)

    def test_equal_sized_region_is_not_suppressed(self) -> None:
        from src.ocr.postprocess import _suppress_furigana

        # Two similarly sized bubbles side-by-side — neither should be suppressed
        left = TextRegion((50, 50, 150, 150), "", detection_confidence=0.9)    # h=100
        right = TextRegion((160, 50, 260, 150), "", detection_confidence=0.9)  # h=100

        result = _suppress_furigana([left, right])
        self.assertEqual(len(result), 2)

    def test_small_region_without_overlap_is_not_suppressed(self) -> None:
        from src.ocr.postprocess import _suppress_furigana

        # Small box above a large box but no horizontal overlap
        small = TextRegion((300, 50, 400, 70), "", detection_confidence=0.8)   # h=20, no x overlap
        large = TextRegion((100, 72, 200, 152), "", detection_confidence=0.9)  # h=80

        result = _suppress_furigana([small, large])
        self.assertEqual(len(result), 2)

    def test_single_region_returns_unchanged(self) -> None:
        from src.ocr.postprocess import _suppress_furigana

        region = TextRegion((100, 50, 200, 150), "", detection_confidence=0.9)
        result = _suppress_furigana([region])
        self.assertEqual(result, [region])


class ReadingOrderTranslationTest(unittest.TestCase):
    def test_translation_candidates_sorted_in_reading_order(self) -> None:
        """_translation_candidates must return regions in manga reading order (right→left, top→bottom)."""
        from src.models import OCRResult
        from src.pipeline import _translation_candidates

        # right bubble is to the right of left bubble, same vertical row
        right = OCRResult(bbox=(200, 10, 300, 100), source_text="A", confidence=0.95, character_confidences=[0.95])
        left = OCRResult(bbox=(50, 15, 150, 105), source_text="B", confidence=0.95, character_confidences=[0.95])
        bottom = OCRResult(bbox=(100, 300, 250, 400), source_text="C", confidence=0.95, character_confidences=[0.95])

        # Pass in reversed order; candidates should come back right→left→bottom
        candidates = _translation_candidates([left, bottom, right], minimum_confidence=0.0)
        self.assertEqual([c.source_text for c in candidates], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()