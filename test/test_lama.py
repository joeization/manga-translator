import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.inpainting.lama import LamaInpainter
from src.models import TextRegion

class LamaInpainterTests(unittest.TestCase):
    def test_lama_inpainter_initialization_and_inpaint(self) -> None:
        model_path = Path("models/inpainting_lama/lama-manga.onnx")
        try:
            inpainter = LamaInpainter(model_path=model_path, device="cpu")
        except Exception as err:
            self.skipTest(f"LaMa model initialization skipped: {err}")

        # Create a test image with a black square text in the center of a colored image
        img_np = np.full((100, 100, 3), (120, 150, 200), dtype=np.uint8)
        img_np[40:60, 40:60] = (0, 0, 0)
        image = Image.fromarray(img_np)

        region = TextRegion(
            bbox=(40, 40, 60, 60),
            source_text="Test",
            source_bbox=(40, 40, 60, 60),
        )

        inpainted_image = inpainter.inpaint(image, [region])
        result_np = np.array(inpainted_image)

        self.assertEqual(result_np.shape, (100, 100, 3))
        # The center area should no longer be pure black (0, 0, 0)
        center_mean = np.mean(result_np[45:55, 45:55])
        self.assertGreater(center_mean, 20.0)

    def test_cluster_bboxes_merges_overlapping_and_adjacent_boxes(self) -> None:
        from src.inpainting.lama import _cluster_bboxes
        # Two boxes that overlap when expanded by margin=10
        bboxes = [(10, 10, 20, 20), (25, 25, 35, 35)]
        clusters = _cluster_bboxes(bboxes, margin=10, img_w=100, img_h=100)
        self.assertEqual(len(clusters), 1)
        # Bounding box should span from 0 (10-10) to 45 (35+10)
        self.assertEqual(clusters[0], (0, 0, 45, 45))

    def test_cluster_bboxes_keeps_disjoint_boxes_separate(self) -> None:
        from src.inpainting.lama import _cluster_bboxes
        bboxes = [(10, 10, 20, 20), (80, 80, 90, 90)]
        clusters = _cluster_bboxes(bboxes, margin=5, img_w=100, img_h=100)
        self.assertEqual(len(clusters), 2)

    def test_extract_text_glyph_mask_finds_dark_text(self) -> None:
        from src.inpainting.lama import _extract_text_glyph_mask
        img_np = np.full((100, 100, 3), 245, dtype=np.uint8)
        # Place dark text in (40:60, 40:60)
        img_np[45:55, 45:55] = (20, 20, 20)
        region = TextRegion(bbox=(40, 40, 60, 60), source_text="test")
        mask, bbox = _extract_text_glyph_mask(img_np, region, 100, 100)
        self.assertIsNotNone(mask)
        assert mask is not None
        # Mask should cover the center dark text
        assert bbox is not None
        bl, bt, _, _ = bbox
        self.assertTrue(mask[45 - bt : 55 - bt, 45 - bl : 55 - bl].all())
        # Mask should NOT cover the entire outer boundary
        self.assertFalse(mask[0, 0])

    def test_extract_text_glyph_mask_returns_none_for_uniform_background(self) -> None:
        from src.inpainting.lama import _extract_text_glyph_mask
        img_np = np.full((100, 100, 3), 245, dtype=np.uint8)
        region = TextRegion(bbox=(40, 40, 60, 60), source_text="test")
        mask, bbox = _extract_text_glyph_mask(img_np, region, 100, 100)
        self.assertIsNone(mask)
        self.assertIsNone(bbox)

    def test_extract_text_glyph_mask_intersects_with_segmentation(self) -> None:
        import cv2
        from src.inpainting.lama import _extract_text_glyph_mask
        img_np = np.full((100, 100, 3), 245, dtype=np.uint8)
        # Text from (20, 20) to (80, 80)
        img_np[20:80, 20:80] = (20, 20, 20)
        # Bubble segmentation mask is a circle centered at (50, 50) with radius 20
        seg_mask_u8 = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(seg_mask_u8, (50, 50), 20, 1, thickness=-1)
        seg_mask = seg_mask_u8.astype(bool)

        region = TextRegion(
            bbox=(20, 20, 80, 80),
            source_text="test",
            layout_bbox=(0, 0, 100, 100),
            layout_mask=seg_mask,
        )
        mask, bbox = _extract_text_glyph_mask(img_np, region, 100, 100)
        self.assertIsNotNone(mask)
        assert mask is not None and bbox is not None
        bl, bt, br, bb = bbox
        # Check that mask is never True outside the circular seg_mask
        for y in range(bt, bb):
            for x in range(bl, br):
                if mask[y - bt, x - bl]:
                    self.assertTrue(seg_mask[y, x], f"Pixel ({x}, {y}) should be within seg_mask!")

    def test_estimate_adaptive_dilation_immune_to_punctuation(self) -> None:
        from src.inpainting.utils import estimate_adaptive_dilation

        # Mask with 3 main 30x30 characters and 10 small 4x4 punctuation dots
        mask = np.zeros((200, 100), dtype=bool)
        mask[20:50, 30:60] = True
        mask[60:90, 30:60] = True
        mask[100:130, 30:60] = True
        for y in range(140, 190, 5):
            mask[y : y + 4, 45:49] = True

        dilation_with_punctuation = estimate_adaptive_dilation(
            (0, 0, 100, 200),
            "AAA……！",
            1200,
            1800,
            glyph_mask=mask,
        )

        # Same mask but with NO punctuation dots
        clean_mask = np.zeros((200, 100), dtype=bool)
        clean_mask[20:50, 30:60] = True
        clean_mask[60:90, 30:60] = True
        clean_mask[100:130, 30:60] = True

        clean_dilation = estimate_adaptive_dilation(
            (0, 0, 100, 200),
            "AAA",
            1200,
            1800,
            glyph_mask=clean_mask,
        )

        # Punctuation must NOT drag down dilation!
        self.assertEqual(dilation_with_punctuation, clean_dilation)
        self.assertGreaterEqual(dilation_with_punctuation, 3)

    def test_extract_glyph_mask_does_not_fill_segmentation(self) -> None:
        import cv2
        from src.inpainting.lama import _extract_text_glyph_mask

        # Large 200x200 speech bubble
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        bubble_mask = np.zeros((200, 200), dtype=bool)
        cv2.circle(bubble_mask.view(np.uint8), (100, 100), 80, 1, -1)

        # Small 40x50 text box in center
        reg = TextRegion(
            bbox=(80, 75, 120, 125),
            source_text="A",
            layout_bbox=(0, 0, 200, 200),
            layout_mask=bubble_mask,
        )
        # Draw character 'A' (dark text) inside the detect box
        img[90:110, 95:105] = (20, 20, 20)

        mask, bbox = _extract_text_glyph_mask(img, reg, 200, 200)
        self.assertIsNotNone(mask)
        assert mask is not None
        # Mask must only cover text + small margin, NOT the whole bubble!
        bubble_area = np.count_nonzero(bubble_mask)
        mask_area = np.count_nonzero(mask)
        self.assertLess(mask_area, 0.20 * bubble_area)

    def test_extract_glyph_mask_does_not_dilate_segmentation_border(self) -> None:
        import cv2
        from src.inpainting.lama import _extract_text_glyph_mask

        # Speech bubble with black perimeter border
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        cv2.circle(img, (100, 100), 80, (0, 0, 0), 3)

        bubble_mask = np.zeros((200, 200), dtype=bool)
        cv2.circle(bubble_mask.view(np.uint8), (100, 100), 80, 1, -1)

        # Text detect rect touching the bubble right border
        reg = TextRegion(
            bbox=(70, 90, 180, 110),
            source_text="text",
            layout_bbox=(0, 0, 200, 200),
            layout_mask=bubble_mask,
        )
        # Draw some text inside
        img[95:105, 80:120] = (10, 10, 10)

        mask, bbox = _extract_text_glyph_mask(img, reg, 200, 200)
        self.assertIsNotNone(mask)
        assert mask is not None
        assert bbox is not None
        pl, pt, pr, pb = bbox

        # Check that top and bottom bubble perimeter pixels are NOT masked or dilated!
        # Top perimeter of bubble is at (100, 20)
        if 20 >= pt and 20 < pb and 100 >= pl and 100 < pr:
            self.assertFalse(mask[20 - pt, 100 - pl])

    def test_inpaint_mask_never_exceeds_detect_rect_or_segmentation(self) -> None:
        import cv2
        from src.inpainting.lama import _extract_text_glyph_mask

        # Complex background image with screentones/lines
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        for y in range(0, 300, 4):
            img[y, :, :] = 140

        bubble_mask = np.zeros((160, 160), dtype=bool)
        cv2.circle(bubble_mask.view(np.uint8), (80, 80), 70, 1, -1)

        # Region with bubble segmentation
        reg_with_bubble = TextRegion(
            bbox=(70, 80, 170, 140),
            source_text="TEST",
            layout_bbox=(50, 50, 210, 210),
            layout_mask=bubble_mask,
        )

        mask, bbox = _extract_text_glyph_mask(img, reg_with_bubble, 300, 300)
        self.assertIsNotNone(mask)
        assert mask is not None and bbox is not None
        pl, pt, pr, pb = bbox

        full_inpaint = np.zeros((300, 300), dtype=bool)
        full_inpaint[pt:pb, pl:pr] = mask

        # 1. Zero pixels outside layout mask (speech bubble)
        full_rect = np.zeros((300, 300), dtype=bool)
        full_rect[80:140, 70:170] = True
        full_layout = np.zeros((300, 300), dtype=bool)
        full_layout[50:210, 50:210] = bubble_mask
        self.assertEqual(np.count_nonzero(full_inpaint & ~full_layout), 0)

        # 2. Outward strokes are bounded to expanded text rect inside bubble
        k_exp = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        expanded_rect = cv2.dilate(full_rect.astype(np.uint8), k_exp).astype(bool)
        self.assertEqual(np.count_nonzero(full_inpaint & ~(expanded_rect & full_layout)), 0)

        # Region without bubble segmentation (open scene / narrative box)
        reg_no_bubble = TextRegion(
            bbox=(70, 80, 170, 140),
            source_text="TEST",
        )
        mask2, bbox2 = _extract_text_glyph_mask(img, reg_no_bubble, 300, 300)
        self.assertIsNotNone(mask2)
        assert mask2 is not None and bbox2 is not None
        pl2, pt2, pr2, pb2 = bbox2
        full_inpaint2 = np.zeros((300, 300), dtype=bool)
        full_inpaint2[pt2:pb2, pl2:pr2] = mask2
        self.assertEqual(np.count_nonzero(full_inpaint2 & ~full_rect), 0)


if __name__ == "__main__":
    unittest.main()
