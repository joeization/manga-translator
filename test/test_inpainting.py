from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image

from src.inpainting.opencv import OpenCVInpainter, _expand_bbox
from src.models import TextRegion


class InpaintingGeometryTests(unittest.TestCase):
    def test_clear_padding_expands_and_clips_detection_bounds(self) -> None:
        self.assertEqual(_expand_bbox((20, 30, 50, 60), 100, 100, 12), (8, 18, 62, 72))
        self.assertEqual(_expand_bbox((1, 2, 99, 98), 100, 100, 12), (0, 0, 100, 100))

    def test_glyph_mask_detects_dark_text_on_light_background(self) -> None:
        roi = np.full((20, 20, 3), (220, 210, 200), dtype=np.uint8)
        roi[8:12, 8:12] = (20, 20, 20)
        allowed = np.ones((20, 20), dtype=bool)

        inpainter = OpenCVInpainter(160, 235, 0.7, 3, 1, 4, 80, 13, "glyph", 0.25, 3)
        bg_color = np.array([220.0, 210.0, 200.0])
        bg_std = np.array([2.0, 2.0, 2.0])

        mask = inpainter._glyph_mask(roi, allowed, (6, 6, 14, 14), bg_color, bg_std)

        self.assertTrue(np.all(mask[8:12, 8:12]))
        self.assertFalse(np.any(mask[:6, :]))

    def test_segmentation_extent_replaces_detection_padding_for_inpainting(self) -> None:
        image = Image.new("RGB", (40, 40), (220, 220, 220))
        region = TextRegion(
            (16, 16, 24, 24),
            "",
            layout_bbox=(4, 4, 36, 36),
            layout_mask=np.ones((32, 32), dtype=bool),
        )
        inpainter = OpenCVInpainter(160, 235, 0.7, 3, 1, 4, 80, 13, "glyph", 0.25, 3)

        inpainter.inpaint(image, [region])

        self.assertEqual(region.inpaint_bbox, (4, 4, 36, 36))

    def test_gradient_background_inpainting_preserves_gradient_transition(self) -> None:
        # Create a vertical color gradient background (0=50, 100=200)
        img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            val = int(50 + (150 * y / 100))
            img_arr[y, :, :] = val

        # Place dark text in the middle (y:40..60, x:40..60)
        img_arr[40:60, 40:60] = 0

        image = Image.fromarray(img_arr)
        region = TextRegion(
            (40, 40, 60, 60),
            "TEST",
            source_bbox=(40, 40, 60, 60),
        )

        inpainter = OpenCVInpainter(
            dark_threshold=160,
            white_threshold=235,
            white_ratio=0.7,
            inpaint_radius=3,
            mask_dilation=1,
            ocr_clear_padding=12,
            bubble_padding=80,
            bubble_close_kernel=13,
            bubble_clear_mode="glyph",
            bubble_min_overlap=0.25,
            bubble_border_width=3,
            solid_fill_std_threshold=5.0,
            inpaint_algorithm="telea",
        )

        inpainted_img = inpainter.inpaint(image, [region])
        res = np.array(inpainted_img)

        # Check that top of inpainted region (y=45) is darker than bottom of inpainted region (y=55)
        top_val = float(np.mean(res[45, 45:55]))
        bottom_val = float(np.mean(res[55, 45:55]))
        self.assertGreater(bottom_val, top_val + 5.0)

    def test_plain_white_region_rejects_open_rectangular_region(self) -> None:
        inpainter = OpenCVInpainter(160, 235, 0.7, 3, 1, 4, 80, 13, "interior", 0.25, 3)
        roi = np.full((40, 40, 3), 255, dtype=np.uint8)
        # Rectangular mask filling the entire box
        mask = np.ones((40, 40), dtype=bool)
        # Should be rejected because it's not an enclosed bubble shape -> fall back to glyph
        self.assertFalse(inpainter._is_plain_white_region(roi, mask))

    def test_plain_white_region_rejects_screentone_or_textured_background(self) -> None:
        inpainter = OpenCVInpainter(160, 235, 0.7, 3, 1, 4, 80, 13, "interior", 0.25, 3)
        # Create an enclosed elliptical bubble mask
        mask_u8 = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(mask_u8, (30, 30), 20, 1, thickness=-1)
        mask = mask_u8.astype(bool)

        # Background inside the bubble has screentone/texture (alternating 190 and 245)
        roi = np.full((60, 60, 3), 245, dtype=np.uint8)
        roi[15:45:2, 15:45:2] = (190, 190, 190)
        # Should be rejected due to texture/low white ratio on background -> fall back to glyph
        self.assertFalse(inpainter._is_plain_white_region(roi, mask))

    def test_plain_white_region_accepts_clean_white_bubble(self) -> None:
        inpainter = OpenCVInpainter(160, 235, 0.7, 3, 1, 4, 80, 13, "interior", 0.25, 3)
        mask_u8 = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(mask_u8, (30, 30), 20, 1, thickness=-1)
        mask = mask_u8.astype(bool)

        roi = np.full((60, 60, 3), 250, dtype=np.uint8)
        # Dark text in the center
        roi[28:32, 25:35] = (20, 20, 20)
        self.assertTrue(inpainter._is_plain_white_region(roi, mask))

    def test_glyph_inpaint_fills_with_true_background_color(self) -> None:
        # Image with a tinted cream background (210, 200, 190)
        img_arr = np.full((60, 60, 3), (210, 200, 190), dtype=np.uint8)
        # Dark text in the center
        img_arr[25:35, 25:35] = (20, 20, 20)
        image = Image.fromarray(img_arr)
        region = TextRegion((25, 25, 35, 35), "test", source_bbox=(25, 25, 35, 35))

        inpainter = OpenCVInpainter(
            dark_threshold=160,
            white_threshold=235,
            white_ratio=0.7,
            inpaint_radius=3,
            mask_dilation=1,
            ocr_clear_padding=12,
            bubble_padding=80,
            bubble_close_kernel=13,
            bubble_clear_mode="glyph",
            bubble_min_overlap=0.25,
            bubble_border_width=3,
            solid_fill_std_threshold=5.0,
            inpaint_algorithm="telea",
        )
        inpainted = inpainter.inpaint(image, [region])
        res = np.array(inpainted)
        # Center should be filled with the cream background (210, 200, 190), NOT white (255, 255, 255)
        center_color = res[30, 30]
        self.assertAlmostEqual(float(center_color[0]), 210.0, delta=2.0)
        self.assertAlmostEqual(float(center_color[1]), 200.0, delta=2.0)
        self.assertAlmostEqual(float(center_color[2]), 190.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()