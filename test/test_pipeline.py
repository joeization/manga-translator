from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.models import TextRegion
from src.pipeline import MangaTranslationPipeline, _translation_candidates, compute_ocr_quality_score, draw_debug_image


class _OCR:
    def detect(self, image: Image.Image) -> list[TextRegion]:
        return [TextRegion((0, 0, 1, 1), "raw OCR")]


class _FailingTranslator:
    def translate(self, texts: list[str], context: str | list[str] | None = None) -> list[str]:
        raise RuntimeError("Ollama unavailable")


class _UnusedStage:
    def __getattr__(self, name: str):
        raise AssertionError(f"{name} should not run after translation failure")


class MangaTranslationPipelineTests(unittest.TestCase):
    def test_debug_image_overlays_final_inpainting_mask(self) -> None:
        image = Image.new("RGB", (10, 10), "white")
        region = TextRegion(
            (0, 0, 10, 10),
            "",
            layout_bbox=(0, 0, 10, 10),
            layout_mask=np.ones((10, 10), dtype=bool),
            inpaint_bbox=(0, 0, 10, 10),
            inpaint_mask=np.ones((10, 10), dtype=bool),
        )

        debug_image = draw_debug_image(image, [region])

        self.assertNotEqual(debug_image.getpixel((5, 5)), image.getpixel((5, 5)))

    def test_low_confidence_region_is_not_a_translation_candidate(self) -> None:
        # Vertically separated so reading order is deterministic (top-to-bottom).
        low_confidence = TextRegion((0, 0, 50, 50), "LOW_CONF", confidence=0.42)
        high_confidence = TextRegion((0, 60, 50, 110), "HIGH_CONF_1", confidence=0.90)
        another_high_confidence = TextRegion((0, 120, 50, 170), "HIGH_CONF_2", confidence=0.96)
        unavailable_confidence = TextRegion((0, 180, 50, 230), "MIN_CONF")

        self.assertEqual(
            _translation_candidates([low_confidence, high_confidence, another_high_confidence, unavailable_confidence]),
            [high_confidence, another_high_confidence, unavailable_confidence],
        )

    def test_combined_quality_score_rejects_hallucinated_uneven_confidences(self) -> None:
        # Hallucination on empty bubble: model generates common tokens with high confidence,
        # but others with very low confidence, resulting in high std dev and low combined quality score
        uneven_char_confidences = [0.95, 0.20, 0.90, 0.35]
        sentence_conf = 0.52  # Borderline geometric mean that would falsely pass simple threshold

        score = compute_ocr_quality_score(sentence_conf, uneven_char_confidences)
        self.assertIsNotNone(score)
        assert score is not None
        # Must be penalized below the default 0.50 threshold due to high std dev
        self.assertLess(score, 0.50)

        hallucinated_region = TextRegion(
            (0, 0, 10, 10),
            "LOW_CONF",
            confidence=sentence_conf,
            character_confidences=uneven_char_confidences,
        )
        self.assertEqual(_translation_candidates([hallucinated_region], minimum_confidence=0.50), [])

    def test_combined_quality_score_accepts_consistent_confidences(self) -> None:
        # Real text: characters have consistently high probabilities with low variance
        consistent_char_confidences = [0.90, 0.94, 0.91, 0.93]
        sentence_conf = 0.92

        score = compute_ocr_quality_score(sentence_conf, consistent_char_confidences)
        self.assertIsNotNone(score)
        assert score is not None
        self.assertGreaterEqual(score, 0.88)

        good_region = TextRegion(
            (0, 0, 10, 10),
            "HIGH_CONF",
            confidence=sentence_conf,
            character_confidences=consistent_char_confidences,
        )
        self.assertEqual(_translation_candidates([good_region], minimum_confidence=0.50), [good_region])

    def test_combined_quality_score_fallback_when_character_confidences_none(self) -> None:
        score = compute_ocr_quality_score(0.75, None)
        self.assertEqual(score, 0.75)

    def test_combined_quality_score_rejects_weak_baberu_hallucination(self) -> None:
        char_confidences = [0.5520579218864441, 0.6995126008987427, 0.9391855597496033, 0.6556254029273987, 0.9956251978874207, 0.9514972567558289]
        sentence_conf = 0.7800
        text = "AAAAA、"
        score = compute_ocr_quality_score(sentence_conf, char_confidences, source_text=text)
        self.assertIsNotNone(score)
        assert score is not None
        # Score is ~0.64, well below the 0.75 threshold
        self.assertLess(score, 0.75)
        reg = TextRegion((0, 0, 10, 10), text, confidence=sentence_conf, character_confidences=char_confidences)
        self.assertEqual(_translation_candidates([reg]), [])

    def test_combined_quality_score_preserves_genuine_text_with_punctuation(self) -> None:
        # Punctuation dot/colon at the end with lower confidence (0.517)
        char_confidences = [0.9973722696304321, 0.9996356964111328, 0.9998389482498169, 0.9999175071716309, 0.9998641014099121, 0.5170303583145142]
        sentence_conf = 0.8954
        text = "AAAAA："

        score = compute_ocr_quality_score(sentence_conf, char_confidences, source_text=text)
        self.assertIsNotNone(score)
        assert score is not None
        # Punctuation must not drag down score; should be ~0.94, well above 0.75
        self.assertGreaterEqual(score, 0.90)

        reg = TextRegion((0, 0, 10, 10), text, confidence=sentence_conf, character_confidences=char_confidences)
        self.assertEqual(_translation_candidates([reg]), [reg])

    def test_translation_failure_returns_the_original_image(self) -> None:
        source = Image.new("RGB", (2, 2), (12, 34, 56))
        pipeline = MangaTranslationPipeline(
            _OCR(),
            _FailingTranslator(),
            _UnusedStage(),
            _UnusedStage(),
            _UnusedStage(),
        )

        result, regions = pipeline.process(source)

        self.assertEqual(regions, [])
        self.assertEqual(result.tobytes(), source.tobytes())


    def test_combined_quality_score_preserves_pure_punctuation(self) -> None:
        # Fullwidth periods / dots "．．．．" should not be penalized by character variance
        sentence_conf = 0.6913
        char_confidences = [0.6913, 0.6913, 0.6913, 0.6913]
        text = "．．．．"
        score = compute_ocr_quality_score(sentence_conf, char_confidences, source_text=text)
        self.assertIsNotNone(score)
        assert score is not None
        self.assertAlmostEqual(score, 0.6913, places=3)

    def test_restore_skipped_regions_pastes_original_image_patch(self) -> None:
        from src.pipeline import _restore_skipped_regions

        # Create original image with distinct blue patch at (10, 10, 30, 30)
        original_img = Image.new("RGB", (100, 100), "white")
        orig_arr = np.array(original_img)
        orig_arr[10:30, 10:30] = [0, 0, 255]
        original_img = Image.fromarray(orig_arr)

        # Inpainted/rendered image where the region was cleared to red
        rendered_img = Image.new("RGB", (100, 100), "white")
        rend_arr = np.array(rendered_img)
        rend_arr[10:30, 10:30] = [255, 0, 0]
        rendered_img = Image.fromarray(rend_arr)

        # Skipped region covering (10, 10, 30, 30)
        skipped_reg = TextRegion((10, 10, 30, 30), "SKIPPED_TEXT")
        restored = _restore_skipped_regions(rendered_img, original_img, [skipped_reg], [])

        # Verify the blue patch from original_img was restored over the red patch
        restored_arr = np.array(restored)
        self.assertTrue(np.all(restored_arr[15, 15] == [0, 0, 255]))

    def test_pipeline_restores_skipped_regions_end_to_end(self) -> None:
        class _MultiOCR:
            def detect(self, image: Image.Image) -> list[TextRegion]:
                return [
                    TextRegion((0, 0, 20, 20), "GOOD_TEXT", confidence=0.95),
                    TextRegion((50, 50, 70, 70), "SKIPPED_TEXT", confidence=0.10),
                ]

        class _MockTranslator:
            def translate(self, texts: list[str], context: str | list[str] | None = None) -> list[str]:
                return ["TRANS_TEXT"] * len(texts)

        class _MockInpainter:
            def inpaint(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
                arr = np.array(image)
                # Inpainter clears entire image to green
                arr[:, :] = [0, 255, 0]
                return Image.fromarray(arr)

        class _MockRenderer:
            def render(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
                return image

        class _MockAnchorDetector:
            def detect(self, image: Image.Image, regions: list[TextRegion]) -> None:
                pass

        # Original image has black pixel at (60, 60) inside skipped region
        orig = Image.new("RGB", (100, 100), "white")
        orig_arr = np.array(orig)
        orig_arr[60, 60] = [0, 0, 0]
        orig = Image.fromarray(orig_arr)

        pipeline = MangaTranslationPipeline(
            _MultiOCR(),
            _MockTranslator(),
            _MockInpainter(),
            _MockRenderer(),
            _MockAnchorDetector(),
            minimum_ocr_translation_confidence=0.50,
        )

        result, regions = pipeline.process(orig)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].source_text, "GOOD_TEXT")

        # The skipped region at (60, 60) must have its original black pixel restored from orig
        res_arr = np.array(result)
        self.assertTrue(np.all(res_arr[60, 60] == [0, 0, 0]))


if __name__ == "__main__":
    unittest.main()