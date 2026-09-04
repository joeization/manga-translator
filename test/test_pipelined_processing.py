from __future__ import annotations

import unittest
from dataclasses import dataclass
from threading import Event

from PIL import Image

from src.models import TextRegion
from src.pipeline import MangaTranslationPipeline, PipelineError


@dataclass
class _MockItem:
    name: str
    stem: str
    sub_dir: str
    image: Image.Image
    should_fail: bool = False

    def load(self) -> Image.Image:
        if self.should_fail:
            raise RuntimeError("Corrupt image in OCR")
        return self.image.copy()


class _MockOCR:
    def detect(self, image: Image.Image) -> list[TextRegion]:
        return [TextRegion((0, 0, 10, 10), "SRC_TXT")]


class _MockTranslator:
    def translate(self, texts: list[str], context: str | list[str] | None = None) -> list[str]:
        return [f"TRANS:{t}" for t in texts]


class _MockAnchorDetector:
    def detect(self, image: Image.Image, regions: list[TextRegion]) -> None:
        pass


class _MockInpainter:
    def inpaint(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        return image.copy()


class _MockRenderer:
    def render(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        # Mark image with a distinct pixel to prove it was rendered
        rendered = image.copy()
        rendered.putpixel((0, 0), (123, 123, 123))
        return rendered


class _FailingOCR:
    def __init__(self, fail_names: set[str]) -> None:
        self._fail_names = fail_names

    def detect(self, image: Image.Image) -> list[TextRegion]:
        if getattr(image, "_fail", False):
            raise RuntimeError("Corrupt image in OCR")
        return [TextRegion((0, 0, 10, 10), "ok")]


class PipelinedProcessingTests(unittest.TestCase):
    def test_pipelined_processes_all_items_in_order(self) -> None:
        pipeline = MangaTranslationPipeline(
            _MockOCR(),
            _MockTranslator(),
            _MockInpainter(),
            _MockRenderer(),
            _MockAnchorDetector(),
        )
        items = [
            _MockItem(name=f"page_{i}.jpg", stem=f"page_{i}", sub_dir="", image=Image.new("RGB", (20, 20), (i, i, i)))
            for i in range(5)
        ]

        results = list(pipeline.process_pipelined(items))
        self.assertEqual(len(results), 5)

        for i, (item, orig, trans, regions, err) in enumerate(results):
            self.assertEqual(item.name, f"page_{i}.jpg")
            self.assertIsNone(err)
            self.assertIsNotNone(orig)
            self.assertIsNotNone(trans)
            assert trans is not None
            # Check renderer modified pixel (0,0)
            self.assertEqual(trans.getpixel((0, 0)), (123, 123, 123))
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0].translated_text, "TRANS:SRC_TXT")

    def test_pipelined_handles_stage_failure_cleanly(self) -> None:
        ocr = _FailingOCR(fail_names={"page_1.jpg"})
        pipeline = MangaTranslationPipeline(
            ocr,
            _MockTranslator(),
            _MockInpainter(),
            _MockRenderer(),
            _MockAnchorDetector(),
        )

        img_ok = Image.new("RGB", (20, 20), (10, 10, 10))
        img_bad = Image.new("RGB", (20, 20), (20, 20, 20))

        items = [
            _MockItem("page_0.jpg", "page_0", "", img_ok),
            _MockItem("page_1.jpg", "page_1", "", img_bad, should_fail=True),
            _MockItem("page_2.jpg", "page_2", "", img_ok),
        ]

        results = list(pipeline.process_pipelined(items))
        self.assertEqual(len(results), 3)

        # page 0 succeeds
        self.assertIsNone(results[0][4])
        # page 1 fails in OCR stage
        self.assertIsNotNone(results[1][4])
        assert results[1][4] is not None
        self.assertEqual(results[1][4].stage, "OCR")
        # page 2 still succeeds uninterrupted
        self.assertIsNone(results[2][4])

    def test_pipelined_cancels_responsively(self) -> None:
        pipeline = MangaTranslationPipeline(
            _MockOCR(),
            _MockTranslator(),
            _MockInpainter(),
            _MockRenderer(),
            _MockAnchorDetector(),
        )
        cancel_event = Event()
        cancel_event.set()  # Cancel immediately

        items = [
            _MockItem(f"page_{i}.jpg", f"page_{i}", "", Image.new("RGB", (20, 20)))
            for i in range(10)
        ]

        results = list(pipeline.process_pipelined(items, cancel_event=cancel_event))
        # Should terminate quickly without processing all 10 items
        self.assertLess(len(results), 10)


if __name__ == "__main__":
    unittest.main()

