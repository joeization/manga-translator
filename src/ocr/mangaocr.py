from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

from src.models import OCRResult


class MangaOCR:
    """Adapter around manga-ocr that keeps its dependency out of the pipeline."""

    def __init__(self, model_dir: Path, detector: object) -> None:
        self._model_dir = model_dir
        self._detector = detector
        self._engine = None

    def detect(self, image: Image.Image) -> list[OCRResult]:
        engine = self._get_engine()
        regions: list[OCRResult] = []
        for candidate in self._detector.detect(image):
            region = candidate if isinstance(candidate, OCRResult) else OCRResult(bbox=candidate, source_text="")
            text = engine(image.crop(region.source_bbox or region.bbox)).strip()
            if text:
                region.source_text = text
                regions.append(region)
        return regions

    def _get_engine(self):
        try:
            from manga_ocr import MangaOcr
        except ImportError as error:
            raise RuntimeError(
                "MangaOCR is unavailable. Install the project requirements with the selected Python "
                "interpreter before processing images."
            ) from error

        if self._engine is None:
            self._model_dir.mkdir(parents=True, exist_ok=True)
            model_path = self._model_dir / "manga-ocr-base"
            if not (model_path / "config.json").is_file():
                snapshot_download("kha-white/manga-ocr-base", local_dir=model_path)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._engine = MangaOcr(pretrained_model_name_or_path=str(model_path))
        return self._engine