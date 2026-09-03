from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

from src.models import OCRResult

from .region_splitter import crop_for_ocr, region_has_text, resolve_overlapping_regions, split_text_regions

logger = logging.getLogger(__name__)


class MangaOCR:
    """Adapter around manga-ocr that keeps its dependency out of the pipeline."""

    def __init__(self, model_dir: Path, detector: object) -> None:
        self._model_dir = model_dir
        self._detector = detector
        self._engine = None

    def detect(self, image: Image.Image) -> list[OCRResult]:
        engine = self._get_engine()
        import numpy as np

        image_array = np.asarray(image.convert("RGB"))
        candidates = self._detector.detect(image)
        all_split_regions: list[OCRResult] = []
        for candidate in candidates:
            region = candidate if isinstance(candidate, OCRResult) else OCRResult(bbox=candidate, source_text="")
            all_split_regions.extend(split_text_regions(image, region, image_array=image_array))

        all_split_regions = resolve_overlapping_regions(all_split_regions, threshold=0.35)
        all_split_regions = [r for r in all_split_regions if region_has_text(image_array, r)]
        if not all_split_regions:
            return []

        crops = [crop_for_ocr(image, text_region) for text_region in all_split_regions]
        results = engine(crops)
        if isinstance(results, str):
            results = [results]

        regions: list[OCRResult] = []
        for text_region, text in zip(all_split_regions, results, strict=True):
            clean_text = text.strip() if text else ""
            if clean_text:
                text_region.source_text = clean_text
                logger.info("MangaOCR result: %s (confidence: unavailable)", clean_text)
                regions.append(text_region)
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