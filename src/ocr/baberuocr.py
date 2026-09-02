from __future__ import annotations

import unicodedata
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from transformers import LogitsProcessor, LogitsProcessorList

from src.models import OCRResult

from .baberu.configuration_baberu import BaberuOCRConfig
from .baberu.modeling_baberu import BaberuOCRModel
from .baberu.tokenization_baberu import BaberuTokenizer


class CapContentRun(LogitsProcessor):
    def __init__(self, content_ids: set[int], max_run: int = 12) -> None:
        self._content_ids = content_ids
        self._max_run = max_run

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for batch_index in range(input_ids.shape[0]):
            token_id = int(input_ids[batch_index, -1])
            if token_id not in self._content_ids:
                continue
            run_length = 0
            for position in range(input_ids.shape[1] - 1, -1, -1):
                if int(input_ids[batch_index, position]) != token_id:
                    break
                run_length += 1
                if run_length >= self._max_run:
                    scores[batch_index, token_id] = float("-inf")
                    break
        return scores


class BaberuOCR:
    """Recognize post-processed manga regions with the local Baberu checkpoint."""

    def __init__(self, model_dir: Path, detector: object, device: str | None = None) -> None:
        self._model_dir = model_dir
        self._detector = detector
        self._requested_device = device
        self._model: BaberuOCRModel | None = None
        self._tokenizer: BaberuTokenizer | None = None
        self._image_processor = None
        self._device: torch.device | None = None
        self._content_ids: set[int] = set()

    def detect(self, image: Image.Image) -> list[OCRResult]:
        self._get_engine()
        regions: list[OCRResult] = []
        for candidate in self._detector.detect(image):
            region = candidate if isinstance(candidate, OCRResult) else OCRResult(bbox=candidate, source_text="")
            text = self._recognize(image.crop(region.source_bbox or region.bbox)).strip()
            if text:
                region.source_text = text
                regions.append(region)
        return regions

    def _get_engine(self) -> None:
        if self._model is not None:
            return
        vision_model_dir = self._model_dir / "vision_encoder"
        required_paths = (
            self._model_dir / "config.json",
            self._model_dir / "model.safetensors",
            self._model_dir / "tokenizer" / "vocab.json",
            vision_model_dir / "config.json",
            vision_model_dir / "model.safetensors",
            vision_model_dir / "preprocessor_config.json",
        )
        missing_paths = [str(path) for path in required_paths if not path.is_file()]
        if missing_paths:
            raise RuntimeError(f"Baberu OCR model is incomplete. Missing: {', '.join(missing_paths)}")

        device = torch.device(self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        config = BaberuOCRConfig.from_pretrained(self._model_dir)
        config.vision_model_name = str(vision_model_dir)
        model = BaberuOCRModel(config)
        model.load_state_dict(load_safetensors(self._model_dir / "model.safetensors"), strict=False)
        model.to(device=device, dtype=dtype).eval()

        from transformers import AutoImageProcessor

        tokenizer = BaberuTokenizer.from_pretrained(self._model_dir / "tokenizer", local_files_only=True)
        self._model = model
        self._tokenizer = tokenizer
        self._image_processor = AutoImageProcessor.from_pretrained(
            vision_model_dir,
            do_center_crop=False,
            size={"height": 224, "width": 224},
            crop_size=None,
            local_files_only=True,
        )
        self._device = device
        self._content_ids = {
            token_id
            for token_id in range(tokenizer.vocab_size)
            if _is_content_character(tokenizer.decode([token_id], skip_special_tokens=False))
        }

    @torch.inference_mode()
    def _recognize(self, image: Image.Image) -> str:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._image_processor is not None
        assert self._device is not None
        pixel_values = self._image_processor(image.convert("RGB"), return_tensors="pt")["pixel_values"].to(self._device)
        input_ids = torch.tensor([[self._tokenizer.bos_token_id]], device=self._device)
        output_ids = self._model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.2,
            logits_processor=LogitsProcessorList([CapContentRun(self._content_ids)]),
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            use_cache=True,
        )
        return self._tokenizer.decode(output_ids[0, input_ids.shape[1]:].cpu().tolist(), skip_special_tokens=True)


def _is_content_character(character: str) -> bool:
    return len(character) == 1 and character not in "ーｰ〜~" and unicodedata.category(character)[0] in "LN"