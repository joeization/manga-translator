# Copyright 2026 genshiai-daichi / Baberu OCR Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import math
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
from .region_splitter import crop_for_ocr, resolve_overlapping_regions, split_text_regions

logger = logging.getLogger(__name__)


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
        import numpy as np

        image_array = np.asarray(image.convert("RGB"))
        candidates = self._detector.detect(image)
        all_split_regions: list[OCRResult] = []
        for candidate in candidates:
            region = candidate if isinstance(candidate, OCRResult) else OCRResult(bbox=candidate, source_text="")
            all_split_regions.extend(split_text_regions(image, region, image_array=image_array))

        all_split_regions = resolve_overlapping_regions(all_split_regions, threshold=0.35)
        regions: list[OCRResult] = []
        for text_region in all_split_regions:
            text, character_confidences = self._recognize(crop_for_ocr(image, text_region))
            text, character_confidences = _strip_text_and_confidences(text, character_confidences)
            if text:
                text_region.source_text = text
                text_region.character_confidences = character_confidences
                if character_confidences is not None:
                    text_region.confidence = _sentence_confidence(character_confidences)
                    text_region.metadata = {"confidence_type": "geometric_mean_generated_token_probability"}
                logger.info(
                    "Baberu OCR result: %s (confidence: %s, character confidences: %s)",
                    text,
                    f"{text_region.confidence:.4f}" if text_region.confidence is not None else "unavailable",
                    character_confidences if character_confidences is not None else "unavailable",
                )
                regions.append(text_region)
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
    def _recognize(self, image: Image.Image) -> tuple[str, list[float] | None]:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._image_processor is not None
        assert self._device is not None
        pixel_values = self._image_processor(image.convert("RGB"), return_tensors="pt")["pixel_values"].to(self._device)
        input_ids = torch.tensor([[self._tokenizer.bos_token_id]], device=self._device)
        output = self._model.generate(
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
            return_dict_in_generate=True,
            output_scores=True,
        )
        return _decode_with_character_confidences(
            output.sequences[0, input_ids.shape[1]:].cpu().tolist(),
            output.scores,
            self._tokenizer,
            self._tokenizer.eos_token_id,
        )


def _is_content_character(character: str) -> bool:
    return len(character) == 1 and character not in "ーｰ〜~" and unicodedata.category(character)[0] in "LN"


def _decode_with_character_confidences(
    token_ids: list[int], scores: tuple[torch.Tensor, ...], tokenizer: BaberuTokenizer, eos_token_id: int
) -> tuple[str, list[float] | None]:
    if len(token_ids) != len(scores):
        return tokenizer.decode(token_ids, skip_special_tokens=True), None

    characters: list[str] = []
    character_confidences: list[float] = []
    for index, (token_id, logits) in enumerate(zip(token_ids, scores, strict=True)):
        if token_id == eos_token_id and index == len(token_ids) - 1:
            continue
        character = tokenizer.decode([token_id], skip_special_tokens=False)
        if len(character) != 1:
            return tokenizer.decode(token_ids, skip_special_tokens=True), None
        probability = torch.softmax(logits[0].float(), dim=-1)[token_id].item()
        characters.append(character)
        character_confidences.append(float(probability))
    return "".join(characters), character_confidences


def _strip_text_and_confidences(text: str, character_confidences: list[float] | None) -> tuple[str, list[float] | None]:
    left_trim = len(text) - len(text.lstrip())
    right_trim = len(text) - len(text.rstrip())
    stripped_text = text.strip()
    if character_confidences is None or len(character_confidences) != len(text):
        return stripped_text, None
    end = len(character_confidences) - right_trim if right_trim else len(character_confidences)
    return stripped_text, character_confidences[left_trim:end]


def _sentence_confidence(character_confidences: list[float]) -> float:
    if not character_confidences or any(confidence <= 0 for confidence in character_confidences):
        return 0.0
    return math.exp(sum(math.log(confidence) for confidence in character_confidences) / len(character_confidences))