"""
LaMa Manga ONNX Inpainter using ONNX Runtime (mayocream/lama-manga-onnx).
Pure ONNX Runtime implementation with zero iopaint dependency.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.models import TextRegion

from .base import Inpainter
from .utils import estimate_ink_color

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


class LamaInpainter(Inpainter):
    """LaMa Manga neural inpainting using ONNX Runtime (GPU CUDA)."""

    def __init__(self, model_path: Path | None = None, device: str = "gpu") -> None:
        if not _ONNX_AVAILABLE:
            raise ImportError("onnxruntime package is required. Install via 'pip install onnxruntime-gpu'.")

        self._device = device.lower()
        self._model_path = self._resolve_model_path(model_path)
        logger.info("Initializing LaMa Manga ONNX model on GPU from: %s", self._model_path)

        # Register PyTorch CUDA/cuDNN DLL paths so ONNX Runtime loads CUDAExecutionProvider on Windows
        try:
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ("CUDAExecutionProvider" in available_providers and self._device != "cpu")
            else ["CPUExecutionProvider"]
        )

        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        self.session = ort.InferenceSession(str(self._model_path), sess_options=opts, providers=providers)
        logger.info("ONNX Runtime session initialized on providers: %s", self.session.get_providers())

        self._input_names = [inp.name for inp in self.session.get_inputs()]
        self._output_names = [out.name for out in self.session.get_outputs()]

    def _resolve_model_path(self, model_path: Path | None) -> Path:
        if model_path and model_path.is_file():
            return model_path
        if model_path and model_path.is_dir():
            files = list(model_path.glob("*.onnx"))
            if files:
                return files[0]

        candidates = [
            Path("models/inpainting_lama/lama-manga.onnx"),
            Path("models/lama-manga.onnx"),
            Path("models/inpainting_lama/lama_manga.onnx"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            "lama-manga.onnx not found. Place lama-manga.onnx in models/inpainting_lama/"
        )

    def inpaint(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        if not regions:
            return image

        img_rgb = np.array(image.convert("RGB"))
        h, w = img_rgb.shape[:2]

        # Estimate and preserve original text ink_color for downstream renderer
        for region in regions:
            ink_color = estimate_ink_color(img_rgb, region)
            region.metadata = {**(region.metadata or {}), "ink_color": ink_color}

        # Build composite mask for original text glyphs (prioritizing glyph strokes over full bubble)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        valid_bboxes: list[tuple[int, int, int, int]] = []
        for region in regions:
            mask, bbox = _extract_text_glyph_mask(img_rgb, region, h, w)
            if mask is not None and bbox is not None:
                valid_bboxes.append(bbox)
                rl, rt, rr, rb = bbox
                mask_uint8 = mask.astype(np.uint8) * 255
                mh, mw = mask_uint8.shape[:2]
                rh, rw = rb - rt, rr - rl
                mh_clip, mw_clip = min(mh, rh), min(mw, rw)
                if mh_clip > 0 and mw_clip > 0:
                    full_mask[rt : rt + mh_clip, rl : rl + mw_clip] = np.maximum(
                        full_mask[rt : rt + mh_clip, rl : rl + mw_clip],
                        mask_uint8[:mh_clip, :mw_clip],
                    )

        if not np.any(full_mask):
            return image

        # Cluster nearby bboxes for high-resolution ROI patch inpainting
        roi_clusters = _cluster_bboxes(valid_bboxes, margin=48, img_w=w, img_h=h)

        output_pixels = img_rgb.copy()
        for crop_l, crop_t, crop_r, crop_b in roi_clusters:
            crop_img = img_rgb[crop_t:crop_b, crop_l:crop_r]
            crop_mask = full_mask[crop_t:crop_b, crop_l:crop_r]
            if not np.any(crop_mask):
                continue

            inpainted_crop = self._run_onnx_patch(crop_img, crop_mask)
            mask_bool = crop_mask > 0
            output_pixels[crop_t:crop_b, crop_l:crop_r][mask_bool] = inpainted_crop[mask_bool]

        return Image.fromarray(output_pixels, mode="RGB")

    def _run_onnx_patch(self, image_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
        """Run GPU ONNX Runtime inpainting on an ROI patch."""
        patch_h, patch_w = image_patch.shape[:2]

        resized_img = cv2.resize(image_patch, (512, 512), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask_patch, (512, 512), interpolation=cv2.INTER_NEAREST)

        img_f = resized_img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_f, (2, 0, 1))[np.newaxis, ...]

        mask_bin = (resized_mask > 127).astype(np.float32)
        mask_chw = mask_bin[np.newaxis, np.newaxis, ...]

        feeds = {}
        for name in self._input_names:
            if "mask" in name.lower() or name == self._input_names[1]:
                feeds[name] = mask_chw.astype(np.float32)
            else:
                feeds[name] = img_chw.astype(np.float32)

        out = self.session.run(self._output_names, feeds)
        result = out[0]

        if result.ndim == 4:
            result = result[0]
        if result.shape[0] == 3:
            result = np.transpose(result, (1, 2, 0))

        if result.max() <= 1.0:
            result = result * 255.0

        out_uint8 = np.clip(np.round(result), 0, 255).astype(np.uint8)
        return cv2.resize(out_uint8, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC)


def _extract_text_glyph_mask(
    img_rgb: np.ndarray,
    region: TextRegion,
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Extract precise text glyph mask from original image patch."""
    bbox = region.inpaint_bbox or region.source_bbox or region.bbox
    rl, rt, rr, rb = bbox
    rl, rt = max(0, rl), max(0, rt)
    rr, rb = min(img_w, rr), min(img_h, rb)
    if rr <= rl or rb <= rt:
        return None, None

    roi = img_rgb[rt:rb, rl:rr]
    if roi.size == 0:
        return None, None

    # 1. Prioritize explicit ocr_mask or inpaint_mask if available
    if isinstance(region.ocr_mask, np.ndarray) and region.ocr_mask.shape[:2] == roi.shape[:2]:
        mask = region.ocr_mask.astype(bool)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        if np.any(mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2).astype(bool)
            return dilated, (rl, rt, rr, rb)

    if isinstance(region.inpaint_mask, np.ndarray) and region.inpaint_mask.shape[:2] == roi.shape[:2]:
        mask = region.inpaint_mask.astype(bool)
        if np.any(mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2).astype(bool)
            return dilated, (rl, rt, rr, rb)

    # 2. Extract text glyph strokes via local contrast & luminance thresholding
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    edge_mask = np.ones(roi.shape[:2], dtype=bool)
    if roi.shape[0] > 6 and roi.shape[1] > 6:
        edge_mask[3:-3, 3:-3] = False
    bg_color = np.median(roi[edge_mask], axis=0) if np.any(edge_mask) else np.median(roi.reshape(-1, 3), axis=0)
    bg_luma = float(0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])

    color_dist = np.linalg.norm(roi.astype(float) - bg_color, axis=2)

    if bg_luma >= 128:
        glyph_mask = (gray <= 170) | (color_dist >= 30.0)
    else:
        glyph_mask = (gray >= 110) | (color_dist >= 30.0)

    glyph_uint8 = glyph_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_glyph = cv2.dilate(glyph_uint8, kernel, iterations=2) > 0

    if np.any(dilated_glyph):
        return dilated_glyph, (rl, rt, rr, rb)

    return np.ones((rb - rt, rr - rl), dtype=bool), (rl, rt, rr, rb)


def _cluster_bboxes(
    bboxes: list[tuple[int, int, int, int]], margin: int, img_w: int, img_h: int
) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []

    merged = [
        (max(0, l - margin), max(0, t - margin), min(img_w, r + margin), min(img_h, b + margin))
        for l, t, r, b in bboxes
    ]

    while True:
        changed = False
        new_merged: list[tuple[int, int, int, int]] = []
        for cur_l, cur_t, cur_r, cur_b in merged:
            placed = False
            for idx, (ml, mt, mr, mb) in enumerate(new_merged):
                if max(cur_l, ml) < min(cur_r, mr) and max(cur_t, mt) < min(cur_b, mb):
                    new_merged[idx] = (min(cur_l, ml), min(cur_t, mt), max(cur_r, mr), max(cur_b, mb))
                    placed = True
                    changed = True
                    break
            if not placed:
                new_merged.append((cur_l, cur_t, cur_r, cur_b))
        merged = new_merged
        if not changed:
            break

    return merged
