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
from .utils import UnionFind, estimate_adaptive_dilation, estimate_ink_color, slice_mask_to_roi

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


class LamaInpainter(Inpainter):
    """LaMa Manga neural inpainting using ONNX Runtime (GPU CUDA)."""

    def __init__(self, model_path: Path | None = None, device: str = "gpu", mask_dilation: int = 3) -> None:
        if not _ONNX_AVAILABLE:
            raise ImportError("onnxruntime package is required. Install via 'pip install onnxruntime-gpu'.")

        self._device = device.lower()
        self._mask_dilation = max(3, mask_dilation)
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
        output_pixels = img_rgb.copy()
        full_mask = np.zeros((h, w), dtype=np.uint8)
        valid_bboxes: list[tuple[int, int, int, int]] = []
        fast_path_count = 0
        for region in regions:
            mask, bbox = _extract_text_glyph_mask(img_rgb, region, h, w, mask_dilation=self._mask_dilation)
            if mask is not None and bbox is not None:
                region.inpaint_bbox = bbox
                region.inpaint_mask = mask
                rl, rt, rr, rb = bbox
                mask_uint8 = mask.astype(np.uint8) * 255
                mh, mw = mask_uint8.shape[:2]
                rh, rw = rb - rt, rr - rl
                mh_clip, mw_clip = min(mh, rh), min(mw, rw)
                if mh_clip <= 0 or mw_clip <= 0:
                    continue

                # Fast-path: Check if the non-text background of this bubble is pure white or flat solid color
                seg_mask = None
                if region.layout_mask is not None and region.layout_bbox is not None:
                    seg_mask = slice_mask_to_roi((rl, rt, rr, rb), region.layout_bbox, region.layout_mask)

                roi_patch = img_rgb[rt : rt + mh_clip, rl : rl + mw_clip]
                is_flat, flat_bg_color = _is_flat_background(roi_patch, mask[:mh_clip, :mw_clip], seg_mask)

                if is_flat and flat_bg_color is not None:
                    # Instant fill on pure white / flat solid background (bypasses neural inference)
                    mask_bool = mask[:mh_clip, :mw_clip]
                    output_pixels[rt : rt + mh_clip, rl : rl + mw_clip][mask_bool] = flat_bg_color
                    fast_path_count += 1
                else:
                    # Complex background (artwork / screentones) queued for neural inpainting
                    valid_bboxes.append(bbox)
                    full_mask[rt : rt + mh_clip, rl : rl + mw_clip] = np.maximum(
                        full_mask[rt : rt + mh_clip, rl : rl + mw_clip],
                        mask_uint8[:mh_clip, :mw_clip],
                    )

        if fast_path_count > 0:
            logger.info("LaMa: fast-path filled %d flat/white regions instantly", fast_path_count)

        if not np.any(full_mask):
            return Image.fromarray(output_pixels, mode="RGB")

        # Cluster nearby bboxes for high-resolution ROI patch inpainting
        roi_clusters = _cluster_bboxes(valid_bboxes, margin=48, img_w=w, img_h=h)
        logger.info("LaMa: running neural inpainting on %d regions across %d ROI clusters", len(valid_bboxes), len(roi_clusters))

        # Collect active ROI clusters for high-resolution patch inpainting
        active_clusters: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
        for crop_l, crop_t, crop_r, crop_b in roi_clusters:
            crop_mask = full_mask[crop_t:crop_b, crop_l:crop_r]
            if np.any(crop_mask):
                crop_img = img_rgb[crop_t:crop_b, crop_l:crop_r]
                active_clusters.append(((crop_l, crop_t, crop_r, crop_b), crop_img, crop_mask))

        if not active_clusters:
            return Image.fromarray(output_pixels, mode="RGB")

        # Batched neural inpainting on GPU
        inpainted_crops = self._run_onnx_patches(
            [c[1] for c in active_clusters],
            [c[2] for c in active_clusters],
        )

        for ((crop_l, crop_t, crop_r, crop_b), _, crop_mask), inpainted_crop in zip(active_clusters, inpainted_crops):
            mask_bool = crop_mask > 0
            output_pixels[crop_t:crop_b, crop_l:crop_r][mask_bool] = inpainted_crop[mask_bool]

        return Image.fromarray(output_pixels, mode="RGB")

    def _run_onnx_patch(self, image_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
        """Run GPU ONNX Runtime inpainting on an ROI patch."""
        return self._run_onnx_patches([image_patch], [mask_patch])[0]

    def _run_onnx_patches(
        self,
        image_patches: list[np.ndarray],
        mask_patches: list[np.ndarray],
        max_batch_size: int = 8,
    ) -> list[np.ndarray]:
        """Run GPU ONNX Runtime inpainting on multiple ROI patches using batched inference."""
        if not image_patches:
            return []

        results: list[np.ndarray] = []
        n = len(image_patches)

        for batch_start in range(0, n, max_batch_size):
            batch_end = min(n, batch_start + max_batch_size)
            batch_imgs_raw = image_patches[batch_start:batch_end]
            batch_masks_raw = mask_patches[batch_start:batch_end]
            b_size = len(batch_imgs_raw)

            batch_imgs = np.empty((b_size, 3, 512, 512), dtype=np.float32)
            batch_masks = np.empty((b_size, 1, 512, 512), dtype=np.float32)

            for i in range(b_size):
                r_img = cv2.resize(batch_imgs_raw[i], (512, 512), interpolation=cv2.INTER_AREA)
                r_mask = cv2.resize(batch_masks_raw[i], (512, 512), interpolation=cv2.INTER_NEAREST)

                img_f = r_img.astype(np.float32) / 255.0
                batch_imgs[i] = np.transpose(img_f, (2, 0, 1))
                batch_masks[i, 0] = (r_mask > 127).astype(np.float32)

            feeds = {}
            for name in self._input_names:
                if "mask" in name.lower() or name == self._input_names[1]:
                    feeds[name] = batch_masks
                else:
                    feeds[name] = batch_imgs

            out = self.session.run(self._output_names, feeds)
            batch_out = out[0]

            for i in range(b_size):
                patch_h, patch_w = batch_imgs_raw[i].shape[:2]
                res_i = batch_out[i]
                if res_i.shape[0] == 3:
                    res_i = np.transpose(res_i, (1, 2, 0))
                if res_i.max() <= 1.0:
                    res_i = res_i * 255.0
                out_uint8 = np.clip(np.round(res_i), 0, 255).astype(np.uint8)
                results.append(cv2.resize(out_uint8, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC))

        return results


def _is_flat_background(
    roi: np.ndarray,
    mask: np.ndarray,
    seg_mask: np.ndarray | None = None,
) -> tuple[bool, np.ndarray | None]:
    """Check if the non-text background of an ROI is uniform (pure white or solid color)."""
    mh, mw = mask.shape[:2]
    rh, rw = roi.shape[:2]
    h, w = min(mh, rh), min(mw, rw)
    if h <= 0 or w <= 0:
        return False, None

    roi_sub = roi[:h, :w]
    mask_sub = mask[:h, :w]

    if seg_mask is not None:
        sh, sw = seg_mask.shape[:2]
        sh_clip, sw_clip = min(h, sh), min(w, sw)
        bg_selector = seg_mask[:sh_clip, :sw_clip] & (~mask_sub[:sh_clip, :sw_clip])
        roi_eval = roi_sub[:sh_clip, :sw_clip]
    else:
        bg_selector = ~mask_sub
        roi_eval = roi_sub

    bg_pixels = roi_eval[bg_selector]
    if bg_pixels.shape[0] < 16:
        return False, None

    gray = cv2.cvtColor(roi_eval, cv2.COLOR_RGB2GRAY)
    bg_gray = gray[bg_selector]
    mean_val = float(np.mean(bg_gray))
    std_val = float(np.std(bg_gray))

    # Condition 1: Pure white speech bubble (standard in 80%+ manga dialogues)
    if mean_val >= 235 and std_val <= 10.0:
        bg_color = np.median(bg_pixels, axis=0).astype(roi.dtype)
        return True, bg_color

    # Condition 2: Solid flat color / tone (e.g. solid black or flat screentone tone)
    if std_val <= 3.5:
        bg_color = np.median(bg_pixels, axis=0).astype(roi.dtype)
        return True, bg_color

    return False, None


def _extract_text_glyph_mask(
    img_rgb: np.ndarray,
    region: TextRegion,
    img_h: int,
    img_w: int,
    mask_dilation: int = 1,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Extract precise text glyph mask, expanding to swallow outlines and clipped to bubble segmentation."""
    bbox = region.inpaint_bbox or region.source_bbox or region.bbox
    rl, rt, rr, rb = bbox

    # Dynamically estimate adaptive dilation from detect box, character count, and image resolution
    adaptive_dilation = estimate_adaptive_dilation(
        region.bbox,
        region.source_text,
        img_w,
        img_h,
        base_dilation=mask_dilation,
    )
    pad = max(6, adaptive_dilation + 2)

    pl, pt = max(0, rl - pad), max(0, rt - pad)
    pr, pb = min(img_w, rr + pad), min(img_h, rb + pad)
    if pr <= pl or pb <= pt:
        return None, None

    roi = img_rgb[pt:pb, pl:pr]
    if roi.size == 0:
        return None, None

    # Detect rect in ROI coordinates
    detect_rect = np.zeros(roi.shape[:2], dtype=bool)
    dt = max(0, rt - pt)
    db = min(pb - pt, rb - pt)
    dl = max(0, rl - pl)
    dr = min(pr - pl, rr - pl)
    detect_rect[dt:db, dl:dr] = True

    # Compute bubble segmentation mask in ROI local coordinates if available
    seg_mask: np.ndarray | None = None
    if isinstance(region.layout_mask, np.ndarray) and region.layout_bbox is not None:
        seg_mask = slice_mask_to_roi((pl, pt, pr, pb), region.layout_bbox, region.layout_mask)

    # Allowed boundary for glyph extraction and dilation:
    # Inside speech bubble (seg_mask), allow outward stroke extension (外擴) up to expanded_rect.
    # Without speech bubble (open scenes / background art), strictly confine to detect_rect.
    if seg_mask is not None and np.any(seg_mask):
        k_exp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (adaptive_dilation * 2 + 1, adaptive_dilation * 2 + 1))
        expanded_rect = cv2.dilate(detect_rect.astype(np.uint8), k_exp).astype(bool)
        allowed_boundary = expanded_rect & seg_mask
    else:
        allowed_boundary = detect_rect

    # 1. Prioritize explicit ocr_mask if available (clipped to interior)
    if isinstance(region.ocr_mask, np.ndarray) and region.ocr_mask.shape[:2] == roi.shape[:2]:
        mask = region.ocr_mask.astype(bool)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        if np.any(mask):
            ks = adaptive_dilation * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            dilated = dilated & allowed_boundary
            if np.any(dilated):
                return dilated, (pl, pt, pr, pb)

    # 2. Extract text glyph strokes and stroke outlines (描邊 / 袋文字)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    p75 = float(np.percentile(gray, 75))

    # Detect stroke outlines (描邊) via local color deviation from surrounding background
    bg_color = np.median(roi, axis=(0, 1))
    color_dist = np.linalg.norm(roi.astype(float) - bg_color, axis=2)
    outline_mask = color_dist >= 25.0

    # Dark text or text with outlines on light background (standard manga)
    if p75 >= 128:
        raw_glyph_mask = (gray <= min(190, p75 - 25)) | outline_mask
    else:  # Light text on dark background
        p25 = float(np.percentile(gray, 25))
        raw_glyph_mask = (gray >= max(100, p25 + 25)) | outline_mask

    # Crucial: Restrict raw glyph extraction strictly to allowed boundary (inside rect & segmentation interior)
    raw_glyph_mask = raw_glyph_mask & allowed_boundary

    # Filter out tiny screentone noise specks using fast 2x2 morphological opening
    k_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    glyph_mask = cv2.morphologyEx(raw_glyph_mask.astype(np.uint8), cv2.MORPH_OPEN, k_noise).astype(bool)
    if not np.any(glyph_mask):
        glyph_mask = raw_glyph_mask

    if np.any(glyph_mask):
        # Dilate glyph mask using actual measured glyph font size and stroke width to swallow text outlines (描邊)
        actual_dilation = estimate_adaptive_dilation(
            region.bbox,
            region.source_text,
            img_w,
            img_h,
            base_dilation=mask_dilation,
            glyph_mask=glyph_mask,
        )
        ks = actual_dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        dilated_glyph = cv2.dilate(glyph_mask.astype(np.uint8), kernel, iterations=1) > 0

        # Confine glyph dilation strictly inside allowed boundary (never outside rect, never outside segmentation)
        dilated_glyph = dilated_glyph & allowed_boundary

        if np.any(dilated_glyph):
            return dilated_glyph, (pl, pt, pr, pb)

    # 3. Fallback for detect box (rect inpaint):
    # Strictly clip detect box within segmentation interior (NEVER dilate rect outwards)
    if seg_mask is not None and np.any(seg_mask):
        clipped_rect = detect_rect & allowed_boundary
        if np.any(clipped_rect):
            return clipped_rect, (pl, pt, pr, pb)

    return None, None


def _cluster_bboxes(
    bboxes: list[tuple[int, int, int, int]], margin: int, img_w: int, img_h: int
) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []

    clusters = [
        (max(0, l - margin), max(0, t - margin), min(img_w, r + margin), min(img_h, b + margin))
        for l, t, r, b in bboxes
    ]

    while True:
        n = len(clusters)
        if n <= 1:
            break
        uf = UnionFind(n)
        has_merge = False
        for i in range(n):
            il, it, ir, ib = clusters[i]
            for j in range(i + 1, n):
                jl, jt, jr, jb = clusters[j]
                if max(il, jl) < min(ir, jr) and max(it, jt) < min(ib, jb):
                    uf.union(i, j)
                    has_merge = True

        if not has_merge:
            break

        new_clusters: list[tuple[int, int, int, int]] = []
        for g_idxs in uf.groups().values():
            gl = min(clusters[i][0] for i in g_idxs)
            gt = min(clusters[i][1] for i in g_idxs)
            gr = max(clusters[i][2] for i in g_idxs)
            gb = max(clusters[i][3] for i in g_idxs)
            new_clusters.append((gl, gt, gr, gb))
        clusters = new_clusters

    return clusters
