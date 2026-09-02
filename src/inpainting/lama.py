"""
LaMa (Large Mask inpainting) model adapter — not yet implemented.

To enable:
1. Download a LaMa checkpoint and set LAMA_MODEL_PATH in .env.
2. Implement the ``inpaint`` method using the LaMa inference API.
3. Set INPAINT_ENGINE=lama in .env.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.models import TextRegion

from .base import Inpainter


class LamaInpainter(Inpainter):
    """
    Placeholder for LaMa-based inpainting.

    LaMa produces high-quality reconstructions of complex backgrounds and
    detailed artwork that OpenCV TELEA cannot reproduce.  Implement this class
    once a suitable LaMa checkpoint and inference library are available.
    """

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        raise NotImplementedError(
            "LaMa inpainting is not yet implemented. "
            "Set INPAINT_ENGINE=opencv to use the OpenCV inpainter, "
            "or implement LamaInpainter in src/inpainting/lama.py."
        )

    def inpaint(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        raise NotImplementedError
