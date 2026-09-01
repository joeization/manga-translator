from __future__ import annotations

from PIL import Image

from src.models import TextRegion

from .base import Inpainter


class NoopInpainter(Inpainter):
    def inpaint(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        return image