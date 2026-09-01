from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from src.models import TextRegion


class Inpainter(ABC):
    @abstractmethod
    def inpaint(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        """Remove source text from the supplied regions and return the updated image."""