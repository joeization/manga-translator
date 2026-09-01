from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from src.models import TextRegion


class Renderer(ABC):
    @abstractmethod
    def render(self, image: Image.Image, regions: list[TextRegion]) -> Image.Image:
        """Render translated region text onto an inpainted image."""