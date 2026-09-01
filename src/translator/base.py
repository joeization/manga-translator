from __future__ import annotations

from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def translate(self, texts: list[str]) -> list[str]:
        """Translate texts while preserving their input order."""