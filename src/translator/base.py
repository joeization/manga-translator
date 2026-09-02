from __future__ import annotations

from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def translate(self, texts: list[str], context_texts: list[str] | None = None) -> list[str]:
        """Translate texts while preserving input order and optionally using page context."""