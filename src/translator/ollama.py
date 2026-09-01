from __future__ import annotations

import json
from pathlib import Path

import requests

from .base import Translator


class TranslationError(RuntimeError):
    """Raised when Ollama cannot return a complete, ordered translation batch."""


class OllamaTranslator(Translator):
    def __init__(self, host: str, model: str, source_language: str, target_language: str, prompt_path: Path) -> None:
        self._endpoint = f"{host.rstrip('/')}/api/chat"
        self._model = model
        self._source_language = source_language
        self._target_language = target_language
        try:
            self._system_prompt = prompt_path.read_text(encoding="utf-8")
        except OSError as error:
            raise TranslationError(f"Could not read translation prompt file {prompt_path}: {error}") from error

    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        try:
            response = requests.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": self._build_system_prompt()},
                        {"role": "user", "content": self._build_input(texts)},
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return self._parse_response(content, len(texts))
        except TranslationError:
            raise
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            raise TranslationError(f"Ollama request to {self._endpoint} failed: {error}") from error

    def _build_system_prompt(self) -> str:
        return (
            self._system_prompt
            .replace("{{source_language}}", self._source_language)
            .replace("{{target_language}}", self._target_language)
        )

    @staticmethod
    def _build_input(texts: list[str]) -> str:
        entries = [{"id": index, "text": text} for index, text in enumerate(texts)]
        return f"Input: {json.dumps(entries, ensure_ascii=False)}"

    @staticmethod
    def _parse_response(content: str, expected_count: int) -> list[str]:
        payload = json.loads(content)
        entries = payload["translations"]
        if not isinstance(entries, list) or len(entries) != expected_count:
            raise TranslationError("Ollama returned an incomplete translation batch.")

        ordered: list[str | None] = [None] * expected_count
        for entry in entries:
            index, text = entry["id"], entry["text"]
            if not isinstance(index, int) or not 0 <= index < expected_count or not isinstance(text, str):
                raise TranslationError("Ollama returned an invalid translation entry.")
            if ordered[index] is not None:
                raise TranslationError("Ollama returned duplicate translation ids.")
            ordered[index] = text

        if any(text is None for text in ordered):
            raise TranslationError("Ollama did not return every translation id.")
        return [text for text in ordered if text is not None]