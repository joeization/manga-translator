from __future__ import annotations

import logging
from pathlib import Path

import requests

from .base import Translator

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """Raised when Ollama cannot return a complete, ordered line batch."""


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


class _OllamaLineClient:
    """Sends a one-line-per-entry request to Ollama and parses a matching line-per-entry reply."""

    def __init__(self, host: str, model: str, prompt_path: Path) -> None:
        self._endpoint = f"{host.rstrip('/')}/api/chat"
        self._model = model
        try:
            self._system_prompt = prompt_path.read_text(encoding="utf-8")
        except OSError as error:
            raise TranslationError(f"Could not read prompt file {prompt_path}: {error}") from error

    def _process(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        return self._request(texts)

    def _build_system_prompt(self) -> str:
        return self._system_prompt

    def _request(self, texts: list[str]) -> list[str]:
        user_input = self._build_input(texts)
        logger.info("Ollama request input:\n%s", user_input)
        try:
            response = requests.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_ctx": 8192,
                        "num_predict": max(512, len(texts) * 256),
                    },
                    "messages": [
                        {"role": "system", "content": self._build_system_prompt()},
                        {"role": "user", "content": user_input},
                    ],
                },
                timeout=120,
            )
        except requests.RequestException as error:
            raise TranslationError(f"Ollama request to {self._endpoint} failed: {error}") from error

        if not response.ok:
            body = response.text[:500]
            raise TranslationError(f"Ollama request to {self._endpoint} failed with status {response.status_code}: {body}")

        if not response.text.strip():
            raise TranslationError(f"Ollama request to {self._endpoint} returned an empty response body.")

        try:
            content = response.json()["message"]["content"]
        except (KeyError, TypeError, ValueError) as error:
            body = response.text[:500]
            raise TranslationError(f"Ollama response from {self._endpoint} was not valid: {error} (body: {body!r})") from error

        logger.info("Ollama response content:\n%s", content)

        return self._parse_response(content, len(texts))

    @staticmethod
    def _build_input(texts: list[str]) -> str:
        return "Input:\n" + "\n".join(texts)

    @staticmethod
    def _parse_response(content: str, expected_count: int) -> list[str]:
        lines = [line.strip() for line in _strip_code_fence(content).splitlines() if line.strip()]
        # Some models leak the chat template's role marker as a stray first line.
        if lines and lines[0].lower() in ("user", "assistant", "system"):
            lines = lines[1:]
        if len(lines) != expected_count:
            raise TranslationError(
                f"Ollama returned {len(lines)} lines, expected {expected_count} (content: {content[:500]!r})"
            )
        return lines


class OllamaTranslator(Translator, _OllamaLineClient):
    def __init__(self, host: str, model: str, source_language: str, target_language: str, prompt_path: Path) -> None:
        super().__init__(host, model, prompt_path)
        self._source_language = source_language
        self._target_language = target_language

    def translate(self, texts: list[str]) -> list[str]:
        return self._process(texts)

    def _build_system_prompt(self) -> str:
        return (
            self._system_prompt
            .replace("{{source_language}}", self._source_language)
            .replace("{{target_language}}", self._target_language)
        )


class OllamaCorrector(_OllamaLineClient):
    def correct(self, texts: list[str]) -> list[str]:
        return self._process(texts)

