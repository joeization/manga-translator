from __future__ import annotations

import logging
from pathlib import Path
from statistics import median

import requests

from src.models import OCRResult

from .base import Translator

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """Raised when Ollama cannot return a valid entry response."""


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


class _OllamaEntryClient:
    """Sends one Ollama request per entry while retaining the page's text as context."""

    def __init__(self, host: str, model: str, prompt_path: Path, session: requests.Session | None = None) -> None:
        self._endpoint = f"{host.rstrip('/')}/api/chat"
        self._model = model
        self._session = session if session is not None else requests.Session()
        try:
            self._system_prompt = prompt_path.read_text(encoding="utf-8")
        except OSError as error:
            raise TranslationError(f"Could not read prompt file {prompt_path}: {error}") from error

    def _process(self, texts: list[str], context_texts: list[str] | None = None, correct: bool = False) -> list[str]:
        if not texts:
            return []
        context = "\n".join(context_texts if context_texts is not None else texts)
        return [self._request(context, text, correct) for text in texts]

    def _build_system_prompt(self, context: str) -> str:
        return self._system_prompt.replace("{{context}}", context)

    def _build_user_message(self, source_text: str) -> str:
        return source_text

    def _request(self, context: str, source_text: str, correct: bool) -> str:
        system_prompt = self._build_system_prompt(context)
        logger.info("Ollama request for entry: %s", source_text)
        try:
            if correct:
                temperature = 0
                top_p = 0.9
            else:
                temperature = 0.3
                top_p = 0.8
            response = self._session.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_ctx": 8192,
                        "num_predict": 256,
                        "stop": ["\n"],
                    },
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._build_user_message(source_text)},
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

        result = _strip_code_fence(content)
        if not result:
            raise TranslationError(f"Ollama response from {self._endpoint} contained no text.")
        if len(result.splitlines()) != 1:
            raise TranslationError(
                f"Ollama response from {self._endpoint} contained multiple lines for one entry: {result[:500]!r}"
            )
        return result


class OllamaTranslator(Translator, _OllamaEntryClient):
    def __init__(
        self,
        host: str,
        model: str,
        source_language: str,
        target_language: str,
        prompt_path: Path,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(host, model, prompt_path, session=session)
        self._source_language = source_language
        self._target_language = target_language

    def translate(self, texts: list[str], context_texts: list[str] | None = None) -> list[str]:
        return self._process(texts, context_texts, correct=False)

    def _build_system_prompt(self, context: str) -> str:
        return (
            super()
            ._build_system_prompt(context)
            .replace("{{source_language}}", self._source_language)
            .replace("{{target_language}}", self._target_language)
        )

    def _build_user_message(self, source_text: str) -> str:
        return f"Translate this {self._source_language} text into {self._target_language}: {source_text}"


class OllamaCorrector(_OllamaEntryClient):
    def __init__(
        self,
        host: str,
        model: str,
        source_language: str,
        prompt_path: Path,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(host, model, prompt_path, session=session)
        self._source_language = source_language

    def correct(self, texts: list[str], ocr_results: list[OCRResult] | None = None) -> list[str]:
        context = "\n".join(texts)
        entry_lengths = [_meaningful_character_count(text) for text in texts]
        results = ocr_results if ocr_results is not None else [None] * len(texts)
        if len(results) != len(texts):
            raise ValueError("OCR results must match the number of texts.")
        available_confidences = [result.confidence for result in results if result is not None and result.confidence is not None]
        corrections: list[str] = []
        for text, result in zip(texts, results, strict=True):
            if not _should_correct_ocr_entry(text, entry_lengths, result, available_confidences):
                logger.info("Skipping OCR correction for simple entry: %s", text)
                corrections.append(text)
                continue
            corrections.append(self._request(context, text))
        return corrections

    def _build_system_prompt(self, context: str) -> str:
        return super()._build_system_prompt(context).replace("{{source_language}}", self._source_language)

    def _build_user_message(self, source_text: str) -> str:
        return f"{self._source_language} OCR correction only; do not translate: {source_text}"

    def _request(self, context: str, source_text: str) -> str:
        result = super()._request(context, source_text, correct=True)
        logger.info("OCR correction result: %s -> %s", source_text, result)
        if _contains_japanese_kana(source_text) and not _contains_japanese_kana(result):
            logger.warning("OCR correction replaced Japanese text with a non-Japanese response; using the original OCR text.")
            return source_text
        return result


def _contains_japanese_kana(text: str) -> bool:
    return any("\u3040" <= character <= "\u30ff" for character in text)


def _should_correct_ocr_entry(
    text: str, entry_lengths: list[int], result: OCRResult | None, available_confidences: list[float]
) -> bool:
    characters = [character for character in text if character.isalnum()]
    if not characters:
        return False
    if result is not None and result.confidence is not None and len(available_confidences) > 1:
        return result.confidence < median(available_confidences)
    return len(characters) > median(entry_lengths) and len(set(characters)) ** 2 > len(characters)


def _meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)

