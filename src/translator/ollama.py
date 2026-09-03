from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

import requests

from .base import Translator

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """Raised when Ollama cannot return a valid entry response."""


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_indexed_output(content: str, expected_count: int) -> list[str | None]:
    """Parse model output with [1], [2] or 1. tags into a list of translations."""
    results: list[str | None] = [None] * expected_count
    pattern = re.compile(r"^\s*\[?(\d+)\]?[\s.:：、\-—]*(.*)$")
    current_idx: int | None = None
    current_lines: list[str] = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() in ("user", "assistant", "system"):
            continue
        match = pattern.match(line)
        if match:
            if current_idx is not None and 1 <= current_idx <= expected_count:
                results[current_idx - 1] = " ".join(current_lines).strip()
            num_str, rest = match.groups()
            idx = int(num_str)
            current_idx = idx
            current_lines = [rest.strip()] if rest.strip() else []
        else:
            if current_idx is not None:
                current_lines.append(line)

    if current_idx is not None and 1 <= current_idx <= expected_count:
        results[current_idx - 1] = " ".join(current_lines).strip()

    return results


def format_response(text: str) -> str:
    """Remove leading ellipsis/dots/dashes from translated text. Preserves pure-ellipsis text."""
    if not text:
        return ""
    clean = re.sub(r"^[\s.…⋯︙·\-—~～]+", "", text).strip()
    # If stripping removed everything, keep the original (pure ellipsis / silence)
    ret = clean if clean else text.strip()
    # Normalize repeated dots/dashes only when there is actual content (not pure ellipsis)
    if clean:
        ret = ret.replace("．．．", "…")
        ret = ret.replace("ーーー", "ー")
        ret = ret.replace("---", "ー")
        ret = ret.replace("～～～", "~")
        ret = ret.replace("~~~", "~")
    return ret

class OllamaTranslator(Translator):
    def __init__(
        self,
        host: str,
        model: str,
        source_language: str,
        target_language: str,
        prompt_path: Path,
        session: requests.Session | None = None,
    ) -> None:
        self._endpoint = f"{host.rstrip('/')}/api/chat"
        self._model = model
        self._source_language = source_language
        self._target_language = target_language
        self._session = session if session is not None else requests.Session()
        try:
            self._system_prompt = prompt_path.read_text(encoding="utf-8")
        except OSError as error:
            raise TranslationError(f"Could not read prompt file {prompt_path}: {error}") from error

    def translate(self, texts: list[str], context: str | list[str] | None = None) -> list[str]:
        if not texts:
            return []

        if isinstance(context, list):
            context_str = "\n".join(c for c in context if c.strip())
        elif isinstance(context, str):
            context_str = context.strip()
        else:
            context_str = ""

        # Normalize texts: replace internal newlines within a single bubble so each is on one line
        sanitized_texts = [text.replace("\n", " ").strip() for text in texts]

        # 1. Attempt whole-page translation with previous page context
        try:
            whole_input = "\n".join(sanitized_texts)
            content = self._request(
                context=context_str,
                source_text=whole_input,
                stop=None,
                max_tokens=max(512, len(sanitized_texts) * 128),
            )
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if lines and lines[0].lower() in ("user", "assistant", "system"):
                lines = lines[1:]

            if len(lines) == len(sanitized_texts):
                logger.info("Whole-page translation succeeded (%d lines).", len(lines))
                return [format_response(l) for l in lines]

            # Also check if model produced indexed lines [1], [2] etc.
            indexed_try = _parse_indexed_output(content, len(sanitized_texts))
            if all(r is not None and r.strip() for r in indexed_try):
                logger.info("Whole-page indexed translation succeeded (%d lines).", len(indexed_try))
                return [format_response(r.strip()) for r in indexed_try if r is not None]

            logger.warning(
                "Whole-page translation returned %d lines, expected %d. Falling back to single-sentence translation without context.",
                len(lines),
                len(sanitized_texts),
            )
        except Exception as error:
            logger.warning(
                "Whole-page translation failed (%s). Falling back to single-sentence translation without context.",
                error,
            )

        # 2. Fast numbered batch attempt when there are multiple dialogues (> 2)
        if len(sanitized_texts) > 2:
            try:
                numbered_input = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(sanitized_texts))
                content = self._request(
                    context=context_str,
                    source_text=numbered_input,
                    stop=None,
                    max_tokens=max(512, len(sanitized_texts) * 128),
                )
                parsed = _parse_indexed_output(content, len(sanitized_texts))
                if all(p is not None and p.strip() for p in parsed):
                    logger.info("Numbered batch translation succeeded (%d lines).", len(parsed))
                    return [format_response(p.strip()) for p in parsed if p is not None]

                missing_indices = [i for i, p in enumerate(parsed) if p is None or not p.strip()]
                if len(missing_indices) < len(sanitized_texts) // 2:
                    logger.info("Numbered batch recovered %d of %d lines, translating %d missing lines individually.",
                                len(sanitized_texts) - len(missing_indices), len(sanitized_texts), len(missing_indices))
                    for idx in missing_indices:
                        single_res = self._request(
                            context="",
                            source_text=sanitized_texts[idx],
                            stop=["\n"],
                            max_tokens=256,
                        )
                        res_lines = [l.strip() for l in single_res.splitlines() if l.strip()]
                        clean = res_lines[0] if res_lines else single_res.strip()
                        clean = re.sub(r"^\s*\[?\d+\]?[\s.:：、\-—]*", "", clean).strip()
                        parsed[idx] = clean
                    return [format_response(p) if p is not None else sanitized_texts[i] for i, p in enumerate(parsed)]
            except Exception as error:
                logger.warning("Numbered batch attempt failed (%s). Falling back to line-by-line translation.", error)

        # 3. Fallback: translate line by line without context
        fallback_results: list[str] = []
        for text in sanitized_texts:
            single_res = self._request(
                context="",
                source_text=text,
                stop=["\n"],
                max_tokens=256,
            )
            res_lines = [l.strip() for l in single_res.splitlines() if l.strip()]
            clean = res_lines[0] if res_lines else single_res.strip()
            clean = re.sub(r"^\s*\[?\d+\]?[\s.:：、\-—]*", "", clean).strip()
            fallback_results.append(format_response(clean))

        return fallback_results

    def _build_system_prompt(self) -> str:
        prompt = (
            self._system_prompt
            .replace("{{source_language}}", self._source_language)
            .replace("{{target_language}}", self._target_language)
        )
        if "{{context}}" in prompt:
            prompt = prompt.replace("{{context}}", "")
        return prompt.strip()

    def _build_user_message(self, source_text: str, context: str = "") -> str:
        context = context.strip()
        if context:
            return (
                f"Previous translation context:\n{context}\n\n"
                f"Based on the context and storyline above, translate the following text from {self._source_language} into natural {self._target_language}:\n"
                f"{source_text}"
            )
        return f"Translate the following text from {self._source_language} into natural {self._target_language}:\n{source_text}"

    def _request(
        self,
        context: str,
        source_text: str,
        stop: list[str] | None = None,
        max_tokens: int = 512,
    ) -> str:
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(source_text, context)
        logger.info("Ollama request input:\n%s", user_message)
        options: dict[str, Any] = {
            "temperature": 0.3,
            "top_p": 0.8,
            "num_ctx": 4096,
            "num_predict": max_tokens,
        }
        if stop:
            options["stop"] = stop

        try:
            response = self._session.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "stream": False,
                    "think": False,
                    "options": options,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
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

        result = format_response(_strip_code_fence(content))
        if not result:
            raise TranslationError(f"Ollama response from {self._endpoint} contained no text.")
        return result

