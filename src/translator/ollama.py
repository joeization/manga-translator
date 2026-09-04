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


class OllamaConnectionError(TranslationError):
    """Raised when the Ollama server is unreachable, refused connection, or down."""


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
        ret = re.sub(r"[.．]{2,}", "…", ret)
        ret = re.sub(r"[-—―ー]{2,}", "—", ret)
        ret = re.sub(r"[~～]{2,}", "~", ret)
        # Remove redundant whitespace between adjacent CJK characters while preserving Latin spaces and newlines
        # Remove redundant horizontal whitespace between adjacent ideographs while preserving inter-word spacing and newlines
        ret = re.sub(r"(?<=[\u4e00-\u9fff\u3040-\u30ff])[ \t\u3000]+(?=[\u4e00-\u9fff\u3040-\u30ff])", "", ret)
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

        # 1. Single text: direct translation without numbering
        if len(sanitized_texts) == 1:
            try:
                content = self._request(
                    context=context_str,
                    source_text=sanitized_texts[0],
                    stop=None,
                    max_tokens=256,
                )
                lines = [line.strip() for line in content.splitlines() if line.strip() and line.strip().lower() not in ("user", "assistant", "system")]
                clean = lines[0] if lines else content.strip()
                clean = re.sub(r"^\s*\[?\d+\]?[\s.:：、\-—]*", "", clean).strip()
                logger.info("Single-text translation succeeded.")
                return [format_response(clean)]
            except OllamaConnectionError:
                raise
            except Exception as error:
                logger.warning("Single-line translation failed (%s). Returning source text.", error)
                return sanitized_texts

        # 2. Multi-text batch: use numbered indexing [1] ... [N] for strict boundary alignment
        try:
            numbered_input = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(sanitized_texts))
            content = self._request(
                context=context_str,
                source_text=numbered_input,
                stop=None,
                max_tokens=max(256, len(sanitized_texts) * 32),
            )
            parsed = _parse_indexed_output(content, len(sanitized_texts))

            # Fallback for models that output plain un-indexed lines with matching count
            if not all(p is not None and p.strip() for p in parsed):
                plain_lines = [
                    line.strip() for line in content.splitlines()
                    if line.strip() and line.strip().lower() not in ("user", "assistant", "system")
                ]
                if len(plain_lines) == len(sanitized_texts):
                    parsed = [re.sub(r"^\s*\[?\d+\]?[\s.:：、\-—]*", "", l).strip() for l in plain_lines]

            # If all lines succeeded
            if all(p is not None and p.strip() for p in parsed):
                logger.info("Batch translation succeeded (%d lines).", len(parsed))
                return [format_response(p.strip()) for p in parsed if p is not None]

            # If only a few lines are missing, translate only the missing lines individually
            missing_indices = [i for i, p in enumerate(parsed) if p is None or not p.strip()]
            logger.info("Batch translation recovered %d of %d lines, translating %d missing lines individually.",
                        len(sanitized_texts) - len(missing_indices), len(sanitized_texts), len(missing_indices))
            for idx in missing_indices:
                try:
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
                except OllamaConnectionError:
                    raise
                except Exception as error:
                    logger.warning("Fallback for line %d failed: %s", idx, error)
                    parsed[idx] = sanitized_texts[idx]

            return [format_response(p) if p is not None else sanitized_texts[i] for i, p in enumerate(parsed)]

        except OllamaConnectionError:
            raise
        except Exception as error:
            logger.warning("Batch translation failed (%s). Falling back to line-by-line translation.", error)

        # 3. Complete fallback: translate line by line without context
        fallback_results: list[str] = []
        for text in sanitized_texts:
            try:
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
            except OllamaConnectionError:
                raise
            except Exception as error:
                logger.warning("Fallback request failed: %s", error)
                fallback_results.append(format_response(text))

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
        max_tokens: int = 256,
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
            err_str = str(error).lower()
            if isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout)) or \
               "connection refused" in err_str or "failed to establish a new connection" in err_str or "10061" in err_str:
                raise OllamaConnectionError(f"Ollama server connection refused at {self._endpoint}: {error}") from error
            raise TranslationError(f"Ollama request to {self._endpoint} failed: {error}") from error

        if not response.ok:
            body = response.text[:500]
            if response.status_code in (502, 503):
                raise OllamaConnectionError(f"Ollama service unavailable at {self._endpoint} (status {response.status_code}): {body}")
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

