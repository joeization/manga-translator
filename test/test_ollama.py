from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models import TextRegion
from src.translator.ollama import OllamaTranslator, TranslationError


class _Response:
    ok = True
    text = '{"message": {"content": "unused"}}'

    def __init__(self, content: str, ok: bool = True) -> None:
        self._content = content
        self.ok = ok
        self.status_code = 200 if ok else 500

    def json(self) -> dict[str, dict[str, str]]:
        return {"message": {"content": self._content}}


class OllamaTranslatorTests(unittest.TestCase):
    def test_translator_whole_page_success_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "translation.txt"
            prompt_path.write_text(
                "{{source_language}}>{{target_language}}\n{{context}}",
                encoding="utf-8",
            )
            client = OllamaTranslator("http://ollama", "model", "Japanese", "Traditional Chinese", prompt_path)

            mock_response = _Response("TRANS_1\nTRANS_2")
            with patch("src.translator.ollama.requests.Session.post", return_value=mock_response) as post:
                translations = client.translate(
                    ["SRC_1", "SRC_2？"],
                    context="CTX_PREV",
                )

        self.assertEqual(translations, ["TRANS_1", "TRANS_2"])
        self.assertEqual(post.call_count, 1)
        req = post.call_args.kwargs["json"]
        user_msg = req["messages"][1]["content"]
        self.assertIn("Previous translation context:\nCTX_PREV", user_msg)
        self.assertIn("translate the following text from Japanese into natural Traditional Chinese:", user_msg)
        self.assertIn("[1] SRC_1\n[2] SRC_2？", user_msg)
        self.assertIsNone(req["options"].get("stop"))

    def test_translator_falls_back_to_single_line_without_context_when_lengths_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "translation.txt"
            prompt_path.write_text(
                "{{source_language}}>{{target_language}}\n{{context}}",
                encoding="utf-8",
            )
            client = OllamaTranslator("http://ollama", "model", "Japanese", "Traditional Chinese", prompt_path)

            # Whole page returns only 1 line for 2 input lines -> length mismatch!
            # Then fallback translates line-by-line (2 requests) without context
            responses = [
                _Response("MISMATCH_LINE"),
                _Response("FB_1"),
                _Response("FB_2"),
            ]
            with patch("src.translator.ollama.requests.Session.post", side_effect=responses) as post:
                translations = client.translate(
                    ["SRC_1", "SRC_2"],
                    context="CTX_PREV",
                )

        self.assertEqual(translations, ["FB_1", "FB_2"])
        self.assertEqual(post.call_count, 3)

        # Fallback calls must NOT contain previous page context in user message
        fallback_req_1 = post.call_args_list[1].kwargs["json"]
        fallback_req_2 = post.call_args_list[2].kwargs["json"]
        self.assertNotIn("Previous translation context", fallback_req_1["messages"][1]["content"])
        self.assertNotIn("CTX_PREV", fallback_req_1["messages"][1]["content"])
        self.assertNotIn("Previous translation context", fallback_req_2["messages"][1]["content"])
        self.assertNotIn("CTX_PREV", fallback_req_2["messages"][1]["content"])
        self.assertIn("Translate the following text from Japanese into natural Traditional Chinese:\nSRC_1", fallback_req_1["messages"][1]["content"])
        self.assertIn("Translate the following text from Japanese into natural Traditional Chinese:\nSRC_2", fallback_req_2["messages"][1]["content"])
        self.assertEqual(fallback_req_1["options"]["stop"], ["\n"])
        self.assertEqual(fallback_req_2["options"]["stop"], ["\n"])

    def test_translator_empty_texts_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "translation.txt"
            prompt_path.write_text("prompt", encoding="utf-8")
            client = OllamaTranslator("http://ollama", "model", "Japanese", "Traditional Chinese", prompt_path)
            self.assertEqual(client.translate([]), [])

    def test_translator_numbered_batch_succeeds_in_single_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "translation.txt"
            prompt_path.write_text("{{source_language}}>{{target_language}}\n{{context}}", encoding="utf-8")
            client = OllamaTranslator("http://ollama", "model", "Japanese", "Traditional Chinese", prompt_path)

            # 3 dialogues: numbered batch succeeds in 1 single request
            mock_response = _Response("[1] OUT_A\n[2] OUT_B\n[3] OUT_C")
            with patch("src.translator.ollama.requests.Session.post", return_value=mock_response) as post:
                translations = client.translate(["IN_A", "IN_B", "IN_C"])

        self.assertEqual(translations, ["OUT_A", "OUT_B", "OUT_C"])
        self.assertEqual(post.call_count, 1)

    def test_translator_numbered_batch_recovers_missing_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "translation.txt"
            prompt_path.write_text("{{source_language}}>{{target_language}}\n{{context}}", encoding="utf-8")
            client = OllamaTranslator("http://ollama", "model", "Japanese", "Traditional Chinese", prompt_path)

            # 3 dialogues: numbered batch returns [1] and [3], missing [2]. Recovered via 1 fallback request (total 2 requests).
            responses = [
                _Response("[1] OUT_A\n[3] OUT_C"),
                _Response("OUT_B"),
            ]
            with patch("src.translator.ollama.requests.Session.post", side_effect=responses) as post:
                translations = client.translate(["IN_A", "IN_B", "IN_C"])

        self.assertEqual(translations, ["OUT_A", "OUT_B", "OUT_C"])
        self.assertEqual(post.call_count, 2)

    def test_ocr_metadata_defaults_to_unavailable(self) -> None:
        result = TextRegion((0, 0, 1, 1), "text")

        self.assertIsNone(result.confidence)
        self.assertIsNone(result.character_confidences)
        self.assertIsNone(result.metadata)

    def test_strip_leading_ellipsis(self) -> None:
        from src.translator.ollama import format_response

        self.assertEqual(format_response("...AAA"), "AAA")
        self.assertEqual(format_response("……AAA？"), "AAA？")
        self.assertEqual(format_response("⋯⋯AAA？"), "AAA？")
        self.assertEqual(format_response("——AAA"), "AAA")
        self.assertEqual(format_response("……"), "……")
        self.assertEqual(format_response("..."), "...")


if __name__ == "__main__":
    unittest.main()