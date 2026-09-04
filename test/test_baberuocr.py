from __future__ import annotations

import unittest

import torch

from src.ocr.baberuocr import _decode_with_character_confidences, _sentence_confidence


class _CharacterTokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        tokens = {2: "<eos>", 4: "A"}
        decoded = "".join(tokens[token_id] for token_id in token_ids)
        return decoded.replace("<eos>", "") if skip_special_tokens else decoded


class BaberuConfidenceTests(unittest.TestCase):
    def test_generated_token_score_is_aligned_with_its_character(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0]])

        text, confidences = _decode_with_character_confidences(
            [4, 2],
            (logits, logits),
            _CharacterTokenizer(),
            eos_token_id=2,
        )

        self.assertEqual(text, "A")
        self.assertIsNotNone(confidences)
        assert confidences is not None
        self.assertEqual(len(confidences), 1)
        self.assertAlmostEqual(confidences[0], float(torch.softmax(logits[0], dim=-1)[4]))

    def test_sentence_confidence_uses_all_character_probabilities(self) -> None:
        self.assertAlmostEqual(_sentence_confidence([0.81, 0.25]), 0.45)

    def test_decode_batch_item_with_padding(self) -> None:
        # Batch size 2, 3 steps: step 0 has 'A' and 'B', step 1 has '<eos>' and '<pad>', step 2 has '<pad>'
        step0 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
        step1 = torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        step2 = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        class _MultiTokenizer:
            def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
                tokens = {0: "<pad>", 2: "<eos>", 4: "A", 5: "B"}
                decoded = "".join(tokens.get(t, "") for t in token_ids)
                return decoded.replace("<eos>", "").replace("<pad>", "") if skip_special_tokens else decoded

        text0, conf0 = _decode_with_character_confidences(
            [4, 2, 0], (step0, step1, step2), _MultiTokenizer(), eos_token_id=2, batch_idx=0, pad_token_id=0
        )
        self.assertEqual(text0, "A")
        self.assertIsNotNone(conf0)
        assert conf0 is not None
        self.assertEqual(len(conf0), 1)

        text1, conf1 = _decode_with_character_confidences(
            [5, 0, 0], (step0, step1, step2), _MultiTokenizer(), eos_token_id=2, batch_idx=1, pad_token_id=0
        )
        self.assertEqual(text1, "B")
        self.assertIsNotNone(conf1)
        assert conf1 is not None
        self.assertEqual(len(conf1), 1)


if __name__ == "__main__":
    unittest.main()