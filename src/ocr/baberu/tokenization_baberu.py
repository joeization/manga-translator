# Copyright 2026 genshiai-daichi / Baberu OCR Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Character-level tokenizer for baberu-ocr v2.

Loads a charset file (JSON array of single characters), prepends special
tokens, and exposes a HuggingFace-compatible PreTrainedTokenizer.

Token layout:
    0: <pad>
    1: <bos>
    2: <eos>
    3: <unk>
    4..N: characters from charset (e.g., '\\n', ' ', '!', ..., 'あ', '世', ...)

Designed for multilingual OCR (JA + ZH + EN). Pure character-level: every
codepoint maps to one token. Unicode codepoints above U+FFFF are stored as
single Python str elements and tokenized as one token (verified for chars
like '𠝹' U+2057D).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

DEFAULT_SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
}


class BaberuTokenizer(PreTrainedTokenizer):
    """Character-level tokenizer with prepended special tokens.

    Args:
        vocab_file: Path to a JSON array of characters (e.g., baberu_charset.txt).
        pad_token / bos_token / eos_token / unk_token: Special token strings.

    Example:
        >>> tok = BaberuTokenizer(vocab_file="data/baberu_charset.txt")
        >>> tok("こんにちは Hello").input_ids
        [1, 1234, 5678, ..., 4, 'H'_id, ...]  # includes <bos> ... (no <eos> by default)
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        pad_token: str = DEFAULT_SPECIAL_TOKENS["pad_token"],
        bos_token: str = DEFAULT_SPECIAL_TOKENS["bos_token"],
        eos_token: str = DEFAULT_SPECIAL_TOKENS["eos_token"],
        unk_token: str = DEFAULT_SPECIAL_TOKENS["unk_token"],
        add_bos_token: bool = True,
        add_eos_token: bool = True,
        **kwargs,
    ):
        if vocab_file is None:
            raise ValueError("vocab_file must be provided")
        if not Path(vocab_file).is_file():
            raise FileNotFoundError(f"vocab_file not found: {vocab_file}")

        charset = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
        if not isinstance(charset, list):
            raise ValueError("vocab_file must contain a JSON array of strings")

        # Coerce special tokens to plain strings — from_pretrained may pass
        # AddedToken instances which would break string-keyed dict lookups.
        pad_str = str(pad_token)
        bos_str = str(bos_token)
        eos_str = str(eos_token)
        unk_str = str(unk_token)

        # Build vocab with special tokens first, then charset.
        self._token_to_id: dict[str, int] = {
            pad_str: 0,
            bos_str: 1,
            eos_str: 2,
            unk_str: 3,
        }
        for ch in charset:
            if ch in self._token_to_id:
                continue
            self._token_to_id[ch] = len(self._token_to_id)
        self._id_to_token: dict[int, str] = {i: t for t, i in self._token_to_id.items()}

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._token_to_id)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        token_str = str(token)
        if token_str in self._token_to_id:
            return self._token_to_id[token_str]
        # Fallback to unk; tolerate self.unk_token being an AddedToken.
        return self._token_to_id.get(str(self.unk_token), 3)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, str(self.unk_token))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """Wrap a sequence with [BOS] ... [EOS] tokens.

        For OCR we only ever have a single sequence (token_ids_1 unused).
        """
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        result = bos + token_ids_0 + eos
        if token_ids_1 is not None:
            result = result + token_ids_1 + eos
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        bos = [1] if self.add_bos_token else []
        eos = [1] if self.add_eos_token else []
        mask = bos + [0] * len(token_ids_0) + eos
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + eos
        return mask

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "")
            + self.vocab_files_names["vocab_file"]
        )
        # Save only the charset (special tokens are recreated on load).
        special_ids = {0, 1, 2, 3}
        charset = [
            self._id_to_token[i]
            for i in range(len(self._id_to_token))
            if i not in special_ids
        ]
        out_path.write_text(json.dumps(charset, ensure_ascii=False, indent=2))
        return (str(out_path),)
