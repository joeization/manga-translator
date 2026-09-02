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

"""baberu-ocr v2: prefix-token VLM for multilingual manga OCR.

Architecture (May 2026 SOTA):
    Vision (DINOv2) -> MLP projector -> prefix tokens -> [BOS] text [EOS]
                                              \\
                                               -> GQA causal LM (6 layers)

Decoder details:
  - GQA self-attention (Llama-style) with 1D RoPE (text positions)
  - SwiGLU FFN
  - RMSNorm sandwich-norm (Gemma2-style: norm before AND after each sub-block)
  - Tied input/output embeddings
  - Logit soft-cap (Gemma2-style: tanh(logit/cap)*cap)
  - Z-loss aux for logit-drift stability
  - KV cache for fast decode

Two entry points:
  - ``BaberuCausalLM``  : text-only mode (Step 1 pretraining on FineWeb2)
  - ``BaberuOCRModel``  : vision + decoder for OCR (Steps 3-1..3-3)

2D RoPE for vision prefix is planned (see config.vision_rope_2d) but currently
falls back to 1D positions over the combined [vision || text] sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_baberu import BaberuOCRConfig


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


class BaberuRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x_f32 = x_f32 * torch.rsqrt(var + self.eps)
        return (self.weight * x_f32).to(input_dtype)


class BaberuMLP(nn.Module):
    """SwiGLU FFN: down(silu(gate) * up)."""

    def __init__(self, config: BaberuOCRConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Rotary position embedding (1D)
# ---------------------------------------------------------------------------


class BaberuRotaryEmbedding(nn.Module):
    """Precomputed 1D RoPE table. Llama-style."""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._build_cache(max_seq_len, device=torch.device("cpu"))

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Recompute inv_freq from theta/dim rather than reading the buffer. It is
        # registered non-persistent (not in the checkpoint), so under
        # `from_pretrained` it is created on the meta device then materialised by
        # `to_empty()` to *uninitialised* memory — no longer `is_meta`, but filled
        # with garbage that silently breaks RoPE (positions collapse, output turns
        # to junk). Recomputing here is cheap and always correct.
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, T] long
        seq_max = int(position_ids.max().item()) + 1
        if (
            self._cos_cached is None
            or seq_max > self.max_seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._build_cache(max(seq_max, self.max_seq_len_cached, 64), device=x.device, dtype=x.dtype)
        cos = self._cos_cached[position_ids]  # [B, T, dim]
        sin = self._sin_cached[position_ids]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, n_heads, T, head_dim]; cos, sin: [B, T, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# GQA self-attention with KV cache
# ---------------------------------------------------------------------------


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for GQA. [B, kv_heads, T, D] -> [B, q_heads, T, D]."""
    if n_rep == 1:
        return hidden_states
    bsz, kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, kv_heads * n_rep, slen, head_dim)


def _cache_length(past_key_values) -> int:
    """Return number of tokens already in the cache, handling DynamicCache and legacy tuples."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "layers"):
        layers = past_key_values.layers
        if not layers or not layers[0].is_initialized:
            return 0
        return layers[0].keys.shape[-2]
    if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
        return past_key_values[0][0].shape[-2]
    return 0


class BaberuAttention(nn.Module):
    def __init__(self, config: BaberuOCRConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attn_softcap = config.attn_logit_softcap
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, T, S] additive (-inf for masked)
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        k_expanded = _repeat_kv(k, self.num_kv_groups)
        v_expanded = _repeat_kv(v, self.num_kv_groups)

        # Scaled dot-product attention.  Q: [B, H, Tq, D], K/V: [B, H, Tk, D]
        if self.attn_softcap is not None:
            # SDPA/flash kernels cannot apply Gemma-style attention logit
            # soft-capping, so keep the explicit (memory-heavy) path only when
            # soft-capping is enabled.
            attn_scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / (self.head_dim ** 0.5)
            attn_scores = torch.tanh(attn_scores / self.attn_softcap) * self.attn_softcap
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask[:, :, :, : k.shape[-2]]
            attn_weights = F.softmax(attn_scores.to(torch.float32), dim=-1).to(q.dtype)
            if self.attention_dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
            attn_output = torch.matmul(attn_weights, v_expanded)  # [B, H, Tq, D]
        else:
            # FlashAttention / mem-efficient SDPA: never materialises the
            # [B, H, Tq, Tk] score matrix, which is the bulk of the memory for
            # long sequences. Mathematically identical to the explicit path.
            attn_mask = attention_mask[:, :, :, : k.shape[-2]] if attention_mask is not None else None
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=attn_mask,
                is_causal=attn_mask is None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.head_dim ** -0.5,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, present


# ---------------------------------------------------------------------------
# Decoder layer (sandwich norm)
# ---------------------------------------------------------------------------


class BaberuDecoderLayer(nn.Module):
    """One layer of the causal decoder.

    With ``sandwich_norm=True``, both pre- and post-RMSNorm wrap each
    sub-block (attention and FFN), in the style of Gemma 2.
    """

    def __init__(self, config: BaberuOCRConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.sandwich = config.sandwich_norm
        self.input_norm = BaberuRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BaberuAttention(config, layer_idx)
        self.post_attn_norm = BaberuRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ffn_norm = BaberuRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BaberuMLP(config)
        self.post_ffn_norm = BaberuRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        x = self.input_norm(hidden_states)
        x, present = self.attn(
            hidden_states=x,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if self.sandwich:
            x = self.post_attn_norm(x)
        hidden_states = residual + x

        residual = hidden_states
        x = self.pre_ffn_norm(hidden_states)
        x = self.mlp(x)
        if self.sandwich:
            x = self.post_ffn_norm(x)
        hidden_states = residual + x
        return hidden_states, present


# ---------------------------------------------------------------------------
# Base preTrained model
# ---------------------------------------------------------------------------


class BaberuPreTrainedModel(PreTrainedModel):
    config_class = BaberuOCRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BaberuDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ---------------------------------------------------------------------------
# Backbone (embeddings + decoder layers + final norm)
# ---------------------------------------------------------------------------


class BaberuModel(BaberuPreTrainedModel):
    """Decoder backbone, language-only (no LM head, no vision).

    Accepts either ``input_ids`` (token IDs that go through the embedding
    table) or ``inputs_embeds`` (pre-embedded, used when prepending vision).
    """

    def __init__(self, config: BaberuOCRConfig):
        super().__init__(config)
        config.validate()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [BaberuDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = BaberuRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BaberuRotaryEmbedding(
            config.head_dim, config.max_position_embeddings, theta=config.rope_theta
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    @staticmethod
    def _build_causal_mask(
        q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Lower-triangular additive mask: -inf above the diagonal aligned to kv_len."""
        # Q has length q_len, K has length kv_len. The last q_len queries see the
        # last q_len keys causally; earlier keys (from KV cache) are all visible.
        mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
        # Compute offset so that diagonal at (q, q + (kv_len - q_len)) is allowed.
        offset = kv_len - q_len
        for i in range(q_len):
            mask[i, : i + 1 + offset] = 0.0
        return mask  # [Tq, Tk]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] of 1/0
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        bsz, q_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Normalize transformers Cache objects to a list of (k, v) tuples.
        if past_key_values is not None and hasattr(past_key_values, "layers"):
            layers = past_key_values.layers
            past_key_values = (
                [(l.keys, l.values) for l in layers if l.is_initialized]
                if layers
                else None
            )
            if past_key_values is not None and len(past_key_values) == 0:
                past_key_values = None

        past_len = _cache_length(past_key_values)
        kv_len = past_len + q_len

        if position_ids is None:
            position_ids = torch.arange(past_len, kv_len, device=device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        # Causal mask. Reshape to broadcast across batch and heads.
        causal_mask = self._build_causal_mask(q_len, kv_len, device, inputs_embeds.dtype)
        attn_mask = causal_mask[None, None, :, :].expand(bsz, 1, q_len, kv_len).clone()
        if attention_mask is not None:
            # attention_mask: [B, kv_len] with 1 for keep, 0 for pad
            pad_mask = (1 - attention_mask[:, None, None, :]).to(inputs_embeds.dtype) * float("-inf")
            # Replace NaN from (-inf * 0)
            pad_mask = torch.nan_to_num(pad_mask, nan=0.0)
            attn_mask = attn_mask + pad_mask[:, :, :, :kv_len]

        hidden_states = inputs_embeds
        next_cache = [] if use_cache else None
        all_hidden_states = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_past = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states, cos, sin, attn_mask, layer_past, use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = layer(
                    hidden_states=hidden_states,
                    cos=cos,
                    sin=sin,
                    attention_mask=attn_mask,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
            if use_cache:
                next_cache.append(present)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs = outputs + (tuple(next_cache),)
            if output_hidden_states:
                outputs = outputs + (tuple(all_hidden_states),)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(next_cache) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=None,
        )


# ---------------------------------------------------------------------------
# Causal LM (text-only, for Step 1 pretraining)
# ---------------------------------------------------------------------------


def _apply_softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(logits / cap) * cap


def _z_loss(logits: torch.Tensor, ignore_index: int = -100, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Auxiliary z-loss: -1e-4 * log(Z)^2 averaged over non-ignored positions.

    Encourages logit normalizer log(Z) -> 0, stabilizing training. Apply on
    pre-softcap logits to keep gradient signal meaningful.
    """
    if labels is not None:
        mask = (labels != ignore_index).float()
        log_z = torch.logsumexp(logits.float(), dim=-1)  # [B, T]
        denom = mask.sum().clamp(min=1.0)
        return ((log_z**2) * mask).sum() / denom
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return (log_z**2).mean()


@dataclass
class BaberuCausalLMOutput(CausalLMOutputWithPast):
    z_loss: Optional[torch.FloatTensor] = None


class BaberuCausalLM(BaberuPreTrainedModel, GenerationMixin):
    """Decoder-only causal LM. Used for Step 1 text pretraining."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: BaberuOCRConfig):
        super().__init__(config)
        self.model = BaberuModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaberuCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else True
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits_raw = self.lm_head(hidden_states)
        logits = (
            _apply_softcap(logits_raw, self.config.final_logit_softcap)
            if self.config.final_logit_softcap
            else logits_raw
        )

        loss: Optional[torch.Tensor] = None
        z_loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if self.config.z_loss_weight and self.config.z_loss_weight > 0:
                # Use pre-softcap logits for z-loss to retain gradient.
                shift_raw = logits_raw[:, :-1, :].contiguous()
                z_loss = _z_loss(shift_raw, ignore_index=-100, labels=shift_labels)
                loss = ce + self.config.z_loss_weight * z_loss
            else:
                loss = ce

        if not return_dict:
            output = (logits,) + ((outputs.past_key_values,) if outputs.past_key_values is not None else ())
            return ((loss,) + output) if loss is not None else output

        return BaberuCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
            z_loss=z_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        # Determine how many positions are already in the cache. An empty
        # DynamicCache is not None but has past_len=0, so we must inspect.
        past_len = _cache_length(past_key_values)
        # When KV cache has content, only the last token needs to be fed.
        if past_len > 0:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "inputs_embeds": None}
        if inputs_embeds is not None and past_len == 0:
            model_inputs = {"input_ids": None, "inputs_embeds": inputs_embeds}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache", True),
            }
        )
        return model_inputs


# ---------------------------------------------------------------------------
# Vision projector
# ---------------------------------------------------------------------------


class BaberuVisionProjector(nn.Module):
    """MLP projector: vision_hidden -> hidden_size."""

    def __init__(self, config: BaberuOCRConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


# ---------------------------------------------------------------------------
# Full OCR model: vision encoder + projector + causal LM
# ---------------------------------------------------------------------------


class BaberuOCRModel(BaberuPreTrainedModel, GenerationMixin):
    """Full OCR model. Reads an image, generates text.

    Input layout fed to the decoder is:
        [vision tokens (projected DINOv2 outputs) | BOS | text tokens | EOS]

    The vision tokens are computed once per image (per generate() call) and
    cached in past_key_values along with all decoded text steps.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: BaberuOCRConfig):
        super().__init__(config)
        if config.vision_model_name is None:
            raise ValueError(
                "BaberuOCRModel requires config.vision_model_name. For text-only "
                "pretraining, use BaberuCausalLM instead."
            )
        # Build the vision encoder from its *config* (structure only, no weight
        # download), which is safe under `from_pretrained`'s meta-device init —
        # calling `AutoModel.from_pretrained` here breaks loading via
        # `AutoModel.from_pretrained(..., trust_remote_code=True)`. When this
        # model is built fresh (training, params on a real device) we populate
        # the pretrained DINOv2 weights; when loaded from a checkpoint the params
        # start on meta and the vision weights come from the checkpoint, so we
        # skip the (redundant) download.
        vision_config = AutoConfig.from_pretrained(config.vision_model_name)
        self.vision_encoder = AutoModel.from_config(vision_config)
        if next(self.vision_encoder.parameters()).device.type != "meta":
            pretrained = AutoModel.from_pretrained(config.vision_model_name)
            self.vision_encoder.load_state_dict(pretrained.state_dict())
            del pretrained
        if config.freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            self.vision_encoder.eval()
        self.projector = BaberuVisionProjector(config)
        self.model = BaberuModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, N_vision, hidden_size]."""
        out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        # DINOv2 returns last_hidden_state of shape [B, N+1, vision_hidden_size]
        # where the +1 is the CLS token. We drop it for OCR.
        feats = out.last_hidden_state[:, 1:, :]
        return self.projector(feats)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,  # text tokens (no vision prefix)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        labels: Optional[torch.LongTensor] = None,  # text-only labels; we shift inside
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaberuCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # On the first forward we must prepend the vision prefix; once it is in
        # the KV cache we only feed new text. Branch on cache *length*, not
        # ``is None``: HF ``generate`` passes an empty (but non-None) Cache on
        # the first step, so an ``is None`` check would skip the vision prefix
        # entirely and every image would decode the same unconditional text.
        if _cache_length(past_key_values) == 0:
            if pixel_values is None:
                raise ValueError("pixel_values is required on the first forward (no KV cache).")
            vision_embeds = self._encode_image(pixel_values)  # [B, Nv, H]
            text_embeds = (
                self.model.embed_tokens(input_ids) if input_ids is not None else None
            )
            if text_embeds is not None:
                inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            else:
                inputs_embeds = vision_embeds
            # Build attention mask that covers both.
            if attention_mask is not None and input_ids is not None:
                vision_mask = torch.ones(
                    vision_embeds.shape[:2], device=vision_embeds.device, dtype=attention_mask.dtype
                )
                full_mask = torch.cat([vision_mask, attention_mask], dim=1)
            else:
                full_mask = None
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                past_key_values=None,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            n_vision = vision_embeds.shape[1]
        else:
            # KV cache already holds the 256 vision tokens + prior text; only the
            # new text token goes in. HF's attention_mask counts text tokens only
            # (it is unaware of the vision prefix), so its length won't match
            # kv_len — drop it. There is no padding in single-sequence greedy
            # generation, so the internal causal mask is sufficient; position_ids
            # default to arange(past_len, kv_len), giving the new token its true
            # post-vision position.
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            n_vision = 0

        hidden_states = outputs.last_hidden_state
        logits_raw = self.lm_head(hidden_states)
        logits = (
            _apply_softcap(logits_raw, self.config.final_logit_softcap)
            if self.config.final_logit_softcap
            else logits_raw
        )

        loss: Optional[torch.Tensor] = None
        z_loss: Optional[torch.Tensor] = None
        if labels is not None:
            # The logits include vision-prefix positions; slice them out for the loss.
            text_logits = logits[:, n_vision:, :]
            text_logits_raw = logits_raw[:, n_vision:, :]
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if self.config.z_loss_weight and self.config.z_loss_weight > 0:
                shift_raw = text_logits_raw[:, :-1, :].contiguous()
                z_loss = _z_loss(shift_raw, ignore_index=-100, labels=shift_labels)
                loss = ce + self.config.z_loss_weight * z_loss
            else:
                loss = ce

        if not return_dict:
            output = (logits,) + ((outputs.past_key_values,) if outputs.past_key_values is not None else ())
            return ((loss,) + output) if loss is not None else output

        return BaberuCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
            z_loss=z_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        past_len = _cache_length(past_key_values)
        if past_len > 0:
            input_ids = input_ids[:, -1:]
            pixel_values = None  # already encoded in the cache
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }
