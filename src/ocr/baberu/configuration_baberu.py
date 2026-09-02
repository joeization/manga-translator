"""Configuration for baberu-ocr v2.

Single config that drives both the text-only decoder (BaberuCausalLM, used
for Step 1 pretraining) and the full vision-grounded OCR model
(BaberuOCRModel = DINOv2 + MLP projector + decoder).

See docs/design/ocr-v2-design.md for the full architecture rationale.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class BaberuOCRConfig(PretrainedConfig):
    """Config for baberu-ocr v2 (~115M params, multilingual JA+ZH+EN).

    Defaults match the May 2026 SOTA design:
      - Vision: DINOv2-ViT-B/14 (86M, Apache 2.0)
      - Projector: MLP 768 -> hidden_size
      - Decoder: 6-layer GQA causal LM, hidden=512, FFN=1536
      - RoPE 1D (text), 2D-RoPE for vision optional
      - RMSNorm sandwich-norm, SwiGLU, logit soft-cap, z-loss, tied embeddings

    Use ``vision_model_name=None`` to disable the vision branch
    (text-only mode for Step 1 pretraining).
    """

    model_type = "baberu_ocr"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # ---- Vision encoder ----
        vision_model_name: str | None = "facebook/dinov2-base",
        vision_hidden_size: int = 768,
        vision_image_size: int = 224,
        vision_patch_size: int = 14,
        vision_num_tokens: int = 256,  # (224/14)^2
        freeze_vision_encoder: bool = True,
        # ---- Projector ----
        projector_act: str = "gelu",
        # ---- Decoder ----
        vocab_size: int = 14630,
        hidden_size: int = 512,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int | None = None,  # default = hidden_size // num_attention_heads
        max_position_embeddings: int = 2048,
        hidden_act: str = "silu",
        # ---- Norm ----
        rms_norm_eps: float = 1e-6,
        sandwich_norm: bool = True,
        # ---- RoPE ----
        rope_theta: float = 10000.0,
        vision_rope_2d: bool = False,  # 2D RoPE for vision prefix (planned)
        # ---- Stability ----
        final_logit_softcap: float = 30.0,
        attn_logit_softcap: float | None = None,
        z_loss_weight: float = 1e-4,
        # ---- Embeddings ----
        tie_word_embeddings: bool = True,
        # ---- Init ----
        initializer_range: float = 0.02,
        # ---- Special tokens ----
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        # ---- Cache / dtype ----
        use_cache: bool = True,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        self.vision_model_name = vision_model_name
        self.vision_hidden_size = vision_hidden_size
        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size
        self.vision_num_tokens = vision_num_tokens
        self.freeze_vision_encoder = freeze_vision_encoder

        self.projector_act = projector_act

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act

        self.rms_norm_eps = rms_norm_eps
        self.sandwich_norm = sandwich_norm

        self.rope_theta = rope_theta
        self.vision_rope_2d = vision_rope_2d

        self.final_logit_softcap = final_logit_softcap
        self.attn_logit_softcap = attn_logit_softcap
        self.z_loss_weight = z_loss_weight

        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # Stored separately because PretrainedConfig doesn't claim unk.
        self.unk_token_id = unk_token_id

    # Convenience for sanity checks
    def validate(self) -> None:
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible "
            f"by num_key_value_heads ({self.num_key_value_heads}) for GQA"
        )
        assert self.head_dim * self.num_attention_heads == self.hidden_size, (
            f"head_dim ({self.head_dim}) * num_attention_heads "
            f"({self.num_attention_heads}) must equal hidden_size ({self.hidden_size})"
        )
