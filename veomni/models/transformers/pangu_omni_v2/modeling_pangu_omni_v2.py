# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Pangu Omni v2 modeling — Week 2.

Verbatim port of the text-only backbone classes from the Pangu reference
(`modeling_openpangu_v2.py`):

- `OpenPanguV2PreTrainedModel`  — base class with Pangu-specific flags
- `OpenPanguV2Model`            — text MoE backbone (37 layers in 30B-A2B)
- `OpenPanguV2ForCausalLM`      — adds `lm_head` for token-level generation

The multimodal entrypoints (`OpenPanguVLForConditionalGeneration` and
`OpenPanguUltraOmniForConditionalGeneration`) are still placeholders that
raise `NotImplementedError` — they land in Week 3 alongside vision /
audio encoder ports.

## Naming policy

All class names are kept verbatim from the upstream reference so:
- `patchgen` rewrite rules match upstream symbols 1:1.
- oracle log grep / state_dict key naming stays trivial.
- `pangu_oracle_check.py --mode veomni` can instantiate the same class
  graph as `--mode hf` (just from `_pangu_common` instead of the dynamic
  HF module load).

## All Day 1-4 _pangu_common modules participate

This file composes:
- `OpenPanguV2RMSNorm`               — input_layernorm + pre_mlp_layernorm + final norm
- `OpenPanguV2RotaryEmbedding`       — partial RoPE
- `OpenPanguV2DecoderLayer`          — wraps attention + MHC + MoE (Day 5)
- `mHCModule` (merge-only variant)   — final stream collapse from (B, S, n*H) -> (B, S, H)

## Known upstream quirk: `self.mhc_num_stream`

The upstream `OpenPanguV2Model.forward` reads `self.mhc_num_stream` on
line 1089 (`inputs_embeds = torch.cat([inputs_embeds] * self.mhc_num_stream, dim=-1)`)
but `__init__` only sets `self.num_stream` (line 1061). This works on
upstream because `PreTrainedModel.__getattr__` in transformers 5.0
forwards unknown attribute access to `self.config`. We do not rely on
that — we read `self.config.mhc_num_stream` directly, which is the
canonical source of truth and gives bit-for-bit identical behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from .._pangu_common import (
    OpenPanguV2Attention,
    OpenPanguV2DecoderLayer,
    OpenPanguV2RMSNorm,
    OpenPanguV2RotaryEmbedding,
    mHCModule,
)
from .._pangu_common.pangu_moe import (
    veomni_moe_experts_forward,  # noqa: F401 — see comment below
)
from .configuration_pangu_omni_v2 import OpenPanguOmniConfig


# The MoE-experts OpSlot lives in ``_pangu_common.pangu_moe`` (alongside
# the ``OpenPanguV2Experts`` class that uses it), but VeOmni's
# ``_bind_veomni_ops`` (in ``models/auto.py``) discovers slots via
# ``dir(modeling_module)``. ``modeling_module`` is the module of the
# top-level model class (``OpenPanguV2ForCausalLM`` lives here), not the
# shared ``_pangu_common.pangu_moe``. We therefore re-export the slot
# here so the discover-and-bind step finds it and binds it according to
# ``args.model.ops_implementation.moe_implementation``.
#
# Without this re-export, setting ``moe_implementation: npu`` would
# silently leave the slot unbound and the eager forward would run —
# fatal under ``ep_size > 1`` because the eager loop indexes global
# expert IDs into the locally-sharded ``gate_up_proj`` (see the OpSlot
# guard in ``OpenPanguV2Experts.forward`` for the full story).

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Base class (verbatim port of OpenPanguV2PreTrainedModel)
# ---------------------------------------------------------------------------


@auto_docstring
class OpenPanguV2PreTrainedModel(PreTrainedModel):
    """Pangu V2 base class — verbatim port of `modeling_openpangu_v2.py:1024-1040`.

    Carries the Pangu-specific flags that control gradient checkpointing
    no-split boundaries, attention-backend support, and load-time key
    filtering.

    Quirk: `_keys_to_ignore_on_load_unexpected = [r"model\\.layers\\.44.*"]`
    — 30B-A2B only has 37 layers (0-36), so this regex matches nothing in
    the current ckpt. Preserved verbatim (likely a copy-paste artifact
    from a different Pangu variant).
    """

    config_class = OpenPanguOmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OpenPanguV2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OpenPanguV2DecoderLayer,
        "attentions": OpenPanguV2Attention,
    }
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.44.*"]


# ---------------------------------------------------------------------------
# Text backbone (verbatim port of OpenPanguV2Model)
# ---------------------------------------------------------------------------


@auto_docstring
class OpenPanguV2Model(OpenPanguV2PreTrainedModel):
    """Text-only MoE backbone — verbatim port of `OpenPanguV2Model`.

    For the 30B-A2B model:
    - 37 decoder layers (layers 0-1 dense MLP, layers 2-36 MoE)
    - hidden_size = 2560 (per-stream)
    - mhc_num_stream = 4 → residual stream width = 10240
    - vocab_size = 152064 (shared with embed_tokens / lm_head)

    Hidden-state flow (use_mhc=True):
    ```
    input_ids -> embed_tokens -> (B, S, H)
              -> cat * n_stream -> (B, S, n*H)        # MHC tile
              -> 37 × decoder_layer                   # each: hc_pre -> attn/mlp@H -> hc_post
              -> merge_mhc_module.hc_pre -> (B, S, H) # final stream collapse
              -> norm -> (B, S, H)
    ```

    See `_pangu_common.pangu_decoder_layer` for the per-layer MHC pre/post.
    """

    config_class = OpenPanguOmniConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OpenPanguV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OpenPanguV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OpenPanguV2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        layer_types = getattr(self.config, "layer_types", None)
        self.has_sliding_layers = layer_types is not None and "sliding_attention" in layer_types

        self.use_mhc = getattr(config, "use_mhc", False)
        if self.use_mhc:
            self.num_stream = config.mhc_num_stream
            # `merge_layer_only_pre=True` -> the post path is identity. This
            # module is used ONCE after the decoder stack to collapse
            # (B, S, n*H) -> (B, S, H) before the final norm + lm_head.
            self.merge_mhc_module = mHCModule(
                config=config,
                merge_layer_only_pre=True,
            )

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

            if self.use_mhc:
                # Upstream uses `self.mhc_num_stream` here (line 1089) which
                # relies on PreTrainedModel's __getattr__ -> config fallback.
                # We read from config explicitly — bit-for-bit identical.
                inputs_embeds = torch.cat([inputs_embeds] * self.config.mhc_num_stream, dim=-1)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # MRoPE shape reconciliation for the mask call.
        #
        # When this backbone is used inside ``OpenPanguVL`` / ``OpenPanguOmni``,
        # ``position_ids`` arrives as a 3D MRoPE tensor ``[3, B, S]`` (T/H/W
        # axes — see ``compute_omni_rope_index``). The downstream
        # ``self.rotary_emb`` is ``OpenPanguVLRotaryEmbedding`` which requires
        # exactly 3 dims, so we must keep the 3D layout for that call.
        #
        # ``transformers.masking_utils.create_causal_mask`` however documents
        # ``position_ids`` as ``[B, S]`` and uses it only for packed-sequence
        # detection via ``find_packed_sequence_indices`` (which itself does
        # ``position_ids[:, :1]`` and ``torch.diff(..., dim=-1)`` — both
        # ill-defined on a 3D MRoPE tensor where temporal-axis values are
        # constant within a single image patch). Following the canonical
        # transformers pattern (Qwen2.5-VL ``Qwen2_5_VLTextModel.forward``,
        # transformers 5.x line "If inputs are not packed (usual 3D positions),
        # do not prepare mask from position_ids"), we pass ``None`` to the
        # mask in the multimodal/3D case. This means:
        #   - Non-packed multimodal input: mask uses ``cache_position`` only
        #     (correct — single sequence per batch row).
        #   - Packed multimodal input: caller must pass ``attention_mask``
        #     explicitly (which short-circuits the position_ids path entirely
        #     in ``_preprocess_mask_arguments``). Pangu does not yet support
        #     the 4D ``[4, B, S]`` packed layout that Qwen2.5-VL uses to
        #     prepend a text-only axis for packing.
        if position_ids.ndim == 3:
            mask_position_ids = None
        else:
            mask_position_ids = position_ids

        # `attention_mask` may already be a dict if passed by `generate`.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": mask_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # RoPE is computed once on the full residual stream — partial RoPE
        # only rotates the leading `rotary_ndims` of each head, so the
        # `(B, S, n*H)` width doesn't affect cos/sin (which only depend on
        # `position_ids`, `head_dim`, and `partial_rotary_factor`).
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        # Final MHC collapse: (B, S, n*H) -> (B, S, H)
        if self.use_mhc:
            hidden_states, _, _ = self.merge_mhc_module.hc_pre(hidden_states)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# Causal LM head (verbatim port of OpenPanguV2ForCausalLM)
# ---------------------------------------------------------------------------


@auto_docstring
class OpenPanguV2ForCausalLM(OpenPanguV2PreTrainedModel, GenerationMixin):
    """Causal LM head over OpenPanguV2Model — verbatim port.

    Adds a single linear `lm_head: (hidden_size, vocab_size)` on top of
    `OpenPanguV2Model`, plus the `loss_function` hook for training (uses
    `transformers`' default cross-entropy via `PreTrainedModel.loss_function`).
    """

    config_class = OpenPanguOmniConfig
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config) -> None:
        super().__init__(config)
        self.model = OpenPanguV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Module) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Module:
        return self.model

    def get_parallel_plan(self):
        """Return the VeOmni expert-parallel plan for this model.

        Delegates to ``parallel_plan.get_parallel_plan`` so the EP shard
        spec lives in one place (matches the qwen3_moe / deepseek_v3
        pattern; see ``pangu_omni_v2/parallel_plan.py`` for the FQN
        verification and design notes). VeOmni's ``torch_parallelize``
        calls this method when ``any_extra_parallel_enabled`` is set
        (i.e. ``train.accelerator.ep_size > 1`` in the yaml).

        Note: Pangu's checkpoint converter (``checkpoint_tensor_converter``)
        always produces the fused 3D ``gate_up_proj`` layout, so we
        always pass ``use_gate_up_proj=True``. The non-fused branch in
        ``parallel_plan.py`` is kept for API parity, not used today.
        """
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(use_gate_up_proj=True)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute the requested logits range (e.g. last `logits_to_keep`
        # positions for generation — saves the full vocab projection when
        # only the last token is needed).
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # VeOmni's `install_loss_mapping` (ops/__init__.py:apply_ops_config)
            # rebinds `LOSS_MAPPING["ForCausalLM"]` to a wrapper that returns
            # `(loss, logits, log_probs, entropy)` instead of the plain
            # tensor that mainline transformers' `ForCausalLMLoss` returns
            # — see `ops/kernels/cross_entropy/__init__.py:74-100`. Trainers
            # always go through `apply_ops_config`, so we unpack the 4-tuple
            # here and discard the extra returns (they're used by RL /
            # inspection paths, not by VeOmni's SFT loss path).
            #
            # Matches qwen3_moe's `modeling_qwen3_moe.py:230-235`. When we
            # cut the patchgen output (Week 4) this becomes the canonical
            # OpSlot guard you see in
            # `qwen3_moe/generated/patched_modeling_qwen3_moe_gpu.py:768-784`.
            loss, logits, log_probs, entropy = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Multimodal entrypoints — Week 3 placeholders
# ---------------------------------------------------------------------------


_MULTIMODAL_NOT_IMPLEMENTED_MSG = (
    "Pangu Omni v2 multimodal entrypoints are pending Week 3 implementation. "
    "The text backbone (OpenPanguV2Model / OpenPanguV2ForCausalLM) is fully "
    "implemented in Week 2 — use those for text-only inference. Vision "
    "(GatedMerger) and audio (HuanyuAudioEncoder + fbank) encoders land "
    "Week 3. See docs/pangu_veomni_adaptation/PHASE1_DESIGN.md."
)


def _get_open_pangu_vl_class():
    """Lazy import of `OpenPanguVL` (Week 3.4.d) from the multimodal
    modeling module — avoids pulling the vision import graph into the
    text-only path."""
    from .modeling_openpangu_vl import OpenPanguVL

    return OpenPanguVL


def _get_open_pangu_omni_class():
    """Lazy import of `OpenPanguOmni` (Week 3.5) — adds the audio tower
    and audio merge branch on top of `OpenPanguVL`.

    Importantly, this is **only** invoked at dispatcher time
    (``register_pangu_omni_v2_modeling`` in ``__init__.py``), NOT at
    class-definition time. ``modeling_openpangu_omni`` imports from
    ``modeling_openpangu_vl`` at module load, so calling this resolver
    inside a class-def base would introduce a 3-way circular
    (``modeling_openpangu_vl`` ↔ ``modeling_pangu_omni_v2`` ↔
    ``modeling_openpangu_omni``). Keeping it dispatcher-time means the
    dispatcher fires AFTER both ``modeling_openpangu_vl`` and
    ``modeling_pangu_omni_v2`` have fully loaded.
    """
    from .modeling_openpangu_omni import OpenPanguOmni

    return OpenPanguOmni


class OpenPanguVLForConditionalGeneration(_get_open_pangu_vl_class()):
    """Vision + LLM (Pangu VL variant) — Week 3.4.d.

    Subclass of `OpenPanguVL` (the multimodal merge model defined in
    `modeling_openpangu_vl.py`). Carries the verbatim upstream class
    name so it matches `config.architectures[0]` when the on-disk
    config marks a VL-only variant; the dispatcher in
    ``__init__.py::register_pangu_omni_v2_modeling`` returns this class
    on that architecture string.

    No additional logic on top of `OpenPanguVL` — keeping it a thin
    subclass means future divergence (VL-only weight loading quirks,
    say) has a stable home. This variant has **no audio tower** by
    design; passing `input_features` will fail in `forward`.
    """

    pass


class OpenPanguUltraOmniForConditionalGeneration(_get_open_pangu_vl_class()):
    """Top-level Pangu Omni v2 — Week 3.5 dispatcher stub.

    This class is **only** used when callers explicitly instantiate
    it (``OpenPanguUltraOmniForConditionalGeneration(config)``) or via
    static type hints. Inherits from ``OpenPanguVL`` so the symbol is
    stable for legacy import paths and ``isinstance`` checks against
    the pre-Week-3.5 layout still work.

    The **real** dispatcher entry — used by VeOmni's loader to build
    the production 30B-A2B model — is
    ``register_pangu_omni_v2_modeling("OpenPanguUltraOmniForConditionalGeneration")``
    which now returns ``OpenPanguOmni`` directly (Week 3.5
    audio-aware). See ``__init__.py::register_pangu_omni_v2_modeling``
    for the routing logic.

    Why this indirection: making ``OpenPanguUltraOmniForConditionalGeneration``
    inherit ``OpenPanguOmni`` at class-def time would introduce a
    3-way circular import (see ``_get_open_pangu_omni_class``
    docstring). Instead, the class itself stays VL-shaped (image-only
    capable) and the dispatcher gets to pick the audio-aware class
    lazily.
    """

    pass


__all__ = [
    "OpenPanguV2PreTrainedModel",
    "OpenPanguV2Model",
    "OpenPanguV2ForCausalLM",
    "OpenPanguVLForConditionalGeneration",
    "OpenPanguUltraOmniForConditionalGeneration",
]
