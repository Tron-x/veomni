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

"""Decoder layer for the Pangu model family.

## What this layer does

`OpenPanguV2DecoderLayer` is the per-block unit of `OpenPanguV2Model`. It
wraps a self-attention sub-block and an MLP-or-MoE sub-block with
`OpenPanguV2RMSNorm` pre-norms and (optionally) `mHCModule` MHC streams.

## Hidden-state layout (with use_mhc=True, the 30B-A2B configuration)

The decoder layer's I/O shape is **`(B, S, n*H)`** (not `(B, S, H)`).
The MHC modules compress to `(B, S, H)` for the sub-block compute and
expand back to `(B, S, n*H)` for the residual stream. See
`_pangu_common.pangu_mhc` for the algorithm.

```
in: (B, S, n*H)
  residual_attn = in
  if use_mhc: in_attn, h_post_a, h_res_a = attn_mhc.hc_pre(in)   # (B, S, H)
  else:       in_attn = in                                       # (B, S, H), n=1 effectively
  in_attn = input_layernorm(in_attn)                             # (B, S, H)
  attn_out = self_attn(in_attn, ...)                             # (B, S, H)
  if sandwich_norm: attn_out = pre_mlp_layernorm(attn_out)
  if use_mhc: x = attn_mhc.hc_post(attn_out, residual_attn, h_post_a, h_res_a)  # (B, S, n*H)
  else:       x = residual_attn + attn_out                       # (B, S, H)

  residual_mlp = x
  if use_mhc: in_mlp, h_post_m, h_res_m = mlp_mhc.hc_pre(x)      # (B, S, H)
  else:       in_mlp = x                                         # (B, S, H)
  in_mlp = pre_mlp_layernorm(in_mlp)                             # (B, S, H)
  mlp_out = mlp(in_mlp)                                          # (B, S, H)
  if sandwich_norm: mlp_out = post_mlp_layernorm(mlp_out)
  if use_mhc: out = mlp_mhc.hc_post(mlp_out, residual_mlp, h_post_m, h_res_m)  # (B, S, n*H)
  else:       out = residual_mlp + mlp_out                       # (B, S, H)

  if has_block_post_layernorm: out = block_post_layernorm(out)
out: same shape as in
```

## first_k_dense_replace

Pangu replaces the first `first_k_dense_replace` MoE blocks with plain
dense MLPs. For 30B-A2B, `first_k_dense_replace=2`:
- layer 0, 1: `mlp = OpenPanguV2MLP(config)` (intermediate=6144)
- layer 2..36: `mlp = OpenPanguV2SparseMoeBlock(config)` (384 experts)

## use_mla

Pangu's `OpenPanguV2DecoderLayer` constructor can dispatch to
`OpenPanguV2MLAAttention` (Multi-Head Latent Attention) when
`config.use_mla=True`. **MLA is not used by 30B-A2B** (state_dict has
0 `q_a_layernorm` / 0 `kv_a_layernorm` keys), and MLA is not yet ported.
The constructor raises `NotImplementedError` for `use_mla=True`. Adding
MLA support is a future expansion when a Pangu variant needs it.

## Naming policy

Class name is verbatim `OpenPanguV2DecoderLayer`. Family-neutral alias
`PanguDecoderLayer = OpenPanguV2DecoderLayer` is exported from
`_pangu_common.__init__`.

## Inheritance

The upstream class inherits from `transformers.modeling_layers.GradientCheckpointingLayer`
to opt into transformers 5.0 gradient-checkpointing. We preserve this for
compatibility with the Pangu reference and with transformers utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .pangu_attention import OpenPanguV2Attention
from .pangu_mhc import mHCModule
from .pangu_moe import OpenPanguV2MLP, OpenPanguV2SparseMoeBlock
from .pangu_rms_norm import OpenPanguV2RMSNorm


if TYPE_CHECKING:
    from transformers.cache_utils import Cache

# Fallback to nn.Module if the transformers version doesn't expose
# GradientCheckpointingLayer — unit tests need to work on stub configs.
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:  # pragma: no cover
    import torch.nn as _nn

    GradientCheckpointingLayer = _nn.Module  # type: ignore[assignment,misc]


class OpenPanguV2DecoderLayer(GradientCheckpointingLayer):
    """Pangu V2 decoder layer with MHC + K-norm attention + (dense or MoE) MLP.

    Verbatim port of `OpenPanguV2DecoderLayer` from the Pangu reference
    (`modeling_openpangu_v2.py:908-1020`). See module docstring for
    hidden-state layout and config dispatch.

    Constructor dispatches:
    - `config.use_mla=True` -> `NotImplementedError` (MLA not yet ported).
    - `layer_idx >= config.first_k_dense_replace` -> `OpenPanguV2SparseMoeBlock`.
    - `layer_idx <  config.first_k_dense_replace` -> `OpenPanguV2MLP`.

    Optional sub-mechanisms (all preserved verbatim, none active in 30B-A2B
    except use_mhc):
    - `use_mhc` (True in 30B-A2B): wrap attention/MLP with two `mHCModule`s.
    - `sandwich_norm` (default False): extra `pre_mlp_layernorm` after
      attention and `post_mlp_layernorm` after MLP.
    - `block_post_layernorm_idx` (default None): final `block_post_layernorm`
      on layers whose index is in this list.

    A family-neutral alias `PanguDecoderLayer = OpenPanguV2DecoderLayer` is
    exported from `_pangu_common.__init__`.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        if getattr(config, "use_mla", False):
            raise NotImplementedError(
                "OpenPanguV2DecoderLayer: use_mla=True path not yet implemented. "
                "MLA attention (`OpenPanguV2MLAAttention`) is a separate verbatim "
                "port pending. The 30B-A2B model uses use_mla=False; future Pangu "
                "variants needing MLA should add the MLA port + register here."
            )
        self.self_attn = OpenPanguV2Attention(config=config, layer_idx=layer_idx)

        layer_types = getattr(config, "layer_types", None)
        self.attention_type = (
            layer_types[layer_idx] if layer_types is not None and layer_idx < len(layer_types) else None
        )

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = OpenPanguV2SparseMoeBlock(config)
        else:
            self.mlp = OpenPanguV2MLP(config)

        self.input_layernorm = OpenPanguV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_layernorm = OpenPanguV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.sandwich_norm = getattr(config, "sandwich_norm", False)
        if self.sandwich_norm:
            # Upstream re-instantiates `pre_mlp_layernorm` here (line 930) —
            # that's a no-op overwrite but we preserve it for verbatim parity.
            self.pre_mlp_layernorm = OpenPanguV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_mlp_layernorm = OpenPanguV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MHC wraps both attention and MLP sub-blocks
        self.use_mhc = getattr(config, "use_mhc", False)
        if self.use_mhc:
            self.attn_mhc_module = mHCModule(config)
            self.mlp_mhc_module = mHCModule(config)

        block_post_layernorm_idx = getattr(config, "block_post_layernorm_idx", None)
        self.has_block_post_layernorm = block_post_layernorm_idx is not None and layer_idx in block_post_layernorm_idx
        if self.has_block_post_layernorm:
            # Norm dim depends on MHC: stream-expanded width if MHC is on.
            mhc_num_stream = getattr(config, "mhc_num_stream", 1)
            block_post_layernorm_hidden_size = config.hidden_size * (mhc_num_stream if self.use_mhc else 1)
            self.block_post_layernorm = OpenPanguV2RMSNorm(block_post_layernorm_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # ---- Attention sub-block ----
        residual = hidden_states

        if self.use_mhc:
            hidden_states, h_post_attn, h_res_attn = self.attn_mhc_module.hc_pre(hidden_states)

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self.sandwich_norm:
            hidden_states = self.pre_mlp_layernorm(hidden_states)

        if self.use_mhc:
            hidden_states = self.attn_mhc_module.hc_post(hidden_states, residual, h_post_attn, h_res_attn)
        else:
            hidden_states = residual + hidden_states

        # ---- MLP sub-block ----
        residual = hidden_states

        if self.use_mhc:
            hidden_states, h_post_mlp, h_res_mlp = self.mlp_mhc_module.hc_pre(hidden_states)

        # Upstream has identical pre_mlp_layernorm call in both branches
        # (line 1004-1007) — sandwich_norm just renames the same operation
        # for clarity. Verbatim preserves this.
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        if self.use_mhc:
            hidden_states = self.mlp_mhc_module.hc_post(hidden_states, residual, h_post_mlp, h_res_mlp)
        else:
            hidden_states = residual + hidden_states

        if self.has_block_post_layernorm:
            hidden_states = self.block_post_layernorm(hidden_states)

        return hidden_states


__all__ = ["OpenPanguV2DecoderLayer"]
