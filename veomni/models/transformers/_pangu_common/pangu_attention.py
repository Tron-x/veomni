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

"""Attention module for the Pangu model family.

## Components

This module ports three pieces from the Pangu reference modeling file
(`modeling_openpangu_v2.py:363-555`):

1. `repeat_kv` (line 363-372) — standard GQA helper.
2. `eager_attention_forward` (line 375-398) — vanilla softmax attention
   used as the default `_attn_implementation`. Flash / SDPA paths are
   dispatched through `transformers.ALL_ATTENTION_FUNCTIONS`.
3. `OpenPanguV2Attention` (line 401-555) — the main attention module.

## Pangu-specific design points

The attention module composes several Pangu-flavored mechanisms on top of
a standard GQA + RoPE backbone:

- **Fused QKV projection** (`qkv_proj` returns `q || k || v`, split by
  `[q_size, k_size, v_size]`). `v_head_dim` may differ from `head_dim`
  (in the 30B-A2B model both are 128).
- **K-norm** (`k_layernorm`, per-head `OpenPanguV2RMSNorm(head_dim)`)
  applied to `key_states` **before** RoPE. The Pangu reference always
  instantiates and always calls this; the `use_k_norm` field in
  `config.json` is informational and not consulted by the modeling code.
  We mirror that behavior verbatim.
- **Partial RoPE** (`apply_partial_rotary_pos_emb`) — rotates only the
  leading `rotary_ndims = head_dim * partial_rotary_factor` dims of q/k.
  See `_pangu_common.pangu_partial_rope`.
- **Optional `param_sink`** (`param_sink_number > 0`) — learnable
  prefix-pad key/value rows broadcast over the batch (attention-sink
  technique). Off in the 30B-A2B model; kept verbatim for future variants.
- **Optional `attn_groupnorm`** — extra `OpenPanguV2RMSNorm(head_dim)` on
  the per-head output before reshape+o_proj. Off in 30B-A2B.
- **Optional `attn_elementwise_gate`** — sigmoid-gated output. Off in
  30B-A2B.

## Naming policy

The class is named `OpenPanguV2Attention` to stay 1:1 with the Pangu
reference. A family-neutral alias `PanguAttention = OpenPanguV2Attention`
is exported from `_pangu_common.__init__`.

## OpSlot dispatch

The upstream `@use_kernelized_func(apply_rotary_pos_emb)` decorator on the
attention class is dropped — VeOmni handles RoPE kernel swap via the
`rotary_pos_emb_implementation` OpSlot, and softmax-attention kernel swap
via `_attn_implementation` (which dispatches through transformers'
`ALL_ATTENTION_FUNCTIONS`). The verbatim port preserves both dispatch
points so kernel swap stays available at the call site.

## Module-level vs config-level optional fields

Pangu's `OpenPanguV2Config` defines several optional fields that gate
attention sub-behaviors (`v_head_dim`, `attn_groupnorm`,
`attn_elementwise_gate`, `param_sink_number`, `attention_bias`,
`sliding_window`, `layer_types`). For the 30B-A2B model and any other
config that omits these, the constructor falls back to safe defaults:

| Config field              | Default in this port      | Source       |
|---------------------------|---------------------------|--------------|
| `v_head_dim`              | `head_dim`                | upstream     |
| `attn_groupnorm`          | `False`                   | this port    |
| `attn_elementwise_gate`   | `False`                   | this port    |
| `param_sink_number`       | `0`                       | this port    |
| `attention_bias`          | `False`                   | this port    |
| `attention_dropout`       | `0.0`                     | this port    |
| `sliding_window`          | `None`                    | this port    |
| `layer_types`             | `None`                    | this port    |
| `_attn_implementation`    | `"eager"`                 | upstream     |

These "this port" defaults are not in the upstream — upstream relies on
`OpenPanguV2Config` always having them set. We add `getattr` fallbacks so
this module works with minimal configs (e.g. unit-test fake configs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn as nn

from .pangu_partial_rope import (
    apply_partial_rotary_pos_emb,  # re-exported for callers that want it directly
)
from .pangu_rms_norm import OpenPanguV2RMSNorm


if TYPE_CHECKING:
    from transformers.cache_utils import Cache

# transformers.utils.Unpack moved across versions; import defensively so the
# module loads on minimal stubs (e.g. unit-test configs without full transformers).
try:
    from transformers.utils import Unpack  # noqa: F401
except ImportError:  # pragma: no cover
    Unpack = None  # type: ignore[assignment]

# `ALL_ATTENTION_FUNCTIONS` is the transformers-side dispatcher registry for
# flash / sdpa / xformers / etc. It is queried by name; the entries are
# populated by transformers as backends register themselves.
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:  # pragma: no cover
    ALL_ATTENTION_FUNCTIONS = {}  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# GQA helper
# ---------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads `n_rep` times along the head axis.

    Verbatim port of `repeat_kv` from the Pangu reference (line 363-372).
    Equivalent to `torch.repeat_interleave(x, dim=1, repeats=n_rep)` but
    uses expand+reshape to avoid the memory copy for `n_rep == 1`.

    Shape: `(batch, num_kv_heads, seq, head_dim)` ->
           `(batch, num_kv_heads * n_rep, seq, head_dim)`
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Eager attention forward
# ---------------------------------------------------------------------------


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla softmax attention (the `eager` `_attn_implementation`).

    Verbatim port of `eager_attention_forward` from the Pangu reference
    (line 375-398). Softmax is always computed in float32 for numerical
    stability and cast back to `query.dtype` (this matches the upstream
    behavior and is preserved bit-for-bit).

    Args:
        module: the calling attention module, used to read
            `num_key_value_groups` and `training`.
        query: `[B, num_heads, S_q, head_dim]`
        key, value: `[B, num_kv_heads, S_kv, head_dim]` (will be repeated
            to `num_heads` via `repeat_kv` for GQA).
        attention_mask: additive mask broadcastable to
            `[B, 1, S_q, S_kv]` (typically `-inf` outside causal triangle).
        scaling: `head_dim ** -0.5` typically.
        dropout: attention dropout probability.

    Returns:
        (attn_output, attn_weights) with `attn_output` transposed back to
        `[B, S_q, num_heads, head_dim]` (note: not yet reshape-fused into
        `[B, S_q, hidden_size]`; the caller does that).
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------


class OpenPanguV2Attention(nn.Module):
    """Pangu V2 attention with K-norm + partial RoPE + optional add-ons.

    Verbatim port of `OpenPanguV2Attention` from the Pangu reference
    (line 401-555). The upstream `@use_kernelized_func(apply_rotary_pos_emb)`
    class decorator is dropped — VeOmni's OpSlot mechanism handles RoPE
    kernel swap. Inline partial-RoPE split/apply/cat (upstream line 494-507)
    is replaced by a single call to `apply_partial_rotary_pos_emb` from
    `_pangu_common.pangu_partial_rope` for sharing across all Pangu variants.

    See module docstring for design rationale, config-field defaults,
    and OpSlot dispatch points.

    A family-neutral alias `PanguAttention = OpenPanguV2Attention` is
    exported from `_pangu_common.__init__`.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True

        # layer_types / sliding_window are populated by the model-level config
        # for hybrid-window architectures. Default to None on flat configs.
        layer_types = getattr(config, "layer_types", None)
        self.layer_type = layer_types[layer_idx] if layer_types is not None and layer_idx < len(layer_types) else None
        self.sliding_window = (
            getattr(config, "sliding_window", None) if self.layer_type == "sliding_attention" else None
        )

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        rope_params = getattr(config, "rope_parameters", None) or {}
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor",
            getattr(config, "partial_rotary_factor", 1.0),
        )
        self.rotary_ndims = int(self.head_dim * partial_rotary_factor)

        self.v_head_dim = config.v_head_dim if getattr(config, "v_head_dim", None) is not None else config.head_dim

        self.q_size = self.num_attention_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim
        self.qkv_size = self.q_size + self.k_size + self.v_size

        attention_bias = getattr(config, "attention_bias", False)
        self.qkv_proj = nn.Linear(config.hidden_size, self.qkv_size, bias=attention_bias)
        self.k_layernorm = OpenPanguV2RMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=attention_bias,
        )

        self.attn_groupnorm = getattr(config, "attn_groupnorm", False)
        self.attn_elementwise_gate = getattr(config, "attn_elementwise_gate", False)
        self.param_sink_number = getattr(config, "param_sink_number", 0)

        if self.param_sink_number > 0:
            param_dtype = getattr(config, "torch_dtype", torch.float32)
            self.param_sink_key = nn.Parameter(
                torch.empty(
                    (self.param_sink_number, self.num_key_value_heads, self.head_dim),
                    dtype=param_dtype,
                )
            )
            self.param_sink_value = nn.Parameter(
                torch.empty(
                    (self.param_sink_number, self.num_key_value_heads, self.v_head_dim),
                    dtype=param_dtype,
                )
            )

        if self.attn_groupnorm:
            self.groupnorm = OpenPanguV2RMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps)

        if self.attn_elementwise_gate:
            self.attention_gate = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.attn_elementwise_gate:
            gate_score = self.attention_gate(hidden_states)
        else:
            gate_score = None

        input_shape = hidden_states.shape[:-1]

        mixed_qkv = self.qkv_proj(hidden_states)

        query_states, key_states, value_states = torch.split(
            mixed_qkv, [self.q_size, self.k_size, self.v_size], dim=-1
        )

        query_states = query_states.view(*input_shape, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(*input_shape, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)

        # K-norm: per-head RMSNorm on key states, before RoPE. Pangu reference
        # always instantiates and always calls this; `config.use_k_norm` is
        # informational metadata and is not consulted by the modeling code.
        key_states = self.k_layernorm(key_states)

        cos, sin = position_embeddings
        # Partial rotary embedding — rotates only the leading rotary_ndims
        # of head_dim, leaves the rest as identity.
        query_states, key_states = apply_partial_rotary_pos_emb(
            query_states, key_states, cos, sin, rotary_ndims=self.rotary_ndims
        )

        if past_key_values is not None:
            # sin and cos are RoPE-specific; cache_position is needed for static cache.
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.param_sink_number > 0:
            # Prepend learnable attention-sink rows to KV. Same as upstream.
            batch_size, _kv_seq_len = key_states.shape[0], key_states.shape[2]
            param_sink_key = (
                self.param_sink_key.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1).to(key_states.device)
            )
            param_sink_value = (
                self.param_sink_value.permute(1, 0, 2)
                .unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .to(value_states.device)
            )
            key_states = torch.cat([param_sink_key, key_states], dim=2)
            value_states = torch.cat([param_sink_value, value_states], dim=2)

            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(attention_mask, (self.param_sink_number, 0), value=0.0)

        # Dispatch to the configured attention implementation (eager / sdpa /
        # flash / flash3 / xformers / npu / ...). VeOmni's `_attn_implementation`
        # setting flows into here via `config._attn_implementation`.
        attention_interface: Callable = eager_attention_forward
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        if attn_impl != "eager" and attn_impl in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        if self.attn_groupnorm:
            attn_output = self.groupnorm(attn_output)
        if self.attn_elementwise_gate:
            attn_output = attn_output * gate_score.sigmoid().view(attn_output.shape)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


__all__ = [
    "OpenPanguV2Attention",
    "eager_attention_forward",
    "repeat_kv",
]
