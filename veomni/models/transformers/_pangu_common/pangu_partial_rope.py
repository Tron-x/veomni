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

"""Partial rotary positional embedding for the Pangu model family.

## What is "partial RoPE"

Standard RoPE applies rotary transforms to all `head_dim` dimensions of
query/key vectors. Partial RoPE applies them only to the leading
`rotary_ndims = head_dim * partial_rotary_factor` dimensions and leaves the
trailing `head_dim - rotary_ndims` dimensions untouched. This is the same
mechanism used by Phi3 / GraniteMoE / PhiMoE in mainline transformers, just
spelled differently in Pangu's config schema.

## Pangu-specific config schema

Pangu's `config.json` exposes:

- `head_dim` (e.g. 128)
- `qk_rope_dim` (e.g. 32) — informational; equals `head_dim * partial_rotary_factor`
- `partial_rotary_factor` (e.g. 0.25) — fraction of dims to rotate
- `rope_theta` (e.g. 400000)
- `rope_scaling = {"rope_type": "default", "rotary_mode": "half", ...}`
  ("rotary_mode" is a Pangu-internal hint; the implementation always uses
  the standard `rotate_half` pattern when `rope_interleaved=False`.)
- `rope_interleaved` (e.g. False) — selects rotate_half vs interleave

The legacy `rope_parameters` dict referenced in the Pangu modeling code is
populated by `OpenPanguOmniConfig.__post_init__` (when ported in Week 1)
from the flat fields above.

## Naming policy

Class / function names are **kept verbatim** from the Pangu reference
(`modeling_openpangu_v2.py:247-360`):

- `OpenPanguV2RotaryEmbedding`
- `apply_rotary_pos_emb`
- `rotate_half`

This is so `patchgen` rewrite rules and oracle log grep match upstream
1:1. A family-neutral alias `PanguRotaryEmbedding = OpenPanguV2RotaryEmbedding`
is exported from `_pangu_common.__init__` for code that wants to express
"shared across all Pangu variants" semantics.

One symbol does **not** exist in the upstream reference because Pangu
inlined the partial-routing logic into `OpenPanguV2Attention.forward`:

- `apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_ndims, ...)`

That helper is named in the natural mainline-transformers style
(`apply_rotary_pos_emb` -> `apply_partial_rotary_pos_emb`); the Pangu-flavored
alias `apply_pangu_partial_rope` is exported alongside.

## Port strategy

Phase 1 is a **verbatim port** of the Pangu reference implementation so
oracle parity is bit-for-bit by construction. Vendor-only decorators
(`@use_kernel_func_from_hub`, `maybe_autocast`) are reduced to no-op
shims here so the module loads cleanly on stock transformers 5.0.0. The
math and the dtype/device control flow are unchanged.

GPU/NPU swap-in for optimized kernels happens at the VeOmni OpSlot layer
(`rotary_pos_emb_implementation` in `OpsImplementationConfig`); the
high-level partial-routing logic in this module stays pure PyTorch.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update


# ---------------------------------------------------------------------------
# RoPE inverse-frequency computation
# ---------------------------------------------------------------------------


def compute_default_rope_parameters(
    config,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
) -> tuple[torch.Tensor, float]:
    """Compute inv_freq for Pangu partial RoPE.

    Verbatim port of `OpenPanguV2RotaryEmbedding.compute_default_rope_parameters`
    from the Pangu reference (`modeling_openpangu_v2.py:267-296`). Lifted to
    module level (the upstream defines it as `@staticmethod`) so it can be
    registered with `ROPE_INIT_FUNCTIONS` or invoked independently.

    The reader is robust to two config schemas:

    1. `config.rope_parameters` dict (the original Pangu schema)
    2. Flat config attributes (`config.rope_theta`, `config.partial_rotary_factor`,
       `config.head_dim`) — used as fallback when the dict is missing.

    Args:
        config: Pangu config with `rope_theta`, `partial_rotary_factor`,
            and either `head_dim` or (`hidden_size` + `num_attention_heads`).
        device: device for the returned tensor.
        seq_len: unused for the default RoPE type; retained for signature
            compatibility with `ROPE_INIT_FUNCTIONS`.

    Returns:
        (inv_freq, attention_scaling) where `inv_freq` has shape
        `[rotary_ndims / 2]` and `attention_scaling` is `1.0` for this RoPE type.
    """
    rope_params = getattr(config, "rope_parameters", None) or {}

    base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = rope_params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    rotary_ndims = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rotary_ndims, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / rotary_ndims)
    )
    return inv_freq, attention_factor


# ---------------------------------------------------------------------------
# Rotary embedding module
# ---------------------------------------------------------------------------


class OpenPanguV2RotaryEmbedding(nn.Module):
    """Rotary embedding for Pangu partial RoPE.

    Verbatim port of `OpenPanguV2RotaryEmbedding` from the Pangu reference
    (`modeling_openpangu_v2.py:247-311`). Class name is preserved verbatim
    so patchgen rewrite rules and grep with the upstream stay 1:1.

    Behavior:
    - Caches `inv_freq` and `original_inv_freq` (the latter for dynamic
      rope rescaling hooks; see `@dynamic_rope_update`).
    - Forward returns `(cos, sin)` of shape `[B, seq, rotary_ndims]` in
      `x.dtype`, computed in float32 then cast back (matches Pangu's
      `maybe_autocast(..., enabled=False)` block — we use
      `torch.amp.autocast` for the same effect).

    A family-neutral alias `PanguRotaryEmbedding` is exported from
    `_pangu_common.__init__` for adapter code that wants to express
    "shared across all Pangu variants" semantics.
    """

    inv_freq: torch.Tensor  # type hint for register_buffer

    # Mirror the upstream staticmethod so call sites that go through the
    # class (e.g. `OpenPanguV2RotaryEmbedding.compute_default_rope_parameters(...)`)
    # keep working bit-for-bit.
    compute_default_rope_parameters = staticmethod(compute_default_rope_parameters)

    def __init__(self, config, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_params = getattr(config, "rope_parameters", None) or {}
        self.rope_type: str = rope_params.get("rope_type", "default")

        # Always use our Pangu-aware computation for the default rope_type;
        # delegate to the global ROPE_INIT_FUNCTIONS registry only for
        # non-default rope_types (linear / dynamic / yarn / etc.) where the
        # mainline transformers implementation suffices.
        rope_init_fn: Callable
        if self.rope_type == "default":
            rope_init_fn = compute_default_rope_parameters
        else:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @torch.no_grad()
    @dynamic_rope_update  # advanced RoPE types (e.g. dynamic / yarn) hook in here
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        # Pangu forces float32 here (line 305 in the reference) to avoid
        # bf16/fp16 precision loss in the (inv_freq @ position_ids) product
        # for long contexts. We replicate that explicitly with autocast off.
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Apply: standard rotary + partial-routing helpers
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim by negating and swapping.

    Verbatim port of `rotate_half` from the Pangu reference (line 330-334).
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to (q, k).

    Verbatim port of `apply_rotary_pos_emb` from the Pangu reference
    (line 337-360). Removed the `@use_kernel_func_from_hub` decorator —
    in VeOmni the kernel swap is driven by `OpsImplementationConfig`
    (the OpSlot mechanism), not by transformers kernel-hub annotations.

    Shapes:
        q, k: `[B, num_heads, seq, head_dim]` (unsqueeze_dim=1) or
              `[B, seq, num_heads, head_dim]` (unsqueeze_dim=2)
        cos, sin: `[B, seq, head_dim]`
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_ndims: int,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial RoPE to (q, k).

    Splits each vector along the last dim at `rotary_ndims`: the leading
    slice gets RoPE applied, the trailing slice passes through unchanged.
    The two slices are then concatenated to restore the original shape.

    This **factors out** the inline split/apply/cat pattern in
    `OpenPanguV2Attention.forward` (Pangu reference line 493-507) so the
    family-level attention port and any future variant that reuses partial
    RoPE share a single implementation. There is no upstream function with
    this exact signature; the name follows the natural mainline transformers
    extension `apply_rotary_pos_emb` -> `apply_partial_rotary_pos_emb`.

    A Pangu-flavored alias `apply_pangu_partial_rope` is exported alongside.

    Args:
        q, k: query / key tensors, last dim is full `head_dim`.
        cos, sin: rotary tables of shape `[B, seq, rotary_ndims]`.
        rotary_ndims: number of leading dims to apply RoPE to
            (`= head_dim * partial_rotary_factor`).
        unsqueeze_dim: passed through to `apply_rotary_pos_emb`.

    Returns:
        (q_out, k_out) with same shape as the inputs.
    """
    if rotary_ndims == q.shape[-1]:
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
    k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=unsqueeze_dim)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


__all__ = [
    "OpenPanguV2RotaryEmbedding",
    "apply_partial_rotary_pos_emb",
    "apply_rotary_pos_emb",
    "compute_default_rope_parameters",
    "rotate_half",
]
