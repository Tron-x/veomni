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

"""Pangu Omni v2 vision tower and multimodal merge layer.

This module is the VeOmni-side counterpart of the Pangu reference
`modeling_openpangu_vl.py`. The reference file packs both the vision
encoder (lines 72-617) and the text-side multimodal merge layer
(lines 618-end) into a single ~1900-line module; we mirror that here so
that downstream patchgen can diff our adapter against upstream
line-for-line.

## Vision Primitives

Ports of the vision-tower primitives:

- `PanguEmbeddedRMSNorm` — variance-only RMSNorm with `eps=1e-6` default
  (vision uses 1e-6, distinct from the text backbone's 1e-5).
- `OpenPanguRMSNorm` — alias of `PanguEmbeddedRMSNorm` for symmetry with
  the text-side `OpenPanguV2RMSNorm`.
- `OpenPanguVLMLP` — SwiGLU-style gated MLP (gate_proj + up_proj +
  down_proj) used inside `OpenPanguVLVisionBlock`. Optional bias.
- `OpenPanguVisionPatchEmbed` / `OpenPanguVLPatchEmbed` — Conv3d-based
  patch embedding that handles both image (single temporal patch) and
  video (multi-frame) inputs. The `if hidden_states.shape[-1] !=
  self.input_size` branch handles the single-frame case by replicating
  to a 2-temporal-patch tensor.
- `OpenPanguVisionRotaryEmbedding` — 1D RoPE inv_freq buffer; the
  2D vision RoPE (`cos`/`sin` over the H/W grid) is built in
  `OpenPanguVisionTransformerPretrainedModel`.
- `OpenPanguVLPatchMerger` — final-stage merger that downsamples by
  `spatial_merge_size**2` and projects to the text-backbone hidden
  size. Optional `use_gatedmerger` adds a SiLU-gated branch (matches
  Pangu 30B-A2B's `vision_config.use_gatedmerger=True`).
- Helpers: `rotate_half`, `apply_rotary_pos_emb_vision`, `repeat_kv`,
  `eager_attention_forward`.

## Vision Transformer

- `OpenPanguVLVisionAttention` — multi-head self-attention with vision
  RoPE; NPU fast path falls back to `eager_attention_forward` on GPU.
- `OpenPanguVLVisionBlock` — pre-LN attention + MLP, gradient-checkpointed.
- `OpenPanguVisionTransformerPretrainedModel` — full ViT-style tower
  with rotary position grid, fullattn-block selection, and merger.

## NPU-specific code stripped

The reference top-level imports `torch_npu` unconditionally and the
attention block dispatches to `torch_npu.npu_fusion_attention` when
`NPU_ATTN_INFR` is True. Our adapter guards both behind try/except so
the module imports cleanly on GPU; the NPU fast path is preserved when
the runtime detects an Ascend device.

## Why not put this in `_pangu_common/`?

`_pangu_common/` is for primitives shared across multiple Pangu *model
variants* (text MoE, VL, Omni). The vision tower is presently consumed
by exactly one adapter package (`pangu_omni_v2/`), so co-locating
under `pangu_omni_v2/modeling_vl.py` keeps the dispatcher
graph flat. If a second adapter package emerges that also needs the
vision tower, we'll lift these primitives into
`_pangu_common/pangu_vision/`.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# NPU is optional; the reference unconditionally imports torch_npu, but
# downstream Forge / Monarch jobs running on GPU shouldn't take a hard
# dependency. Guard the import so the module loads cleanly anywhere.
try:
    import torch_npu  # noqa: F401
    from transformers.utils import is_torch_npu_available

    if is_torch_npu_available() and "910" in torch.npu.get_device_name():
        NPU_ATTN_INFR = True
    else:
        NPU_ATTN_INFR = False
except ImportError:
    torch_npu = None  # type: ignore[assignment]
    NPU_ATTN_INFR = False

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import is_torchdynamo_compiling, logging

from .._pangu_common.pangu_moe import (
    veomni_moe_experts_forward,  # noqa: F401 — see comment below
)
from .configuration_pangu_omni_v2 import OpenPanguOmniConfig, OpenPanguOmniVisionConfig


# Re-export the MoE-experts OpSlot for ``_bind_veomni_ops`` discovery
# when the loaded class is ``OpenPanguVL`` — same rationale as
# ``modeling_omni.py`` and ``modeling_text.py``. See
# the latter's "OpSlot re-export" docstring for the full reasoning.
# Without this, ``moe_implementation: fused_npu`` would silently leave
# the slot unbound on the multimodal VL path and the eager forward in
# ``OpenPanguV2Experts`` would index global expert IDs into a locally
# EP-sharded tensor (``IndexError``).

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm primitives
# ---------------------------------------------------------------------------


class PanguEmbeddedRMSNorm(nn.Module):
    """Variance-only RMSNorm with default `eps=1e-6` (vision-side).

    Equivalent to T5LayerNorm. Distinct from the text backbone's
    `OpenPanguV2RMSNorm` only in the default `eps` (vision uses 1e-6,
    text uses 1e-5). Both implementations are identical otherwise — we
    intentionally do NOT deduplicate so that future divergence (e.g.
    vision adding an `affine_bias` term) stays localized.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class OpenPanguRMSNorm(PanguEmbeddedRMSNorm):
    """Alias kept for symmetry with the reference's two-name convention.

    Reference `modeling_vl.py` defines both
    `PanguEmbeddedRMSNorm` (used inside the ViT) and `OpenPanguRMSNorm`
    (used inside `OpenPanguVLPatchMerger`). They are the same class.
    """


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class OpenPanguVLMLP(nn.Module):
    """SwiGLU-style gated MLP used inside `OpenPanguVLVisionBlock`.

    Layout mirrors the reference at line 94 of `modeling_vl.py`:
    when `hidden_act == "silu"` we have a 3-projection gated MLP
    (`gate_proj`, `up_proj`, `down_proj`); otherwise we fall back to a
    2-projection MLP (`up_proj` followed by activation, then
    `down_proj`).

    Pangu 30B-A2B's `vision_config.hidden_act == "gelu"` according to
    `config.json::vision_config`, so on that model the gate_proj path is
    NOT taken; we keep the silu path for forward compatibility with
    other Pangu variants.
    """

    def __init__(self, config, bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        if self.hidden_act == "silu":
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "silu":
            x_gate = self.gate_proj(hidden_state)
            x_gate = self.act_fn(x_gate)
            x_up = self.up_proj(hidden_state)
            intermediate_parallel = x_gate * x_up
        else:
            x_up = self.up_proj(hidden_state)
            intermediate_parallel = self.act_fn(x_up)
        x_down = self.down_proj(intermediate_parallel)
        return x_down


# ---------------------------------------------------------------------------
# Patch embedding (image / video)
# ---------------------------------------------------------------------------


class OpenPanguVisionPatchEmbed(nn.Module):
    """Conv3d-based patch embedding supporting both images and videos.

    The Conv3d kernel is `(temporal_patch_size, patch_size, patch_size)`
    with matching stride, so a single-frame image (T=1) and a multi-frame
    video (T>1) both flatten cleanly into per-patch embeddings.

    The `hidden_states.shape[-1] != self.input_size` branch handles the
    case where the caller passed pre-flattened single-temporal-patch
    tensors (length = `patch_size * patch_size * in_channels`) instead
    of `2*patch*patch*in_channels`; we replicate to fill the
    temporal axis.
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
        self.input_size = self.patch_size * self.patch_size * in_channels * self.temporal_patch_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.input_size:
            hidden_states = torch.cat(
                [
                    hidden_states.reshape(-1, self.patch_size * self.patch_size),
                    hidden_states.reshape(-1, self.patch_size * self.patch_size),
                ],
                dim=-1,
            ).reshape(-1, self.input_size)
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class OpenPanguVLPatchEmbed(OpenPanguVisionPatchEmbed):
    """Alias of `OpenPanguVisionPatchEmbed` (reference keeps both names)."""


# ---------------------------------------------------------------------------
# Rotary position embedding (vision)
# ---------------------------------------------------------------------------


class OpenPanguVisionRotaryEmbedding(nn.Module):
    """1D rotary inv_freq buffer used by the vision attention layer.

    Vision uses a 2D RoPE assembled from row/column frequencies; this
    module is the per-axis 1D primitive. The 2D grid is built in
    `OpenPanguVisionTransformerPretrainedModel.rot_pos_emb` (Week
    3.2.b).
    """

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


# ---------------------------------------------------------------------------
# Patch merger (final-stage spatial downsampling + projection)
# ---------------------------------------------------------------------------


class OpenPanguVLPatchMerger(nn.Module):
    """Spatial merger that downsamples by `spatial_merge_size**2` and
    projects to `dim` (the text-backbone hidden size).

    Two variants controlled by `use_gatedmerger`:

    - **`use_gatedmerger=False`** (vanilla): a 2-layer MLP
      (`Linear -> GELU -> Linear`) mapping
      `context_dim * sm^2 -> context_dim * sm^2 -> dim`.

    - **`use_gatedmerger=True`** (Pangu 30B-A2B default per
      `vision_config.use_gatedmerger=True`): the second Linear emits
      `2 * dim` channels, which are then chunked into `(x, gate)` and
      combined as `x * silu(gate)`. This is structurally equivalent to
      a SwiGLU output stage.

    Pre-LN with `OpenPanguRMSNorm(context_dim, eps=1e-6)`.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        use_gatedmerger: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.gated = use_gatedmerger
        self.ln_q = OpenPanguRMSNorm(context_dim, eps=1e-6)
        self.gate_act = nn.SiLU() if use_gatedmerger else None
        outdim = dim * 2 if use_gatedmerger else dim
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, outdim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        if self.gated:
            x, gate = torch.chunk(x, 2, dim=-1)
            x = x * self.gate_act(gate)
        return x


# ---------------------------------------------------------------------------
# Rotary helpers
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half (negated)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D vision RoPE to query / key.

    `cos` / `sin` are `(seq_len, head_dim)`-shaped tensors; we
    `unsqueeze(-2)` to broadcast across the head axis and cast to fp32
    for numerically safe multiplication before casting back.
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Attention helpers shared with the vision attention block.
# ---------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat-interleave the KV head axis by `n_rep`.

    Equivalent of `torch.repeat_interleave(x, dim=1, repeats=n_rep)`.
    Shape: `(batch, num_kv_heads, seqlen, head_dim) -> (batch,
    num_kv_heads * n_rep, seqlen, head_dim)`.

    GQA is not used in the vision tower (`num_kv_groups=1` in
    `OpenPanguVLVisionAttention`), so this helper is effectively a
    no-op there, but the multimodal `OpenPanguVLAttention`
    DOES use GQA, sharing this helper.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,  # noqa: ANN003 — accepts FlashAttention extras like cu_seq_lens_q
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-eager attention used by `OpenPanguVLVisionAttention` and
    optionally by the multimodal merge layer.

    Mirrors the reference at line 231. Softmax in fp32 then cast back to
    `query.dtype`.
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


# ===========================================================================
# Vision attention block + transformer tower.
#
# Verbatim port of `modeling_vl.py:257-617`. The NPU
# fast-path inside `OpenPanguVLVisionAttention.forward` (lines 307-329
# of the reference) calls `torch_npu.npu_fusion_attention`; in our
# adapter we guard that branch by `NPU_ATTN_INFR` (set False when
# `torch_npu` isn't installed at module-load time) so the same code
# runs unmodified on GPU.
# ===========================================================================


def _init_copy(target: torch.Tensor, source: torch.Tensor) -> None:
    """In-place copy used by `_init_weights` on rotary inv_freq buffers.

    Reference code imports `init.copy_` from a vendor fork
    (`transformers.initialization` shipped with Ascend transformers).
    On mainline transformers that module doesn't exist, so reference
    falls back to a `_InitShim` that wraps `target.copy_(source)` in
    `torch.no_grad()`. We use the same fallback verbatim — under FSDP2 /
    DCP loading, the inv_freq buffer is non-persistent and will be
    re-derived on the next forward, so init policy here matters only
    for the eager-construct path.
    """
    with torch.no_grad():
        target.copy_(source)


class OpenPanguVLVisionAttention(nn.Module):
    """Vision self-attention with cu_seqlens, RoPE, NPU/eager dispatch.

    Mirrors the reference at line 257-349 of `modeling_vl.py`.

    - **QKV projection** is a single `nn.Linear(dim, 3*dim, bias=True)`
      that's then `reshape -> permute -> unbind` into three head-axis
      tensors.

    - **Rotary embedding** is applied via `apply_rotary_pos_emb_vision`
      on the un-batched `(seq, num_heads, head_dim)` layout. Reference
      supports both the legacy `rotary_pos_emb` 2D-tensor path and the
      current `position_embeddings: tuple(cos, sin)` path; we preserve
      both for API compatibility.

    - **Attention backend** is dispatched by `self.config._attn_implementation`:
      `"eager"` (our reference fallback), or any backend registered in
      `ALL_ATTENTION_FUNCTIONS` (FA2, SDPA, etc.). When
      `not self.training and NPU_ATTN_INFR`, we take the NPU fused
      path via `torch_npu.npu_fusion_attention` — on GPU this branch
      is dead because `NPU_ATTN_INFR=False`.

    - **GQA**: `num_key_value_groups=1` (vision tower doesn't use GQA;
      that's text-side multimodal attention's job).

    - **Causal**: `is_causal=False` — vision sees all patches.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        if position_embeddings is None:
            # Legacy path — derive cos/sin from `rotary_pos_emb` tensor.
            # Reference logs a warning here saying v4.54 removes this
            # path; we keep it for parity but downstream callers should
            # prefer `position_embeddings`.
            logger.warning_once(
                "Vision attention received `rotary_pos_emb` instead of "
                "`position_embeddings`. The latter is preferred; the former path "
                "will be removed in a future release."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Reshape to (batch=1, num_heads, seq, head_dim) for attention dispatch.
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if not self.training and NPU_ATTN_INFR:
            # NPU fused path — only fires on real Ascend NPUs where
            # `torch_npu` is genuinely installed. On GPU/CPU,
            # `NPU_ATTN_INFR=False` and this branch is dead.
            from einops import rearrange  # local import to keep GPU env import-light

            if isinstance(cu_seqlens, torch.Tensor):
                cu_seqlens = cu_seqlens.tolist()

            q, k, v = (rearrange(x, "b n s d -> (b s) n d") for x in [query_states, key_states, value_states])
            attn_output = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                self.num_heads,
                "TND",
                pse=None,
                padding_mask=None,
                atten_mask=None,
                scale=self.scaling,
                pre_tockens=1048576,
                next_tockens=0,
                keep_prob=1.0,
                inner_precise=0,
                sparse_mode=0,
                actual_seq_qlen=cu_seqlens,
                actual_seq_kvlen=cu_seqlens,
            )[0]
        else:
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class OpenPanguVLVisionBlock(GradientCheckpointingLayer):
    """Pre-LN attention + MLP block used by `OpenPanguVisionTransformer`.

    Mirrors the reference at line 352-378. RMSNorm with eps=1e-6 on
    both pre-attention and pre-MLP; MLP uses bias=True (this is the
    vision-tower bias convention — the text backbone's MLP uses
    bias=False).

    Inherits `GradientCheckpointingLayer` so the per-block recompute
    happens transparently when the parent
    `OpenPanguVisionTransformerPretrainedModel.gradient_checkpointing
    = True`. On forward, GradientCheckpointingLayer calls our `forward`
    directly when checkpointing is off, or via
    `torch.utils.checkpoint.checkpoint` when on — no extra code needed.
    """

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = OpenPanguRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = OpenPanguRMSNorm(config.hidden_size, eps=1e-6)
        self.attn = OpenPanguVLVisionAttention(config=config)
        self.mlp = OpenPanguVLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class OpenPanguPreTrainedModel(PreTrainedModel):
    """Base PreTrainedModel for OpenPanguVL / OpenPanguOmni stacks.

    Reference: `modeling_vl.py:382`. Carries the standard
    PreTrainedModel knobs (FA2 / SDPA / cache class support) plus a
    custom `_init_weights` that special-cases vision RoPE inv_freq
    buffers — those are non-persistent and need to be re-computed on
    each fresh instantiation rather than left as the random tensor
    that `nn.Parameter` defaults to.
    """

    config_class = OpenPanguOmniConfig
    supports_gradient_checkpointing = True
    # ``OpenPanguVLDecoderLayer`` (the upstream reference's name) is dead
    # code in the 30B-A2B inference path — see comment near line 1295.
    # The text backbone (``OpenPanguVLTextModel`` → ``_OpenPanguV2Model``)
    # actually uses ``OpenPanguV2DecoderLayer``, so that's what we declare
    # here. ``ConformerEncoderLayerBlock`` is the actual audio-tower
    # layer class (``modeling_huanyu_audio_encoder.py``); the previous
    # entry ``Qwen2AudioEncoderLayer`` was a copy/paste from the Qwen2.5-
    # Omni reference and never matched any module in this graph,
    # accidentally lumping the entire 24-layer audio tower into the
    # root FSDP unit. FSDP2 looks up class names at wrap time and
    # silently changes wrap granularity if a listed name doesn't
    # match any module, causing large multi-card logp drift
    # multimodal forward where ``OpenPanguVLDecoderLayer`` match misses
    # caused FSDP2 to wrap at a coarser boundary spanning the visual-feature
    # merge in ``OpenPanguOmniModel.forward``.
    _no_split_modules = [
        "OpenPanguV2DecoderLayer",
        "OpenPanguVLVisionBlock",
        "ConformerEncoderLayerBlock",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module) -> None:
        if isinstance(module, OpenPanguVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            _init_copy(module.inv_freq, inv_freq)

    def get_no_input_cast_modules_in_mixed_precision(self):
        """Skip the FSDP2 ``cast_forward_inputs`` step at each per-layer
        boundary inside the vision and audio towers while keeping the
        ``param_dtype=bf16`` cast intact.

        Why this exact granularity (and not the coarser
        ``get_ignore_modules_in_mixed_precision`` hook):

        - The vision tower has 26 ``OpenPanguVLVisionBlock``\\s. Each takes a
          ``position_embeddings = (cos, sin)`` tuple computed *once* at the
          top of the tower as fp32 (the rotary embedding deliberately runs in
          fp32 to match HF + single-card behavior). With FSDP2's default
          ``MixedPrecisionPolicy(cast_forward_inputs=True)``, these fp32
          tensors are downcast to bf16 at every block boundary — 26
          round-trips of fp32→bf16→fp32 that produce ~1 bf16 ULP of drift
          in ``visual_output`` vs single-card.

        - The audio tower (``HuanyuAudioEncoder``) follows the same pattern:
          24 ``ConformerEncoderLayerBlock``\\s receive a fp32
          ``rotary_pos_emb = (cos, sin)`` tuple computed once via
          ``select_cos_sin`` (the rotary buffer ``emb`` is built in fp32 to
          match HF). Per-layer FSDP wrap would similarly downcast it 24
          times. (Today the audio tower has zero per-layer FSDP wrap because
          ``_no_split_modules`` used to list ``Qwen2AudioEncoderLayer`` —
          a copy-paste from Qwen2.5-Omni that never matched anything, so
          the whole tower folded into root FSDP. Fixing
          ``_no_split_modules`` to list the real class would re-introduce
          the cos/sin downcast issue *unless* the new class is also added
          here; declaring both at once lets us fix the dead-code bug while
          preserving bit-exact audio_tower_out vs single-card.)

        - Dropping the *entire* mp policy (the ``get_ignore_modules_in_mixed_precision``
          path) also disables the ``param_dtype=bf16`` cast, so the per-block
          RMSNorm / Linear weights stay at their storage dtype (fp32 under
          VeOmni's "fp32 storage + bf16 compute" design). That mismatches
          single-card — which builds with ``torch_dtype=bf16`` and ends up
          with bf16 storage — producing a *different* 1 bf16 ULP drift at
          ``b00_post_norm1`` and onward (see bisection in
          ``README.md`` §"Multi-card visual drift").

        - The "no input cast" path is the surgical fix: keep ``param_dtype=bf16``
          so compute matches single-card (bf16 weights × bf16 hidden_states),
          but skip ``cast_forward_inputs`` so the fp32 ``(cos, sin)`` tuple
          survives unchanged across the per-layer boundaries. Result: 8-card
          ``visual_output`` and ``audio_tower_out`` are both bit-exact with
          single-card on Pangu Omni 30B-A2B vision and audio parity
          samples.

        ``ConformerEncoderLayerBlock`` is imported lazily — importing it at
        module scope would create a circular dependency
        (``modeling_huanyu_audio_encoder`` → its own utilities → eventually
        back into this module's namespace via the registry).
        """
        from .modeling_huanyu_audio_encoder import ConformerEncoderLayerBlock

        return (OpenPanguVLVisionBlock, ConformerEncoderLayerBlock)


class OpenPanguVisionTransformerPretrainedModel(OpenPanguPreTrainedModel):
    """Pangu Omni v2 vision tower (ViT-style with window attention + merger).

    Verbatim port of `modeling_vl.py:400-609`. Key components
    assembled here:

    - **`patch_embed`** — `OpenPanguVLPatchEmbed` (Conv3d). Handles
      images (T=1) and videos (T>1) uniformly.

    - **`rotary_pos_emb`** — 1D inv_freq buffer over `head_dim // 2`;
      the per-token 2D rotary is computed in `rot_pos_emb()` from
      `grid_thw`.

    - **`blocks`** — `config.depth` × `OpenPanguVLVisionBlock`.

    - **`merger`** — either a single `OpenPanguVLPatchMerger` with
      `use_gatedmerger=True` (Pangu 30B-A2B path), or an `nn.ModuleList`
      of plain mergers when intermediate features from multiple
      `select_layer` indices are summed (legacy multi-stage fusion).

    Forward pipeline:

    1. `patch_embed` flattens patches into hidden vectors.
    2. `rot_pos_emb(grid_thw)` builds 2D positional cos/sin grid.
    3. `get_window_index` reshuffles tokens into local-attention windows.
    4. For each block: full-attention vs. window-attention chosen by
       `layer_num in self.fullatt_block_indexes`. The attention mask
       is generated by `_prepare_attention_mask` to be FA2-compatible
       (None for FA2, 4D `[1, 1, S, S]` -inf-masked block-diagonal for
       eager).
    5. `merger` projects to `out_hidden_size` (= text-backbone hidden).
    6. Reverse window permutation via `torch.argsort(window_index)` to
       restore canonical token order before returning.

    Output shape: `(num_image_tokens // spatial_merge_unit, out_hidden_size)`,
    where `num_image_tokens = sum(t*h*w)` over all images in batch and
    `spatial_merge_unit = spatial_merge_size**2`.
    """

    config_class = OpenPanguOmniVisionConfig
    _no_split_modules = ["OpenPanguVLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = OpenPanguVLPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = OpenPanguVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([OpenPanguVLVisionBlock(config) for _ in range(config.depth)])

        # `select_layer` is a list of negative indices into self.blocks
        # — e.g. [-1, -3] picks the last block and the 3rd-from-last.
        # When `use_gatedmerger=True`, only the last block's output is
        # passed to the (single) merger; intermediates aren't used.
        self.select_layer = getattr(config, "mm_unit_vision_select_layer", [-1, -3])
        self.select_index = [config.depth + i for i in self.select_layer]
        self.select_index = self.select_index[::-1]
        self.select_layer = [-1 * (i + 1) for i in range(len(self.select_index))]

        self.use_gatedmerger = config.use_gatedmerger
        if config.use_gatedmerger:
            self.merger = OpenPanguVLPatchMerger(
                dim=config.out_hidden_size,
                context_dim=config.hidden_size,
                spatial_merge_size=config.spatial_merge_size,
                use_gatedmerger=True,
            )
        else:
            self.merger = nn.ModuleList(
                [
                    OpenPanguVLPatchMerger(
                        dim=config.out_hidden_size,
                        context_dim=config.hidden_size,
                        spatial_merge_size=config.spatial_merge_size,
                    )
                    for _ in range(len(self.select_layer))
                ]
            )
        self.gradient_checkpointing = False
        self.take_indices = self.select_index

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Build 2D RoPE freqs for the (t, h, w) grids in this batch.

        For each (t, h, w):
        - Generate hpos_ids (h-axis positions) and wpos_ids (w-axis
          positions), each reshaped into (h/sm) × sm × (w/sm) × sm
          blocks then permuted to (h/sm) × (w/sm) × sm × sm — this
          interleaves patches within each spatial_merge_size block so
          merger receives spatially-coherent groups.
        - Stack `[hpos_ids, wpos_ids]` (last dim = 2) and repeat t
          times for video frames.

        After concatenating across images, look up `rotary_pos_emb_full`
        at those positions and flatten to a `(seq_len, head_dim)`
        tensor of RoPE thetas.
        """
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Compute window-attention index reshuffling and cu_window_seqlens.

        Within each image, tokens are grouped into `vit_merger_window_size`-sized
        windows. Tokens are re-indexed so that all tokens within a window
        are contiguous in the final sequence, with padding tokens (index
        -100 in the intermediate buffer) dropped. `cu_window_seqlens`
        records cumulative window boundaries for cu_seqlens-style
        attention.

        `vit_merger_window_size = window_size // spatial_merge_size //
        patch_size`. Reference Pangu 30B-A2B has window_size=112,
        spatial_merge_size=2, patch_size=14 → vit_merger_window_size=4.
        """
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> Optional[torch.Tensor]:
        """Build a 4D attention mask for non-FA2 backends.

        FA2 uses `cu_seqlens / max_seqlen` directly and accepts None.
        For eager/SDPA we materialize a `[1, 1, S, S]` mask filled with
        `-inf` everywhere except inside the block-diagonal segments
        defined by `cu_seqlens`. NOTE: This is an *approximation* of
        FA2 varlen attention — within each cu_seqlens block we use
        bidirectional attention.
        """
        if self.config._attn_implementation == "flash_attention_2":
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the full vision tower over `(seq, hidden_size)` patches.

        Args:
            hidden_states: `(num_patches, patch_size**2 * in_channels * temporal_patch_size)`
                — flattened patch tensor, NOT image pixels. The image
                processor is responsible for producing this layout.
            grid_thw: `(num_images, 3)` — (t, h, w) for each image.

        Returns:
            `(num_image_tokens // spatial_merge_unit, out_hidden_size)`.
        """
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        intermediates = []
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens_now)
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )
            if layer_num in self.take_indices:
                intermediates.append(hidden_states)

        if self.use_gatedmerger:
            hidden_states = self.merger(hidden_states)
        else:
            image_embeddings_list = []
            for idx, sl in enumerate(self.select_layer):
                image_embeddings_list.append(self.merger[idx](intermediates[sl]))
            hidden_states = sum(image_embeddings_list)

        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def dummy_forward(self):
        """Minimal forward over a 1×2×2 patch grid (1 image, 4
        patches) so FSDP all-gathers on this tower fire on ranks
        that have no images in the current micro-batch.

        Mirrors ``Qwen2_5OmniVisionEncoder.dummy_forward`` (see
        modeling_qwen2_5_omni.py line 646). Patch tensor shape is
        ``(num_patches, patch_size**2 * in_channels * temporal_patch_size)``
        — for Pangu's default ``patch_size=14, in_channels=3,
        temporal_patch_size=2`` that's ``(4, 1176)``. The
        ``grid_thw=(1, 2, 2)`` tells the tower this is a single
        image with a 2×2 patch grid, which after
        ``spatial_merge_size=2`` -> 1 token. Just enough to keep
        the vision blocks happy without large compute cost.
        """
        if getattr(self, "_dummy_data", None) is None:
            patch_size = self.config.patch_size
            in_channels = self.config.in_channels
            temporal_patch_size = self.config.temporal_patch_size
            patch_dim = patch_size * patch_size * in_channels * temporal_patch_size
            num_patches = 4
            param = next(self.parameters())
            hidden_states = torch.zeros((num_patches, patch_dim), dtype=param.dtype, device=param.device)
            grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=param.device)
            self._dummy_data = {"hidden_states": hidden_states, "grid_thw": grid_thw}
        return self(**self._dummy_data)


# ===========================================================================
# Multimodal RoPE (3D mrope, the text-side companion to the
# vision tower's 2D vision RoPE).
# ===========================================================================
#
# `OpenPanguVLRotaryEmbedding` is the multimodal-aware sibling of the
# stock 1D rotary embedding from `transformers.modeling_rope_utils`.
# Where a normal RoPE produces a single `(cos, sin)` pair indexed by
# linear position ids, mrope produces per-axis `(cos, sin)` pairs and
# *blends* them along the head-dim according to `mrope_section`, so
# that within a vision token's representation the temporal / height /
# width axes each occupy their own slice of the head-dim and rotate
# independently from one another and from text-only tokens.
#
# Two code paths exist, picked by the config flag `mrope_interleaved`:
#
# 1. **Default mrope** (mrope_interleaved=False): split cos/sin into
#    `len(mrope_section) * 2 == 6` chunks, take chunks `0,1,2,0,1,2`
#    (axis indices) and concatenate. Net effect: head-dim is divided
#    into 6 contiguous segments where consecutive triples come from
#    the same axis-rotation.
#
# 2. **Interleaved mrope** (mrope_interleaved=True, **the path Pangu
#    30B-A2B actually uses** per its config.json):
#    `mrope_dim = get_mrope_interleaved_id_list(t, h, w, force_last=True)`
#    builds a length-(t+h+w) round-robin permutation that interleaves
#    the axes more finely than the contiguous-triples form, then the
#    same `cat([m[idx[i]] for i, m in enumerate(emb.split(...))])`
#    blend is applied with `mrope_section_3d = [1] * len(mrope_dim)`.
#
# The companion `apply_multimodal_rotary_pos_emb(q, k, cos, sin,
# mrope_section)` re-runs the **non-interleaved** blend right before
# applying the rotation to q/k, which is redundant for the interleaved
# path (the cos/sin returned by `OpenPanguVLRotaryEmbedding.forward`
# already have the interleaved blend baked in) but necessary for the
# default path. Mirror the reference 1:1 — both paths exist in
# production checkpoints (different model variants may use different
# rope_scaling).
#
# `dynamic_rope_update` is a thin decorator from transformers that
# refreshes `inv_freq` when seq_len exceeds the cached `max_seq_len`.
# We import it lazily via try/except so older transformers versions
# without it fall back to a no-op.


try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
except ImportError:  # pragma: no cover — defensive fallback for very old transformers
    ROPE_INIT_FUNCTIONS = {}

    def dynamic_rope_update(fn):
        return fn


class OpenPanguVLRotaryEmbedding(nn.Module):
    """Multimodal 3D rotary embedding.

    Builds inv_freq per the rope_type (default / yarn / linear / etc.
    via ``ROPE_INIT_FUNCTIONS``), then in ``forward`` expands to 3
    rotation axes (temporal, height, width) and blends them according
    to either:

    - ``mrope_section`` (contiguous-triples blend, called "normal
      mrope"), or
    - ``mrope_dim = get_mrope_interleaved_id_list(t, h, w, force_last)``
      (round-robin per-position blend, called "interleaved mrope").

    Returns ``(cos, sin)`` shaped ``(bsz, seq_len, head_dim)``.
    """

    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            self.mrope_interleaved = config.rope_scaling.get("mrope_interleaved", False)
        else:
            self.rope_type = "default"
            self.mrope_interleaved = False
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        # Dispatch the inv_freq initializer: "default" goes through our
        # local fallback (kept verbatim from the reference) so that
        # transformers versions without that key in ROPE_INIT_FUNCTIONS
        # still work; other rope_types delegate to transformers.
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default" and ROPE_INIT_FUNCTIONS:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        # Pre-compute the interleaved permutation index list. Only used
        # when mrope_interleaved=True; cheap to compute upfront and
        # avoids re-running the placement algorithm on every forward.
        self.mrope_section = None
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.mrope_section = config.rope_scaling.get("mrope_section", None)
        if self.mrope_interleaved:
            if not self.mrope_section:
                raise AssertionError("when you use interleave mrope, mrope_section cannot be None.")
            if len(self.mrope_section) == 2:
                h_num, w_num = self.mrope_section[0], self.mrope_section[1]
                mrope_dim = self.get_mrope_interleaved_id_list(h_num, w_num, 0)
            elif len(self.mrope_section) == 3:
                t_num, h_num, w_num = (
                    self.mrope_section[0],
                    self.mrope_section[1],
                    self.mrope_section[2],
                )
                mrope_dim = self.get_mrope_interleaved_id_list(t_num, h_num, w_num, force_last=True)
            else:
                raise AssertionError("Cannot support the length of mrope section is not 2 or 3.")
            mrope_dim = mrope_dim * 2  # tile for the cat-(freqs, freqs) layout
            self.mrope_dim = mrope_dim

    @staticmethod
    def compute_default_rope_parameters(
        config=None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        **rope_kwargs,
    ) -> tuple["torch.Tensor", float]:
        """Default rope inv_freq computation (mirror of
        ``transformers.modeling_rope_utils._compute_default_rope_parameters``).

        Honours ``partial_rotary_factor`` so the head-dim slice that
        receives rotation can be shorter than the full head-dim — Pangu
        text backbone uses ``partial_rotary_factor=0.25`` for its
        partial-RoPE attention (see ``_pangu_common/pangu_partial_rope.py``)
        but the VL text backbone is full RoPE (factor=1.0); we honour
        whatever the config says.
        """
        if config is not None and len(rope_kwargs) > 0:
            raise ValueError(
                "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
                f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
            )
        if len(rope_kwargs) > 0:
            base = rope_kwargs["base"]
            dim = rope_kwargs["dim"]
        elif config is not None:
            base = config.rope_theta
            partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids: shape (3, bs, seq_len) — three axes (T, H, W)
        # inv_freq_expanded: (3, bs, head_dim//2, 1)
        # position_ids_expanded: (3, bs, 1, seq_len)
        # → freqs after matmul + transpose: (3, bs, seq_len, head_dim//2)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)

            if self.mrope_interleaved:
                mrope_section_3d = [1] * len(self.mrope_dim)
                mrope_dim = self.mrope_dim
                emb = torch.cat(
                    [m[mrope_dim[i]] for i, m in enumerate(emb.split(mrope_section_3d, dim=-1))],
                    dim=-1,
                )

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

            if not self.mrope_interleaved and self.mrope_section:
                mrope_section = self.mrope_section * 2
                cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
                sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def get_mrope_interleaved_id_list(a: int, b: int, c: int, force_last: bool = False) -> list[int]:
        """Round-robin axis permutation used by the interleaved mrope.

        Given counts ``(a, b, c)`` for the three modalities (T, H, W,
        or H, W, 0 in 2-axis fallback), produce a length-(a+b+c) list of
        axis indices that:

        1. Never repeats the same axis in adjacent positions (when
           possible — relaxed if only one axis has remaining slots).
        2. Among non-repeat candidates, picks the axis whose placement
           ratio ``placed[k] / counts[k]`` is currently smallest (with
           the axis index ``k`` as a tiebreaker for determinism).
        3. If ``force_last=True``, reserves one slot for axis 0 at the
           tail (Pangu 30B-A2B uses this with ``mrope_section=[6,5,5]``
           so the very last position belongs to axis 0 = temporal).
        """
        if force_last:
            a -= 1

        counts = {0: a, 1: b, 2: c}
        placed = dict.fromkeys(counts, 0)
        rem = counts.copy()
        seq: list[int] = []
        last = None

        total = a + b + c
        for _ in range(total):
            cands = [k for k in rem if rem[k] > 0 and k != last]
            if not cands:
                cands = [k for k in rem if rem[k] > 0]
            try:
                best = min(cands, key=lambda k: (placed[k] / counts[k], k))
            except KeyError:
                best = 0
            seq.append(best)
            placed[best] += 1
            rem[best] -= 1
            last = best

        if force_last:
            seq.append(0)

        return seq


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D mrope (default / non-interleaved) to q and k.

    Used by ``OpenPanguVLAttention.forward``. NB: this function applies
    the **same blend** that ``OpenPanguVLRotaryEmbedding.forward``
    already applies to ``cos``/``sin`` *only when
    mrope_interleaved=False*. For the interleaved path, the blend here
    is mathematically redundant because the cos/sin returned by the
    embedding already have the interleaved permutation baked in.
    Reference reapplies the blend unconditionally; we preserve that
    exact behaviour rather than gating on ``mrope_interleaved`` (would
    diverge from oracle).
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ===========================================================================
# ProjectionSingle + VLTextModel + output dataclasses.
# ===========================================================================
#
# `ProjectionSingle` is the **vision token → text-embedding** linear
# projection used by `OpenPanguVLModel.visual.vision_projection`. The
# reference uses a SiLU activation before the linear (atypical — most
# VLM projections are linear-then-activation). Mirror 1:1.
#
# `OpenPanguVLTextModel` is a trivial subclass of `OpenPanguV2Model`
# that swaps in `OpenPanguVLRotaryEmbedding` (mrope) in place of the
# default `OpenPanguV2RotaryEmbedding` (1D RoPE). All other text
# backbone behaviour (MHC, MoE, partial RoPE inside attn, etc.) is
# inherited unchanged.
#
# Why does swapping `rotary_emb` give us multimodal RoPE without
# touching attention?  Because `OpenPanguV2Model.forward` computes
# `position_embeddings = self.rotary_emb(...)` once at the top of
# forward and passes the resulting `(cos, sin)` tuple down to every
# `OpenPanguV2DecoderLayer.forward` → `OpenPanguV2Attention.forward` →
# `apply_rotary_pos_emb_qk_partial(...)`. The mrope **blend is baked
# into `cos`/`sin` inside `OpenPanguVLRotaryEmbedding.forward`** (see
# Multimodal RoPE docstring) — by the time `apply_rotary_pos_emb_qk_partial`
# sees them they're already (bsz, seq, head_dim) shaped just like a
# regular 1D RoPE, so the standard partial-RoPE apply works as-is.
# That's the multimodal merge trick: 3D mrope masquerading as 1D RoPE
# via in-forward blending.
#
# `OpenPanguVLModelOutputWithPast` / `OpenPanguVLCausalLMOutputWithPast`
# are dataclass output containers that add a `rope_deltas` field to
# track the offset between the visible sequence length and the
# multimodal RoPE position id range. Used by `prepare_inputs_for_
# generation` to keep KV-cache positions in sync across decode steps.
#
# NB: We deliberately **do not port** `OpenPanguVLAttention` and
# `OpenPanguVLDecoderLayer` — parity testing established
# they're dead code in the 30B-A2B inference path. The reference's
# class hierarchy keeps them around but `OpenPanguVLTextModel` (which
# inherits `OpenPanguV2Model`) uses `OpenPanguV2DecoderLayer` with
# `OpenPanguV2Attention` everywhere, so the VL{Attention,DecoderLayer}
# code path is never instantiated. Re-adding them later is trivial if
# a Pangu variant emerges that does use them.


from dataclasses import dataclass  # noqa: E402 — kept beside the dataclass it scopes for readability


try:
    from transformers.modeling_outputs import ModelOutput
except ImportError:  # pragma: no cover
    ModelOutput = object  # type: ignore[misc,assignment]


@dataclass
class OpenPanguVLModelOutputWithPast(ModelOutput):
    """Output of `OpenPanguVLModel.forward`.

    Adds `rope_deltas` on top of the standard last-hidden + cache
    fields. `rope_deltas[i]` is the difference between the multimodal
    rope index range and `len(input_ids[i])` for sample i, used to
    advance KV-cache positions during incremental decoding."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.LongTensor] = None


@dataclass
class OpenPanguVLCausalLMOutputWithPast(ModelOutput):
    """Output of `OpenPanguVL.forward` (the ForCausalLM wrapper).

    Adds `loss`, `logits` on top of `OpenPanguVLModelOutputWithPast`
    fields. `rope_deltas` is propagated from the inner model output
    for use by `prepare_inputs_for_generation`."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.LongTensor] = None


class ProjectionSingle(nn.Module):
    """Vision-tokens → text-embedding linear projection.

    The reference does `act(x) → fc1(x)` (SiLU then Linear). This is
    *atypical* — most VLM projections do `linear → activation` or `linear`
    alone. Pangu's convention is preserved here verbatim because any
    pre-trained `vision_projection` weight is calibrated for this exact
    pre-activation layout.

    `i_hidden_size` is the vision tower's output dim (post-merger),
    `t_hidden_size` is the text backbone's hidden dim. When MHC is
    enabled, `t_hidden_size = text_hidden_size * mhc_num_stream` so
    the projection produces an MHC-shape residual directly."""

    def __init__(self, i_hidden_size: int, t_hidden_size: int) -> None:
        super().__init__()
        self.act = F.silu
        self.fc1 = nn.Linear(i_hidden_size, t_hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.act(hidden_states)
        return self.fc1(x)


# Forward-declare to break the circular import: OpenPanguVLTextModel
# must inherit OpenPanguV2Model, which lives in modeling_text;
# we lazy-import inside the class definition's parent resolution so a
# top-level import doesn't cycle when that module imports VL pieces.
def _get_openpangu_v2_model_cls():
    from veomni.models.transformers.pangu_omni_v2.modeling_text import (
        OpenPanguV2Model,
    )

    return OpenPanguV2Model


_OpenPanguV2Model = _get_openpangu_v2_model_cls()


class OpenPanguVLTextModel(_OpenPanguV2Model):  # type: ignore[misc,valid-type]
    """Text backbone for VL inference — same as `OpenPanguV2Model` but
    with `rotary_emb` swapped for `OpenPanguVLRotaryEmbedding` (mrope).

    The mrope blend is applied **inside the rotary forward** (see Week
    3.4.a docstring), so the resulting cos/sin look like a regular
    1D RoPE pair from the perspective of the inherited V2 decoder
    layers and attention modules. Net effect: same code path as
    text-only V2 inference, but with multimodal-aware position
    encoding."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.rotary_emb = OpenPanguVLRotaryEmbedding(config=config)


# ===========================================================================
# 3D position id computation for VL multimodal sequences.
# ===========================================================================
#
# `compute_vl_rope_index` (the standalone form) and the
# `OpenPanguVLModel.get_rope_index` method (a thin shim that calls it)
# implement Pangu's variant of Qwen2-VL's 3D mrope position id
# scheme:
#
# - For text-only tokens: temporal/height/width position ids are all
#   the same linear index (the standard 1D rope ids).
# - For image-token blocks: temporal=0..0 (single frame), height /
#   width are 2D grid indices that move per token; the entire image
#   block advances the rope counter by `max(llm_H, llm_W)` so the
#   following text picks up after the image's spatial extent.
# - For video-token blocks: temporal advances per frame, height /
#   width are 2D grid indices per frame; surrounding
#   `vision_start_token_id` / `vision_end_token_id` placeholders also
#   participate in the rope counter (a small detail that's easy to
#   miss).
#
# The `mrope_position_delta` returned alongside `position_ids` is the
# diff between the max position-id seen and the visible input length;
# generation code uses it to advance the rope counter correctly across
# decode steps.
#
# `_get_llm_pos_ids_for_vision` is a helper that builds the (t, h, w)
# per-token position ids for a single image's tokens. Kept as a free
# function for direct testability.
#
# **Numerical caveat**: every dispatch (text / image / video) and
# every off-by-one in the per-image `_get_llm_pos_ids_for_vision`
# call is potentially silently broken in ways that only show up at
# evaluation time (the rope ids are integer-valued so type errors
# won't surface in unit tests). We pin **many** parametrized parity
# cases below to guard against drift.


def _get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list,
    grid_hs: torch.Tensor,
    grid_ws: torch.Tensor,
) -> torch.Tensor:
    """Build per-token (t, h, w) position ids for one image's tokens.

    Used by `compute_vl_rope_index` when it hits an image-token block.
    Returns a tensor of shape `(3, num_image_tokens)` where the three
    rows are temporal / height / width position ids, all offset by
    `start_idx` so the image block begins at the position immediately
    after the previous token's rope counter."""
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = (
        torch.arange(llm_grid_h).to(llm_grid_h.device).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
    )
    w_index = (
        torch.arange(llm_grid_w).to(llm_grid_h.device).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
    )
    t_index_tensor = (
        torch.Tensor(t_index).to(llm_grid_h.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).long().flatten()
    )
    _llm_pos_ids = torch.stack([t_index_tensor, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def compute_vl_rope_index(
    *,
    input_ids: Optional[torch.LongTensor],
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    spatial_merge_size: int,
    tokens_per_second: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone form of `OpenPanguVLModel.get_rope_index`.

    See the 3D mrope docstring above for the scheme. The
    `OpenPanguVLModel.get_rope_index` method is a thin wrapper that
    reads the config fields and forwards to this helper — keeping the
    bulk of the logic at module scope makes parity testing trivial
    (no need to instantiate a full VLModel + load weights just to
    compute a few hundred position ids)."""
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        video_idx = 0
        image_idx = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            input_tokens = input_ids_i.tolist()
            src_item = input_tokens
            new_src_item: list = []
            llm_pos_ids_list: list = []

            idx = 0
            while idx < len(src_item):
                new_src_item_len = len(new_src_item)
                start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if src_item[idx] not in [video_token_id, image_token_id]:
                    new_src_item.append(src_item[idx])
                    llm_pos_ids = torch.tensor([start_idx], dtype=torch.long).expand(3, -1)
                    llm_pos_ids_list.append(llm_pos_ids.to(position_ids.device))
                elif src_item[idx] == image_token_id:
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * tokens_per_second).long()
                    llm_pos_ids = _get_llm_pos_ids_for_vision(
                        start_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    llm_pos_ids_list.append(llm_pos_ids.to(position_ids.device))
                    vision_seqlen = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    new_src_item.extend([image_token_id] * vision_seqlen)
                    image_idx += 1
                else:
                    # video_token_id branch — per-frame position ids
                    T = video_grid_thw[video_idx][0].item()
                    H = video_grid_thw[video_idx][1].item()
                    W = video_grid_thw[video_idx][2].item()
                    llm_H = H // spatial_merge_size
                    llm_W = W // spatial_merge_size
                    tokens_per_frame = llm_H * llm_W
                    start_pos = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                    current_pos = start_pos
                    final_frame_time = T - 1
                    for t in range(T):
                        if t != 0:
                            new_src_item.append(vision_start_token_id)
                            bot_pos = torch.full((3, 1), current_pos, dtype=torch.long)
                            llm_pos_ids_list.append(bot_pos.to(position_ids.device))
                            current_pos += 1
                        grid_h = torch.arange(llm_H).view(-1, 1).expand(-1, llm_W).flatten()
                        grid_w = torch.arange(llm_W).view(1, -1).expand(llm_H, -1).flatten()
                        frame_pos = torch.stack(
                            [
                                torch.full_like(grid_h, 0, dtype=torch.long),
                                grid_h,
                                grid_w,
                            ]
                        )
                        frame_pos_with_offset = frame_pos + current_pos
                        new_src_item.extend([video_token_id] * tokens_per_frame)
                        llm_pos_ids_list.append(frame_pos_with_offset.to(position_ids.device))
                        current_pos += max(llm_H, llm_W)
                        if t != final_frame_time:
                            new_src_item.append(vision_end_token_id)
                            eot_pos = torch.full((3, 1), current_pos, dtype=torch.long)
                            llm_pos_ids_list.append(eot_pos.to(position_ids.device))
                            current_pos += 1
                    video_idx += 1
                idx += len(new_src_item) - new_src_item_len
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_delta = llm_positions.max() + 1 - len(total_input_ids[i])
            mrope_position_deltas.append(mrope_position_delta)
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


# ===========================================================================
# OpenPanguVLModel (vision+text merge) + OpenPanguVL (top-level CausalLM).
# ===========================================================================
#
# This is the **merge layer** of the multimodal pipeline: it owns the
# vision tower (`OpenPanguVisionTransformerPretrainedModel`), the text
# backbone (`OpenPanguVLTextModel`), and the `vision_projection` that
# maps vision tower output → text hidden size. Forward path:
#
#   input_ids (with IMAGE/VIDEO placeholders already expanded to vision_seqlen)
#         │
#         ├─ get_input_embeddings(input_ids) ────────────────► inputs_embeds
#         │                                                          │
#         ├─ pixel_values? ─► visual(pixel_values, image_grid_thw)   │
#         │                  ─► vision_projection                    │
#         │                  ─► masked_scatter(IMAGE_TOKEN_ID) ──────┤
#         │                                                          │
#         ├─ pixel_values_videos? ─► same path via video_grid_thw ───┤
#         │                                                          ▼
#         └─ get_rope_index(input_ids, ...) ──► 3D position_ids
#                                                          │
#                            language_model(position_ids, inputs_embeds)
#                                                          │
#                                                       lm_head ──► logits
#
# What we skip vs the reference:
#
# * `_parse_preprocess_params` — calls `AutoProcessor.from_pretrained`
#   to pull rescale/normalize/image_mean/image_std from the image
#   processor config. These are inference-time conveniences only; they
#   are not used inside `forward()` so dropping them keeps the adapter
#   importable without a tokenizer/processor shipped on disk. If a
#   downstream caller (e.g. an inference server) needs those values,
#   they should be read from `processor.image_processor.*` directly,
#   not from the model.
#
# * `OpenPanguVL.prepare_inputs_for_generation` — overrides the base
#   class to suppress pixel_values on non-prefill steps. We skip this
#   for single-forward parity scope. It can be
#   ported later when we wire HF generate() through Forge.
#
# What's different vs reference (intentional):
#
# * The reference's `OpenPanguVLModel.get_rope_index` and
#   `_get_llm_pos_ids_for_vision` are method-form duplications of the
#   logic we already extracted into module-level `compute_vl_rope_index`
#   and `_get_llm_pos_ids_for_vision` free functions. The
#   method versions here are thin wrappers around those — single source
#   of truth, no logic duplication.
# ---------------------------------------------------------------------------


def _get_open_pangu_vl_model_base():
    """Lazy import of `OpenPanguPreTrainedModel` defined above in this file.

    Used to declare both `OpenPanguVLModel` and `OpenPanguVL` as
    subclasses of `OpenPanguPreTrainedModel` while keeping the module
    layout flat (avoids a circular-import dance through the package
    `__init__`)."""
    return OpenPanguPreTrainedModel


class OpenPanguVLModel(_get_open_pangu_vl_model_base()):
    """Vision + text merge model, used by both inference (`OpenPanguVL`)
    and downstream wrappers (`OpenPanguOmni`).

    Owns three sub-modules:

    - `self.visual` — `OpenPanguVisionTransformerPretrainedModel` (the
      vision transformer. Built from `config.vision_config`.
    - `self.language_model` — `OpenPanguVLTextModel`. Built
      from `config.text_config`.
    - `self.visual.vision_projection` — `ProjectionSingle` attached to
      the vision tower after construction. Output dim depends on MHC:
      `text_config.hidden_size * mhc_num_stream` when `use_mhc`, else
      just `text_config.hidden_size`.

    Per the reference:
    - `base_model_prefix = ""` — no prefix when loading state_dict.
    - `_no_split_modules` covers FSDP / TP shard boundaries. We use
      the names of classes that actually appear in this module's
      runtime graph — ``OpenPanguV2DecoderLayer`` (text), the
      vision block, and the audio block — rather than the reference's
      ``OpenPanguVLDecoderLayer`` (dead code, never actually
      instantiated by ``OpenPanguVLTextModel``). FSDP2 silently changes
      wrap granularity if a listed name doesn't match any module.
    """

    base_model_prefix = ""
    config_class = OpenPanguOmniConfig
    _no_split_modules = [
        "OpenPanguV2DecoderLayer",
        "OpenPanguVLVisionBlock",
        "ConformerEncoderLayerBlock",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.visual = OpenPanguVisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = OpenPanguVLTextModel(config.text_config)

        # Cached rope deltas — populated on the first prefill, reused on
        # subsequent decode steps when `position_ids` is left as None.
        self.rope_deltas = None
        self.use_mhc = self.config.use_mhc
        self.mhc_num_stream = self.config.mhc_num_stream
        if self.use_mhc:
            self.visual.vision_projection = ProjectionSingle(
                config.vision_config.out_hidden_size,
                config.text_config.hidden_size * self.mhc_num_stream,
            )
        else:
            self.visual.vision_projection = ProjectionSingle(
                config.vision_config.out_hidden_size,
                config.text_config.hidden_size,
            )
        self.post_init()
        # NOTE: `_parse_preprocess_params` from the reference is deliberately
        # skipped — see module-level docstring for rationale.

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,  # noqa: ARG002 — reserved
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Thin wrapper over the module-level :func:`compute_vl_rope_index`
        free function — reads the four token-id and the
        ``spatial_merge_size`` / ``tokens_per_second`` scalars off the
        config, then delegates the actual position-id math.

        ``second_per_grid_ts`` is accepted in the signature for upstream
        BC but is unused: the reference video branch never threads it
        into the per-frame offset (the offset is always
        ``current_pos + max(llm_H, llm_W)``, independent of
        ``second_per_grid_ts``). Kept here so callers don't have to
        change their kwarg shape.
        """
        return compute_vl_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            tokens_per_second=getattr(self.config, "tokens_per_second", 1.0),
        )

    def _get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index,
        grid_hs: torch.Tensor,
        grid_ws: torch.Tensor,
    ) -> torch.Tensor:
        """Method wrapper over the module-level helper — preserves the
        reference's method-form API in case downstream callers reach in
        through ``self._get_llm_pos_ids_for_vision``."""
        return _get_llm_pos_ids_for_vision(start_idx, vision_idx, spatial_merge_size, t_index, grid_hs, grid_ws)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """Encode videos through the vision tower + projection, then
        split per-video so the downstream `masked_scatter` works."""
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        video_embeds = self.visual.vision_projection(video_embeds)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """Encode images through the vision tower + projection, then
        split per-image so the downstream `masked_scatter` works."""
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = self.visual.vision_projection(image_embeds)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,  # noqa: ARG002 — BC, unused
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if self.use_mhc:
                # MHC: duplicate inputs along hidden dim — mhc_num_stream copies.
                inputs_embeds = self.get_input_embeddings()(input_ids).repeat(1, 1, self.mhc_num_stream)

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                    raise ValueError(
                        "Image features and image tokens do not match: "
                        f"tokens: {n_image_tokens}, features {n_image_features}"
                    )
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0)
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                    raise ValueError(
                        "Video features and video tokens do not match: "
                        f"tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas_new = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas_new
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = OpenPanguVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class OpenPanguVL(OpenPanguPreTrainedModel, GenerationMixin):
    """Top-level multimodal CausalLM. Wraps `OpenPanguVLModel` with an
    `lm_head` Linear that projects `text_config.hidden_size` →
    `text_config.vocab_size`.

    The `_checkpoint_conversion_mapping` declares how on-disk param
    names map into this module's hierarchy: any param starting with
    ``visual.`` lands under ``model.visual.``, and any ``model.*`` that
    isn't ``language_model``/``visual``/``lm_head`` is re-routed to
    ``model.language_model.``. This matches the reference's checkpoint
    layout exactly so a state-dict shipped from the Pangu reference
    drops in without renaming.
    """

    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual|lm_head))": "model.language_model",
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = OpenPanguVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def get_parallel_plan(self):
        """Return the VeOmni expert-parallel plan for OpenPanguVL.

        Same backbone layout as ``OpenPanguOmni`` (text MoE under
        ``model.language_model.*``, dense vision tower under
        ``model.visual.*``), so we delegate to the shared builder with
        the multimodal FQN prefix. See ``parallel_plan.py`` and
        ``OpenPanguOmni.get_parallel_plan`` for the full FQN-verification
        rationale.
        """
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(use_gate_up_proj=True, prefix="model.language_model")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # See `modeling_text.py:OpenPanguV2ForCausalLM.forward`
            # for the rationale — VeOmni's `LOSS_MAPPING["ForCausalLM"]`
            # wrapper returns a 4-tuple, not a tensor. Mirrors qwen3_moe.
            loss, logits, log_probs, entropy = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size
            )

        return OpenPanguVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


# Re-export the NPU/GPU dispatch flag so downstream callers
# multimodal merge + tests) can branch on it without re-probing.
__all__ = [
    "PanguEmbeddedRMSNorm",
    "OpenPanguRMSNorm",
    "OpenPanguVLMLP",
    "OpenPanguVisionPatchEmbed",
    "OpenPanguVLPatchEmbed",
    "OpenPanguVisionRotaryEmbedding",
    "OpenPanguVLPatchMerger",
    "rotate_half",
    "apply_rotary_pos_emb_vision",
    "repeat_kv",
    "eager_attention_forward",
    "OpenPanguVLVisionAttention",
    "OpenPanguVLVisionBlock",
    "OpenPanguPreTrainedModel",
    "OpenPanguVisionTransformerPretrainedModel",
    "OpenPanguVLRotaryEmbedding",
    "apply_multimodal_rotary_pos_emb",
    "ProjectionSingle",
    "OpenPanguVLTextModel",
    "OpenPanguVLModelOutputWithPast",
    "OpenPanguVLCausalLMOutputWithPast",
    "compute_vl_rope_index",
    "_get_llm_pos_ids_for_vision",
    "OpenPanguVLModel",
    "OpenPanguVL",
    "ROPE_INIT_FUNCTIONS",
    "dynamic_rope_update",
    "ALL_ATTENTION_FUNCTIONS",
    "NPU_ATTN_INFR",
]
