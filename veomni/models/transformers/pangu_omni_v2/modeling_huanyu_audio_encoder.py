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

"""Pangu Omni v2 audio tower (HuanyuAudioEncoder).

This module is the VeOmni-side counterpart of the audio sub-section
of the Pangu reference `modeling_pangu_omni.py` (lines 66-620). The
reference packs the audio encoder + the multimodal `OpenPanGuOmni`
wrapper into one file; we split them so the audio port is self-
contained and consumed only by the multimodal merge layer.

## Scope

Ports of:

- ``HuanyuRotaryEmbedding`` — 1D RoPE used by audio attention. Pre-
  builds a cached ``emb`` buffer of shape ``[max_len, 1, 1, head_dim]``
  whose cosine/sine views are selected per-frame via
  ``HuanyuAudioEncoder.select_cos_sin``.
- ``VGGBlock`` — 2D Conv + (optional) LayerNorm + ReLU + MaxPool stack
  used as the front-end downsampler over (channels=1, time, mel) mel
  spectrograms.
- ``HuanyuAudioEncoderPreTrainedModel`` — PreTrainedModel base for the
  audio encoder with the same ``_init_weights`` hook as the reference
  (re-derives ``emb`` for ``HuanyuRotaryEmbedding`` modules at load).
- ``HuanyuAudioEncoder`` — full encoder pipeline: VGG downsample →
  ``linear_before_attn`` → N×ConformerEncoderLayerBlock → post-LN →
  optional ``AvgPool1d`` (when ``audio_merge_size == 2``).
- ``eager_attention_forward`` — audio-specific copy from Whisper. NB:
  this variant ``.transpose(1, 2)`` the attn output before returning,
  which differs from the vision variant's ``transpose(1, 2).contiguous()
  .reshape(...)`` pattern; the difference is preserved 1:1.
- ``HuanyuAttention`` — full-mask self-attention with windowed mask
  (``eager_atten_mask`` with ``window=64``), rotary on q/k, NPU fast
  path via ``torch_npu.npu_fusion_attention`` (with ``sparse_mode=4``,
  ``pre_tockens=64``) when ``NPU_ATTN_INFR`` is True.
- ``ConformerEncoderLayerBlock`` — Conformer block in the order
  macaron1 → attention → conv-module (pointwise GLU → depthwise k=5 →
  pointwise) → macaron2 → final-LN.

## NPU-specific code

The reference unconditionally imports ``torch_npu`` at module top and
uses ``torch_npu.npu_rotary_mul`` / ``torch_npu.npu_fusion_attention``
when ``NPU_ATTN_INFR`` is True. We guard the import behind try/except
so the module loads cleanly on GPU; the NPU fast paths are preserved
when the runtime detects an Ascend device.

## Init helper

The reference imports ``init`` from a vendor fork (``transformers``
sub-module ``initialization``) which exposes ``init.copy_(target,
source)``. We inline a tiny equivalent so we don't take that import
dependency.

## Why not co-locate with the vision tower?

The vision and audio towers share no code (different RoPE
parameterisation, different Q/K/V shape conventions, different
fused-attn calls) and are consumed by independent merge sites in the
multimodal model. Keeping them in separate files matches the
reference's split between ``modeling_openpangu_vl.py`` and
``modeling_pangu_omni.py`` (audio half).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 — kept for parity / future hooks


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

from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel


def _init_copy(target: torch.Tensor, source: torch.Tensor) -> None:
    """Inline replacement for the vendor-fork ``init.copy_`` helper used
    by the reference's ``_init_weights``. Plain ``no_grad`` copy."""
    with torch.no_grad():
        target.copy_(source)


# ---------------------------------------------------------------------------
# Rotary embedding (audio variant — single dim, cached emb buffer)
# ---------------------------------------------------------------------------


class HuanyuRotaryEmbedding(nn.Module):
    """1D rotary embedding used by ``HuanyuAttention``.

    Stores the full ``emb = cat([freqs, freqs], dim=-1)`` tensor of
    shape ``[max_len, 1, 1, head_dim]`` as a non-persistent buffer so
    callers can slice ``emb[:T].cos()`` / ``emb[:T].sin()`` at runtime
    via :meth:`HuanyuAudioEncoder.select_cos_sin`.
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, max_position_embeddings: int = 1024, base: int = 10000) -> None:
        super().__init__()
        self.dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        emb = self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32
        )
        self.register_buffer("emb", emb, persistent=False)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
        return emb.to(dtype=dtype)

    def ensure_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Grow the RoPE cache when packed audio exceeds the reference default."""
        if seq_len <= self.emb.shape[0] and self.emb.device == device and self.emb.dtype == dtype:
            return

        new_len = max(seq_len, self.emb.shape[0] * 2)
        self.emb = self._set_cos_sin_cache(seq_len=new_len, device=device, dtype=dtype)
        self.max_position_embeddings = max(self.max_position_embeddings, new_len)


# ---------------------------------------------------------------------------
# VGG conv block (mel-spectrogram front-end)
# ---------------------------------------------------------------------------


class VGGBlock(nn.Module):
    """Front-end 2D conv block: ``num_conv_layers × (Conv2d [+ LayerNorm]
    + ReLU) + MaxPool2d``.

    Used by :class:`HuanyuAudioEncoder` to downsample the
    ``(channels=1, time, mel)`` mel-spectrogram by ``pooling_kernel_size``
    in both time and frequency axes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int,
        pooling_kernel_size: int,
        num_conv_layers: int,
        layer_norm_dim: int,
        conv_stride: int = 1,
        padding: Optional[int] = None,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = (conv_kernel_size, conv_kernel_size)
        self.pooling_kernel_size = (pooling_kernel_size, pooling_kernel_size)
        self.num_conv_layers = num_conv_layers
        self.padding = tuple(e // 2 for e in self.conv_kernel_size) if padding is None else (padding, padding)
        self.conv_stride = (conv_stride, conv_stride)
        self.layers = nn.ModuleList()
        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(
                in_channels if layer == 0 else out_channels,
                out_channels,
                self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.padding,
            )
            self.layers.append(conv_op)
            if layer_norm:
                self.layers.append(nn.LayerNorm(out_channels))
            self.layers.append(nn.ReLU())

        if self.pooling_kernel_size is not None:
            pool_op = nn.MaxPool2d(kernel_size=self.pooling_kernel_size, ceil_mode=True)
            self.layers.append(pool_op)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = x.permute(2, 1, 0)
                x = layer(x)
                x = x.permute(2, 1, 0)
            else:
                x = layer(x)
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)) and mask is not None:
                x = x * mask
                x = x.to(dtype)
        return x


# ---------------------------------------------------------------------------
# PreTrainedModel base + audio encoder
# ---------------------------------------------------------------------------


class HuanyuAudioEncoderPreTrainedModel(PreTrainedModel):
    """``PreTrainedModel`` shell for the audio encoder.

    The only meaningful hook is ``_init_weights`` which re-derives the
    cached ``emb`` for any :class:`HuanyuRotaryEmbedding` modules; this
    keeps load-from-pretrained idempotent with respect to the cached
    cos/sin table.
    """

    base_model_prefix = "model"
    supports_gradient_checkpointing = False  # reference says True; mirror reference choice
    _no_split_modules = ["HuanyuAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:  # type: ignore[override]
        if isinstance(module, HuanyuRotaryEmbedding):
            inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            t = torch.arange(module.max_position_embeddings, dtype=torch.int64).type_as(inv_freq)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
            _init_copy(module.emb, emb)


class HuanyuAudioEncoder(HuanyuAudioEncoderPreTrainedModel):
    """Pangu / Huanyu audio encoder.

    Pipeline: ``VGG conv stack → linear_before_attn → N ×
    ConformerEncoderLayerBlock → linear_after_attn → LayerNorm →
    optional AvgPool1d (audio_merge_size==2)``.

    ``feature_lens`` carries the per-sample mel-spectrogram frame
    counts (variable length per batch). The block runs each sample's
    VGG independently (avoids ragged 4-D conv) and concatenates them
    back into a single (T_total, d_model) tensor with ``cu_seqlens``
    bookkeeping for downstream attention masking.
    """

    main_input_name = "input_features"
    _no_split_modules = ["Qwen2AudioEncoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.embed_dim = config.d_model
        self.head_dim = self.embed_dim // config.encoder_attention_heads
        n_mels = config.num_mel_bins
        vggblock_config = eval(config.vggblock_enc_config)  # noqa: S307 — config is trusted, mirrors reference
        # Reference hard-codes which Conformer layers carry depthwise-conv mask
        # contribution. Kept verbatim — these are dataset-tuned hyperparameters.
        self.layer_dw_conv_mask = [
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        pooling_size = 1
        if vggblock_config is not None:
            for vgg_config in vggblock_config:
                (
                    out_channels,
                    conv_kernel_size,
                    pooling_kernel_size,
                    num_conv_layers,
                    layer_norm,
                    layer_norm_dim,
                ) = vgg_config
                self.conv_layers.append(
                    VGGBlock(
                        in_channels,
                        out_channels,
                        conv_kernel_size,
                        pooling_kernel_size,
                        num_conv_layers,
                        layer_norm_dim=layer_norm_dim,
                        layer_norm=layer_norm,
                    )
                )
                pooling_size *= pooling_kernel_size
                in_channels = out_channels

        self.linear_before_attn = nn.Linear(
            in_channels * (n_mels // pooling_size),
            self.config.d_model,
        )
        self.layers = nn.ModuleList()
        for _ in range(config.encoder_layers):
            self.layers.append(ConformerEncoderLayerBlock(config))
        self.rotary_emb = HuanyuRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=1024,
            base=10000,
        )
        self.linear_after_attn = nn.Linear(config.d_model, config.d_model)
        self.ln_post = nn.LayerNorm(config.d_model)
        if getattr(config, "audio_merge_size", 1) == 2:
            self.avg_pooler = nn.AvgPool1d(2, stride=2)
        else:
            self.avg_pooler = None

    def _freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def run_VGG_layers(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            x = conv_layer(x, None)
        return x

    def calc_ids(self, seq_len: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.arange(int(i), dtype=seq_len.dtype, device=seq_len.device) for i in seq_len],
            dim=0,
        )

    def select_cos_sin(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NB: ``self.rotary_emb.emb`` has shape [max_len, 1, 1, head_dim].
        # The reference indexes by ``len(position_ids)`` (not the values
        # of position_ids) — mirror that exactly.
        self.rotary_emb.ensure_cache(
            seq_len=len(position_ids),
            device=position_ids.device,
            dtype=self.rotary_emb.emb.dtype,
        )
        return (
            self.rotary_emb.emb[: len(position_ids)].cos().squeeze(-2),
            self.rotary_emb.emb[: len(position_ids)].sin().squeeze(-2),
        )

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[BaseModelOutput, torch.Tensor]:
        cu_inSeqlen = torch.cat(
            (
                torch.zeros(1, device=feature_lens.device, dtype=torch.int32),
                feature_lens.cumsum(0),
            )
        ).to(torch.int32)
        hidden_states = input_features.transpose(-1, -2).unsqueeze(0).contiguous()
        convOut_seq_lens = torch.zeros_like(feature_lens)
        x_list = []
        for i in range(feature_lens.shape[0]):
            x_list.append(self.run_VGG_layers(hidden_states[:, cu_inSeqlen[i] : cu_inSeqlen[i + 1], :]))
            convOut_seq_lens[i] = x_list[i].shape[1]
        cu_Seqlen_ConvOut = torch.cat(
            (
                torch.zeros(1, device=feature_lens.device, dtype=torch.int32),
                convOut_seq_lens.cumsum(0),
            )
        ).to(torch.int32)
        hidden_states = torch.cat(x_list, dim=1).transpose(-2, -3).contiguous().view(cu_Seqlen_ConvOut[-1], -1)

        hidden_states = self.linear_before_attn(hidden_states)
        position_ids = self.calc_ids(convOut_seq_lens)
        rotary_pos_emb = self.select_cos_sin(position_ids)

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                cu_Seqlen_ConvOut,
                self.layer_dw_conv_mask[idx],
                rotary_pos_emb,
                idx,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.linear_after_attn(hidden_states)
        hidden_states = self.ln_post(hidden_states)

        if self.avg_pooler is not None:
            hidden_states = hidden_states.transpose(-1, -2)
            pooled_list = []
            for i in range(feature_lens.shape[0]):
                x = hidden_states[:, cu_Seqlen_ConvOut[i] : cu_Seqlen_ConvOut[i + 1]]
                dtype = x.dtype
                pooled = self.avg_pooler(x.float())
                pooled_list.append(pooled.to(dtype))
            hidden_states = torch.cat(pooled_list, dim=-1).transpose(-1, -2)
            convOut_seq_lens = convOut_seq_lens // 2

        return BaseModelOutput(last_hidden_state=hidden_states, attentions=None), convOut_seq_lens

    def dummy_forward(self):
        """Minimal forward that triggers FSDP all-gathers on this
        tower without producing a useful output.

        Mirrors the pattern used by Qwen2.5-Omni
        (``Qwen2_5OmniAudioEncoder.dummy_forward``,
        modeling_qwen2_5_omni.py line 431). The Pangu / Huanyu
        audio_tower expects ``input_features`` of shape
        ``(n_mels, T_total)`` plus ``feature_lens`` of shape ``(B,)``
        — see ``HuanyuAudioEncoder.forward`` above. The smallest
        ``T_total`` that survives the VGG conv stack is determined
        by ``vggblock_enc_config``:

          - Two 3×3 convs (stride 1) + 2×2 MaxPool per block
          - 2 blocks → 4× temporal downsampling
          - Conv kernel 3 also requires ``T >= 3`` at every stage,
            so input ``T >= 3 * 4 = 12``.

        We pick ``T=32`` to have headroom for any conformer
        positional-encoding bound (the rotary table is registered
        up to 1024 positions, ``T=32`` lands well within that). The
        tensor is cached on ``self._dummy_data`` so we don't pay
        the allocation cost on every step.

        The returned ``last_hidden_state`` is multiplied by 0 by
        the caller (``OpenPanguOmniModel.forward``) before being
        added to ``inputs_embeds``, so it never contributes to the
        loss numerically — its only purpose is to keep all FSDP
        ranks in lock-step on the audio_tower's collective ops.
        """
        if getattr(self, "_dummy_data", None) is None:
            n_mels = self.config.num_mel_bins
            t = 32
            param = next(self.parameters())
            input_features = torch.zeros((n_mels, t), dtype=param.dtype, device=param.device)
            feature_lens = torch.tensor([t], dtype=torch.int64, device=param.device)
            self._dummy_data = {
                "input_features": input_features,
                "feature_lens": feature_lens,
            }
        return self(**self._dummy_data)


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Audio-variant eager attention.

    NB: distinct from the vision variant in two ways:
    1. No ``repeat_kv`` (audio uses MHA, vision uses GQA-friendly path).
    2. ``transpose(1, 2).contiguous()`` at end *without* a final
       ``reshape`` — caller is expected to handle the final
       ``reshape(seq_length, -1)`` (see :class:`HuanyuAttention.forward`).
    """
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class HuanyuAttention(nn.Module):
    """Full-mask self-attention with 1D rotary on q/k.

    Forward shape contract: ``hidden_states`` is ``(T_total, d_model)``
    (NB: 2-D, the batch dim is folded into the variable-length
    ``T_total``). ``cu_seqlens`` carries the per-sample boundaries so
    the windowed attention mask can be built block-diagonally.

    On NPU the fast path uses ``torch_npu.npu_fusion_attention`` with
    ``sparse_mode=4`` (causal-only-within-window, length-aware) and
    ``pre_tockens=64``. On GPU we fall back to ``eager_attention_forward``
    with the explicit windowed-block-diagonal mask built by
    :meth:`eager_atten_mask`.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def apply_rotary_pos_emb(
        self,
        t: torch.Tensor,
        freqs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(freqs, tuple):
            cos_freqs, sin_freqs = freqs
            if len(t) < len(cos_freqs):
                cos_freqs = cos_freqs[: len(t)]
                sin_freqs = sin_freqs[: len(t)]
            _mscale = 1.0
            rot_dim = cos_freqs.shape[-1]
            t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
            cos_ = (cos_freqs * _mscale).to(t.dtype).to(t.device)
            sin_ = (sin_freqs * _mscale).to(t.dtype).to(t.device)
        else:
            if len(t) < len(freqs):
                freqs = freqs[: len(t)]
            _mscale = 1.0
            rot_dim = freqs.shape[-1]
            t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
            cos_ = (torch.cos(freqs) * _mscale).to(t.dtype).to(t.device)
            sin_ = (torch.sin(freqs) * _mscale).to(t.dtype).to(t.device)

        if NPU_ATTN_INFR:
            t = torch_npu.npu_rotary_mul(  # type: ignore[union-attr]
                t.unsqueeze(0), cos_.unsqueeze(0), sin_.unsqueeze(0)
            ).squeeze(0)
        else:
            x1, x2 = torch.chunk(t, 2, -1)
            x_new = torch.cat((-x2, x1), dim=-1)
            t = cos_ * t + sin_ * x_new

        return torch.cat((t, t_pass), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        q = query_states.reshape(seq_length, self.num_heads, -1)
        k = key_states.reshape(seq_length, self.num_heads, -1)
        v = value_states.reshape(seq_length, self.num_heads, -1)
        if rotary_pos_emb is not None:
            q = self.apply_rotary_pos_emb(q, rotary_pos_emb)
            k = self.apply_rotary_pos_emb(k, rotary_pos_emb)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if not self.training and NPU_ATTN_INFR:
            if isinstance(cu_seqlens, torch.Tensor):
                cu_seqlens = cu_seqlens.tolist()
            attn_output = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
                q,
                k,
                v,
                self.num_heads,
                "TND",
                pse=None,
                padding_mask=None,
                atten_mask=~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device="npu")),
                scale=self.scaling,
                pre_tockens=64,
                next_tockens=0,
                keep_prob=1.0,
                inner_precise=0,
                sparse_mode=4,
                actual_seq_qlen=cu_seqlens,
                actual_seq_kvlen=cu_seqlens,
            )[0]
            attn_output = attn_output.reshape(seq_length, -1).contiguous()
        else:
            if q.ndim == 3:
                q = q.unsqueeze(0).transpose(1, 2)
            if k.ndim == 3:
                k = k.unsqueeze(0).transpose(1, 2)
            if v.ndim == 3:
                v = v.unsqueeze(0).transpose(1, 2)
            batch, _, seq_len, _ = k.shape
            attention_mask = self.eager_atten_mask(cu_seqlens, seq_len, window=64).repeat(batch, 1, 1, 1)
            attention_mask = (
                torch.where(attention_mask, 0.0, torch.finfo(hidden_states.dtype).min)
                .to(hidden_states.device)
                .to(hidden_states.dtype)
            )
            attn_output, _ = attention_interface(
                self,
                q,
                k,
                v,
                scaling=self.scaling,
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.dropout,
            )
            attn_output = attn_output.reshape(seq_length, -1)
        output = self.out_proj(attn_output)
        return output

    def eager_atten_mask(self, cu_seqlens: torch.Tensor, seq_len: int, window: int = 64) -> torch.Tensor:
        """Build a windowed block-diagonal boolean attention mask.

        - Within each ``cu_seqlens[i-1]:cu_seqlens[i]`` block, only
          positions ``j - window <= i <= j`` (i.e. ``window`` past +
          self, no future) are unmasked.
        - Outside the block diagonal everything is masked.

        Caller converts the boolean mask to additive
        ``finfo(dtype).min`` for ``softmax``.
        """
        pos = torch.arange(seq_len)
        rel_pos = pos[None, :] - pos[:, None]
        mask = (rel_pos >= -window) & (rel_pos <= 0)
        mask_window = torch.zeros([1, 1, seq_len, seq_len])
        for i in range(1, len(cu_seqlens)):
            mask_window[
                ..., int(cu_seqlens[i - 1]) : int(cu_seqlens[i]), int(cu_seqlens[i - 1]) : int(cu_seqlens[i])
            ] = 1
        mask = mask & mask_window.bool()
        return mask


# ---------------------------------------------------------------------------
# Conformer encoder block
# ---------------------------------------------------------------------------


class ConformerEncoderLayerBlock(nn.Module):
    """Conformer block: ``macaron-FFN → attention → conv-module →
    macaron-FFN → final-LN``.

    Layout follows Pangu's `OpenPanguOmniAudioConfig` defaults
    (``d_model=1024``, ``encoder_attention_heads=16``,
    ``encoder_ffn_dim=4096``, ``encoder_conv1d_kernel_size=5``). Inside
    the conv-module:

    1. ``pointwise_conv_before`` doubles channels for GLU.
    2. ``glu_fn`` halves them back (channel-axis split).
    3. ``depthwise_conv`` (k=5) with hard-coded ``padding=(2,)`` and a
       per-sample ``dw_mask`` controlling whether to use the
       past-aligned slice ``x[..., 1:-1]`` (mask=1) or the right-aligned
       ``x[..., :-2]`` (mask=0).
    4. LayerNorm → SiLU → ``pointwise_conv_after``.

    The depthwise conv path runs each sample independently
    (``cu_seqlens`` boundaries) because the depthwise sliding window
    must not bleed across sample boundaries.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.submodules = None

        self.hidden_size = config.d_model
        self.ffn_hidden_size = config.encoder_ffn_dim
        self.head_num = config.encoder_attention_heads
        self.normalize_before = True
        self.macaron1_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size, eps=config.layernorm_epsilon)
        self.macaron1_fc1 = nn.Linear(self.hidden_size, self.ffn_hidden_size)
        self.macaron1_fc2 = nn.Linear(self.ffn_hidden_size, self.hidden_size)
        self.activation_fn = nn.SiLU()
        self.self_attn_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size, eps=config.layernorm_epsilon)
        self.config.attention_dropout = 0.15
        self.self_attn = HuanyuAttention(self.config)

        self._dropout = 0.15
        self.self_attn_bda = None
        self.head_dim = self.hidden_size // self.head_num

        self.scaling = self.head_dim**-0.5
        self.dropout = 0.15
        kernel_size = config.encoder_conv1d_kernel_size

        k = kernel_size
        self.conv_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size, eps=config.layernorm_epsilon)
        self.pointwise_conv_before = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size * 2,
            kernel_size=1,
            padding=0,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=k,
            groups=1,
            padding=k // 2,
        )
        self.cnn_module_norm = nn.LayerNorm(normalized_shape=self.hidden_size, eps=config.layernorm_epsilon)
        self.pointwise_conv_after = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=1,
            padding=0,
        )
        self.macaron2_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size, eps=config.layernorm_epsilon)
        self.macaron2_fc1 = nn.Linear(self.hidden_size, self.ffn_hidden_size)
        self.macaron2_fc2 = nn.Linear(self.ffn_hidden_size, self.hidden_size)
        self.final_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size, eps=config.layernorm_epsilon)
        self.glu_fn = nn.GLU(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        dw_mask: float,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        idx: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        residual = x
        x = self.macaron1_layer_norm(x)
        x = self.macaron1_fc1(x)
        x = self.activation_fn(x)
        x = self.macaron1_fc2(x)
        x = residual + x * 0.5

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            hidden_states=x,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        x = x + residual
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        x = self.conv_layer_norm(x)
        x = x.transpose(-1, -2)
        x = self.pointwise_conv_before(x)
        x = self.glu_fn(x)
        # NB: depthwise padding gets *mutated* per call in the reference;
        # mirror exactly (this matters because the depthwise kernel is 5
        # and we slice to k=3 effective post-trim).
        self.depthwise_conv.padding = (2,)
        x_list = []
        for i in range(cu_seqlens.shape[0] - 1):
            x_i = self.depthwise_conv(x[:, cu_seqlens[i] : cu_seqlens[i + 1]])
            x1_i = x_i[..., :-2]
            x2_i = x_i[..., 1:-1]
            x_i = x2_i * dw_mask + x1_i * (1 - dw_mask)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=-1)
        x = x.transpose(-1, -2)
        x = self.cnn_module_norm(x)
        x = x.transpose(-1, -2)
        x = self.activation_fn(x)
        x = self.pointwise_conv_after(x)
        x = x.transpose(-1, -2)
        x = residual + x

        residual = x
        x = self.macaron2_layer_norm(x)
        x = self.macaron2_fc1(x)
        x = self.activation_fn(x)
        x = self.macaron2_fc2(x)
        x = residual + x * 0.5

        x = self.final_layer_norm(x)
        return (x,)


__all__ = [
    "HuanyuRotaryEmbedding",
    "VGGBlock",
    "HuanyuAudioEncoderPreTrainedModel",
    "HuanyuAudioEncoder",
    "eager_attention_forward",
    "HuanyuAttention",
    "ConformerEncoderLayerBlock",
    "NPU_ATTN_INFR",
]
