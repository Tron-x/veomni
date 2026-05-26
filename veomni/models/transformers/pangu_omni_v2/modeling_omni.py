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

"""Pangu OpenPanguOmni — vision + text + audio multimodal merge layer.

This module is the VeOmni-side counterpart of the Pangu reference
`modeling_pangu_omni.py`. Production 30B-A2B's
`architectures=["OpenPanguUltraOmniForConditionalGeneration"]` lands here
(via the dispatcher in `modeling_text.py`); image-only inputs
degrade to the underlying `OpenPanguVLModel.forward` path and audio
inputs enter through the new `input_features → masked_scatter` branch.

## Scope

Port of:

- `OpenPanguOmniModel` — extends `OpenPanguVLModel` with `audio_tower`
  (`HuanyuAudioEncoder`) + `audio_tower.proj`
  (`Linear(d_model → hidden_size * mhc_num_stream)`). Overrides
  `forward` to add the audio branch (lines 1150-1158 of HF reference)
  and `get_rope_index` to handle the audio token branch (lines 837-845
  of HF reference).
- `OpenPanguOmni` — top-level CausalLM wrapping `OpenPanguOmniModel` +
  `lm_head`. Mirrors `OpenPanguVL` shape; only difference is the
  forward forwards extra `input_features` / `feature_attention_mask` /
  `audio_feature_lengths` kwargs to the inner model and the
  `_checkpoint_conversion_mapping` adds the `^audio_tower` rule so
  the production checkpoint's flat `audio_tower.*` keys load into our
  nested `model.audio_tower.*` parameters.
- `compute_omni_rope_index` — extension of `compute_vl_rope_index`
  that adds the audio_token_id branch. Position-id
  arithmetic for non-`use_audio_in_video` audio: each audio block of
  `place_num` tokens gets a 3D-broadcast arange offset by
  `start_idx`. The `use_audio_in_video=True` branch (audio interleaved
  with video frames, lines 906-987 of HF reference) is not implemented because production
  30B-A2B has `vision_config.use_audio_in_video=False`, so the
  interleaved path is dead code for our current target. A `NotImplementedError`
  guard makes the gap explicit; future Omni Plus targets that need
  audio-in-video can extend this branch following the HF reference
  byte-for-byte.

## What we skip vs the reference

- `_parse_preprocess_params` — inference-time convenience for pulling
  `image_mean` / `image_std` off the processor; not on the forward path.
- `Qwen2_5OmniThinkerForConditionalGeneration` inheritance — the HF
  reference's `OpenPanGuOmni` inherits `Qwen2_5OmniThinkerForConditionalGeneration`
  for the `generate()` plumbing. We inherit `GenerationMixin` directly,
  same as `OpenPanguVL`. This keeps the class hierarchy flat: HF
  reference's `model.model` = text backbone, `model.visual` = visual,
  `model.audio_tower` = audio, all at top level. Our:
  `OpenPanguOmni → self.model (OpenPanguOmniModel) → self.visual /
  self.language_model / self.audio_tower`. The
  `_checkpoint_conversion_mapping` reconciles the difference at load
  time.
- `Qwen2_5OmniThinkerCausalLMOutputWithPast` — we reuse
  `OpenPanguVLCausalLMOutputWithPast` (same fields, different name).

## NPU-specific code

`HuanyuAudioEncoder` already guards NPU imports in its own module;
this module imports the encoder class and instantiates it
without further NPU coupling. The audio path is verified to be
identical on GPU (`attn_implementation="eager"`) and NPU
(`attn_implementation="sdpa"`) — both kernel choices flow through
`HuanyuAttention.forward` and are covered by the parity/debug tools.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.utils import is_torchdynamo_compiling

from .._pangu_common.pangu_moe import (
    veomni_moe_experts_forward,  # noqa: F401 — see comment below
)
from .modeling_huanyu_audio_encoder import HuanyuAudioEncoder
from .modeling_vl import (
    OpenPanguPreTrainedModel,
    OpenPanguVLCausalLMOutputWithPast,
    OpenPanguVLModel,
    OpenPanguVLModelOutputWithPast,
    _get_llm_pos_ids_for_vision,
)


# The MoE-experts OpSlot lives in ``_pangu_common.pangu_moe`` (alongside
# the ``OpenPanguV2Experts`` class that uses it), but VeOmni's
# ``_bind_veomni_ops`` (in ``models/auto.py``) discovers slots via
# ``dir(modeling_module)`` on the **top-level model class's** module.
#
# When the loaded class is ``OpenPanguOmni`` (= visual + audio_tower +
# language_model), that module is *this file* — not
# ``modeling_text.py`` (which already re-exports the slot for
# the text-only ``OpenPanguV2ForCausalLM`` path). Without the re-export
# below, setting ``moe_implementation: fused_npu`` would silently leave
# the slot unbound and the eager forward in ``OpenPanguV2Experts`` would
# run, which is fatal under ``ep_size > 1`` because it indexes global
# expert IDs into the locally-EP-sharded ``gate_up_proj``
# (``IndexError: index 49 is out of bounds for dimension 0 with size 48``,
# observed in multi-card multimodal smoke testing).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_audio_output_length(
    input_lengths,
    audio_merge_size: int = 1,
) -> int:
    """Per-sample audio token count after VGG + (optional) AvgPool1d.

    Verbatim port of `processor_openpangu_omni.get_audio_output_length`
    from the HF reference: ``output = floor((floor((L+1)/2)+1)/2)`` and,
    when ``audio_merge_size == 2``, ``output = (output - 2) // 2 + 1``.

    The formula reflects the two stride-2 VGG conv layers
    (`(64, 3, 2, 2, True, 40), (128, 3, 2, 2, True, 20)`) plus the final
    `AvgPool1d(2, stride=2)`. Both reference and adapter rely on this
    for `get_rope_index` and for processor-side token expansion.
    """
    input_lengths = (input_lengths + 1) // 2
    output_lengths = (input_lengths + 1) // 2
    if audio_merge_size == 2:
        output_lengths = (output_lengths - 2) // 2 + 1
    return output_lengths


def compute_omni_rope_index(
    *,
    input_ids: Optional[torch.LongTensor],
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,  # noqa: ARG001 — kept in signature for BC; unused (audio path off-by-default)
    vision_end_token_id: int,
    spatial_merge_size: int,
    tokens_per_second: float = 1.0,
    audio_seqlens: Optional[torch.LongTensor] = None,
    audio_merge_size: int = 1,
    use_audio_in_video: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D mrope position-id computation with audio support.

    Extension of `compute_vl_rope_index`. Layout:

    - Text token: 1D positions (3D-broadcast).
    - Image token: 3D (T, H, W) positions per patch — same as VL path.
    - Video token (no audio): same as VL path.
    - Audio token: ``place_num = get_audio_output_length(audio_seqlen,
      audio_merge_size)`` positions, 3D-broadcast arange offset by
      ``start_idx``.

    The `use_audio_in_video=True` branch (audio interleaved with video
    frames) is not implemented. Production 30B-A2B has
    `vision_config.use_audio_in_video=False`, so this path is dead code
    for our current target. Calling with `use_audio_in_video=True`
    raises `NotImplementedError` to make the gap explicit.

    Args:
        input_ids: ``(B, S)`` token ids.
        image_grid_thw: ``(N_images, 3)`` per-image (T, H, W).
        video_grid_thw: ``(N_videos, 3)`` per-video (T, H, W).
        attention_mask: ``(B, S)`` 0/1 mask. Padding positions are
            assigned position-id 1 (matches the reference's
            ``masked_fill_(attention_mask == 0, 1)``).
        image_token_id / video_token_id / audio_token_id /
        vision_start_token_id / vision_end_token_id: scalar token ids
        from `OpenPanguOmniConfig`.
        spatial_merge_size: vision tower spatial merge factor.
        tokens_per_second: temporal granularity for video frames
            (reference uses ``vision_config.tokens_per_second``, default 1).
        audio_seqlens: ``(N_audios,)`` per-audio mel-spectrogram frame
            counts (post-feature-extraction). Required when any audio
            tokens are present.
        audio_merge_size: ``audio_config.audio_merge_size`` (1 or 2).
        use_audio_in_video: must be `False`.

    Returns:
        position_ids: ``(3, B, S)`` LongTensor.
        mrope_position_deltas: ``(B, 1)`` LongTensor — diff between max
        position-id used and seq length, used to chain prefill ↔
        decode rope steps.
    """
    if use_audio_in_video:
        raise NotImplementedError(
            "compute_omni_rope_index: `use_audio_in_video=True` is not "
            "supported (production 30B-A2B has "
            "`vision_config.use_audio_in_video=False`). The interleaved "
            "audio-in-video position-id branch (HF reference lines "
            "906-987) is deferred."
        )

    has_multimodal = image_grid_thw is not None or video_grid_thw is not None or audio_seqlens is not None

    if input_ids is not None and has_multimodal:
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
        audio_idx = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        mrope_position_deltas: list = []
        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            input_tokens = input_ids_i.tolist()
            src_item = input_tokens
            new_src_item: list = []
            llm_pos_ids_list: list = []

            idx = 0
            while idx < len(src_item):
                new_src_item_len = len(new_src_item)
                # Always unwrap to Python int. The HF reference does this for the
                # video branch (L871 ``start_pos = ... .item() + 1``) but not for
                # the main ``start_idx`` — that's a latent bug that surfaces when
                # ``position_ids.device`` is non-CPU and the audio branch hits
                # ``torch.arange(N) + start_idx``: arange is on CPU, start_idx
                # would be a 0-d NPU tensor (inherited from the previous text/image
                # branch's ``.to(position_ids.device)`` call on the prior entry),
                # producing "Expected all tensors to be on the same device".
                # Using ``.item()`` keeps every per-branch tensor build on CPU,
                # matching the reference's effective semantics (the final
                # ``llm_positions.to(position_ids.device)`` transfers once at
                # L312).
                start_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
                if src_item[idx] == audio_token_id:
                    if audio_seqlens is None:
                        raise ValueError(
                            "compute_omni_rope_index: encountered audio_token_id "
                            f"in input_ids[{i}] at position {idx} but "
                            "`audio_seqlens` is None. Pass per-audio mel-frame "
                            "counts so we can expand the placeholder."
                        )
                    audio_seqlen = audio_seqlens[audio_idx]
                    place_num = get_audio_output_length(audio_seqlen, audio_merge_size)
                    place_num_int = int(place_num.item() if isinstance(place_num, torch.Tensor) else place_num)
                    new_src_item.extend([audio_token_id] * place_num_int)
                    llm_pos_ids = torch.arange(place_num_int).expand(3, -1) + start_idx
                    llm_pos_ids_list.append(llm_pos_ids.to(position_ids.device))
                    audio_idx += 1
                elif src_item[idx] not in [video_token_id, image_token_id]:
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
                    # video_token_id branch (no audio-in-video)
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
        mrope_position_deltas_tensor = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas_tensor
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


# ---------------------------------------------------------------------------
# OpenPanguOmniModel (extends OpenPanguVLModel with audio_tower + audio
# branch in forward)
# ---------------------------------------------------------------------------


class OpenPanguOmniModel(OpenPanguVLModel):
    """Vision + text + audio merge model.

    Extends `OpenPanguVLModel` with:

    - `self.audio_tower` — `HuanyuAudioEncoder` instantiated from
      `config.audio_config` with `attn_implementation` matching the
      top-level config (`config._attn_implementation`, default
      ``"eager"``). The encoder's ``main_input_name`` is
      ``"input_features"`` so HF generation utilities know how to feed it.
    - `self.audio_tower.proj` — `Linear(d_model → hidden_size *
      mhc_num_stream)` matching the reference's hidden replacement
      (`OpenPanGuOmni.__init__` lines 636-640). Output dim mirrors
      the MHC multiplier on `inputs_embeds`, so post-projection audio
      features drop cleanly into the masked-scatter target shape.

    Overrides:

    - `get_audio_features` — verbatim of HF reference lines 695-724.
    - `get_rope_index` — delegates to `compute_omni_rope_index` (passes
      audio token id + audio config). Falls back to VL behavior when
      `audio_seqlens=None` (and thus the audio branch in
      `compute_omni_rope_index` is unreachable).
    - `forward` — adds audio scatter branch BEFORE image scatter
      (matches HF reference line 1150 ordering). Audio scatter is
      no-op when `input_features=None`.

    Token-id ordering in the source: the reference scans tokens
    left-to-right and the audio branch only fires on actual
    audio_token_id occurrences, so even though the audio scatter
    happens before image scatter in the *forward* code, the resulting
    `inputs_embeds` is correct because each token id maps to exactly
    one modality's embedding.
    """

    def __init__(self, config):
        super().__init__(config)
        # NOTE: avoid calling `self.post_init()` again here — `super().__init__`
        # (i.e. `OpenPanguVLModel.__init__`) already invoked it for the
        # visual + language_model + vision_projection submodules. The
        # `HuanyuAudioEncoder._from_config` call below runs the encoder's
        # own init pipeline (PreTrainedModel + _init_weights), and the
        # subsequent `audio_tower.proj` is a plain `nn.Linear` whose
        # default kaiming init is fine. Re-running `post_init` would
        # silently re-randomize visual + language_model weights with the
        # current RNG state, breaking any test that relies on a fixed
        # seed across construct → load → forward.
        attn_impl = getattr(config, "_attn_implementation", "eager")
        self.audio_tower = HuanyuAudioEncoder._from_config(config.audio_config, attn_implementation=attn_impl)
        # Override the audio_tower's own projection (which targets
        # `audio_config.d_model → ?`) with our hidden-size-aware one.
        # `mhc_num_stream` defaults to 1 when use_mhc=False so this
        # always lands on the same dim as `inputs_embeds`.
        proj_out = config.text_config.hidden_size * (self.mhc_num_stream if self.use_mhc else 1)
        self.audio_tower.proj = nn.Linear(config.audio_config.d_model, proj_out, bias=True)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Encode audios into continuous embeddings.

        Mirrors HF reference `OpenPanGuOmni.get_audio_features` lines
        695-724. The two-mode handling:

        - When ``feature_attention_mask`` is given, derive
          ``audio_feature_lengths`` from it and re-pack
          ``input_features`` from ``(B, n_mels, T_padded)`` to
          ``(n_mels, T_unpadded_total)`` by selecting unmasked frames.
        - When ``feature_attention_mask`` is None, pass-through; the
          caller is responsible for providing flat
          ``input_features`` + per-audio lengths via
          ``audio_feature_lengths``.

        Dtype handling diverges slightly from the reference: it
        hardcodes ``.to(torch.bfloat16)`` because the production
        deployment is always bf16. We instead cast to the audio
        tower's first-parameter dtype, which lands on bf16 in
        production (audio_tower built from a bf16 model) and on fp32
        in toy/test configs (audio_tower built from an fp32 model).
        Both modes produce the same numerical result at production
        scale; the fp32 path additionally lets CPU parity tests run
        without crashing on bf16 ↔ fp32 conv mismatches.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)

        # Pick the audio tower's running dtype (bf16 in production,
        # fp32 in toy parity tests). Using `next(self.audio_tower.parameters())`
        # is the cheapest way to read it without caching state.
        audio_dtype = next(self.audio_tower.parameters()).dtype

        audio_outputs, _audio_output_lengths = self.audio_tower(
            input_features.to(dtype=audio_dtype),
            feature_lens=audio_feature_lengths,
        )
        audio_features = self.audio_tower.proj(audio_outputs.last_hidden_state)
        return audio_features

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,  # noqa: ARG002 — BC, unused
        attention_mask: Optional[torch.Tensor] = None,
        audio_seqlens: Optional[torch.LongTensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Audio-aware version of `OpenPanguVLModel.get_rope_index`.

        Delegates to `compute_omni_rope_index`. When `audio_seqlens=None`
        and no audio tokens are in `input_ids`, the result matches
        `compute_vl_rope_index` byte-for-byte (the audio branch is
        unreachable).
        """
        audio_merge_size = getattr(self.config.audio_config, "audio_merge_size", 1)
        return compute_omni_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            audio_token_id=self.config.audio_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            tokens_per_second=getattr(self.config.vision_config, "tokens_per_second", 1.0),
            audio_seqlens=audio_seqlens,
            audio_merge_size=audio_merge_size,
            use_audio_in_video=use_audio_in_video,
        )

    def forward(  # type: ignore[override]
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
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,  # noqa: ARG002 — BC, unused
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        # VeOmni precomputed multimodal masks (from
        # ``process_sample_openpangu_omni`` data transform): boolean
        # tensors of shape ``(B, L)`` marking the audio / image /
        # video token positions in ``input_ids``. The reference HF
        # path locates these positions via ``input_ids ==
        # self.config.audio_token_id`` etc., but VeOmni's transform
        # zeros out those sentinel tokens before the model sees
        # them (so they don't blow up the embedding lookup), which
        # means the in-line equality check returns an all-False
        # mask. Mirror Qwen2.5-Omni's Patch.4 pattern (see
        # ``qwen2_5_omni/modeling_qwen2_5_omni.py:807``) and trust
        # the precomputed masks when provided. Fall back to the
        # equality check when the masks are missing — that keeps
        # the original inference path working for HF-shape inputs.
        audio_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
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
                inputs_embeds = self.get_input_embeddings()(input_ids).repeat(1, 1, self.mhc_num_stream)

            # FSDP rank-symmetry hook (audio).
            #
            # Under FSDP2 every rank must enter the same set of
            # all-gather collectives in the same order, otherwise
            # ranks that skip the audio tower because their batch
            # has no audio will hang waiting for ranks that do. We
            # follow Qwen2.5-Omni's Patch.5 pattern (see
            # ``qwen2_5_omni/modeling_qwen2_5_omni.py`` line 806-817):
            # when ``input_features`` is None on this rank but FSDP
            # is enabled globally, run a cached zero-input pass
            # through ``audio_tower`` and add ``0 * fake_embeds`` to
            # ``inputs_embeds`` so the all-gathers fire but the
            # numerics don't change.
            from veomni.distributed.parallel_state import get_parallel_state

            if input_features is not None:
                audio_features = self.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                # See the ``audio_mask`` docstring at the top of this
                # forward — prefer the precomputed VeOmni mask when
                # available (the transform zeroes the original
                # ``audio_token_id`` sentinels in ``input_ids``).
                if audio_mask is not None:
                    audio_mask_local = audio_mask.to(inputs_embeds.device).bool()
                else:
                    audio_mask_local = (input_ids == self.config.audio_token_id).to(inputs_embeds.device)
                n_audio_tokens = audio_mask_local.sum()
                n_audio_features = audio_features.shape[0]
                if not is_torchdynamo_compiling() and n_audio_tokens != n_audio_features:
                    raise ValueError(
                        "Audio features and audio tokens do not match: "
                        f"tokens: {n_audio_tokens}, features {n_audio_features}"
                    )
                mask_unsqueezed = audio_mask_local.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, audio_features)
            elif get_parallel_state().fsdp_enabled:
                fake = self.audio_tower.dummy_forward()
                # ``HuanyuAudioEncoder.forward`` returns (BaseModelOutput, conv_seq_lens);
                # mirror Qwen by only consuming ``last_hidden_state``.
                fake_embeds = fake[0].last_hidden_state.mean() * 0.0
                fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + fake_embeds

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0)
                if image_mask is not None:
                    image_mask_local = image_mask.to(inputs_embeds.device).bool()
                else:
                    image_mask_local = (input_ids == self.config.image_token_id).to(inputs_embeds.device)
                n_image_tokens = image_mask_local.sum()
                n_image_features = image_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                    raise ValueError(
                        "Image features and image tokens do not match: "
                        f"tokens: {n_image_tokens}, features {n_image_features}"
                    )
                mask_unsqueezed = image_mask_local.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)
            elif get_parallel_state().fsdp_enabled:
                fake_embeds = self.visual.dummy_forward().mean() * 0.0
                fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + fake_embeds

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0)
                if video_mask is not None:
                    video_mask_local = video_mask.to(inputs_embeds.device).bool()
                else:
                    video_mask_local = (input_ids == self.config.video_token_id).to(inputs_embeds.device)
                n_video_tokens = video_mask_local.sum()
                n_video_features = video_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                    raise ValueError(
                        "Video features and video tokens do not match: "
                        f"tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask_unsqueezed = video_mask_local.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)
            # NOTE: no FSDP dummy_forward for the video path — Pangu
            # Omni v2 30B-A2B reuses ``self.visual`` for both images
            # and videos, so the image branch above already fires the
            # all-gathers for the same tower. (Qwen2.5-Omni does emit
            # a second dummy here because it splits image/video into
            # separate encoders.)

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
                # Derive audio_seqlens from feature_attention_mask when given;
                # both the reference and our `get_audio_features` use this
                # same chain.
                audio_seqlens = (
                    torch.sum(feature_attention_mask, dim=1)
                    if feature_attention_mask is not None
                    else audio_feature_lengths
                )
                position_ids, rope_deltas_new = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_tensor,
                    audio_seqlens=audio_seqlens,
                    use_audio_in_video=use_audio_in_video,
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


# ---------------------------------------------------------------------------
# OpenPanguOmni (top-level CausalLM)
# ---------------------------------------------------------------------------


class OpenPanguOmni(OpenPanguPreTrainedModel, GenerationMixin):
    """Top-level multimodal (vision + audio + text) CausalLM. Wraps
    `OpenPanguOmniModel` with an `lm_head` Linear that projects
    `text_config.hidden_size` → `text_config.vocab_size`.

    The `_checkpoint_conversion_mapping` extends `OpenPanguVL`'s mapping
    with one extra rule: any param starting with ``audio_tower.`` lands
    under ``model.audio_tower.``. The reference checkpoint has a flat
    layout (`audio_tower.foo` at top level, same as `visual.foo`); our
    nested structure (`OpenPanguOmni → OpenPanguOmniModel → audio_tower /
    visual / language_model`) requires both rules to load cleanly.

    The exclusion clause on the ``^model`` rule is updated to also skip
    ``audio_tower`` — otherwise an `audio_tower.foo` key would be
    rewritten as `model.language_model.audio_tower.foo` and the load
    would silently drop the audio weights.
    """

    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        "^audio_tower": "model.audio_tower",
        r"^model(?!\.(language_model|visual|audio_tower|lm_head))": "model.language_model",
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = OpenPanguOmniModel(config)
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

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_audio_features(input_features, feature_attention_mask, audio_feature_lengths)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @property
    def audio_tower(self):
        return self.model.audio_tower

    def get_position_id_func(self):
        """Per-sample position-id precomputation hook used by VeOmni's
        ``VLMTrainer._build_data_transform``.

        VeOmni computes MRoPE position ids during data preprocessing
        (one CPU worker per sample, before the dataloader). The data
        transform replaces actual image/video/audio tokens with the
        VeOmni sentinels ``IMAGE_INPUT_INDEX`` / ``VIDEO_INPUT_INDEX``
        / ``AUDIO_INPUT_INDEX`` (constants in ``veomni/utils/constants.py``)
        before calling this function, so we tell ``compute_omni_rope_index``
        to treat those sentinels as the multimodal tokens. The real token
        IDs (``vision_start_token_id``, ``vision_end_token_id``,
        ``audio_start_token_id``, ``audio_end_token_id``) stay intact —
        they're never sentinels because the transform tokenizes the chat
        template normally and only swaps the *content-pad* tokens.

        This mirrors Qwen2.5-Omni's
        ``Qwen2_5OmniThinkerForConditionalGeneration.get_position_id_func``
        (``modeling_qwen2_5_omni.py:682``) except that:

        - Pangu's ``compute_omni_rope_index`` is a top-level free function
          (not a bound method), so we don't need the ``SimpleNamespace``
          fake-self / ``partial(get_position_id, main_func, fake_model)``
          double-wrap; a closure suffices.
        - Pangu returns ``(position_ids, rope_deltas)`` with shapes
          ``(3, B, L)`` and ``(B, 3)`` respectively. Data transform calls
          it with B=1 and expects a dict ``{"position_ids": (3, L),
          "rope_deltas": (3,)}``.
        """
        from ....utils.constants import AUDIO_INPUT_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX

        cfg = self.config
        vision_start_token_id = cfg.vision_start_token_id
        vision_end_token_id = cfg.vision_end_token_id
        spatial_merge_size = cfg.vision_config.spatial_merge_size
        tokens_per_second = getattr(cfg.vision_config, "tokens_per_second", 1.0)
        audio_merge_size = getattr(cfg.audio_config, "audio_merge_size", 1)

        def _compute(**kwargs) -> Dict[str, torch.Tensor]:
            position_ids, rope_deltas = compute_omni_rope_index(
                input_ids=kwargs["input_ids"],
                image_grid_thw=kwargs.get("image_grid_thw"),
                video_grid_thw=kwargs.get("video_grid_thw"),
                attention_mask=kwargs.get("attention_mask"),
                # Sentinels — match the data-transform swap below.
                image_token_id=IMAGE_INPUT_INDEX,
                video_token_id=VIDEO_INPUT_INDEX,
                audio_token_id=AUDIO_INPUT_INDEX,
                # Real token IDs — never swapped, kept intact in input_ids.
                vision_start_token_id=vision_start_token_id,
                vision_end_token_id=vision_end_token_id,
                spatial_merge_size=spatial_merge_size,
                tokens_per_second=tokens_per_second,
                audio_seqlens=kwargs.get("audio_seqlens"),
                audio_merge_size=audio_merge_size,
                use_audio_in_video=kwargs.get("use_audio_in_video", False),
            )
            # ``compute_omni_rope_index`` returns ``position_ids`` shape
            # ``(3, B, L)`` and ``rope_deltas`` shape ``(B, 3)``. The
            # data transform feeds B=1 (one sample at a time), so we
            # squeeze the batch dim to match what VeOmni's
            # ``mask_before_position_id_func`` consumers expect.
            assert position_ids.shape[1] == 1, (
                f"get_position_id_func: expected batch=1, got position_ids.shape={tuple(position_ids.shape)}"
            )
            return {
                "position_ids": position_ids.squeeze(1),  # (3, L)
                "rope_deltas": rope_deltas.squeeze(0),  # (3,)
            }

        return _compute

    def get_parallel_plan(self):
        """Return the VeOmni expert-parallel plan for OpenPanguOmni.

        The text backbone lives at ``self.model.language_model.*`` (vs
        ``self.model.*`` in ``OpenPanguV2ForCausalLM``), so we delegate
        to the shared ``parallel_plan.get_parallel_plan`` builder with
        the multimodal FQN prefix. The vision (``self.model.visual``)
        and audio (``self.model.audio_tower``) towers are dense and
        sharded by FSDP2 alone — no extra EP plan needed for them.

        Verified via ``named_parameters()`` on a meta-init build of
        the real 30B-A2B config:

            model.language_model.layers.{i}.mlp.experts.gate_up_proj
                shape (384, 512, 2560)
            model.language_model.layers.{i}.mlp.experts.down_proj
                shape (384, 2560, 256)

        See ``parallel_plan.py`` "Multimodal classes" section.
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
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        # See OpenPanguOmniModel.forward for the audio_mask /
        # image_mask / video_mask rationale (VeOmni Patch.4 mirror).
        audio_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # NOTE: `return_dict` is captured as a named param (rather than
        # left in **kwargs) so that callers passing
        # ``return_dict=True/False`` explicitly don't collide with the
        # `return_dict=True` we forward to `self.model(...)`. The inner
        # model always returns a dataclass; whether OpenPanguOmni
        # downstream returns a dataclass or tuple is gated by the
        # outer `return_dict` value, mirroring HF convention.
        kwargs.pop("return_dict", None)

        # MRoPE position_ids layout fixup.
        #
        # When VeOmni's ``MainCollator`` packs per-sample
        # ``position_ids: (3, L_i)`` into a batch, it cats along
        # ``pack_dim=-1`` then ``unsqueeze(0)`` (see
        # ``PackingCollator.__call__`` in data_collator.py), producing
        # ``(1, 3, sum_L)`` — i.e. shape ``(bs, dim, l)``.
        # ``OpenPanguVLRotaryEmbedding.forward`` however expects
        # ``(dim, bs, l)`` (the layout returned by
        # ``compute_omni_rope_index`` directly, sans collator). Mirror
        # the exact "Patch.6" fix in
        # ``qwen2_5_omni/modeling_qwen2_5_omni.py`` (line 898-900):
        # detect the ``(bs, 3, l)`` layout and transpose it back to
        # ``(3, bs, l)`` before handing off to the language model. Without
        # this fix, the MRoPE forward broadcasts wrong, producing
        # ``cos/sin`` of shape ``(3, bs, l, head_dim)`` instead of
        # ``(bs, l, head_dim)``, which then crashes inside
        # ``apply_partial_rotary_pos_emb`` with a ``torch.cat`` shape
        # mismatch (``q_rot dim 0 = 3`` vs ``q_pass dim 0 = 1``).
        if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
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
            use_audio_in_video=use_audio_in_video,
            audio_mask=audio_mask,
            image_mask=image_mask,
            video_mask=video_mask,
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
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )

        return OpenPanguVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


__all__ = [
    "get_audio_output_length",
    "compute_omni_rope_index",
    "OpenPanguOmniModel",
    "OpenPanguOmni",
]
