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

"""End-to-end integration parity for Pangu Omni v2 multimodal merge model
(Week 3.4.d).

Scope:

- `OpenPanguVLModel(config)` — full multimodal merge layer
  (vision_tower + text_backbone + masked_scatter merge + 3D mrope).
- `OpenPanguVL(config)` — `OpenPanguVLModel + lm_head` (GenerationMixin).

Test matrix:

1. **text_only_forward** — no `pixel_values`, no `image_grid_thw`. Sanity
   that the vision branch is properly skipped and the text-backbone
   integration through `OpenPanguVLTextModel` works end-to-end. Asserts
   bit-for-bit logits parity with the reference.

2. **image_forward** — exactly one image present, ``input_ids`` already
   pre-expanded with the right number of ``IMAGE_TOKEN_ID``s (=
   ``vision_seqlen``). Exercises the full pipeline:
   ``visual(pixel_values) → vision_projection → masked_scatter →
   language_model``. Bit-for-bit logits parity.

3. **vl_construction** — `OpenPanguVL(config)` constructs (lm_head wired
   in) and parameter counts match.

We monkey-patch the reference's `OpenPanguVLModel._parse_preprocess_params`
to a no-op before instantiation; that method calls `AutoProcessor.from_pretrained`
to populate `image_mean/std/rescale` scalars that **are not used in the
forward pass** (the actual rescale/normalize lines in `get_image_features`
are commented out in the reference). Neutering it keeps the test config
self-contained.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
TOLERANCE = 0.0  # bit-for-bit on eager attention path
_REF_MOD_CACHE: tuple[Any, ...] | None = None


# ---------------------------------------------------------------------------
# Module loaders (replicate the import-order ritual from sibling tests)
# ---------------------------------------------------------------------------


def _ours_module():
    # Pre-load `modeling_text` BEFORE `modeling_vl` to
    # avoid a (real) circular import when this test runs in isolation
    # (`pytest tests/test_pangu_vl_model_parity.py`). The cycle is:
    #
    #   modeling_vl  ──(line 1368, _get_openpangu_v2_model_cls)──>
    #   modeling_text ──(line 426, OpenPanguVLForConditionalGeneration
    #                              base = _get_open_pangu_vl_class())──>
    #   modeling_vl  (still mid-load, OpenPanguVL undefined) ✗
    #
    # If `modeling_text` is loaded first, `OpenPanguV2Model` is
    # defined (line 132) by the time `OpenPanguVLForConditionalGeneration`
    # needs `OpenPanguVL`, and the resolution chain unwinds cleanly. In a
    # full pytest session this happens incidentally (alphabetically earlier
    # tests import the omni module first) but standalone runs need an
    # explicit pre-load.
    importlib.import_module("veomni.models.transformers.pangu_omni_v2.modeling_text")
    ours = importlib.import_module("veomni.models.transformers.pangu_omni_v2.modeling_vl")
    ours.NPU_ATTN_INFR = False

    return ours


def _load_reference_vl_module():
    global _REF_MOD_CACHE
    if _REF_MOD_CACHE is not None:
        return _REF_MOD_CACHE
    _ours_module()
    from veomni.models.transformers._pangu_common._test_compat import install_pangu_reference_torch_npu_mock

    install_pangu_reference_torch_npu_mock()
    from transformers import AutoConfig

    AutoConfig.from_pretrained(str(PANGU_MODEL_DIR), trust_remote_code=True)
    pkg_name = "transformers_modules." + PANGU_MODEL_DIR.name
    ref_mod = importlib.import_module(f"{pkg_name}.modeling_openpangu_vl")
    ref_mod.NPU_ATTN_INFR = False

    # Neutralize the reference's `_parse_preprocess_params` so the
    # reference can be instantiated from a toy config without a real
    # `AutoProcessor` directory. The fields it normally sets
    # (image_mean / image_std / do_rescale / do_normalize / rescale_factor)
    # are dead weight in `forward()` — the only use site is inside the
    # `get_image_features` rescale/normalize block which is commented
    # out in the reference. Stubbing this is benign for parity.
    def _stub_parse(self, vision_config):
        self.channel = vision_config.in_channels
        self.patch_size = vision_config.patch_size
        self.do_rescale = False
        self.rescale_factor = 1.0
        self.do_normalize = False
        self.image_mean = (0.0, 0.0, 0.0)
        self.image_std = (1.0, 1.0, 1.0)

    ref_mod.OpenPanguVLModel._parse_preprocess_params = _stub_parse

    _REF_MOD_CACHE = (ref_mod,)
    return _REF_MOD_CACHE


# ---------------------------------------------------------------------------
# Tiny multimodal config — rich enough that ref+ours can both construct
# ---------------------------------------------------------------------------


# Multimodal token ids (test-only; production uses different real ids).
IMAGE_TOKEN_ID = 100
VIDEO_TOKEN_ID = 101
VISION_START_TOKEN_ID = 102
VISION_END_TOKEN_ID = 103
AUDIO_START_TOKEN_ID = 104
AUDIO_END_TOKEN_ID = 105
AUDIO_TOKEN_ID = 106


def _make_vision_config_dict(out_hidden_size: int) -> dict:
    """Tiny vision-tower config. `out_hidden_size` must equal the text
    backbone's `hidden_size` so the `vision_projection` outputs land
    cleanly in the text embedding space."""
    return dict(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        hidden_act="gelu",
        num_heads=4,
        in_channels=3,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=56,  # vit_merger_window_size = 56//2//14 = 2
        use_gatedmerger=True,
        out_hidden_size=out_hidden_size,
        fullatt_block_indexes=[1],
        mm_unit_vision_select_layer=[-1],
        _attn_implementation="eager",
    )


def _make_text_config_dict(hidden_size: int = 128) -> dict:
    """Tiny text-backbone config. `head_dim=32` so mrope_section=[6,5,5]
    (sum=16, ×2=32) fits exactly. partial_rotary_factor=1.0 so the rope
    rotates the full head_dim."""
    return dict(
        vocab_size=128,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        v_head_dim=32,
        intermediate_size=256,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        initializer_range=0.02,
        tie_word_embeddings=False,
        use_cache=False,
        partial_rotary_factor=1.0,
        qk_rope_dim=32,
        rope_theta=10000.0,
        max_position_embeddings=512,
        rope_interleaved=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        },
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
        attn_groupnorm=False,
        attn_elementwise_gate=False,
        param_sink_number=0,
        _attn_implementation="eager",
        torch_dtype="float32",
        # MoE — all-dense for simplicity (first_k_dense_replace covers all layers)
        n_routed_experts=1,
        n_shared_experts=0,
        num_experts_per_tok=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_intermediate_size=128,
        hidden_act="silu",
        _experts_implementation="eager",
        # MHC off
        use_mhc=False,
        mhc_num_stream=1,
        mhc_use_gamma=True,
        mhc_recur_norm=20,
        # Optional branches
        use_mla=False,
        first_k_dense_replace=2,  # all 2 layers dense
        sandwich_norm=False,
        block_post_layernorm_idx=None,
        layer_types=["full_attention", "full_attention"],
    )


def _make_full_config():
    """Build an `OpenPanguOmniConfig` with all sub-configs + multimodal
    token ids set. Must work for both the reference's `OpenPanguVL` (which
    reads ``config.image_token_id``, ``config.vision_config.spatial_merge_size``,
    etc.) and ours."""
    _ours_module()  # ensures sys.path is set so veomni imports below resolve
    from veomni.models.transformers.pangu_omni_v2.configuration_pangu_omni_v2 import (
        OpenPanguOmniConfig,
    )

    hidden_size = 128
    cfg = OpenPanguOmniConfig(
        vision_config=_make_vision_config_dict(out_hidden_size=hidden_size),
        text_config=_make_text_config_dict(hidden_size=hidden_size),
        audio_config={},  # not used in VL forward
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        audio_end_token_id=AUDIO_END_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        tokens_per_second=1.0,
        use_mhc=False,
        mhc_num_stream=1,
        torch_dtype="float32",
    )
    # The reference's vision tower instantiation reads `_attn_implementation`
    # off the vision sub-config; the audio sub-config likewise needs the flag
    # set even when unused (PretrainedConfig sub-configs strip it).
    cfg.vision_config._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.audio_config._attn_implementation = "eager"
    # Transformers v5 standardizes `rope_scaling` into `rope_parameters`
    # and can drop the reference-only `rope_theta` field from toy configs.
    # The upstream Pangu reference still reads it from `rope_parameters`.
    cfg.text_config.rope_parameters["rope_theta"] = cfg.text_config.rope_theta
    return cfg


def _build_pair(cfg):
    """Build ref + ours `OpenPanguVLModel(cfg)`, sync state dicts so
    bit-for-bit comparison is on identical params."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    ref = ref_mod.OpenPanguVLModel(cfg).eval()
    torch.manual_seed(0)
    ours = ours_mod.OpenPanguVLModel(cfg).eval()

    # Sync: drop random weights into both with the same seed.
    ours_sd = dict(ours.state_dict())
    ref_sd = dict(ref.state_dict())
    only_ours = set(ours_sd) - set(ref_sd)
    only_ref = set(ref_sd) - set(ours_sd)
    if only_ours or only_ref:
        raise AssertionError(
            "state_dict mismatch between ref and ours:\n"
            f"  only in ours: {sorted(only_ours)[:10]}\n"
            f"  only in ref:  {sorted(only_ref)[:10]}"
        )
    for k in ours_sd:
        rand = torch.randn_like(ours_sd[k]) * 0.02
        ours_sd[k].copy_(rand)
        ref_sd[k].copy_(rand)
    ours.load_state_dict(ours_sd)
    ref.load_state_dict(ref_sd)
    return ours, ref


# ---------------------------------------------------------------------------
# 1. Text-only forward parity
# ---------------------------------------------------------------------------


def test_open_pangu_vl_model_text_only_forward_parity():
    """No pixel_values / no image_grid_thw / no video_grid_thw. The merge
    layer should fall through to a plain text forward (mrope position
    ids cumsum-only, full ``inputs_embeds`` from the input_ids embedding
    lookup, no `masked_scatter`)."""
    cfg = _make_full_config()
    ours, ref = _build_pair(cfg)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_ours = ours(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        out_ref = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    assert out_ours.last_hidden_state.shape == out_ref.last_hidden_state.shape
    diff = (out_ours.last_hidden_state - out_ref.last_hidden_state).abs().max().item()
    assert diff == TOLERANCE, f"text-only last_hidden_state mismatch: max_diff={diff}"


# ---------------------------------------------------------------------------
# 2. Image forward parity
# ---------------------------------------------------------------------------


def _make_pixel_values_for_grid(image_grid_thw: torch.LongTensor, vision_cfg) -> torch.Tensor:
    """Build a (num_patches, channels*temporal*patch*patch) tensor matching
    `image_grid_thw`. With `temporal_patch_size=2`, `in_channels=3`,
    `patch_size=14`, the second dim is `3*2*14*14 = 1176`."""
    total_patches = int(image_grid_thw.prod(-1).sum().item())
    feat_dim = vision_cfg.in_channels * vision_cfg.temporal_patch_size * vision_cfg.patch_size * vision_cfg.patch_size
    return torch.randn(total_patches, feat_dim) * 0.1


def test_open_pangu_vl_model_image_forward_parity():
    """One image with `image_grid_thw=[1, 4, 4]` → vision_seqlen = 4
    (after `spatial_merge_size=2`). Sequence: text(2) + 4 IMAGE_TOKEN_IDs
    + text(2). Exercises:
      - patch embedding through vision tower
      - vision_projection → text hidden size
      - masked_scatter merge into inputs_embeds
      - 3D mrope position ids with image block
      - text backbone consuming merged inputs_embeds
    """
    cfg = _make_full_config()
    ours, ref = _build_pair(cfg)

    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    # vision_seqlen = 1*4*4 / 2^2 = 4
    input_ids = torch.tensor(
        [[1, 2, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 3, 4]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    torch.manual_seed(42)
    pixel_values = _make_pixel_values_for_grid(image_grid_thw, cfg.vision_config)

    with torch.no_grad():
        out_ours = ours(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.clone(),
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        # Reset cached rope_deltas on ref so the prefill branch is taken
        # again (ours and ref share the same module instance state otherwise).
        ref.rope_deltas = None
        out_ref = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.clone(),
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

    assert out_ours.last_hidden_state.shape == out_ref.last_hidden_state.shape
    diff = (out_ours.last_hidden_state - out_ref.last_hidden_state).abs().max().item()
    assert diff == TOLERANCE, f"image last_hidden_state mismatch: max_diff={diff}"
    # rope_deltas should match too — these are integer-valued counts.
    if out_ours.rope_deltas is not None and out_ref.rope_deltas is not None:
        assert torch.equal(out_ours.rope_deltas, out_ref.rope_deltas)


# ---------------------------------------------------------------------------
# 3. OpenPanguVL construction (lm_head wired in)
# ---------------------------------------------------------------------------


def test_open_pangu_vl_for_causallm_construction():
    """`OpenPanguVL(config)` adds an `lm_head` projection of shape
    `(text_hidden_size, text_vocab_size)`. Verify both sides expose
    the same parameter count and lm_head shape."""
    cfg = _make_full_config()
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    ref = ref_mod.OpenPanguVL(cfg)
    torch.manual_seed(0)
    ours = ours_mod.OpenPanguVL(cfg)

    ref_params = sum(p.numel() for p in ref.parameters())
    our_params = sum(p.numel() for p in ours.parameters())
    assert ref_params == our_params, (ref_params, our_params)

    # lm_head shape
    assert ours.lm_head.weight.shape == ref.lm_head.weight.shape
    expected = (cfg.text_config.vocab_size, cfg.text_config.hidden_size)
    assert tuple(ours.lm_head.weight.shape) == expected


def test_open_pangu_vl_for_causallm_text_only_logits_parity():
    """End-to-end logits parity through `OpenPanguVL` (lm_head included)
    on text-only input. This is the minimum acceptance gate before we
    can claim Week 3.4.d is done: the same input must produce the same
    logits through the full multimodal stack with text-only data."""
    cfg = _make_full_config()
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    ref = ref_mod.OpenPanguVL(cfg).eval()
    torch.manual_seed(0)
    ours = ours_mod.OpenPanguVL(cfg).eval()

    # Sync state_dict
    ours_sd = dict(ours.state_dict())
    ref_sd = dict(ref.state_dict())
    only_ours = set(ours_sd) - set(ref_sd)
    only_ref = set(ref_sd) - set(ours_sd)
    if only_ours or only_ref:
        raise AssertionError(
            f"state_dict mismatch:\n  only in ours: {sorted(only_ours)[:10]}\n  only in ref:  {sorted(only_ref)[:10]}"
        )
    for k in ours_sd:
        rand = torch.randn_like(ours_sd[k]) * 0.02
        ours_sd[k].copy_(rand)
        ref_sd[k].copy_(rand)
    ours.load_state_dict(ours_sd)
    ref.load_state_dict(ref_sd)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_ours = ours(input_ids=input_ids, attention_mask=attention_mask)
        ref.model.rope_deltas = None  # match cold-cache state
        out_ref = ref(input_ids=input_ids, attention_mask=attention_mask)

    diff = (out_ours.logits - out_ref.logits).abs().max().item()
    assert diff == TOLERANCE, f"OpenPanguVL text-only logits mismatch: max_diff={diff}"
