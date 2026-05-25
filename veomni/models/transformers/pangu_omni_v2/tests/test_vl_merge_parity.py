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

"""Bit-for-bit parity for Pangu Omni v2 multimodal merge plumbing
(Week 3.4.b + 3.4.c).

Covers the **live** code paths in `OpenPanguVLModel`'s composition
graph (the pieces that *do* get executed during real VL inference,
unlike `OpenPanguVLAttention` / `OpenPanguVLDecoderLayer` which Week
3.4.a established are dead code in the 30B-A2B inference path):

- `ProjectionSingle` — vision_projection's `act → linear` block
- `OpenPanguVLTextModel` — V2Model + mrope-rotary swap
- `OpenPanguVLModelOutputWithPast` / `OpenPanguVLCausalLMOutputWithPast`
  — dataclass outputs (construction round-trip, no compute)
- `compute_vl_rope_index` / `_get_llm_pos_ids_for_vision` — the 3D
  position id computation that maps `(input_ids, image_grid_thw,
  video_grid_thw)` to a `(3, bsz, seq)` position id tensor

The rope-index function is the **highest-risk live component** in
Week 3.4 since:

1. It's pure-Python tensor-index manipulation (no learned params, no
   norm to "soak up" errors).
2. Outputs are integer-valued position ids, so off-by-one bugs cause
   silently wrong mrope rotations downstream rather than nan/inf
   crashes.
3. Different code paths exist for text-only, image-only,
   image-and-text, video-only, and mixed-modality sequences — each
   needs its own parity case.

We parametrize 6 distinct input shapes covering these branches.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
TOLERANCE = 0.0
_REF_MOD_CACHE: tuple[Any, ...] | None = None


def _ours_module():
    # Pre-load ``modeling_pangu_omni_v2`` to break the
    # ``modeling_openpangu_vl ↔ modeling_pangu_omni_v2`` import cycle
    # when this file runs standalone. Detailed reasoning in
    # ``tests/test_pangu_vl_model_parity.py::_ours_module``.
    from veomni.models.transformers.pangu_omni_v2 import modeling_openpangu_vl as ours
    from veomni.models.transformers.pangu_omni_v2 import modeling_pangu_omni_v2  # noqa: F401

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
    _REF_MOD_CACHE = (ref_mod,)
    return _REF_MOD_CACHE


# Token ids — production-aligned but stand-in (test-config doesn't have
# real tokenizer). Anything works as long as ref and ours see the
# same ids.
IMAGE_TOKEN_ID = 100
VIDEO_TOKEN_ID = 101
VISION_START_TOKEN_ID = 102
VISION_END_TOKEN_ID = 103


def _make_stub_for_ref(ref_mod, spatial_merge_size: int = 2, tokens_per_second: float = 1.0):
    """A stand-in ``self`` for ``OpenPanguVLModel.get_rope_index`` so
    we don't need to instantiate a full VLModel (and its vision tower)
    just to invoke the rope-index method on the reference side. The
    reference's ``get_rope_index`` calls ``self._get_llm_pos_ids_for_vision``
    on the image branch, so we bind that helper to the stub here too."""

    class _Stub:
        config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            vision_end_token_id=VISION_END_TOKEN_ID,
            vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size),
            tokens_per_second=tokens_per_second,
        )
        _get_llm_pos_ids_for_vision = ref_mod.OpenPanguVLModel._get_llm_pos_ids_for_vision

    return _Stub()


# ---------------------------------------------------------------------------
# 1. ProjectionSingle parity (act → linear)
# ---------------------------------------------------------------------------


def test_projection_single_parity():
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    torch.manual_seed(0)
    ref_proj = ref_mod.ProjectionSingle(64, 128)
    our_proj = ours_mod.ProjectionSingle(64, 128)
    our_proj.load_state_dict(ref_proj.state_dict())

    x = torch.randn(4, 16, 64)
    ref_out = ref_proj(x.clone())
    our_out = our_proj(x.clone())
    assert ref_out.shape == our_out.shape == (4, 16, 128)
    assert (ref_out - our_out).abs().max().item() == TOLERANCE


# ---------------------------------------------------------------------------
# 2. OpenPanguVLTextModel: same code path as V2Model but with mrope rotary
# ---------------------------------------------------------------------------


def _make_text_config_for_text_model():
    """Build a tiny text-config that is rich enough for the **reference's**
    `OpenPanguV2Model` constructor (which `OpenPanguVLTextModel` inherits
    from) to succeed. The reference reads many more fields than ours
    does — `use_mla`, `sandwich_norm`, `block_post_layernorm_idx`, MoE
    routing scalars, `attn_*` flags — so we mirror the field set used
    by `tests/test_pangu_modeling_parity.py::_build_test_config` and
    additionally pin `head_dim` / `qk_rope_dim` so the mrope section
    `[6,5,5]` fits exactly (sum × 2 = 32)."""
    from veomni.models.transformers.pangu_omni_v2.configuration_pangu_omni_v2 import (
        OpenPanguOmniTextConfig,
    )

    cfg = OpenPanguOmniTextConfig()
    cfg.vocab_size = 128
    cfg.hidden_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.head_dim = 32
    cfg.v_head_dim = 32
    cfg.intermediate_size = 256
    cfg.pad_token_id = None
    cfg.bos_token_id = 0
    cfg.eos_token_id = 1
    cfg.initializer_range = 0.02
    cfg.tie_word_embeddings = False
    cfg.use_cache = False
    cfg.partial_rotary_factor = 1.0  # full rotary on full head_dim=32; matches mrope section sum*2
    cfg.qk_rope_dim = 32
    cfg.rope_theta = 10000.0
    cfg.max_position_embeddings = 512
    cfg.rope_interleaved = False
    cfg.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
    }
    cfg.rope_scaling = {
        "rope_type": "default",
        "mrope_section": [6, 5, 5],
        "mrope_interleaved": True,
    }
    cfg.rms_norm_eps = 1e-6
    cfg.attention_dropout = 0.0
    cfg.attention_bias = False
    cfg.attn_groupnorm = False
    cfg.attn_elementwise_gate = False
    cfg.param_sink_number = 0
    cfg._attn_implementation = "eager"
    cfg.torch_dtype = torch.float32
    # MoE
    cfg.n_routed_experts = 1
    cfg.n_shared_experts = 0
    cfg.num_experts_per_tok = 1
    cfg.topk_group = 1
    cfg.norm_topk_prob = True
    cfg.routed_scaling_factor = 1.0
    cfg.moe_intermediate_size = 128
    cfg.hidden_act = "silu"
    cfg._experts_implementation = "eager"
    # MHC
    cfg.use_mhc = False
    cfg.mhc_num_stream = 1
    cfg.mhc_use_gamma = True
    cfg.mhc_recur_norm = 20
    # Optional branches
    cfg.use_mla = False
    cfg.first_k_dense_replace = 2  # all dense, no MoE
    cfg.sandwich_norm = False
    cfg.block_post_layernorm_idx = None
    cfg.layer_types = ["full_attention", "full_attention"]
    return cfg


def test_open_pangu_vl_text_model_construction():
    """VLTextModel must inherit V2Model behavior with the rotary_emb
    swapped. Verify the swap took (correct class) and that the model
    is constructible end-to-end."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    cfg = _make_text_config_for_text_model()

    torch.manual_seed(0)
    ref_tm = ref_mod.OpenPanguVLTextModel(cfg)
    our_tm = ours_mod.OpenPanguVLTextModel(cfg)

    # The rotary_emb on both should be OpenPanguVLRotaryEmbedding
    assert ref_tm.rotary_emb.__class__.__name__ == "OpenPanguVLRotaryEmbedding"
    assert our_tm.rotary_emb.__class__.__name__ == "OpenPanguVLRotaryEmbedding"
    # And param counts must match (V2Model identical structure)
    ref_params = sum(p.numel() for p in ref_tm.parameters())
    our_params = sum(p.numel() for p in our_tm.parameters())
    assert ref_params == our_params, (ref_params, our_params)


# ---------------------------------------------------------------------------
# 3-8. compute_vl_rope_index parity across all branches
# ---------------------------------------------------------------------------


def _call_ref_get_rope_index(
    ref_mod,
    input_ids,
    image_grid_thw,
    video_grid_thw,
    attention_mask,
    *,
    spatial_merge_size=2,
    tokens_per_second=1.0,
):
    stub = _make_stub_for_ref(ref_mod, spatial_merge_size, tokens_per_second)
    return ref_mod.OpenPanguVLModel.get_rope_index(
        stub, input_ids, image_grid_thw, video_grid_thw, None, attention_mask
    )


def _call_our_compute_vl_rope_index(
    ours_mod,
    input_ids,
    image_grid_thw,
    video_grid_thw,
    attention_mask,
    *,
    spatial_merge_size=2,
    tokens_per_second=1.0,
):
    return ours_mod.compute_vl_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        spatial_merge_size=spatial_merge_size,
        tokens_per_second=tokens_per_second,
    )


def test_compute_vl_rope_index_text_only_no_mask():
    """Pure text input with no attention_mask — fastest branch
    (`else` branch returning expanded arange)."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    ref_pos, ref_delta = _call_ref_get_rope_index(ref_mod, input_ids, None, None, None)
    our_pos, our_delta = _call_our_compute_vl_rope_index(ours_mod, input_ids, None, None, None)
    assert torch.equal(ref_pos, our_pos)
    assert torch.equal(ref_delta, our_delta)


def test_compute_vl_rope_index_text_only_with_mask():
    """Pure text + attention_mask present — exercises the cumsum branch."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.long)
    ref_pos, ref_delta = _call_ref_get_rope_index(ref_mod, input_ids, None, None, attention_mask)
    our_pos, our_delta = _call_our_compute_vl_rope_index(ours_mod, input_ids, None, None, attention_mask)
    assert torch.equal(ref_pos, our_pos)
    assert torch.equal(ref_delta, our_delta)


def test_compute_vl_rope_index_single_image():
    """One image, single batch sample. Image block: T=1, H=2, W=2 →
    after spatial_merge_size=2: llm_H=1, llm_W=1 → 1 vision token
    expanded from 1 IMAGE_TOKEN_ID placeholder."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    # Sequence: text(1) + image(1 placeholder, expands to 1 token) + text(3)
    input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, 2, 3, 4]], dtype=torch.long)
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    ref_pos, ref_delta = _call_ref_get_rope_index(ref_mod, input_ids, image_grid_thw, None, None)
    our_pos, our_delta = _call_our_compute_vl_rope_index(ours_mod, input_ids, image_grid_thw, None, None)
    assert torch.equal(ref_pos, our_pos), (ref_pos, our_pos)
    assert torch.equal(ref_delta, our_delta)


def test_compute_vl_rope_index_larger_image():
    """T=1, H=4, W=4 → after spatial_merge_size=2: llm_H=2, llm_W=2
    → ``vision_seqlen = 1*2*2 = 4``. The caller is responsible for
    pre-expanding the image placeholder to ``vision_seqlen`` consecutive
    ``IMAGE_TOKEN_ID``s in ``input_ids`` (Qwen2-VL convention). Exercises
    ``_get_llm_pos_ids_for_vision`` with a non-trivial grid."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    input_ids = torch.tensor(
        [[1, 2, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 3, 4]],
        dtype=torch.long,
    )
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    ref_pos, ref_delta = _call_ref_get_rope_index(ref_mod, input_ids, image_grid_thw, None, None)
    our_pos, our_delta = _call_our_compute_vl_rope_index(ours_mod, input_ids, image_grid_thw, None, None)
    assert torch.equal(ref_pos, our_pos), f"\nref:\n{ref_pos}\nours:\n{our_pos}"
    assert torch.equal(ref_delta, our_delta)


def test_compute_vl_rope_index_batched_with_mask():
    """Batched: sample 0 has image, sample 1 is text-only with
    attention_mask. Stresses per-batch dispatch within the loop."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    input_ids = torch.tensor(
        [
            [1, 2, IMAGE_TOKEN_ID, 3, 4],
            [5, 6, 7, 8, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    ref_pos, ref_delta = _call_ref_get_rope_index(ref_mod, input_ids, image_grid_thw, None, attention_mask)
    our_pos, our_delta = _call_our_compute_vl_rope_index(ours_mod, input_ids, image_grid_thw, None, attention_mask)
    assert torch.equal(ref_pos, our_pos), f"\nref:\n{ref_pos}\nours:\n{our_pos}"
    assert torch.equal(ref_delta, our_delta)


def test_compute_vl_rope_index_video():
    """Video: T=2 frames, H=2, W=2 → after spatial_merge_size=2:
    ``llm_H=1, llm_W=1, tokens_per_frame=1``. Per the reference's video
    expansion convention, between adjacent frames the caller must insert
    both a ``vision_end_token_id`` (right placeholder of the earlier
    frame) **and** a ``vision_start_token_id`` (left placeholder of the
    next frame). For T=2 frames, the video block expands to::

        VIDEO_TOKEN, VISION_END, VISION_START, VIDEO_TOKEN

    so the full input_ids becomes 6 tokens long (text(1) + 4-token
    video block + text(1)).
    """
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    input_ids = torch.tensor(
        [[1, VIDEO_TOKEN_ID, VISION_END_TOKEN_ID, VISION_START_TOKEN_ID, VIDEO_TOKEN_ID, 2]],
        dtype=torch.long,
    )
    video_grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.long)
    ref_pos, ref_delta = _call_ref_get_rope_index(ref_mod, input_ids, None, video_grid_thw, None)
    our_pos, our_delta = _call_our_compute_vl_rope_index(ours_mod, input_ids, None, video_grid_thw, None)
    assert torch.equal(ref_pos, our_pos), f"\nref:\n{ref_pos}\nours:\n{our_pos}"
    assert torch.equal(ref_delta, our_delta)


# ---------------------------------------------------------------------------
# 9. _get_llm_pos_ids_for_vision standalone parity
# ---------------------------------------------------------------------------


def test_get_llm_pos_ids_for_vision_parity():
    """Standalone parity for the per-image position id helper.

    Construct a `OpenPanguVLModel` stub so the reference's method can
    bind, and call its method form. Compare against our free-function
    form."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    stub = _make_stub_for_ref(ref_mod, spatial_merge_size=2)
    grid_hs = torch.tensor([4])
    grid_ws = torch.tensor([4])
    t_index = torch.tensor([0])

    ref_pos = ref_mod.OpenPanguVLModel._get_llm_pos_ids_for_vision(stub, 10, 0, 2, t_index, grid_hs, grid_ws)
    our_pos = ours_mod._get_llm_pos_ids_for_vision(10, 0, 2, t_index, grid_hs, grid_ws)
    assert torch.equal(ref_pos, our_pos), f"\nref:\n{ref_pos}\nours:\n{our_pos}"


# ---------------------------------------------------------------------------
# 10. Output dataclass round-trip
# ---------------------------------------------------------------------------


def test_output_dataclasses_round_trip():
    """Construct both output containers and verify the fields hold."""
    ours_mod = _ours_module()
    a = ours_mod.OpenPanguVLModelOutputWithPast(
        last_hidden_state=torch.zeros(2),
        rope_deltas=torch.zeros(2, 1, dtype=torch.long),
    )
    assert a.last_hidden_state.shape == (2,)
    assert a.rope_deltas.shape == (2, 1)

    b = ours_mod.OpenPanguVLCausalLMOutputWithPast(
        loss=torch.tensor(0.0),
        logits=torch.zeros(2, 5, 128),
        rope_deltas=torch.zeros(2, 1, dtype=torch.long),
    )
    assert b.logits.shape == (2, 5, 128)
    assert b.rope_deltas.shape == (2, 1)
