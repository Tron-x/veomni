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

"""Bit-for-bit parity for Pangu Omni v2 multimodal RoPE (Week 3.4.a).

Covers ``OpenPanguVLRotaryEmbedding`` (3D mrope with two code paths —
default and interleaved) and ``apply_multimodal_rotary_pos_emb`` (the
companion blend-and-rotate function used in
``OpenPanguVLAttention.forward``).

mrope is the **highest-risk** component of Week 3 because it's where:

1. Position ids become **3D** (per-axis T/H/W) instead of the usual
   1D linear ids; any off-by-one in `mrope_section` blending breaks
   alignment between vision-token rotations and text-token rotations.
2. The Pangu 30B-A2B production config sets
   ``mrope_interleaved=True``, which goes through the
   ``get_mrope_interleaved_id_list`` path — a deterministic but
   non-obvious round-robin axis permutation. A subtle drift here
   (e.g. tie-breaking with different ordering) shifts every rotation
   downstream and silently degrades multimodal accuracy without any
   visible error.

So we pin both paths with bit-for-bit parity tests against the Pangu
reference, including:

- The id-list generator at its three canonical configurations
  (``(h, w, 0)`` 2-axis, ``(t, h, w)`` 3-axis vanilla, and ``(t, h, w,
  force_last=True)`` the production setting).
- ``OpenPanguVLRotaryEmbedding`` cos/sin output for both interleaved
  and default branches.
- ``apply_multimodal_rotary_pos_emb`` round-trip against reference
  with a randomized q/k pair.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
TOLERANCE = 0.0
_REF_MOD_CACHE: tuple[Any, ...] | None = None


def _ours_module():
    from veomni.models.transformers.pangu_omni_v2 import modeling_vl as ours

    return ours


def _text_config_class():
    from veomni.models.transformers.pangu_omni_v2.configuration_pangu_omni_v2 import (
        OpenPanguOmniTextConfig,
    )

    return OpenPanguOmniTextConfig


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


def _make_text_config(*, mrope_interleaved: bool, mrope_section: list[int]):
    """Minimal text-config that satisfies ``OpenPanguVLRotaryEmbedding.__init__``.

    Only the rope-related fields and basic head-dim math are needed
    for the rope embedding test; the rest of the text config (MoE,
    layer types, etc.) is irrelevant here."""
    # head_dim must satisfy sum(mrope_section) == head_dim // 2 (so that
    # the blend `mrope_section * 2` sums exactly to the full head_dim
    # after the `cat((freqs, freqs), dim=-1)` in forward). For
    # mrope_section=[6,5,5] (sum=16) this means head_dim=32. We achieve
    # head_dim=32 via hidden_size=128, num_attention_heads=4.
    # Production model uses partial_rotary_factor with qk_rope_dim=32
    # so the same relation holds (sum(mrope_section) == qk_rope_dim/2).
    target_head_dim = sum(mrope_section) * 2
    hidden_size = target_head_dim * 4  # 4 heads
    cls = _text_config_class()
    cfg = cls(
        vocab_size=128,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": mrope_section,
            "mrope_interleaved": mrope_interleaved,
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


# ---------------------------------------------------------------------------
# 1. get_mrope_interleaved_id_list parity (the tiebreaker is the critical one)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,c,force_last",
    [
        (5, 5, 0, False),  # 2-axis (H, W) fallback (mrope_section has len=2)
        (5, 5, 5, False),  # 3-axis no-force-last
        (6, 5, 5, True),  # 3-axis force_last (Pangu 30B-A2B production)
        (4, 8, 3, False),  # asymmetric counts (regression for tie-breaker)
        (3, 5, 8, True),  # asymmetric + force_last
    ],
)
def test_mrope_interleaved_id_list_parity(a, b, c, force_last):
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    ref_list = ref_mod.OpenPanguVLRotaryEmbedding.get_mrope_interleaved_id_list(a, b, c, force_last=force_last)
    our_list = ours_mod.OpenPanguVLRotaryEmbedding.get_mrope_interleaved_id_list(a, b, c, force_last=force_last)
    assert ref_list == our_list, (
        f"id_list({a},{b},{c},force_last={force_last}) differs:\n  ref: {ref_list}\n  ours: {our_list}"
    )


# ---------------------------------------------------------------------------
# 2. OpenPanguVLRotaryEmbedding parity — default (non-interleaved)
# ---------------------------------------------------------------------------


def test_open_pangu_vl_rotary_embedding_parity_default():
    """``mrope_interleaved=False`` path: cos/sin are returned **without**
    the contiguous-triples blend applied; the blend happens later in
    ``apply_multimodal_rotary_pos_emb``. Verify the raw emb is
    bit-for-bit identical."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    cfg = _make_text_config(mrope_interleaved=False, mrope_section=[6, 5, 5])

    torch.manual_seed(0)
    ref_rope = ref_mod.OpenPanguVLRotaryEmbedding(cfg)
    our_rope = ours_mod.OpenPanguVLRotaryEmbedding(cfg)
    # inv_freq must match
    assert (ref_rope.inv_freq - our_rope.inv_freq).abs().max().item() == 0.0

    x = torch.randn(1, 8, cfg.hidden_size, dtype=torch.float32)
    # position_ids must be 3D: (3, batch, seq_len)
    position_ids = torch.arange(8).view(1, 1, -1).expand(3, 1, -1).contiguous()

    ref_cos, ref_sin = ref_rope(x, position_ids)
    our_cos, our_sin = our_rope(x, position_ids)
    assert ref_cos.shape == our_cos.shape
    assert (ref_cos - our_cos).abs().max().item() == TOLERANCE
    assert (ref_sin - our_sin).abs().max().item() == TOLERANCE


# ---------------------------------------------------------------------------
# 3. OpenPanguVLRotaryEmbedding parity — interleaved (production path)
# ---------------------------------------------------------------------------


def test_open_pangu_vl_rotary_embedding_parity_interleaved_30b_a2b():
    """``mrope_interleaved=True`` with ``mrope_section=[6,5,5]`` —
    **the production setting** for Pangu 30B-A2B per its config.json.

    This is the path that gets exercised in real inference; the
    interleaved blend is baked into cos/sin returned by the embedding
    (not deferred to ``apply_multimodal_rotary_pos_emb`` like the
    default path)."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()
    cfg = _make_text_config(mrope_interleaved=True, mrope_section=[6, 5, 5])

    torch.manual_seed(0)
    ref_rope = ref_mod.OpenPanguVLRotaryEmbedding(cfg)
    our_rope = ours_mod.OpenPanguVLRotaryEmbedding(cfg)
    # Compare the pre-computed permutation index
    assert ref_rope.mrope_dim == our_rope.mrope_dim

    x = torch.randn(2, 12, cfg.hidden_size, dtype=torch.float32)
    # Vary position ids across the 3 axes to stress-test the blend
    pos_t = torch.arange(12).unsqueeze(0).expand(2, -1)
    pos_h = (torch.arange(12) * 3 % 12).unsqueeze(0).expand(2, -1)
    pos_w = (torch.arange(12) * 5 % 12).unsqueeze(0).expand(2, -1)
    position_ids = torch.stack([pos_t, pos_h, pos_w])  # (3, 2, 12)

    ref_cos, ref_sin = ref_rope(x, position_ids)
    our_cos, our_sin = our_rope(x, position_ids)
    assert ref_cos.shape == our_cos.shape
    assert (ref_cos - our_cos).abs().max().item() == TOLERANCE
    assert (ref_sin - our_sin).abs().max().item() == TOLERANCE


# ---------------------------------------------------------------------------
# 4. apply_multimodal_rotary_pos_emb parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mrope_section", [[16, 24, 24], [6, 5, 5], [4, 4, 8]])
def test_apply_multimodal_rotary_pos_emb_parity(mrope_section):
    """Verify the q/k rotation step bit-for-bit, with q/k shapes
    ``(bsz, heads, seq_len, head_dim)`` and head_dim sized to fit
    ``sum(mrope_section) * 2``."""
    (ref_mod,) = _load_reference_vl_module()
    ours_mod = _ours_module()

    head_dim = sum(mrope_section) * 2
    bsz, heads, seq_len = 1, 4, 16

    torch.manual_seed(42)
    q = torch.randn(bsz, heads, seq_len, head_dim)
    k = torch.randn(bsz, heads, seq_len, head_dim)
    # cos/sin shape: (3, bsz, seq_len, head_dim) — pre-blend
    cos = torch.randn(3, bsz, seq_len, head_dim)
    sin = torch.randn(3, bsz, seq_len, head_dim)

    ref_q, ref_k = ref_mod.apply_multimodal_rotary_pos_emb(
        q.clone(), k.clone(), cos.clone(), sin.clone(), mrope_section
    )
    our_q, our_k = ours_mod.apply_multimodal_rotary_pos_emb(
        q.clone(), k.clone(), cos.clone(), sin.clone(), mrope_section
    )
    assert ref_q.shape == our_q.shape and ref_k.shape == our_k.shape
    assert (ref_q - our_q).abs().max().item() == TOLERANCE
    assert (ref_k - our_k).abs().max().item() == TOLERANCE


# ---------------------------------------------------------------------------
# 5. NB: NO end-to-end test of forward → apply_multimodal_rotary_pos_emb
# ---------------------------------------------------------------------------
#
# Concretely: ``OpenPanguVLRotaryEmbedding.forward`` collapses the
# 3-axis dim *inside forward* (both interleaved and default branches),
# so its output cos/sin have shape ``(bsz, seq, head_dim)``. The
# companion ``apply_multimodal_rotary_pos_emb`` then re-splits on
# ``dim=-1`` and indexes ``m[i % 3]`` — which now picks the **batch**
# dim instead of the (already collapsed) 3-axis dim. This:
#
# 1. ``IndexError`` for ``bsz < 3``.
# 2. Semantic non-sense for ``bsz >= 3`` (it mashes three different
#    batch samples together as if they were three mrope axes).
#
# We verified the reference itself exhibits this behaviour — it's
# **not a porting bug on our side**, it's a quirk of the reference.
# In production this code path is **dead**: ``OpenPanguVLAttention``
# is never instantiated (the text backbone is ``OpenPanguV2Model``
# whose ``OpenPanguV2DecoderLayer`` uses ``OpenPanguV2Attention``,
# which calls **regular** ``apply_rotary_pos_emb`` with the (bsz,
# seq, head_dim)-shaped cos/sin returned by ``OpenPanguVLRotaryEmbedding``
# — exactly the right shape for that path).
#
# So we test ``apply_multimodal_rotary_pos_emb`` in isolation with
# synthetic 3-axis cos input (test #4 above), and we test
# ``OpenPanguVLRotaryEmbedding.forward`` in isolation (tests #2-3), but
# we deliberately do **not** chain them end-to-end since that chain
# isn't on any production code path.
