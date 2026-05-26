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

"""Bit-for-bit parity tests for the Pangu Omni v2 audio tower.

These tests check the verbatim port of ``HuanyuAudioEncoder`` (and its
sub-modules ``HuanyuRotaryEmbedding``, ``VGGBlock``, ``HuanyuAttention``,
``ConformerEncoderLayerBlock``) at
``veomni/models/transformers/pangu_omni_v2/modeling_huanyu_audio_encoder.py``
matches the Pangu reference at
``/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model/modeling_pangu_omni.py``
lines 66-620.

Each test instantiates both reference and our copy with the same
toy ``OpenPanguOmniAudioConfig``, copies reference state into our
copy via ``load_state_dict``, runs an identical input through both,
and asserts ``max_abs_diff == 0.0`` (bit-for-bit). The audio encoder
is small enough (~163k params at toy config, ~30M at production-tail
24-layer config) that we can afford a full fwd parity per test.

A ``slow``-marked shape-smoke at the end uses the real 30B-A2B audio
sub-config from ``config.json`` (``d_model=768``, ``encoder_layers=24``,
etc.) and checks param count + forward shape; full numeric parity at
that scale is covered by the dedicated end-to-end test
``test_pangu_oracle_check.py`` once it's wired for the audio path
(Week 3.6).

NPU-fast-paths (``torch_npu.npu_fusion_attention`` /
``torch_npu.npu_rotary_mul``) are guarded behind ``NPU_ATTN_INFR``
which is False in this CI environment; we exercise the eager / GPU
fallback throughout, which is what the reference falls into on
non-NPU hosts.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
TOLERANCE = 0.0  # bit-for-bit
_REF_MOD_CACHE: tuple[Any, ...] | None = None


def _ours_module():
    """Import VeOmni's audio port. Importing this also triggers VeOmni's
    ``veomni/utils/device.py`` to compute ``IS_NPU_AVAILABLE = False``
    when no real ``torch_npu`` is installed — that must happen BEFORE
    the ``install_pangu_reference_torch_npu_mock()`` call below or
    VeOmni would see the mock and crash on ``torch.npu.config`` access."""
    from veomni.models.transformers.pangu_omni_v2 import (
        modeling_huanyu_audio_encoder as ours,
    )

    return ours


def _audio_config_class():
    from veomni.models.transformers.pangu_omni_v2.configuration_pangu_omni_v2 import (
        OpenPanguOmniAudioConfig,
    )

    return OpenPanguOmniAudioConfig


def _load_reference_audio_module():
    """Load the Pangu reference's ``modeling_pangu_omni.py`` audio
    classes via the transformers trust_remote_code mechanism.

    Important ordering: (1) import our VeOmni copy first so its
    ``IS_NPU_AVAILABLE`` caches as False, (2) install the ``torch_npu``
    sys.modules placeholder for the reference's unconditional
    ``import torch_npu``, (3) load the reference module via
    ``transformers.AutoConfig``. See ``conftest.py`` docstrings on
    ``install_pangu_reference_torch_npu_mock`` for the race-condition
    rationale.
    """
    global _REF_MOD_CACHE
    if _REF_MOD_CACHE is not None:
        return _REF_MOD_CACHE

    if not PANGU_MODEL_DIR.exists():
        pytest.skip(f"Pangu reference model directory not found: {PANGU_MODEL_DIR}")

    _ours_module()  # force VeOmni device-init to run first

    from veomni.models.transformers._pangu_common._test_compat import install_pangu_reference_torch_npu_mock

    install_pangu_reference_torch_npu_mock()

    from transformers import AutoConfig

    AutoConfig.from_pretrained(str(PANGU_MODEL_DIR), trust_remote_code=True)
    pkg_name = "transformers_modules." + PANGU_MODEL_DIR.name
    ref_mod = importlib.import_module(f"{pkg_name}.modeling_pangu_omni")
    _REF_MOD_CACHE = (ref_mod,)
    return _REF_MOD_CACHE


def _toy_audio_config():
    """Smallest config that still exercises every branch.

    - 2 VGG blocks (2x2 pool each → pooling_size=4)
    - n_mels=16, d_model=64, head_dim=8 (heads=8)
    - encoder_conv1d_kernel_size=3 (matches production; with override
      ``padding=(2,)`` the depthwise conv preserves length L→L)
    - 2 conformer layers (enough to exercise the per-layer
      ``layer_dw_conv_mask`` switching between mask=0 and mask=1)
    - audio_merge_size=2 (exercises avg_pooler tail)
    """
    cls = _audio_config_class()
    cfg = cls(
        d_model=64,
        encoder_attention_heads=8,
        num_mel_bins=16,
        encoder_layers=2,
        encoder_ffn_dim=128,
        encoder_conv1d_kernel_size=3,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        vggblock_enc_config="[(8, 3, 2, 2, True, 16), (16, 3, 2, 2, True, 8)]",
        audio_merge_size=2,
    )
    cfg._attn_implementation = "eager"
    cfg.initializer_range = 0.02
    return cfg


def _copy_state(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """Copy ``src.state_dict()`` into ``dst`` without re-allocating
    buffers. Used after parallel instantiation when both objects have
    identical structure."""
    dst.load_state_dict(src.state_dict())


# ---------------------------------------------------------------------------
# 1. HuanyuRotaryEmbedding parity
# ---------------------------------------------------------------------------


def test_huanyu_rotary_embedding_parity():
    """``HuanyuRotaryEmbedding`` builds an ``emb`` buffer of shape
    ``[max_len, 1, 1, head_dim]``. Verify our buffer matches the
    reference value-for-value at construction time, and that
    ``cos()``/``sin()`` views match the encoder's ``select_cos_sin``
    contract."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    ref_rope = ref_mod.HuanyuRotaryEmbedding(head_dim=8, max_position_embeddings=32, base=10000)
    our_rope = ours_mod.HuanyuRotaryEmbedding(head_dim=8, max_position_embeddings=32, base=10000)

    assert ref_rope.emb.shape == our_rope.emb.shape == (32, 1, 1, 8)
    assert (ref_rope.emb - our_rope.emb).abs().max().item() == 0.0
    assert (ref_rope.inv_freq - our_rope.inv_freq).abs().max().item() == 0.0


def test_huanyu_rotary_embedding_grows_for_packed_audio():
    """Packed training batches can exceed the reference's 1024-step cache."""
    ours_mod = _ours_module()
    cfg = _toy_audio_config()
    enc = ours_mod.HuanyuAudioEncoder(cfg).eval()

    enc.rotary_emb = ours_mod.HuanyuRotaryEmbedding(
        head_dim=cfg.d_model // cfg.encoder_attention_heads,
        max_position_embeddings=32,
        base=10000,
    )
    original = enc.rotary_emb.emb.clone()
    position_ids = torch.arange(48)

    cos, sin = enc.select_cos_sin(position_ids)

    assert cos.shape == sin.shape == (48, 1, cfg.d_model // cfg.encoder_attention_heads)
    assert enc.rotary_emb.emb.shape[0] >= 48
    assert torch.equal(enc.rotary_emb.emb[:32], original)


# ---------------------------------------------------------------------------
# 2. VGGBlock parity
# ---------------------------------------------------------------------------


def test_vgg_block_parity():
    """Toy 2x2-pool VGG block: input (1, 16, 32) → output
    (out_channels=8, 8, 16) after one block with 2x2 pool."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    ref_block = ref_mod.VGGBlock(
        in_channels=1,
        out_channels=8,
        conv_kernel_size=3,
        pooling_kernel_size=2,
        num_conv_layers=2,
        layer_norm_dim=16,
        layer_norm=True,
    )
    our_block = ours_mod.VGGBlock(
        in_channels=1,
        out_channels=8,
        conv_kernel_size=3,
        pooling_kernel_size=2,
        num_conv_layers=2,
        layer_norm_dim=16,
        layer_norm=True,
    )
    _copy_state(our_block, ref_block)

    x = torch.randn(1, 16, 32)
    ref_out = ref_block(x.clone(), None)
    our_out = our_block(x.clone(), None)
    assert ref_out.shape == our_out.shape
    diff = (ref_out - our_out).abs().max().item()
    assert diff == TOLERANCE, f"VGGBlock parity FAILED: max_abs_diff={diff}"


# ---------------------------------------------------------------------------
# 3. eager_attention_forward parity (audio variant)
# ---------------------------------------------------------------------------


def test_eager_attention_forward_parity():
    """Audio's ``eager_attention_forward`` is the Whisper variant.
    Distinct from the vision variant in that it doesn't ``repeat_kv``
    (audio uses MHA, not GQA-friendly path). Test on a toy shape and
    verify bit-for-bit match."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    B, H, T, D = 1, 4, 16, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mask = torch.zeros(B, 1, T, T)  # full mask = 0 = no penalty

    class DummyMod:
        training = False

    ref_out, _ = ref_mod.eager_attention_forward(DummyMod(), q, k, v, mask)
    our_out, _ = ours_mod.eager_attention_forward(DummyMod(), q, k, v, mask)
    assert ref_out.shape == our_out.shape
    diff = (ref_out - our_out).abs().max().item()
    assert diff == TOLERANCE, f"eager_attention_forward parity FAILED: max_abs_diff={diff}"


# ---------------------------------------------------------------------------
# 4. HuanyuAttention.eager_atten_mask shape parity
# ---------------------------------------------------------------------------


def test_huanyu_attention_eager_mask_shape():
    """``eager_atten_mask`` builds a windowed block-diagonal boolean
    mask. Verify shape and basic block-diagonal structure: positions
    within the same sample block can attend, cross-sample positions
    cannot."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    cfg = _toy_audio_config()
    ref_attn = ref_mod.HuanyuAttention(cfg).eval()
    our_attn = ours_mod.HuanyuAttention(cfg).eval()

    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
    mask_ref = ref_attn.eager_atten_mask(cu_seqlens, seq_len=16, window=64)
    mask_ours = our_attn.eager_atten_mask(cu_seqlens, seq_len=16, window=64)
    assert mask_ref.shape == mask_ours.shape == (1, 1, 16, 16)
    assert (mask_ref ^ mask_ours).sum().item() == 0, "eager_atten_mask mismatch"
    # Sanity: cross-block (position 0 → position 10) must be masked
    assert not mask_ref[0, 0, 0, 10]


# ---------------------------------------------------------------------------
# 5. HuanyuAttention end-to-end parity
# ---------------------------------------------------------------------------


def test_huanyu_attention_parity():
    """Single-batch ``HuanyuAttention`` forward with rotary + per-sample
    cu_seqlens. Exercises the eager / GPU fallback path (NPU fused
    attention is bypassed because ``NPU_ATTN_INFR=False`` in CI)."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    cfg = _toy_audio_config()
    ref_attn = ref_mod.HuanyuAttention(cfg).eval()
    our_attn = ours_mod.HuanyuAttention(cfg).eval()
    _copy_state(our_attn, ref_attn)

    T = 16
    hidden = torch.randn(T, cfg.d_model)
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
    # build a toy rotary pos emb (cos, sin) tuple matching the audio
    # encoder's select_cos_sin output: shape (T, 1, head_dim)
    rope = ours_mod.HuanyuRotaryEmbedding(
        head_dim=cfg.d_model // cfg.encoder_attention_heads, max_position_embeddings=64
    )
    pos_ids = torch.arange(T)
    cos = rope.emb[: len(pos_ids)].cos().squeeze(-2)
    sin = rope.emb[: len(pos_ids)].sin().squeeze(-2)
    rotary_pos_emb = (cos, sin)

    ref_out = ref_attn(
        hidden_states=hidden.clone(),
        cu_seqlens=cu_seqlens,
        attention_mask=None,
        rotary_pos_emb=rotary_pos_emb,
    )
    our_out = our_attn(
        hidden_states=hidden.clone(),
        cu_seqlens=cu_seqlens,
        attention_mask=None,
        rotary_pos_emb=rotary_pos_emb,
    )
    assert ref_out.shape == our_out.shape
    diff = (ref_out - our_out).abs().max().item()
    assert diff == TOLERANCE, f"HuanyuAttention parity FAILED: max_abs_diff={diff}"


# ---------------------------------------------------------------------------
# 6. ConformerEncoderLayerBlock parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dw_mask", [0.0, 1.0])
def test_conformer_layer_parity(dw_mask):
    """Conformer block forward — macaron1 → attn → conv-module
    (pointwise GLU → depthwise k=3 → pointwise) → macaron2 → final-LN.

    Parametrize the per-layer ``dw_mask`` since the depthwise conv
    branch selects between past-aligned (``mask=1``) and right-aligned
    (``mask=0``) slices — different code paths."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    cfg = _toy_audio_config()
    ref_blk = ref_mod.ConformerEncoderLayerBlock(cfg).eval()
    our_blk = ours_mod.ConformerEncoderLayerBlock(cfg).eval()
    _copy_state(our_blk, ref_blk)

    T = 8
    hidden = torch.randn(T, cfg.d_model)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    rope = ours_mod.HuanyuRotaryEmbedding(
        head_dim=cfg.d_model // cfg.encoder_attention_heads, max_position_embeddings=64
    )
    cos = rope.emb[:T].cos().squeeze(-2)
    sin = rope.emb[:T].sin().squeeze(-2)

    (ref_out,) = ref_blk(hidden.clone(), cu_seqlens, dw_mask, (cos, sin), 0)
    (our_out,) = our_blk(hidden.clone(), cu_seqlens, dw_mask, (cos, sin), 0)
    assert ref_out.shape == our_out.shape
    diff = (ref_out - our_out).abs().max().item()
    assert diff == TOLERANCE, f"Conformer parity FAILED (dw_mask={dw_mask}): max_abs_diff={diff}"


# ---------------------------------------------------------------------------
# 7. End-to-end HuanyuAudioEncoder parity
# ---------------------------------------------------------------------------


def test_huanyu_audio_encoder_parity_balanced():
    """End-to-end audio encoder forward parity with balanced
    ``feature_lens = [16, 16]``. Exercises VGG → linear → 2 conformer
    layers → linear → ln → avg_pooler (audio_merge_size=2) tail."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    torch.manual_seed(0)
    cfg = _toy_audio_config()
    ref_enc = ref_mod.HuanyuAudioEncoder(cfg).eval()
    our_enc = ours_mod.HuanyuAudioEncoder(cfg).eval()
    _copy_state(our_enc, ref_enc)

    input_features = torch.randn(cfg.num_mel_bins, 32)
    feature_lens = torch.tensor([16, 16], dtype=torch.long)
    ref_out, ref_lens = ref_enc(input_features.clone(), feature_lens.clone())
    our_out, our_lens = our_enc(input_features.clone(), feature_lens.clone())

    assert ref_out.last_hidden_state.shape == our_out.last_hidden_state.shape
    assert torch.equal(ref_lens, our_lens)
    diff = (ref_out.last_hidden_state - our_out.last_hidden_state).abs().max().item()
    assert diff == TOLERANCE, f"end-to-end parity FAILED: max_abs_diff={diff}"


def test_huanyu_audio_encoder_parity_variable_length():
    """End-to-end parity with variable-length ``feature_lens``. Stresses
    the per-sample VGG loop, the per-sample depthwise-conv loop in the
    conformer block, and the per-sample avg_pooler loop in the tail."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    torch.manual_seed(1)
    cfg = _toy_audio_config()
    ref_enc = ref_mod.HuanyuAudioEncoder(cfg).eval()
    our_enc = ours_mod.HuanyuAudioEncoder(cfg).eval()
    _copy_state(our_enc, ref_enc)

    input_features = torch.randn(cfg.num_mel_bins, 32)
    feature_lens = torch.tensor([20, 12], dtype=torch.long)
    ref_out, ref_lens = ref_enc(input_features.clone(), feature_lens.clone())
    our_out, our_lens = our_enc(input_features.clone(), feature_lens.clone())

    assert ref_out.last_hidden_state.shape == our_out.last_hidden_state.shape
    assert torch.equal(ref_lens, our_lens)
    diff = (ref_out.last_hidden_state - our_out.last_hidden_state).abs().max().item()
    assert diff == TOLERANCE, f"variable-length parity FAILED: max_abs_diff={diff}"


def test_huanyu_audio_encoder_parity_no_avg_pooler():
    """End-to-end parity with ``audio_merge_size=1`` (no avg_pooler
    tail)."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    torch.manual_seed(2)
    cfg = _toy_audio_config()
    cfg.audio_merge_size = 1
    ref_enc = ref_mod.HuanyuAudioEncoder(cfg).eval()
    our_enc = ours_mod.HuanyuAudioEncoder(cfg).eval()
    _copy_state(our_enc, ref_enc)
    assert our_enc.avg_pooler is None and ref_enc.avg_pooler is None

    input_features = torch.randn(cfg.num_mel_bins, 32)
    feature_lens = torch.tensor([16, 16], dtype=torch.long)
    ref_out, ref_lens = ref_enc(input_features.clone(), feature_lens.clone())
    our_out, our_lens = our_enc(input_features.clone(), feature_lens.clone())

    assert ref_out.last_hidden_state.shape == our_out.last_hidden_state.shape
    assert torch.equal(ref_lens, our_lens)
    diff = (ref_out.last_hidden_state - our_out.last_hidden_state).abs().max().item()
    assert diff == TOLERANCE


# ---------------------------------------------------------------------------
# 8. 30B-A2B shape smoke (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_huanyu_audio_encoder_shape_smoke_30b_a2b():
    """Build the audio encoder at the real 30B-A2B audio sub-config
    (``d_model=768``, ``encoder_layers=24``, etc.) and verify
    construction succeeds, parameter count is within expected
    millions-range, and a small forward pass produces the right
    output shape. Numeric parity at this scale is covered by the
    Week 3.6 oracle path (real ckpt, real audio sample).

    This test is marked slow because it builds a 24-layer Conformer
    encoder (~30M params on CPU) twice (ref + ours). Doable in a
    handful of seconds; the marker just keeps fast unit-test runs
    snappy."""
    (ref_mod,) = _load_reference_audio_module()
    ours_mod = _ours_module()

    cls = _audio_config_class()
    cfg = cls(
        d_model=768,
        encoder_attention_heads=8,
        num_mel_bins=40,
        encoder_layers=24,
        encoder_ffn_dim=3072,
        encoder_conv1d_kernel_size=3,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        vggblock_enc_config="[(64, 3, 2, 2, True, 40), (128, 3, 2, 2, True, 20)]",
        audio_merge_size=2,
    )
    cfg._attn_implementation = "eager"
    cfg.initializer_range = 0.02

    torch.manual_seed(42)
    ref_enc = ref_mod.HuanyuAudioEncoder(cfg).eval()
    our_enc = ours_mod.HuanyuAudioEncoder(cfg).eval()
    _copy_state(our_enc, ref_enc)

    ref_params = sum(p.numel() for p in ref_enc.parameters())
    our_params = sum(p.numel() for p in our_enc.parameters())
    assert ref_params == our_params, (ref_params, our_params)
    # Actual ~370M params for 24-layer d_model=768 ffn=3072 Conformer
    # (dominated by macaron1+macaron2 each at 2*768*3072 ≈ 4.7M per
    # layer × 24 layers × 2 macarons ≈ 226M, plus attn 24*4*768^2 ≈
    # 57M, plus conv module ≈ 31M, plus VGG + linear glue ≈ 1.5M).
    # The total ≈ 0.74 GB at bf16 — adds to vision 0.94 GB and text
    # 60 GB to land at ~62 GB on the 65 GB NPU HBM (still safe).
    assert 300_000_000 < ref_params < 450_000_000, f"unexpected param count: {ref_params}"

    input_features = torch.randn(cfg.num_mel_bins, 32)
    feature_lens = torch.tensor([16, 16], dtype=torch.long)
    ref_out, ref_lens = ref_enc(input_features.clone(), feature_lens.clone())
    our_out, our_lens = our_enc(input_features.clone(), feature_lens.clone())

    assert ref_out.last_hidden_state.shape == our_out.last_hidden_state.shape
    assert torch.equal(ref_lens, our_lens)
    diff = (ref_out.last_hidden_state - our_out.last_hidden_state).abs().max().item()
    assert diff == TOLERANCE, f"30B-A2B shape smoke parity FAILED: max_abs_diff={diff}"
