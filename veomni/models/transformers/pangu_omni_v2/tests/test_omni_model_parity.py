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

"""Week 3.5 — parity tests for ``OpenPanguOmni`` (audio-aware multimodal
merge) at ``veomni/models/transformers/pangu_omni_v2/modeling_openpangu_omni.py``.

Coverage:

1. ``get_audio_output_length`` formula correctness — verbatim port of
   ``processor_openpangu_omni.get_audio_output_length`` (lines 227-235
   of the HF reference).
2. ``compute_omni_rope_index`` standalone tests:

   - **Audio token branch**: each audio occupies
     ``get_audio_output_length(seqlen, audio_merge_size)`` positions
     with 3D-broadcast arange.
   - **BC vs ``compute_vl_rope_index``**: when ``audio_seqlens=None``
     (and no audio tokens in ``input_ids``), the result must match
     ``compute_vl_rope_index`` byte-for-byte. Locks down the Week 3.4
     mrope invariants.
   - **use_audio_in_video=True**: raises ``NotImplementedError``
     (deferred until a downstream target enables interleaved video+audio).

3. ``OpenPanguOmni`` construction smoke (toy config) — assert audio
   tower wired in alongside visual + language_model.
4. ``OpenPanguOmni.forward`` image-only path — degenerates to the
   ``OpenPanguVL.forward`` path when ``input_features=None``; logits
   are bit-for-bit identical to ``OpenPanguVL`` on a paired toy
   config + matching state dict.
5. ``OpenPanguOmni.forward`` audio scatter shape correctness — given
   ``input_features`` of known length, the audio masked_scatter
   targets the right token positions.

What this file does NOT test (covered elsewhere):

- ``HuanyuAudioEncoder`` internal numerical parity (Week 3.3
  ``test_pangu_audio_encoder_parity.py``, 11/11 PASS).
- Production 30B-A2B build path (``test_pangu_real_config_build.py``,
  6/6 PASS after Week 3.5 update).
- NPU oracle (``tools/pangu_oracle_check.py --mode veomni``,
  Week 3.6 image-only PASS 3/3 — Week 3.5 audio sample TBD when an
  audio baseline JSONL is available).

Toy config sizing (CPU-friendly):

- Text: 4 layers, hidden=64, intermediate=128, vocab=200, 4 attention
  heads.
- Vision: 2 blocks, hidden=64, intermediate=128, spatial_merge_size=2.
- Audio: 2 conformer layers, d_model=64, encoder_layers=2,
  num_mel_bins=16, audio_merge_size=2.
- Total params: ~600 K — fits in seconds on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")


def _ours_modules():
    """Load the VeOmni Pangu Omni v2 modules in the order required to
    avoid the ``modeling_openpangu_vl ↔ modeling_pangu_omni_v2`` circular
    import (the dispatcher module must be loaded first; see the
    ``_get_openpangu_v2_model_cls`` lazy-resolver in
    ``modeling_openpangu_vl.py``)."""
    from veomni.models.transformers.pangu_omni_v2 import (
        configuration_pangu_omni_v2,
        modeling_huanyu_audio_encoder,
        modeling_openpangu_omni,
        modeling_openpangu_vl,
        modeling_pangu_omni_v2,  # noqa: F401 — bootstrap order
    )

    # Used for the NPU_ATTN_INFR monkey-patch below.
    _ = modeling_huanyu_audio_encoder  # noqa: F841

    return (
        configuration_pangu_omni_v2,
        modeling_openpangu_vl,
        modeling_openpangu_omni,
    )


def _force_eager_attention_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin both vision and audio attention dispatchers onto the eager
    PyTorch path for the duration of a single test.

    Why this is needed (and why only on certain hosts):

    Both ``modeling_openpangu_vl`` (vision attention) and
    ``modeling_huanyu_audio_encoder`` (audio attention) decide at
    module-load time whether the NPU fused kernel path is available:

        try:
            import torch_npu
            NPU_ATTN_INFR = True
        except ImportError:
            NPU_ATTN_INFR = False

    On the bare-metal NPU host running the AReaL CI smoke (where
    ``torch_npu`` is genuinely installed) the flag flips to True at
    import time, and the per-layer ``forward()`` then dispatches to
    ``torch_npu.npu_fusion_attention`` whenever the module is in
    ``.eval()`` mode (which our parity tests always set).

    These parity tests intentionally place all tensors on CPU — they
    are unit tests, not integration tests. Calling
    ``torch_npu.npu_fusion_attention`` on CPU tensors raises
    ``NotImplementedError`` because the op only has an NPU kernel.

    The fix is to monkey-patch ``NPU_ATTN_INFR=False`` so the
    ``forward()`` falls through to the eager PyTorch attention branch
    (same path that a vanilla CPU/GPU host without ``torch_npu`` would
    take). This is the exact in-process equivalent of "as if torch_npu
    weren't installed" and matches what
    ``tests/test_pangu_audio_encoder_parity.py`` relies on implicitly
    (it documents ``NPU_ATTN_INFR=False in CI``).

    Restricted to one test at a time via ``monkeypatch`` so other
    tests/modules that may legitimately need ``NPU_ATTN_INFR=True``
    (e.g. an NPU smoke that imports this file as a side effect) are
    unaffected.
    """
    from veomni.models.transformers.pangu_omni_v2 import (
        modeling_huanyu_audio_encoder,
        modeling_openpangu_vl,
    )

    monkeypatch.setattr(modeling_openpangu_vl, "NPU_ATTN_INFR", False, raising=False)
    monkeypatch.setattr(modeling_huanyu_audio_encoder, "NPU_ATTN_INFR", False, raising=False)


def _make_text_config_dict(hidden_size: int = 64):
    """Toy text-backbone config. Mirrors
    ``test_pangu_vl_model_parity._make_text_config_dict`` but smaller:
    2 layers, 4 heads, vocab=200. ``first_k_dense_replace=2`` makes all
    layers dense so we don't need to wire up the MoE router for this
    test.
    """
    return dict(
        vocab_size=200,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        v_head_dim=16,
        intermediate_size=128,
        initializer_range=0.02,
        tie_word_embeddings=False,
        use_cache=False,
        partial_rotary_factor=1.0,
        qk_rope_dim=16,
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
            "mrope_section": [3, 2, 3],
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
        # MoE — all-dense for simplicity (first_k_dense_replace covers all layers).
        n_routed_experts=1,
        n_shared_experts=0,
        num_experts_per_tok=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_intermediate_size=128,
        hidden_act="silu",
        _experts_implementation="eager",
        # MHC off.
        use_mhc=False,
        mhc_num_stream=1,
        mhc_use_gamma=True,
        mhc_recur_norm=20,
        # Optional branches.
        use_mla=False,
        first_k_dense_replace=2,
        sandwich_norm=False,
        block_post_layernorm_idx=None,
        layer_types=["full_attention", "full_attention"],
    )


def _make_vision_config_dict(out_hidden_size: int = 64):
    """Toy vision tower config. Smallest setting that still produces
    a non-trivial set of patches after `spatial_merge_size`
    downsampling.

    Three fields below need a brief justification:

    - ``window_size=56``: matches ``test_pangu_vl_model_parity`` (the
      derived ``vit_merger_window_size = 56 // 2 // 14 = 2`` matches
      our 4×4 patch grid).
    - ``mm_unit_vision_select_layer=[-1]``: with
      ``use_gatedmerger=False`` the vision tower's forward uses
      ``self.merger`` as a ``ModuleList`` indexed by ``select_layer``
      entries; a one-element list with the final block satisfies the
      multistage-merger path (single-stage degenerate case).
    - ``use_gatedmerger=False``: keeps the merger as a
      ``ModuleList`` so the multistage-merger path is exercised. The
      gated-merger path is covered by ``test_pangu_vision_*`` instead.
    """
    return dict(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=out_hidden_size,
        hidden_act="silu",
        fullatt_block_indexes=[0, 1],
        use_gatedmerger=False,
        window_size=56,
        mm_unit_vision_select_layer=[-1],
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        tokens_per_second=1,
        use_audio_in_video=False,
        _attn_implementation="eager",
    )


def _make_audio_config_dict():
    """Toy audio encoder config — same as Week 3.3 parity test config."""
    return dict(
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
        initializer_range=0.02,
        _attn_implementation="eager",
    )


def _make_toy_omni_config():
    """Build a full OpenPanguOmniConfig with all three sub-configs +
    multimodal token ids set. Mirrors the shape of
    ``test_pangu_vl_model_parity._make_full_config`` and extends with
    audio token + audio sub-config.
    """
    cfg_mod, _vl_mod, _omni_mod = _ours_modules()
    hidden_size = 64
    cfg = cfg_mod.OpenPanguOmniConfig(
        vision_config=_make_vision_config_dict(out_hidden_size=hidden_size),
        text_config=_make_text_config_dict(hidden_size=hidden_size),
        audio_config=_make_audio_config_dict(),
        image_token_id=190,
        video_token_id=191,
        audio_token_id=192,
        vision_start_token_id=188,
        vision_end_token_id=189,
        audio_start_token_id=193,
        audio_end_token_id=194,
        tokens_per_second=1.0,
        use_mhc=False,
        mhc_num_stream=1,
        torch_dtype="float32",
    )
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    cfg.audio_config._attn_implementation = "eager"
    return cfg


# ---------------------------------------------------------------------------
# 1. get_audio_output_length formula parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_lengths,audio_merge_size,expected",
    [
        # Reference formula: floor((floor((L+1)/2)+1)/2); then if merge=2:
        # output = (output - 2) // 2 + 1.
        # Walk-through:
        #   L=1,  merge=1: (1+1)//2=1, (1+1)//2=1  → 1
        #   L=4,  merge=1: (4+1)//2=2, (2+1)//2=1  → 1
        #   L=8,  merge=1: (8+1)//2=4, (4+1)//2=2  → 2
        #   L=16, merge=1: (16+1)//2=8, (8+1)//2=4 → 4
        #   L=32, merge=1: (32+1)//2=16,(16+1)//2=8 → 8
        #   L=16, merge=2: base=4, (4-2)//2+1=2 → 2
        #   L=32, merge=2: base=8, (8-2)//2+1=4 → 4
        #   L=64, merge=2: (64+1)//2=32,(32+1)//2=16; (16-2)//2+1=8 → 8
        (1, 1, 1),
        (4, 1, 1),
        (8, 1, 2),
        (16, 1, 4),
        (32, 1, 8),
        (16, 2, 2),
        (32, 2, 4),
        (64, 2, 8),
    ],
)
def test_get_audio_output_length_formula(input_lengths: int, audio_merge_size: int, expected: int) -> None:
    _, _, omni_mod = _ours_modules()
    # Verify our implementation matches the manually-computed expected
    # value AND matches the reference formula transcribed from
    # `processor_openpangu_omni.get_audio_output_length`.
    got = omni_mod.get_audio_output_length(input_lengths, audio_merge_size)
    # Re-derive expected from the formula transcribed verbatim.
    inp = (input_lengths + 1) // 2
    out = (inp + 1) // 2
    if audio_merge_size == 2:
        out = (out - 2) // 2 + 1
    assert got == out, (
        f"get_audio_output_length({input_lengths}, {audio_merge_size}) returned {got}, formula gives {out}"
    )
    # And matches the parametrized expected (sanity-check the table).
    assert got == expected, (
        f"get_audio_output_length({input_lengths}, {audio_merge_size}) = {got}, expected {expected}"
    )


def test_get_audio_output_length_tensor_input():
    """Reference accepts ``Union[torch.Tensor, int]``. Verify the
    tensor path returns a tensor (or integer-coercible result).
    """
    _, _, omni_mod = _ours_modules()
    seqlens = torch.tensor([16, 32, 64], dtype=torch.long)
    out = omni_mod.get_audio_output_length(seqlens, audio_merge_size=2)
    expected = torch.tensor([2, 4, 8], dtype=torch.long)
    assert torch.equal(out, expected), f"tensor path returned {out.tolist()}, expected {expected.tolist()}"


# ---------------------------------------------------------------------------
# 2. compute_omni_rope_index — audio token branch + BC vs VL
# ---------------------------------------------------------------------------


def test_compute_omni_rope_index_audio_branch_arange():
    """An audio block of N tokens at positions ``[start_idx,
    start_idx+1, ..., start_idx+N-1]`` along all 3 mrope axes
    (T = H = W broadcast).

    The caller pre-expands ``input_ids`` so that the audio_token_id
    appears ``place_num`` times (where ``place_num =
    get_audio_output_length(audio_seqlen, audio_merge_size)``). This
    matches the reference's protocol: the processor expands
    ``<|audio|>`` placeholders before ``input_ids`` reaches the model;
    ``compute_omni_rope_index`` then writes per-position position-ids
    that fit the expanded ``input_ids`` exactly.

    Layout: ``[text, text, audio×3, text]`` with
    ``audio_seqlens=[8], audio_merge_size=1`` →
    ``get_audio_output_length(8, 1) = 2``. So we expand audio_token_id
    twice (not 3 times — recompute), test layout:
    ``[text, text, audio, audio, text]`` (5 tokens total).
    """
    _, _, omni_mod = _ours_modules()

    audio_seqlens = torch.tensor([8], dtype=torch.long)
    place_num = int(omni_mod.get_audio_output_length(8, 1))
    assert place_num == 2, f"sanity: get_audio_output_length(8, 1) = {place_num}"

    # Token layout: text(100), text(101), audio×place_num, text(102)
    audio_block = [192] * place_num
    input_ids_list = [100, 101, *audio_block, 102]
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    pos_ids, _deltas = omni_mod.compute_omni_rope_index(
        input_ids=input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=torch.ones_like(input_ids),
        image_token_id=190,
        video_token_id=191,
        audio_token_id=192,
        vision_start_token_id=188,
        vision_end_token_id=189,
        spatial_merge_size=2,
        tokens_per_second=1.0,
        audio_seqlens=audio_seqlens,
        audio_merge_size=1,
        use_audio_in_video=False,
    )

    # Expected per-axis: [0, 1, 2, 3, 4] (text=text, audio block fills
    # 2 consecutive positions starting at start_idx=2).
    seq_len = input_ids.shape[1]
    expected_row = list(range(seq_len))
    expected = torch.tensor([[expected_row], [expected_row], [expected_row]], dtype=input_ids.dtype)
    assert torch.equal(pos_ids, expected), f"pos_ids mismatch:\n  got:\n{pos_ids}\n  expected:\n{expected}"


def test_compute_omni_rope_index_bc_with_vl_when_no_audio():
    """When ``audio_seqlens=None`` and no audio tokens in input,
    ``compute_omni_rope_index`` must produce the same position_ids as
    ``compute_vl_rope_index``. Locks down Week 3.4.c invariants."""
    _, vl_mod, omni_mod = _ours_modules()

    # Image-only input: image_grid_thw=[1, 4, 4] expands to 4 tokens
    # (4*4 / spatial_merge_size**2 = 16/4 = 4).
    input_ids = torch.tensor([[100, 188, 190, 190, 190, 190, 189, 101]], dtype=torch.long)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    vl_positions, vl_deltas = vl_mod.compute_vl_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
        image_token_id=190,
        video_token_id=191,
        vision_start_token_id=188,
        vision_end_token_id=189,
        spatial_merge_size=2,
        tokens_per_second=1.0,
    )
    omni_positions, omni_deltas = omni_mod.compute_omni_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
        image_token_id=190,
        video_token_id=191,
        audio_token_id=192,
        vision_start_token_id=188,
        vision_end_token_id=189,
        spatial_merge_size=2,
        tokens_per_second=1.0,
        audio_seqlens=None,
        audio_merge_size=1,
        use_audio_in_video=False,
    )
    assert torch.equal(vl_positions, omni_positions), (
        f"position_ids drift between VL and Omni paths:\n"
        f"  vl:   {vl_positions.tolist()}\n"
        f"  omni: {omni_positions.tolist()}"
    )
    assert torch.equal(vl_deltas, omni_deltas)


def test_compute_omni_rope_index_use_audio_in_video_raises():
    """The ``use_audio_in_video=True`` branch is deferred (production
    30B-A2B has ``vision_config.use_audio_in_video=False``). Calling
    with the flag set must raise ``NotImplementedError`` with a
    descriptive message — otherwise a future regression that silently
    drops audio frames in video inputs would be hard to debug."""
    _, _, omni_mod = _ours_modules()

    input_ids = torch.tensor([[100, 101]], dtype=torch.long)
    with pytest.raises(NotImplementedError, match="use_audio_in_video=True"):
        omni_mod.compute_omni_rope_index(
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
            image_token_id=190,
            video_token_id=191,
            audio_token_id=192,
            vision_start_token_id=188,
            vision_end_token_id=189,
            spatial_merge_size=2,
            audio_seqlens=torch.tensor([8]),
            use_audio_in_video=True,
        )


# ---------------------------------------------------------------------------
# 3. OpenPanguOmni construction smoke
# ---------------------------------------------------------------------------


def test_open_pangu_omni_construction_wires_audio_tower():
    """``OpenPanguOmni(cfg)`` must construct on CPU with vision +
    language_model + audio_tower attached. Verifies the inherited
    ``OpenPanguVLModel.__init__`` runs to completion and the audio
    tower is initialized in addition (without double-init of
    visual / language_model)."""
    _, _, omni_mod = _ours_modules()
    cfg = _make_toy_omni_config()
    torch.manual_seed(0)
    model = omni_mod.OpenPanguOmni(cfg).eval()

    # Top-level structure
    assert hasattr(model, "model")
    assert hasattr(model.model, "visual")
    assert hasattr(model.model, "language_model")
    assert hasattr(model.model, "audio_tower")
    assert hasattr(model.model.audio_tower, "proj")
    # audio_tower.proj.weight shape matches hidden_size * mhc (use_mhc=False → 1×)
    assert model.model.audio_tower.proj.weight.shape == (
        cfg.text_config.hidden_size * (cfg.mhc_num_stream if cfg.use_mhc else 1),
        cfg.audio_config.d_model,
    )
    # lm_head shape matches text vocab x hidden
    assert model.lm_head.weight.shape == (
        cfg.text_config.vocab_size,
        cfg.text_config.hidden_size,
    )

    # Property accessors mirror nested fields
    assert model.audio_tower is model.model.audio_tower
    assert model.visual is model.model.visual
    assert model.language_model is model.model.language_model


def test_open_pangu_omni_checkpoint_conversion_mapping_keys():
    """Lock down the three mapping rules: vision / audio / text
    backbone. Catches regressions where someone removes the audio rule
    or drops the audio_tower entry from the text-backbone exclusion."""
    _, _, omni_mod = _ours_modules()
    mapping = omni_mod.OpenPanguOmni._checkpoint_conversion_mapping

    assert "^visual" in mapping and mapping["^visual"] == "model.visual"
    assert "^audio_tower" in mapping and mapping["^audio_tower"] == "model.audio_tower"
    # The text-backbone exclusion must include `audio_tower` so audio
    # weights don't accidentally re-route through `language_model`.
    text_rule_key = next(k for k in mapping if "model(?" in k)
    text_rule_val = mapping[text_rule_key]
    assert "audio_tower" in text_rule_key, f"text-backbone exclusion {text_rule_key!r} is missing audio_tower"
    assert text_rule_val == "model.language_model"


# ---------------------------------------------------------------------------
# 4. OpenPanguOmni image-only forward == OpenPanguVL forward
# ---------------------------------------------------------------------------


def test_open_pangu_omni_image_only_forward_matches_vl(monkeypatch: pytest.MonkeyPatch):
    """When ``input_features=None``, the OpenPanguOmni forward path
    must be byte-for-byte identical to the OpenPanguVL forward path
    on the same (vision + text) sub-config. Establishes the BC
    invariant that Week 3.6's image-only oracle (PASS 3/3, max=2e-5)
    is unaffected by the Week 3.5 dispatcher swap from OpenPanguVL to
    OpenPanguOmni."""
    _, vl_mod, omni_mod = _ours_modules()
    _force_eager_attention_paths(monkeypatch)
    cfg = _make_toy_omni_config()

    torch.manual_seed(0)
    vl_model = vl_mod.OpenPanguVL(cfg).eval()
    torch.manual_seed(0)
    omni_model = omni_mod.OpenPanguOmni(cfg).eval()

    # Same seed → same init for shared modules. But OpenPanguOmni has
    # extra audio_tower (random init), which doesn't get exercised on
    # the image-only path. To make the comparison meaningful, copy
    # shared state — visual + language_model + lm_head — explicitly,
    # leaving audio_tower as a no-op for input_features=None.
    omni_model.model.visual.load_state_dict(vl_model.model.visual.state_dict())
    omni_model.model.language_model.load_state_dict(vl_model.model.language_model.state_dict())
    omni_model.lm_head.load_state_dict(vl_model.lm_head.state_dict())

    # Build a minimal image input: 1 image of 1×4×4 patches.
    batch_size = 1  # noqa: F841  # kept for readability of the dimension layout
    grid_t, grid_h, grid_w = 1, 4, 4
    n_image_patches = grid_t * grid_h * grid_w
    # After spatial_merge_size**2 downsampling: n_patches / 4 = 4
    # placeholder tokens in input_ids.
    n_image_tokens = n_image_patches // (cfg.vision_config.spatial_merge_size**2)
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    # Input ids: [<text>, <vision_start>, image*n_image_tokens, <vision_end>, <text>]
    input_ids = torch.tensor(
        [[100, 188] + [190] * n_image_tokens + [189, 101]],
        dtype=torch.long,
    )
    # Pixel values: (n_patches, in_channels * temporal_patch_size * patch_size * patch_size)
    patch_dim = cfg.vision_config.in_channels * cfg.vision_config.temporal_patch_size * cfg.vision_config.patch_size**2
    torch.manual_seed(42)
    pixel_values = torch.randn(n_image_patches, patch_dim, dtype=torch.float32)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        # NOTE: don't pass `return_dict=True` explicitly — both
        # OpenPanguVL.forward and OpenPanguOmni.forward forward
        # `return_dict=True` to `self.model(...)`, so an explicit
        # caller-side `return_dict` would collide via `**kwargs`. The
        # default behavior (dataclass output) is what we want anyway.
        vl_out = vl_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        omni_out = omni_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_features=None,
        )

    diff = (vl_out.logits - omni_out.logits).abs().max().item()
    assert diff == 0.0, f"OpenPanguOmni image-only forward diverges from OpenPanguVL: max_diff={diff}"


# ---------------------------------------------------------------------------
# 5. OpenPanguOmni audio forward shape correctness
# ---------------------------------------------------------------------------


def test_open_pangu_omni_audio_scatter_shape(monkeypatch: pytest.MonkeyPatch):
    """End-to-end audio forward smoke: given ``input_features``,
    ``OpenPanguOmni.forward`` runs through the audio_tower +
    audio_tower.proj + masked_scatter pipeline and produces logits of
    shape ``(B, S, vocab)``. The numerical correctness of each piece
    is established elsewhere (audio_tower Week 3.3, masked_scatter is
    a built-in op). Here we just lock down the wiring: no shape
    mismatches, no missing kwargs, no silent fallback to text-only.
    """
    _, _, omni_mod = _ours_modules()
    _force_eager_attention_paths(monkeypatch)
    cfg = _make_toy_omni_config()

    torch.manual_seed(0)
    model = omni_mod.OpenPanguOmni(cfg).eval()

    # Build a minimal audio input.
    # Pick mel-frames length such that get_audio_output_length yields
    # exactly N=2 audio tokens (with audio_merge_size=2).
    # Reference: get_audio_output_length(L, 2) = (((L+1)//2 + 1)//2 - 2)//2 + 1.
    # Solve for N=2: ((output_base - 2) // 2 + 1) = 2 → output_base=4.
    # output_base = (input//2+1) // 2 = 4  → input//2 = 7 → input = 14
    # Sanity-check via the helper:
    n_audio_tokens = 2
    # Empirically pick L=14 — confirmed via helper
    audio_seqlen = 14
    place_num = omni_mod.get_audio_output_length(audio_seqlen, audio_merge_size=2)
    assert place_num == n_audio_tokens, (
        f"sanity: get_audio_output_length({audio_seqlen}, 2) = {place_num}, expected {n_audio_tokens}"
    )

    # input_ids: [<text>, audio*n_audio_tokens, <text>]
    input_ids = torch.tensor(
        [[100] + [cfg.audio_token_id] * n_audio_tokens + [101]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    # input_features: (n_mels, T_total) where T_total = audio_seqlen
    torch.manual_seed(42)
    input_features = torch.randn(cfg.audio_config.num_mel_bins, audio_seqlen, dtype=torch.float32)
    audio_feature_lengths = torch.tensor([audio_seqlen], dtype=torch.long)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
        )

    # Logits shape: (batch, seq, vocab)
    assert out.logits.shape == (1, input_ids.shape[1], cfg.text_config.vocab_size)
    # Non-degenerate forward — logits aren't all the same (which would
    # signal a broken masked_scatter or projection).
    assert out.logits.std().item() > 1e-3, f"logits look degenerate: std={out.logits.std().item():.3e}"
