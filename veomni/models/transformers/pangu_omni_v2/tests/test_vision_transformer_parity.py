"""Bit-for-bit parity tests for Pangu Omni v2 vision-tower block + transformer.

Week 3.2.b coverage:

- `OpenPanguVLVisionAttention` — multi-head self-attention with vision RoPE
  + cu_seqlens. GPU eager path (NPU fused path is dead code on CUDA).
- `OpenPanguVLVisionBlock` — pre-LN attention + MLP (GradientCheckpointing).
- `OpenPanguVisionTransformerPretrainedModel` — full ViT tower with patch
  embed, window attention, merger, reverse-window permutation.
- 30B-A2B production-shape smoke — instantiate with real `vision_config`
  fields (depth=26, hidden=1280, out=10240) and run a forward on a tiny
  image grid. Checks that param shapes / forward shapes are correct;
  numerical parity isn't asserted at production scale (it's exercised at
  toy scale in the parity tests above).

Reference-vs-ours comparison runs on `_attn_implementation="eager"` so the
softmax is in pure PyTorch — FA2 / SDPA backends are not bit-for-bit
equivalent to eager.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
_REF_MOD_CACHE: tuple | None = None


def _ours_module():
    """Import our VeOmni-side vision modeling module (lock NPU=False first)."""
    from veomni.models.transformers.pangu_omni_v2 import modeling_openpangu_vl as ours

    return ours


def _load_reference_vl_module():
    """Import the Pangu reference vision module after VeOmni init.

    Same ordering rule as `test_pangu_vision_primitives_parity.py`:

    1. Import our adapter to lock `IS_NPU_AVAILABLE=False` in VeOmni.
    2. Install torch_npu sys.modules placeholder.
    3. Trigger `AutoConfig.from_pretrained` + import reference module.
    """
    global _REF_MOD_CACHE
    if _REF_MOD_CACHE is not None:
        return _REF_MOD_CACHE

    _ours_module()

    from veomni.models.transformers._pangu_common._test_compat import install_pangu_reference_torch_npu_mock

    install_pangu_reference_torch_npu_mock()

    from transformers import AutoConfig

    AutoConfig.from_pretrained(str(PANGU_MODEL_DIR), trust_remote_code=True)
    pkg_name = "transformers_modules." + PANGU_MODEL_DIR.name
    import importlib

    ref_mod = importlib.import_module(f"{pkg_name}.modeling_openpangu_vl")
    _REF_MOD_CACHE = (ref_mod,)
    return _REF_MOD_CACHE


def _make_vision_config(
    hidden_size: int = 64,
    intermediate_size: int = 128,
    hidden_act: str = "gelu",
    num_heads: int = 4,
    depth: int = 4,
    in_channels: int = 3,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
    window_size: int = 56,  # → vit_merger_window_size = 56//2//14 = 2
    use_gatedmerger: bool = True,
    out_hidden_size: int = 96,
    fullatt_block_indexes: list[int] | None = None,
    mm_unit_vision_select_layer: list[int] | None = None,
):
    """Toy vision config carrying just what the tower needs to instantiate.

    Uses `SimpleNamespace`-like attribute access. We do NOT use the real
    `OpenPanguOmniVisionConfig` PretrainedConfig class here because that
    pulls in default field values via PretrainedConfig.__init__ that
    interfere with `getattr(config, 'mm_unit_vision_select_layer', ...)`
    fallback — the field becomes set rather than absent.
    """
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        num_heads=num_heads,
        depth=depth,
        in_channels=in_channels,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
        use_gatedmerger=use_gatedmerger,
        out_hidden_size=out_hidden_size,
        fullatt_block_indexes=fullatt_block_indexes if fullatt_block_indexes is not None else [1, 3],
        mm_unit_vision_select_layer=mm_unit_vision_select_layer
        if mm_unit_vision_select_layer is not None
        else [-1, -3],
        _attn_implementation="eager",
    )


def _copy_weights(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    sd = src.state_dict()
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    assert not missing, f"Missing: {missing}"
    assert not unexpected, f"Unexpected: {unexpected}"


# ---------------------------------------------------------------------------
# Vision attention
# ---------------------------------------------------------------------------


def test_open_pangu_vl_vision_attention_parity() -> None:
    """Single-image attention with two cu_seqlens segments (split image)."""
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    cfg = _make_vision_config(hidden_size=32, num_heads=4)
    ref = ref_mod.OpenPanguVLVisionAttention(cfg).eval()
    mine = ours.OpenPanguVLVisionAttention(cfg).eval()
    _copy_weights(mine, ref)

    seq_len = 16
    head_dim = cfg.hidden_size // cfg.num_heads
    hidden_states = torch.randn(seq_len, cfg.hidden_size)

    # cu_seqlens splits the sequence into two equal segments (block-diag attention).
    cu_seqlens = torch.tensor([0, seq_len // 2, seq_len], dtype=torch.int32)
    cos = torch.randn(seq_len, head_dim)
    sin = torch.randn(seq_len, head_dim)
    position_embeddings = (cos, sin)

    # Build the same _attn_mask that VisionTransformer would for eager path.
    attention_mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(hidden_states.dtype).min)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    with torch.no_grad():
        out_ref = ref(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        out_mine = mine(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
    assert out_ref.shape == out_mine.shape == (seq_len, cfg.hidden_size)
    assert torch.equal(out_ref, out_mine), f"VisionAttention mismatch, max_diff={(out_ref - out_mine).abs().max()}"


# ---------------------------------------------------------------------------
# Vision block
# ---------------------------------------------------------------------------


def test_open_pangu_vl_vision_block_parity() -> None:
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    cfg = _make_vision_config(hidden_size=32, num_heads=4, intermediate_size=64)
    ref = ref_mod.OpenPanguVLVisionBlock(cfg).eval()
    mine = ours.OpenPanguVLVisionBlock(cfg).eval()
    _copy_weights(mine, ref)

    seq_len = 16
    head_dim = cfg.hidden_size // cfg.num_heads
    hidden_states = torch.randn(seq_len, cfg.hidden_size)

    cu_seqlens = torch.tensor([0, seq_len // 2, seq_len], dtype=torch.int32)
    cos = torch.randn(seq_len, head_dim)
    sin = torch.randn(seq_len, head_dim)

    attention_mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(hidden_states.dtype).min)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    with torch.no_grad():
        out_ref = ref(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
        )
        out_mine = mine(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
        )
    assert torch.equal(out_ref, out_mine), f"VisionBlock mismatch, max_diff={(out_ref - out_mine).abs().max()}"


# ---------------------------------------------------------------------------
# Vision transformer (full tower) — toy and 30B-A2B shape smoke
# ---------------------------------------------------------------------------


def _make_vision_pretrained_config(toy: dict):
    """Build a real `OpenPanguOmniVisionConfig` for PreTrainedModel-style instantiation.

    `OpenPanguVisionTransformerPretrainedModel` inherits PreTrainedModel,
    which in `__init__` calls `super().__init__(config, *inputs, **kwargs)`
    that runs HF's `PreTrainedModel.__init__` — that one expects a real
    `PretrainedConfig` instance, not a SimpleNamespace.
    """
    from veomni.models.transformers.pangu_omni_v2.configuration_pangu_omni_v2 import (
        OpenPanguOmniVisionConfig,
    )

    cfg = OpenPanguOmniVisionConfig(**toy)
    return cfg


def _make_grid_thw(t: int = 1, h: int = 4, w: int = 4) -> torch.Tensor:
    """Single-image grid with (t, h, w). Total tokens = t*h*w."""
    return torch.tensor([[t, h, w]], dtype=torch.int64)


def _make_pixel_tokens(
    grid_thw: torch.Tensor, in_channels: int, patch_size: int, temporal_patch_size: int
) -> torch.Tensor:
    """Build a synthetic flattened-patch tensor matching PatchEmbed.input_size.

    Shape: `(num_patches, patch_size**2 * in_channels * temporal_patch_size)`.
    """
    num_patches = int(grid_thw.prod(dim=-1).sum().item())
    input_size = patch_size * patch_size * in_channels * temporal_patch_size
    return torch.randn(num_patches, input_size)


def test_open_pangu_vision_transformer_parity_gated() -> None:
    """End-to-end ViT parity at toy scale with `use_gatedmerger=True`."""
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    toy = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "hidden_act": "gelu",
        "num_heads": 4,
        "depth": 4,
        "in_channels": 3,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "window_size": 56,
        "use_gatedmerger": True,
        "out_hidden_size": 96,
        "fullatt_block_indexes": [1, 3],
        "mm_unit_vision_select_layer": [-1, -3],
        "_attn_implementation": "eager",
    }
    cfg = _make_vision_pretrained_config(toy)

    ref = ref_mod.OpenPanguVisionTransformerPretrainedModel(cfg).eval()
    mine = ours.OpenPanguVisionTransformerPretrainedModel(cfg).eval()
    _copy_weights(mine, ref)

    # 4x4 spatial grid → 16 patches → spatial_merge 2 → 4 output tokens.
    grid_thw = _make_grid_thw(t=1, h=4, w=4)
    pixels = _make_pixel_tokens(grid_thw, in_channels=3, patch_size=14, temporal_patch_size=2)

    with torch.no_grad():
        out_ref = ref(pixels, grid_thw=grid_thw)
        out_mine = mine(pixels, grid_thw=grid_thw)
    n_output_tokens = pixels.shape[0] // (cfg.spatial_merge_size**2)
    assert out_ref.shape == out_mine.shape == (n_output_tokens, cfg.out_hidden_size)
    assert torch.equal(out_ref, out_mine), (
        f"VisionTransformer[gated] mismatch, max_diff={(out_ref - out_mine).abs().max()}"
    )


def test_open_pangu_vision_transformer_parity_multistage() -> None:
    """End-to-end ViT parity with `use_gatedmerger=False` (multi-stage fusion)."""
    torch.manual_seed(1)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    toy = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "hidden_act": "gelu",
        "num_heads": 4,
        "depth": 4,
        "in_channels": 3,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "window_size": 56,
        "use_gatedmerger": False,
        "out_hidden_size": 96,
        "fullatt_block_indexes": [1, 3],
        "mm_unit_vision_select_layer": [-1, -3],
        "_attn_implementation": "eager",
    }
    cfg = _make_vision_pretrained_config(toy)

    ref = ref_mod.OpenPanguVisionTransformerPretrainedModel(cfg).eval()
    mine = ours.OpenPanguVisionTransformerPretrainedModel(cfg).eval()
    _copy_weights(mine, ref)

    grid_thw = _make_grid_thw(t=1, h=4, w=4)
    pixels = _make_pixel_tokens(grid_thw, in_channels=3, patch_size=14, temporal_patch_size=2)

    with torch.no_grad():
        out_ref = ref(pixels, grid_thw=grid_thw)
        out_mine = mine(pixels, grid_thw=grid_thw)
    assert torch.equal(out_ref, out_mine), (
        f"VisionTransformer[multistage] mismatch, max_diff={(out_ref - out_mine).abs().max()}"
    )


@pytest.mark.slow
def test_open_pangu_vision_transformer_shape_smoke_30b_a2b() -> None:
    """30B-A2B production shape smoke — verify param + forward shapes only.

    Uses real `config.json::vision_config` values:
        depth=26, hidden_size=1280, intermediate_size=3840, num_heads=16,
        in_channels=3, patch_size=14, temporal_patch_size=2,
        spatial_merge_size=2, window_size=112, fullatt_block_indexes=[5, 12, 19, 25],
        use_gatedmerger=True, out_hidden_size=3584,
        mm_unit_vision_select_layer=[-1].

    Does NOT run the reference — that would crash on transformers 4.57.1
    `auto_docstring` PEP-604 incompatibility (the slow test is OK; we
    just check our adapter's own shapes). On CPU this takes ~1s on a
    4x4 grid.

    Marked `@pytest.mark.slow` because instantiating depth=26 layers
    allocates ~470M params (~2GB fp32 / 1GB bf16); CI default runs may
    skip via the slow marker.
    """
    ours = _ours_module()

    toy = {
        "hidden_size": 1280,
        "intermediate_size": 3840,
        "hidden_act": "gelu",
        "num_heads": 16,
        "depth": 26,
        "in_channels": 3,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "window_size": 112,
        "use_gatedmerger": True,
        "out_hidden_size": 3584,
        "fullatt_block_indexes": [5, 12, 19, 25],
        "mm_unit_vision_select_layer": [-1],
        "_attn_implementation": "eager",
    }
    cfg = _make_vision_pretrained_config(toy)

    mine = ours.OpenPanguVisionTransformerPretrainedModel(cfg)
    mine = mine.to(dtype=torch.bfloat16).eval()

    # Sanity check param count is in the expected range (~470M for
    # depth=26, hidden=1280). If shape changes break this, the test
    # surfaces it cleanly.
    n_params = sum(p.numel() for p in mine.parameters())
    assert 400_000_000 < n_params < 550_000_000, (
        f"30B-A2B vision tower expected ~470M params, got {n_params / 1e6:.1f}M"
    )

    # Forward on a 4x4 grid (16 patches). Production uses much larger
    # grids per image, but 4x4 is enough to verify the forward path
    # without OOM in CPU/CI.
    grid_thw = _make_grid_thw(t=1, h=4, w=4)
    pixels = _make_pixel_tokens(grid_thw, in_channels=3, patch_size=14, temporal_patch_size=2).to(dtype=torch.bfloat16)

    with torch.no_grad():
        out = mine(pixels, grid_thw=grid_thw)
    expected_tokens = pixels.shape[0] // (cfg.spatial_merge_size**2)
    assert out.shape == (expected_tokens, cfg.out_hidden_size), (
        f"expected ({expected_tokens}, {cfg.out_hidden_size}), got {out.shape}"
    )
    # No NaN / Inf in output (sanity)
    assert torch.isfinite(out).all(), "Vision tower output contains NaN/Inf"
