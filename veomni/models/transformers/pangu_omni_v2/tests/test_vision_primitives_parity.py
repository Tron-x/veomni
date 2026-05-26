"""Bit-for-bit parity tests for Pangu Omni v2 vision-tower primitives.

These tests pair each `pangu_omni_v2.modeling_vl` component
with its counterpart in the upstream Pangu reference at
`/mnt/data_3/models/pangu/configs/modeling_openpangu_vl.py`, ensuring
our verbatim port produces identical numerical output for a fixed
random seed.

Coverage (Week 3.2.a):

- `PanguEmbeddedRMSNorm` — variance-only RMSNorm with eps=1e-6
- `OpenPanguVLMLP` — both `hidden_act="silu"` (gated) and `"gelu"` paths
- `OpenPanguVisionPatchEmbed` — Conv3d patch embedding on synthetic video
- `OpenPanguVisionRotaryEmbedding` — 1D rotary inv_freq buffer
- `OpenPanguVLPatchMerger` — both `use_gatedmerger=False` and `=True`
- `rotate_half` / `apply_rotary_pos_emb_vision` — RoPE helpers
- `repeat_kv` / `eager_attention_forward` — attention helpers (used by
  Week 3.2.b VisionAttention; smoke-tested here so any regression
  surfaces before the more complex block lands)

Reference-module compat shims (transformers 4.57.1 ↔ Pangu 5.0+ symbols)
are installed by `tests/conftest.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
_REF_MOD_CACHE: tuple | None = None


def _ours_module():
    """Import our VeOmni-side vision modeling module.

    MUST be called BEFORE `_load_reference_vl_module()` — VeOmni's
    `veomni/utils/device.py` computes `IS_NPU_AVAILABLE` from
    `importlib.util.find_spec("torch_npu")` at module load. We need
    VeOmni to see torch_npu as ABSENT (so `IS_NPU_AVAILABLE=False`),
    THEN install the torch_npu sys.modules placeholder for the
    reference module's import. Conftest's
    `install_pangu_reference_torch_npu_mock` is called inside
    `_load_reference_vl_module()` for exactly this reason.
    """
    from veomni.models.transformers.pangu_omni_v2 import modeling_vl as ours

    return ours


def _load_reference_vl_module():
    """Import the Pangu reference vision modeling module once.

    Order matters here:

    1. Force VeOmni's `IS_NPU_AVAILABLE = False` to be computed FIRST,
       by importing our adapter (which transitively loads
       `veomni.utils.device`). This snapshots torch_npu as unavailable.

    2. Install the torch_npu sys.modules placeholder so the reference's
       unconditional `import torch_npu` at the top of
       `modeling_vl.py` succeeds.

    3. Trigger `AutoConfig.from_pretrained(..., trust_remote_code=True)`
       to materialize the reference modules into the trust_remote_code
       cache and import them.

    Steps 1 and 2 are inverted from naive ordering — without step 1
    happening before step 2, VeOmni's `device.py:32` does
    `torch.npu.config.allow_internal_format = False` which
    AttributeErrors on CPU/CUDA-only torch builds (no `torch.npu`).
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


# ---------------------------------------------------------------------------
# Tiny vision config helper
# ---------------------------------------------------------------------------


def _make_vision_config(
    hidden_size: int = 64,
    intermediate_size: int = 128,
    hidden_act: str = "gelu",
    num_heads: int = 4,
    spatial_merge_size: int = 2,
    use_gatedmerger: bool = True,
    out_hidden_size: int = 96,
) -> SimpleNamespace:
    """Build a SimpleNamespace mimicking `OpenPanguOmniVisionConfig` for tests.

    Vision-side primitives just read attribute-style fields; SimpleNamespace
    keeps the test setup ~5 lines instead of pulling in the full
    PretrainedConfig machinery and re-stating defaults.
    """
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        num_heads=num_heads,
        spatial_merge_size=spatial_merge_size,
        use_gatedmerger=use_gatedmerger,
        out_hidden_size=out_hidden_size,
        _attn_implementation="eager",
    )


def _copy_weights(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """Copy all named params and float buffers from `src` to `dst`.

    Both modules must have identical parameter / buffer naming and
    shapes. Uses `state_dict` round-trip so torch handles dtype/device
    matching for us.
    """
    sd = src.state_dict()
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    assert not missing, f"Missing keys when copying weights: {missing}"
    assert not unexpected, f"Unexpected keys when copying weights: {unexpected}"


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def test_pangu_embedded_rms_norm_parity() -> None:
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    hidden = 64
    ref = ref_mod.PanguEmbeddedRMSNorm(hidden_size=hidden, eps=1e-6).eval()
    mine = ours.PanguEmbeddedRMSNorm(hidden_size=hidden, eps=1e-6).eval()
    _copy_weights(mine, ref)

    x = torch.randn(2, 16, hidden, dtype=torch.float32)
    with torch.no_grad():
        y_ref = ref(x)
        y_mine = mine(x)
    assert torch.equal(y_ref, y_mine), f"RMSNorm mismatch, max_diff={(y_ref - y_mine).abs().max()}"


def test_open_pangu_rms_norm_is_alias_of_embedded() -> None:
    """`OpenPanguRMSNorm` is documented as an alias of `PanguEmbeddedRMSNorm`.

    Confirm via `isinstance` and by checking forward output identity.
    """
    ours = _ours_module()
    assert issubclass(ours.OpenPanguRMSNorm, ours.PanguEmbeddedRMSNorm)

    torch.manual_seed(0)
    hidden = 32
    a = ours.PanguEmbeddedRMSNorm(hidden_size=hidden, eps=1e-6).eval()
    b = ours.OpenPanguRMSNorm(hidden_size=hidden, eps=1e-6).eval()
    _copy_weights(b, a)
    x = torch.randn(4, hidden)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


# ---------------------------------------------------------------------------
# MLP — both silu (gated) and gelu (non-gated) paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden_act", ["silu", "gelu"])
def test_open_pangu_vl_mlp_parity(hidden_act: str) -> None:
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    cfg = _make_vision_config(hidden_size=32, intermediate_size=64, hidden_act=hidden_act)
    ref = ref_mod.OpenPanguVLMLP(cfg, bias=False).eval()
    mine = ours.OpenPanguVLMLP(cfg, bias=False).eval()
    _copy_weights(mine, ref)

    x = torch.randn(4, 8, 32)
    with torch.no_grad():
        y_ref = ref(x)
        y_mine = mine(x)
    assert torch.equal(y_ref, y_mine), f"MLP[{hidden_act}] mismatch, max_diff={(y_ref - y_mine).abs().max()}"


# ---------------------------------------------------------------------------
# Patch embedding (Conv3d)
# ---------------------------------------------------------------------------


def test_open_pangu_vision_patch_embed_parity() -> None:
    """Run Conv3d patch embed on a synthetic video tensor.

    `input_size = patch_size * patch_size * in_channels * temporal_patch_size
    = 14 * 14 * 3 * 2 = 1176`. Reference and ours should produce
    bit-identical projections after weight sharing.
    """
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    patch, t_patch, in_ch, embed = 14, 2, 3, 64
    ref = ref_mod.OpenPanguVisionPatchEmbed(
        patch_size=patch,
        temporal_patch_size=t_patch,
        in_channels=in_ch,
        embed_dim=embed,
    ).eval()
    mine = ours.OpenPanguVisionPatchEmbed(
        patch_size=patch,
        temporal_patch_size=t_patch,
        in_channels=in_ch,
        embed_dim=embed,
    ).eval()
    _copy_weights(mine, ref)

    n_patches = 6
    x = torch.randn(n_patches, patch * patch * in_ch * t_patch)
    with torch.no_grad():
        y_ref = ref(x)
        y_mine = mine(x)
    assert y_ref.shape == y_mine.shape == (n_patches, embed)
    assert torch.equal(y_ref, y_mine), f"PatchEmbed mismatch, max_diff={(y_ref - y_mine).abs().max()}"


# ---------------------------------------------------------------------------
# Rotary
# ---------------------------------------------------------------------------


def test_open_pangu_vision_rotary_embedding_parity() -> None:
    """1D inv_freq buffer should match exactly given same theta and dim."""
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    dim = 32
    ref = ref_mod.OpenPanguVisionRotaryEmbedding(dim=dim, theta=10000.0).eval()
    mine = ours.OpenPanguVisionRotaryEmbedding(dim=dim, theta=10000.0).eval()
    assert torch.equal(ref.inv_freq, mine.inv_freq), "inv_freq buffer mismatch"

    seqlen = 24
    with torch.no_grad():
        f_ref = ref(seqlen)
        f_mine = mine(seqlen)
    assert torch.equal(f_ref, f_mine), f"rotary freqs mismatch, max_diff={(f_ref - f_mine).abs().max()}"


def test_rotate_half_and_apply_rotary_pos_emb_vision_parity() -> None:
    """Smoke-test the two RoPE helper functions.

    They are pure (no params), so we just need to confirm our ports
    produce identical outputs as the reference on the same input
    tensors.
    """
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    x = torch.randn(2, 16, 4, 32)
    assert torch.equal(ref_mod.rotate_half(x), ours.rotate_half(x))

    seq, dim = 16, 32
    q = torch.randn(seq, 4, dim)
    k = torch.randn(seq, 4, dim)
    cos = torch.randn(seq, dim)
    sin = torch.randn(seq, dim)
    qr_ref, kr_ref = ref_mod.apply_rotary_pos_emb_vision(q, k, cos, sin)
    qr_mine, kr_mine = ours.apply_rotary_pos_emb_vision(q, k, cos, sin)
    assert torch.equal(qr_ref, qr_mine)
    assert torch.equal(kr_ref, kr_mine)


# ---------------------------------------------------------------------------
# Patch merger (both gated and vanilla)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_gatedmerger", [False, True])
def test_open_pangu_vl_patch_merger_parity(use_gatedmerger: bool) -> None:
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    dim = 96  # output (text-side) hidden
    context_dim = 64  # vision-side hidden
    spatial_merge_size = 2

    ref = ref_mod.OpenPanguVLPatchMerger(
        dim=dim,
        context_dim=context_dim,
        spatial_merge_size=spatial_merge_size,
        use_gatedmerger=use_gatedmerger,
    ).eval()
    mine = ours.OpenPanguVLPatchMerger(
        dim=dim,
        context_dim=context_dim,
        spatial_merge_size=spatial_merge_size,
        use_gatedmerger=use_gatedmerger,
    ).eval()
    _copy_weights(mine, ref)

    n_input_tokens = 16  # divisible by spatial_merge_size**2 = 4
    x = torch.randn(n_input_tokens, context_dim)
    with torch.no_grad():
        y_ref = ref(x)
        y_mine = mine(x)
    assert y_ref.shape == y_mine.shape == (n_input_tokens // (spatial_merge_size**2), dim)
    assert torch.equal(y_ref, y_mine), (
        f"PatchMerger[gated={use_gatedmerger}] mismatch, max_diff={(y_ref - y_mine).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------


def test_repeat_kv_parity() -> None:
    """Repeat-interleave the KV head axis. n_rep=1 should be a no-op."""
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    x = torch.randn(2, 4, 16, 32)
    for n_rep in (1, 2, 4):
        assert torch.equal(ref_mod.repeat_kv(x, n_rep), ours.repeat_kv(x, n_rep))


def test_eager_attention_forward_parity() -> None:
    """The `module` arg only needs `num_key_value_groups` and `training`."""
    torch.manual_seed(0)
    (ref_mod,) = _load_reference_vl_module()
    ours = _ours_module()

    batch, n_kv_heads, seq, head_dim = 2, 4, 16, 32
    q = torch.randn(batch, n_kv_heads, seq, head_dim)
    k = torch.randn(batch, n_kv_heads, seq, head_dim)
    v = torch.randn(batch, n_kv_heads, seq, head_dim)
    mask = torch.zeros(batch, 1, seq, seq)

    fake_module = SimpleNamespace(num_key_value_groups=1, training=False)
    scaling = head_dim**-0.5

    with torch.no_grad():
        out_ref, _ = ref_mod.eager_attention_forward(fake_module, q, k, v, attention_mask=mask, scaling=scaling)
        out_mine, _ = ours.eager_attention_forward(fake_module, q, k, v, attention_mask=mask, scaling=scaling)
    assert torch.equal(out_ref, out_mine), f"eager_attention mismatch, max_diff={(out_ref - out_mine).abs().max()}"
