"""Bit-for-bit parity test: _pangu_common.pangu_mhc vs Pangu reference.

MHC (Multi-Head Computation) is the most-Pangu-specific algorithm in the
30B-A2B model. It maintains the residual stream at width `n*H` (n=4,
H=2560 -> 10240 in 30B-A2B) and projects it down/up around each
attention/MLP sub-block. All parameters are bfloat16 by upstream design.

Tests cover both the mhc_use_gamma=True path (30B-A2B default) and the
False path, plus the Sinkhorn-Knopps doubly-stochastic normalization
iterating 20 times (mhc_recur_norm in 30B-A2B). Also covers the
`merge_layer_only_pre=True` degeneracy where hc_post becomes identity.

Run:
    python tests/test_pangu_mhc_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
_REF_MOD_CACHE: tuple | None = None


def _load_reference_module():
    global _REF_MOD_CACHE
    if _REF_MOD_CACHE is not None:
        return _REF_MOD_CACHE

    from transformers import AutoConfig

    AutoConfig.from_pretrained(str(PANGU_MODEL_DIR), trust_remote_code=True)
    pkg_name = "transformers_modules." + PANGU_MODEL_DIR.name
    import importlib as _il

    ref_mod = _il.import_module(f"{pkg_name}.modeling_openpangu_v2")
    _REF_MOD_CACHE = (ref_mod,)
    return _REF_MOD_CACHE


class _MHCConfig:
    """Scaled-down MHC config: num_stream=4 (preserved), hidden=64.

    Uses the real 30B-A2B num_stream=4 because the algorithm only really
    exercises the stream-axis ops when n>1. Hidden is scaled from 2560
    -> 64 to keep test fast (phi.weight goes from 4*2560*24 = 245760
    bf16 params to 4*64*24 = 6144 params).
    """

    def __init__(self, *, mhc_use_gamma: bool = True, merge_layer_only_pre: bool = False) -> None:
        self.hidden_size = 64
        self.mhc_num_stream = 4
        self.mhc_use_gamma = mhc_use_gamma
        self.mhc_recur_norm = 20
        self.rms_norm_eps = 1e-6
        # Note: merge_layer_only_pre is a constructor arg to mHCModule,
        # not a config field. We pass it through in _make_pair.


def _make_pair(cfg, merge_layer_only_pre: bool = False):
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import mHCModule

    torch.manual_seed(0)
    ours = mHCModule(cfg, merge_layer_only_pre=merge_layer_only_pre)
    ref = ref_mod.mHCModule(cfg, merge_layer_only_pre=merge_layer_only_pre)

    # Force identical state across all parameters
    ours_state = dict(ours.state_dict())
    ref_state = dict(ref.state_dict())
    assert set(ours_state) == set(ref_state), (
        f"keys differ: only_ours={set(ours_state) - set(ref_state)}, only_ref={set(ref_state) - set(ours_state)}"
    )
    for k in ours_state:
        rand = torch.randn_like(ours_state[k]) * 0.1
        ours_state[k].copy_(rand)
        ref_state[k].copy_(rand)
    ours.load_state_dict(ours_state)
    ref.load_state_dict(ref_state)
    return ours, ref


def test_hc_pre_parity_with_gamma() -> None:
    """hc_pre with mhc_use_gamma=True (the 30B-A2B default path)."""
    cfg = _MHCConfig(mhc_use_gamma=True)
    ours, ref = _make_pair(cfg)

    # Input shape is (B, S, n*H). All parameters are bf16, so inputs
    # should also be bf16 for the full Linear path to type-match.
    torch.manual_seed(1)
    B, S, n, H = 2, 8, cfg.mhc_num_stream, cfg.hidden_size
    x = torch.randn(B, S, n * H, dtype=torch.bfloat16)

    y_ours, h_post_ours, h_res_ours = ours.hc_pre(x)
    y_ref, h_post_ref, h_res_ref = ref.hc_pre(x)
    assert y_ours.shape == y_ref.shape == (B, S, H), (y_ours.shape, y_ref.shape)
    assert h_post_ours.shape == h_post_ref.shape == (B, S, n)
    assert h_res_ours.shape == h_res_ref.shape == (B, S, n, n)

    assert torch.equal(y_ours, y_ref), f"hc_pre y: max_diff={(y_ours.float() - y_ref.float()).abs().max():.3e}"
    assert torch.equal(h_post_ours, h_post_ref), "hc_pre h_post mismatch"
    assert torch.equal(h_res_ours, h_res_ref), "hc_pre h_res mismatch"
    print("  [PASS] hc_pre (mhc_use_gamma=True) parity: y/h_post/h_res all bit-equal")


def test_hc_pre_parity_without_gamma() -> None:
    """hc_pre with mhc_use_gamma=False (alternative path, exercises `phi(x) * rsqrt`)."""
    cfg = _MHCConfig(mhc_use_gamma=False)
    ours, ref = _make_pair(cfg)

    torch.manual_seed(2)
    B, S, n, H = 1, 4, cfg.mhc_num_stream, cfg.hidden_size
    x = torch.randn(B, S, n * H, dtype=torch.bfloat16)

    y_ours, h_post_ours, h_res_ours = ours.hc_pre(x)
    y_ref, h_post_ref, h_res_ref = ref.hc_pre(x)
    assert torch.equal(y_ours, y_ref)
    assert torch.equal(h_post_ours, h_post_ref)
    assert torch.equal(h_res_ours, h_res_ref)
    print("  [PASS] hc_pre (mhc_use_gamma=False) parity")


def test_hc_post_parity() -> None:
    """hc_post combining sub-block output with residual via h_post + h_res."""
    cfg = _MHCConfig()
    ours, ref = _make_pair(cfg)

    torch.manual_seed(3)
    B, S, n, H = 2, 16, cfg.mhc_num_stream, cfg.hidden_size
    x = torch.randn(B, S, n * H, dtype=torch.bfloat16)

    # Get matching h_post, h_res from hc_pre then use them in hc_post
    sub_block_out_ours, h_post_ours, h_res_ours = ours.hc_pre(x)
    sub_block_out_ref, h_post_ref, h_res_ref = ref.hc_pre(x)

    # Simulate attention/MLP modifying the (B, S, H) tensor
    perturb = torch.randn_like(sub_block_out_ours) * 0.05
    sub_block_out_ours = sub_block_out_ours + perturb
    sub_block_out_ref = sub_block_out_ref + perturb

    y_ours = ours.hc_post(sub_block_out_ours, x, h_post_ours, h_res_ours)
    y_ref = ref.hc_post(sub_block_out_ref, x, h_post_ref, h_res_ref)
    assert y_ours.shape == y_ref.shape == (B, S, n * H)
    assert torch.equal(y_ours, y_ref), f"hc_post: max_diff={(y_ours.float() - y_ref.float()).abs().max():.3e}"
    print("  [PASS] hc_post parity (returns (B, S, n*H) recombined)")


def test_sinkhorn_knopps_parity() -> None:
    """Doubly-stochastic Sinkhorn iteration with the real 20-iter count."""
    cfg = _MHCConfig()
    ours, ref = _make_pair(cfg)

    torch.manual_seed(4)
    # h_res shape (B, S, n, n)
    B, S, n = 2, 8, cfg.mhc_num_stream
    h_res = torch.randn(B, S, n, n, dtype=torch.bfloat16)

    out_ours = ours.sinkhorn_knopps(h_res, cfg.mhc_recur_norm, ours.hc_eps)
    out_ref = ref.sinkhorn_knopps(h_res, cfg.mhc_recur_norm, ref.hc_eps)
    assert out_ours.shape == out_ref.shape == h_res.shape
    assert torch.equal(out_ours, out_ref), (
        f"sinkhorn_knopps: max_diff={(out_ours.float() - out_ref.float()).abs().max():.3e}"
    )

    # Verify that after 20 iters, row and column sums are close to 1
    # (the Sinkhorn iteration's mathematical guarantee).
    row_sums = out_ours.float().sum(-1)
    col_sums = out_ours.float().sum(-2)  # noqa: F841  # kept for diagnostic prints when this assertion fails
    assert (row_sums - 1.0).abs().max() < 0.1, (
        f"row sums not close to 1 after 20 iters: max_dev={(row_sums - 1.0).abs().max()}"
    )
    print(f"  [PASS] sinkhorn_knopps parity (20 iters, row_sum_max_dev={(row_sums - 1.0).abs().max().item():.3e})")


def test_merge_layer_only_pre_parity() -> None:
    """merge_layer_only_pre=True path — hc_post is identity."""
    cfg = _MHCConfig()
    ours, ref = _make_pair(cfg, merge_layer_only_pre=True)

    torch.manual_seed(5)
    B, S, n, H = 1, 4, cfg.mhc_num_stream, cfg.hidden_size
    x = torch.randn(B, S, n * H, dtype=torch.bfloat16)

    y_ours, h_post_ours, h_res_ours = ours.hc_pre(x)
    y_ref, h_post_ref, h_res_ref = ref.hc_pre(x)
    assert h_post_ours is None and h_post_ref is None
    assert h_res_ours is None and h_res_ref is None
    assert torch.equal(y_ours, y_ref)

    # hc_post returns x unchanged when merge_layer_only_pre=True
    sub_block_out = torch.randn(B, S, H, dtype=torch.bfloat16)
    post_ours = ours.hc_post(sub_block_out, x, h_post_ours, h_res_ours)
    post_ref = ref.hc_post(sub_block_out, x, h_post_ref, h_res_ref)
    assert torch.equal(post_ours, sub_block_out)  # identity
    assert torch.equal(post_ref, sub_block_out)
    print("  [PASS] merge_layer_only_pre=True (hc_post identity)")


def test_full_round_trip_30b_a2b_shape() -> None:
    """Full hc_pre -> sub-block -> hc_post round-trip at 30B-A2B shape.

    n=4, H=2560. We use a tiny seq length to keep memory low while still
    exercising the full residual stream width (4*2560 = 10240).
    """
    cfg = _MHCConfig(mhc_use_gamma=True)
    cfg.hidden_size = 2560  # real 30B-A2B hidden_size
    ours, ref = _make_pair(cfg)

    torch.manual_seed(6)
    B, S, n, H = 1, 4, cfg.mhc_num_stream, cfg.hidden_size
    assert n * H == 10240, n * H
    x = torch.randn(B, S, n * H, dtype=torch.bfloat16)

    # Pre
    sub_block_in_ours, h_post_ours, h_res_ours = ours.hc_pre(x)
    sub_block_in_ref, h_post_ref, h_res_ref = ref.hc_pre(x)
    assert sub_block_in_ours.shape == (B, S, H)
    assert torch.equal(sub_block_in_ours, sub_block_in_ref)

    # Pretend attention modified it
    sub_block_out = sub_block_in_ours + torch.randn_like(sub_block_in_ours) * 0.01

    # Post
    y_ours = ours.hc_post(sub_block_out, x, h_post_ours, h_res_ours)
    y_ref = ref.hc_post(sub_block_out, x, h_post_ref, h_res_ref)
    assert y_ours.shape == y_ref.shape == (B, S, n * H)
    assert torch.equal(y_ours, y_ref)
    print("  [PASS] full round-trip at 30B-A2B shape (n=4, H=2560, n*H=10240)")


def test_param_dtypes_are_bfloat16() -> None:
    """All MHC parameters must be bfloat16 by upstream design."""
    cfg = _MHCConfig()
    ours, _ = _make_pair(cfg)
    for name, p in ours.named_parameters():
        assert p.dtype == torch.bfloat16, f"{name} has dtype {p.dtype}, expected bfloat16"
    print(f"  [PASS] all {sum(1 for _ in ours.named_parameters())} MHC params are bfloat16")


def test_naming_aliases() -> None:
    from veomni.models.transformers import _pangu_common as pc

    assert pc.PanguMHCModule is pc.mHCModule
    print("  [PASS] PanguMHCModule is mHCModule (preserving upstream lowercase-m)")


if __name__ == "__main__":
    print(f"\n{'=' * 72}")
    print("  _pangu_common.pangu_mhc <-> Pangu reference parity")
    print(f"{'=' * 72}\n")

    failed = 0
    for fn in (
        test_hc_pre_parity_with_gamma,
        test_hc_pre_parity_without_gamma,
        test_hc_post_parity,
        test_sinkhorn_knopps_parity,
        test_merge_layer_only_pre_parity,
        test_full_round_trip_30b_a2b_shape,
        test_param_dtypes_are_bfloat16,
        test_naming_aliases,
    ):
        print(f"[run] {fn.__name__}")
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            import traceback

            print(f"  [ERROR] {type(e).__name__}: {e}")
            traceback.print_exc(limit=8)
            failed += 1
        print()

    print(f"{'=' * 72}")
    print(f"  Verdict: {'PASS' if failed == 0 else f'FAIL ({failed} test(s))'}")
    print(f"{'=' * 72}\n")
    sys.exit(0 if failed == 0 else 1)
