"""Bit-for-bit parity test: _pangu_common.pangu_moe vs Pangu reference.

Covers all four MoE building blocks at the 30B-A2B model's actual config
(384 routed experts, 8 active per token, 2 shared, intermediate=256, etc.).

The 30B-A2B model has 384 experts but a parity test instantiating 384
expert weights eats ~6 GB just in fp32 Parameters. We use a **scaled-down**
config (16 experts, top_k=4, hidden=256) for the bulk of the tests and
add one **30B-A2B-shape** smoke test with very few tokens so the parity
is still exercised at the real shape (cost ~1.5 GB peak).

Tests:
1. `OpenPanguV2MLP` (dense + shared-experts shape) — fp32/bf16.
2. `OpenPanguV2TopkRouter` — float32 promotion behavior.
3. `OpenPanguV2Experts.forward` — fused-3D-Parameter routed expert dispatch.
4. `OpenPanguV2SparseMoeBlock.route_tokens_to_experts` — group routing
   + bias correction + norm topk + scaling factor.
5. `OpenPanguV2SparseMoeBlock.forward` — full MoE forward.
6. 30B-A2B-shape smoke for `OpenPanguV2SparseMoeBlock.forward`.
7. Naming aliases.

Run:
    python tests/test_pangu_moe_parity.py
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


class _SmallMoEConfig:
    """Scaled-down MoE config (16 experts, 4 active, hidden=256).

    Keeps the routing / bias / shared-experts / norm-topk-prob / scaling
    paths identical to 30B-A2B, just with smaller numbers so the test
    runs in <100 MB and <1 s.
    """

    def __init__(self) -> None:
        self.hidden_size = 256
        self.moe_intermediate_size = 64
        self.intermediate_size = 512
        self.n_routed_experts = 16
        self.n_shared_experts = 2
        self.num_experts_per_tok = 4  # top_k
        self.topk_group = 1  # n_group = 1 hardcoded in upstream
        self.norm_topk_prob = True
        self.routed_scaling_factor = 2.5
        self.hidden_act = "silu"
        # Reference upstream wraps OpenPanguV2Experts with
        # @use_experts_implementation (transformers 5.0 MoE kernel-swap
        # mechanism). It reads config._experts_implementation at __init__
        # time. None / "eager" both keep the verbatim Python forward path.
        self._experts_implementation = "eager"


class _A2BMoEConfig:
    """Actual 30B-A2B MoE config — used in the smoke test only.

    Real numbers: 384 routed experts, 8 top_k, 2 shared, intermediate=256
    in moe / 6144 in dense, hidden=2560.
    """

    def __init__(self) -> None:
        self.hidden_size = 2560
        self.moe_intermediate_size = 256
        self.intermediate_size = 6144
        self.n_routed_experts = 384
        self.n_shared_experts = 2
        self.num_experts_per_tok = 8
        self.topk_group = 1
        self.norm_topk_prob = True
        self.routed_scaling_factor = 2.5
        self.hidden_act = "silu"
        self._experts_implementation = "eager"


def _make_pair(ref_cls, ours_cls, cfg, *args, **kwargs):
    """Construct (ours, ref) modules with identical state."""
    torch.manual_seed(0)
    ours = ours_cls(cfg, *args, **kwargs)
    ref = ref_cls(cfg, *args, **kwargs)
    ours_state = dict(ours.state_dict())
    ref_state = dict(ref.state_dict())
    common = set(ours_state) & set(ref_state)
    assert common == set(ours_state) == set(ref_state), (
        f"state_dict keys differ:\n  only_ours={set(ours_state) - set(ref_state)}\n"
        f"  only_ref={set(ref_state) - set(ours_state)}"
    )
    # Force same values
    for k in common:
        rand = torch.randn_like(ours_state[k])
        ours_state[k].copy_(rand)
        ref_state[k].copy_(rand)
    ours.load_state_dict(ours_state)
    ref.load_state_dict(ref_state)
    return ours, ref


def test_mlp_parity() -> None:
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import OpenPanguV2MLP

    cfg = _SmallMoEConfig()

    # Test as dense MLP (intermediate_size = config.intermediate_size = 512)
    for dtype in (torch.float32, torch.bfloat16):
        ours, ref = _make_pair(ref_mod.OpenPanguV2MLP, OpenPanguV2MLP, cfg)
        ours = ours.eval().to(dtype=dtype)
        ref = ref.eval().to(dtype=dtype)

        torch.manual_seed(1)
        x = torch.randn(2, 32, cfg.hidden_size, dtype=dtype)
        y_ours = ours(x)
        y_ref = ref(x)
        assert torch.equal(y_ours, y_ref), (
            f"dense MLP dtype={dtype}: max_diff={(y_ours.float() - y_ref.float()).abs().max():.3e}"
        )

    # Test as shared-experts MLP (custom intermediate_size)
    custom_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
    ours, ref = _make_pair(ref_mod.OpenPanguV2MLP, OpenPanguV2MLP, cfg, intermediate_size=custom_inter)
    x = torch.randn(2, 16, cfg.hidden_size)
    assert ours.intermediate_size == custom_inter
    assert torch.equal(ours(x), ref(x))
    print(f"  [PASS] MLP parity (dense + shared-experts shape, inter={custom_inter})")


def test_router_parity() -> None:
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import OpenPanguV2TopkRouter

    cfg = _SmallMoEConfig()
    ours, ref = _make_pair(ref_mod.OpenPanguV2TopkRouter, OpenPanguV2TopkRouter, cfg)

    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(2)
        x = torch.randn(3, 8, cfg.hidden_size, dtype=dtype)
        # Note: router always promotes weights+input to float32 internally —
        # output dtype is float32 regardless of input dtype.
        y_ours = ours(x)
        y_ref = ref(x)
        assert y_ours.dtype == torch.float32 == y_ref.dtype, (y_ours.dtype, y_ref.dtype)
        assert y_ours.shape == y_ref.shape == (3 * 8, cfg.n_routed_experts)
        assert torch.equal(y_ours, y_ref), f"router dtype={dtype}: max_diff={(y_ours - y_ref).abs().max():.3e}"
    print("  [PASS] TopkRouter parity (fp32 promotion preserved)")


def test_experts_forward_parity() -> None:
    """Fused-3D-Parameter routed expert dispatch — the heart of MoE."""
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import OpenPanguV2Experts

    cfg = _SmallMoEConfig()
    ours, ref = _make_pair(ref_mod.OpenPanguV2Experts, OpenPanguV2Experts, cfg)

    for dtype in (torch.float32, torch.bfloat16):
        ours_dt = ours.eval().to(dtype=dtype)
        ref_dt = ref.eval().to(dtype=dtype)

        # Re-copy state after to() (cast may have rounded differently in ours
        # vs ref if order differs)
        with torch.no_grad():
            for name, p in ours_dt.named_parameters():
                dict(ref_dt.named_parameters())[name].copy_(p)

        torch.manual_seed(3)
        n_tokens = 64
        hidden = torch.randn(n_tokens, cfg.hidden_size, dtype=dtype)
        # Random top-k indices and weights (sum-to-1 already not required —
        # the experts just use the given weights as-is)
        top_k_index = torch.randint(0, cfg.n_routed_experts, (n_tokens, cfg.num_experts_per_tok))
        top_k_weights = torch.randn(n_tokens, cfg.num_experts_per_tok, dtype=dtype).abs() + 0.1

        y_ours = ours_dt(hidden, top_k_index, top_k_weights)
        y_ref = ref_dt(hidden, top_k_index, top_k_weights)
        if not torch.equal(y_ours, y_ref):
            raise AssertionError(
                f"experts forward dtype={dtype}: max_diff={(y_ours.float() - y_ref.float()).abs().max():.3e}"
            )
    print("  [PASS] Experts forward parity (fused-3D routed expert dispatch)")


def test_route_tokens_to_experts_parity() -> None:
    """Group routing + bias correction + norm + scaling factor."""
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import OpenPanguV2SparseMoeBlock

    cfg = _SmallMoEConfig()
    ours, ref = _make_pair(ref_mod.OpenPanguV2SparseMoeBlock, OpenPanguV2SparseMoeBlock, cfg)
    # Also randomize the e_score_correction_bias buffer to exercise the bias path
    with torch.no_grad():
        bias = torch.randn(cfg.n_routed_experts) * 0.01
        ours.e_score_correction_bias.copy_(bias)
        ref.e_score_correction_bias.copy_(bias)

    torch.manual_seed(4)
    n_tokens = 32
    router_logits = torch.randn(n_tokens, cfg.n_routed_experts)

    idx_ours, w_ours = ours.route_tokens_to_experts(router_logits)
    idx_ref, w_ref = ref.route_tokens_to_experts(router_logits)
    assert torch.equal(idx_ours, idx_ref), "topk indices mismatch"
    assert torch.equal(w_ours, w_ref), f"topk weights: max_diff={(w_ours - w_ref).abs().max():.3e}"
    print("  [PASS] route_tokens_to_experts parity (group + bias + norm + scale)")


def test_sparse_moe_block_forward_parity() -> None:
    """Full MoE forward at small scale."""
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import OpenPanguV2SparseMoeBlock

    cfg = _SmallMoEConfig()

    for dtype in (torch.float32, torch.bfloat16):
        ours, ref = _make_pair(ref_mod.OpenPanguV2SparseMoeBlock, OpenPanguV2SparseMoeBlock, cfg)
        ours = ours.eval().to(dtype=dtype)
        ref = ref.eval().to(dtype=dtype)
        # Re-sync after cast — RMSNorm-style fp32-trapped params remain
        # fp32 across `.to(bf16)`; experts 3D Parameters convert. Re-copy
        # to make absolutely sure ours and ref agree post-cast.
        with torch.no_grad():
            for name, p in ours.named_parameters():
                dict(ref.named_parameters())[name].copy_(p)
            for name, b in ours.named_buffers():
                dict(ref.named_buffers())[name].copy_(b)

        torch.manual_seed(5)
        x = torch.randn(2, 16, cfg.hidden_size, dtype=dtype)
        y_ours = ours(x)
        y_ref = ref(x)
        assert y_ours.shape == y_ref.shape == x.shape
        if not torch.equal(y_ours, y_ref):
            raise AssertionError(
                f"sparse MoE block dtype={dtype}: "
                f"max_diff={(y_ours.float() - y_ref.float()).abs().max():.3e}, "
                f"ours[0,0,:3]={y_ours[0, 0, :3].float().tolist()}, "
                f"ref[0,0,:3]={y_ref[0, 0, :3].float().tolist()}"
            )
    print("  [PASS] SparseMoeBlock.forward parity (small scale, fp32+bf16)")


def test_sparse_moe_block_30b_a2b_shape_smoke() -> None:
    """One smoke test at actual 30B-A2B shape (384 experts, 8 top_k, hidden=2560).

    Cost: peak ~1.5 GB (384 experts * 2*256*2560 fp32 params *2 modules).
    Tokens=2 so the routed expert dispatch is cheap, but the routing
    matrices still go through the full 384-wide topk path.
    """
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import OpenPanguV2SparseMoeBlock

    cfg = _A2BMoEConfig()
    ours, ref = _make_pair(ref_mod.OpenPanguV2SparseMoeBlock, OpenPanguV2SparseMoeBlock, cfg)

    torch.manual_seed(7)
    x = torch.randn(1, 2, cfg.hidden_size)  # B=1, S=2 → 2 tokens
    y_ours = ours(x)
    y_ref = ref(x)
    if not torch.equal(y_ours, y_ref):
        raise AssertionError(f"30B-A2B shape: max_diff={(y_ours.float() - y_ref.float()).abs().max().item():.3e}")
    print(
        f"  [PASS] SparseMoeBlock forward at 30B-A2B shape "
        f"({cfg.n_routed_experts} experts, top_k={cfg.num_experts_per_tok})"
    )


def test_naming_aliases() -> None:
    from veomni.models.transformers import _pangu_common as pc

    assert pc.PanguMLP is pc.OpenPanguV2MLP
    assert pc.PanguExperts is pc.OpenPanguV2Experts
    assert pc.PanguTopkRouter is pc.OpenPanguV2TopkRouter
    assert pc.PanguSparseMoeBlock is pc.OpenPanguV2SparseMoeBlock
    print("  [PASS] PanguMLP / PanguExperts / PanguTopkRouter / PanguSparseMoeBlock aliases")


if __name__ == "__main__":
    print(f"\n{'=' * 72}")
    print("  _pangu_common.pangu_moe <-> Pangu reference parity")
    print(f"{'=' * 72}\n")

    failed = 0
    for fn in (
        test_mlp_parity,
        test_router_parity,
        test_experts_forward_parity,
        test_route_tokens_to_experts_parity,
        test_sparse_moe_block_forward_parity,
        test_sparse_moe_block_30b_a2b_shape_smoke,
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
