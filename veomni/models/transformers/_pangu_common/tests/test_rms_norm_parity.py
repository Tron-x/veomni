"""Bit-for-bit parity test: _pangu_common.pangu_rms_norm vs Pangu reference.

Covers both use cases:
1. **Standard RMSNorm** (input_layernorm / pre_mlp_layernorm, dim = hidden_size)
2. **K-norm** (k_layernorm on per-head key projection, dim = head_dim)

Both must produce bit-for-bit identical outputs vs Pangu's
`OpenPanguV2RMSNorm` across float32 / bf16 / fp16.

Run:
    python tests/test_pangu_rms_norm_parity.py
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


def _make_pair(hidden_size: int, eps: float):
    """Construct (ours, ref) RMSNorm instances with identical weights."""
    from veomni.models.transformers._pangu_common import OpenPanguV2RMSNorm

    (ref_mod,) = _load_reference_module()

    torch.manual_seed(0)
    ours = OpenPanguV2RMSNorm(hidden_size=hidden_size, eps=eps)
    ref = ref_mod.OpenPanguV2RMSNorm(hidden_size=hidden_size, eps=eps)

    # Force identical weights — randomize ours, copy to ref.
    with torch.no_grad():
        ours.weight.copy_(torch.randn_like(ours.weight))
        ref.weight.copy_(ours.weight)
    assert torch.equal(ours.weight, ref.weight)
    return ours, ref


def test_standard_rmsnorm_parity() -> None:
    """input_layernorm / pre_mlp_layernorm use-case: dim = hidden_size = 2560."""
    ours, ref = _make_pair(hidden_size=2560, eps=1e-6)

    failures = []
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for B, S in [(1, 8), (2, 64), (4, 1024)]:
            torch.manual_seed(B * 1000 + S)
            x = torch.randn(B, S, 2560, dtype=dtype)
            y_ours = ours(x)
            y_ref = ref(x)
            # Note: output dtype is dictated by weight (fp32) broadcasting with
            # the input — both ours and ref end up with fp32 output even when
            # input is bf16/fp16. The right invariant is ours == ref, not
            # ours.dtype == input.dtype.
            assert y_ours.dtype == y_ref.dtype, (y_ours.dtype, y_ref.dtype)
            assert y_ours.shape == y_ref.shape == (B, S, 2560), (
                y_ours.shape,
                y_ref.shape,
            )
            if not torch.equal(y_ours, y_ref):
                max_diff = (y_ours.float() - y_ref.float()).abs().max().item()
                failures.append(
                    (str(dtype), B, S, max_diff, y_ours[0, 0, :3].float().tolist(), y_ref[0, 0, :3].float().tolist())
                )
    if failures:
        msg_lines = ["standard RMSNorm parity failed for combos:"]
        for d, B, S, md, yo, yr in failures:
            msg_lines.append(f"  dtype={d} B={B} S={S} max_diff={md:.3e} ours[:3]={yo} ref[:3]={yr}")
        raise AssertionError("\n" + "\n".join(msg_lines))
    print("  [PASS] standard RMSNorm (hidden_size=2560) parity across all combos")


def test_k_norm_parity() -> None:
    """K-norm use-case: dim = head_dim = 128, applied per-head to key states.

    Per-head shape of key_states post-projection-and-reshape is
    `[B, num_kv_heads, S, head_dim]`. RMSNorm is applied to the last dim,
    so the test shape matches that contract.
    """
    ours, ref = _make_pair(hidden_size=128, eps=1e-6)

    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(7)
        B, num_kv_heads, S, head_dim = 2, 4, 64, 128
        key_states = torch.randn(B, num_kv_heads, S, head_dim, dtype=dtype)
        y_ours = ours(key_states)
        y_ref = ref(key_states)
        assert torch.equal(y_ours, y_ref), (
            f"K-norm dtype={dtype}: max_diff={(y_ours.float() - y_ref.float()).abs().max()}"
        )
    print("  [PASS] K-norm (per-head, hidden_size=128) parity")


def test_q_a_layernorm_shape() -> None:
    """q_a_layernorm in Pangu uses hidden_size = q_lora_rank (small).

    This config field isn't set in the 30B-A2B model (no LoRA-style attention
    decomposition), but the module must still work at smaller dimensions.
    """
    ours, ref = _make_pair(hidden_size=64, eps=1e-5)
    x = torch.randn(8, 64)
    assert torch.equal(ours(x), ref(x))
    print("  [PASS] small-dim (hidden_size=64) parity")


def test_extra_repr_matches() -> None:
    """Module repr must match for debugging / state-dict introspection."""
    ours, ref = _make_pair(hidden_size=2560, eps=1.23e-7)
    assert ours.extra_repr() == ref.extra_repr(), (ours.extra_repr(), ref.extra_repr())
    print(f"  [PASS] extra_repr = {ours.extra_repr()!r}")


def test_naming_aliases() -> None:
    """Verify PanguRMSNorm alias points to upstream-verbatim OpenPanguV2RMSNorm."""
    from veomni.models.transformers import _pangu_common as pc

    assert pc.PanguRMSNorm is pc.OpenPanguV2RMSNorm
    print("  [PASS] PanguRMSNorm is OpenPanguV2RMSNorm")


if __name__ == "__main__":
    print(f"\n{'=' * 72}")
    print("  _pangu_common.pangu_rms_norm <-> Pangu reference parity")
    print(f"{'=' * 72}\n")

    failed = 0
    for fn in (
        test_standard_rmsnorm_parity,
        test_k_norm_parity,
        test_q_a_layernorm_shape,
        test_extra_repr_matches,
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
            traceback.print_exc(limit=5)
            failed += 1
        print()

    print(f"{'=' * 72}")
    print(f"  Verdict: {'PASS' if failed == 0 else f'FAIL ({failed} test(s))'}")
    print(f"{'=' * 72}\n")
    sys.exit(0 if failed == 0 else 1)
