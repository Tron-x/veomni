"""Bit-for-bit parity test: _pangu_common.pangu_partial_rope vs Pangu reference.

This test loads the Pangu modeling code from the on-disk model dir, instantiates
the reference `OpenPanguV2RotaryEmbedding` + `apply_rotary_pos_emb`, then runs
the same inputs through our `_pangu_common.pangu_partial_rope` port. Outputs
must match bit-for-bit (`torch.equal`).

Phase 1 verbatim-port invariant: any non-zero diff is a port bug.

Run:
    python tests/test_pangu_partial_rope_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")


_REF_MOD_CACHE: tuple | None = None


def _load_reference_module():
    """Load Pangu reference modeling via HF dynamic-module machinery.

    Reusing HF's `transformers_modules.<repo_name>.modeling_openpangu_v2` is
    the only sane way to honor the relative imports inside the Pangu modeling
    files (`from .configuration_openpangu_omni import ...`). We trigger HF's
    AutoConfig with trust_remote_code=True once to populate the namespace,
    then plain-import the modules.
    """
    global _REF_MOD_CACHE
    if _REF_MOD_CACHE is not None:
        return _REF_MOD_CACHE

    from transformers import AutoConfig

    AutoConfig.from_pretrained(str(PANGU_MODEL_DIR), trust_remote_code=True)
    # HF caches modules under transformers_modules.<repo_name_with_underscores>
    pkg_name = "transformers_modules." + PANGU_MODEL_DIR.name
    import importlib as _il

    ref_mod = _il.import_module(f"{pkg_name}.modeling_openpangu_v2")
    cfg_mod = _il.import_module(f"{pkg_name}.configuration_openpangu_v2")
    _REF_MOD_CACHE = (ref_mod, cfg_mod)
    return _REF_MOD_CACHE


class _FakeConfig:
    """Minimum Pangu-like config for rotary test.

    Mirrors the relevant fields from the real `pangu_omini_30ba2_hf_model/
    config.json` so our rotary embedding hits the exact same code paths
    as the reference.
    """

    def __init__(self) -> None:
        # From config.json top-level
        self.hidden_size = 2560
        self.num_attention_heads = 20
        self.head_dim = 128
        self.partial_rotary_factor = 0.25  # rotary_ndims = 32
        self.rope_theta = 400000.0
        self.max_position_embeddings = 32768
        self.rope_interleaved = False
        # The legacy dict form used by Pangu's reference RoPE
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": 400000.0,
            "partial_rotary_factor": 0.25,
        }


def test_inv_freq_parity() -> None:
    ref_mod, _cfg_mod = _load_reference_module()
    from veomni.models.transformers._pangu_common import compute_default_rope_parameters

    cfg = _FakeConfig()
    ours_inv_freq, ours_scaling = compute_default_rope_parameters(cfg, device=torch.device("cpu"))
    ref_inv_freq, ref_scaling = ref_mod.OpenPanguV2RotaryEmbedding.compute_default_rope_parameters(
        cfg, device=torch.device("cpu")
    )
    assert ours_scaling == ref_scaling, (ours_scaling, ref_scaling)
    assert torch.equal(ours_inv_freq, ref_inv_freq), f"inv_freq mismatch: ours={ours_inv_freq}, ref={ref_inv_freq}"
    print(f"  [PASS] inv_freq parity: shape={tuple(ours_inv_freq.shape)}, scaling={ours_scaling}")


def test_rotary_embedding_forward_parity() -> None:
    ref_mod, _ = _load_reference_module()
    from veomni.models.transformers._pangu_common import OpenPanguV2RotaryEmbedding

    cfg = _FakeConfig()
    torch.manual_seed(0)
    ours = OpenPanguV2RotaryEmbedding(cfg, device=torch.device("cpu"))
    ref = ref_mod.OpenPanguV2RotaryEmbedding(cfg, device=torch.device("cpu"))

    # Sanity: same inv_freq registered
    assert torch.equal(ours.inv_freq, ref.inv_freq)

    # Forward with several batch/seq/dtype combos to make sure the float32
    # autocast block + the final dtype cast match.
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for B, S in [(1, 8), (2, 64), (4, 1024)]:
            x = torch.randn(B, S, 2560, dtype=dtype)
            position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S)
            ours_cos, ours_sin = ours(x, position_ids)
            ref_cos, ref_sin = ref(x, position_ids)
            assert ours_cos.dtype == dtype, (ours_cos.dtype, dtype)
            assert ours_cos.shape == ref_cos.shape == (B, S, 32), (
                ours_cos.shape,
                ref_cos.shape,
            )
            assert torch.equal(ours_cos, ref_cos), (
                f"cos mismatch at dtype={dtype} B={B} S={S}: "
                f"max_diff={(ours_cos.float() - ref_cos.float()).abs().max()}"
            )
            assert torch.equal(ours_sin, ref_sin), (
                f"sin mismatch at dtype={dtype} B={B} S={S}: "
                f"max_diff={(ours_sin.float() - ref_sin.float()).abs().max()}"
            )
    print("  [PASS] rotary forward parity across (dtype x B x S) matrix")


def test_apply_partial_rope_parity() -> None:
    ref_mod, _ = _load_reference_module()
    from veomni.models.transformers._pangu_common import apply_partial_rotary_pos_emb

    cfg = _FakeConfig()
    rotary_ndims = int(cfg.head_dim * cfg.partial_rotary_factor)
    head_dim = cfg.head_dim
    rot_emb_ref = ref_mod.OpenPanguV2RotaryEmbedding(cfg, device=torch.device("cpu"))

    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(42)
        B, num_heads, S = 2, 20, 128
        q = torch.randn(B, num_heads, S, head_dim, dtype=dtype)
        k = torch.randn(B, num_heads, S, head_dim, dtype=dtype)
        position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S)
        cos, sin = rot_emb_ref(q, position_ids)

        # Reference: inline split/apply/cat (matches modeling_openpangu_v2.py:493-507)
        q_rot_ref, q_pass_ref = (
            q[..., :rotary_ndims],
            q[..., rotary_ndims:],
        )
        k_rot_ref, k_pass_ref = (
            k[..., :rotary_ndims],
            k[..., rotary_ndims:],
        )
        q_rot_ref, k_rot_ref = ref_mod.apply_rotary_pos_emb(q_rot_ref, k_rot_ref, cos, sin)
        q_ref = torch.cat((q_rot_ref, q_pass_ref), dim=-1)
        k_ref = torch.cat((k_rot_ref, k_pass_ref), dim=-1)

        # Ours: helper does the same split/apply/cat in one call
        q_ours, k_ours = apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_ndims=rotary_ndims)

        assert torch.equal(q_ours, q_ref), (
            f"partial RoPE q mismatch at dtype={dtype}: max_diff={(q_ours.float() - q_ref.float()).abs().max()}"
        )
        assert torch.equal(k_ours, k_ref), (
            f"partial RoPE k mismatch at dtype={dtype}: max_diff={(k_ours.float() - k_ref.float()).abs().max()}"
        )
    print("  [PASS] partial RoPE apply parity across dtypes")


def test_full_rotary_ndims_passthrough() -> None:
    """When rotary_ndims == head_dim, partial RoPE degenerates to standard RoPE."""
    from veomni.models.transformers._pangu_common import (
        apply_partial_rotary_pos_emb,
        apply_rotary_pos_emb,
    )

    torch.manual_seed(7)
    B, num_heads, S, head_dim = 1, 4, 16, 64
    q = torch.randn(B, num_heads, S, head_dim)
    k = torch.randn(B, num_heads, S, head_dim)
    cos = torch.randn(B, S, head_dim)
    sin = torch.randn(B, S, head_dim)

    q_full, k_full = apply_rotary_pos_emb(q, k, cos, sin)
    q_partial, k_partial = apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_ndims=head_dim)
    assert torch.equal(q_full, q_partial)
    assert torch.equal(k_full, k_partial)
    print("  [PASS] rotary_ndims == head_dim short-circuit")


def test_naming_aliases() -> None:
    """Verify family-neutral aliases point to the upstream-verbatim names."""
    from veomni.models.transformers import _pangu_common as pc

    assert pc.PanguRotaryEmbedding is pc.OpenPanguV2RotaryEmbedding
    assert pc.apply_pangu_partial_rope is pc.apply_partial_rotary_pos_emb
    print("  [PASS] PanguRotaryEmbedding is OpenPanguV2RotaryEmbedding")
    print("  [PASS] apply_pangu_partial_rope is apply_partial_rotary_pos_emb")


if __name__ == "__main__":
    print(f"\n{'=' * 72}")
    print("  _pangu_common.pangu_partial_rope <-> Pangu reference parity")
    print(f"{'=' * 72}\n")

    failed = 0
    for fn in (
        test_inv_freq_parity,
        test_rotary_embedding_forward_parity,
        test_apply_partial_rope_parity,
        test_full_rotary_ndims_passthrough,
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
