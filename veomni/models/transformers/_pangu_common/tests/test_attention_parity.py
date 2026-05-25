"""Bit-for-bit parity test: _pangu_common.pangu_attention vs Pangu reference.

This is the most-substantive parity test of Week 1 — it integrates K-norm,
partial RoPE, GQA repeat_kv, eager softmax attention, and the o_proj
projection into a single end-to-end forward, then validates that our port
matches the Pangu reference `OpenPanguV2Attention.forward(...)` bit-for-bit
under the 30B-A2B model's actual config shape.

Covers:
1. `eager_attention_forward` standalone parity (any GQA-style attention).
2. `repeat_kv` parity.
3. `OpenPanguV2Attention.forward` end-to-end parity at 30B-A2B shape
   (head_dim=128, num_heads=20, num_kv_heads=4, partial_rotary_factor=0.25).
4. `OpenPanguV2Attention.forward` parity with the small `param_sink_number`
   option turned on (off in 30B-A2B but verbatim port must still work).
5. Name aliases (PanguAttention is OpenPanguV2Attention).

Run:
    python tests/test_pangu_attention_parity.py
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


class _FakeConfig:
    """Minimal Pangu-like config for attention test.

    Mirrors the 30B-A2B model fields needed by `OpenPanguV2Attention.__init__`
    and `.forward`. Optional fields default to off (attn_groupnorm=False,
    attn_elementwise_gate=False, param_sink_number=0, sliding_window=None).
    """

    def __init__(self, *, param_sink_number: int = 0) -> None:
        # From config.json top-level
        self.hidden_size = 2560
        self.num_attention_heads = 20  # head_dim = 128, q_size = 2560
        self.num_key_value_heads = 4  # GQA group = 5
        self.head_dim = 128
        self.v_head_dim = 128
        self.partial_rotary_factor = 0.25  # rotary_ndims = 32
        self.rope_theta = 400000.0
        self.max_position_embeddings = 32768
        self.rope_interleaved = False
        self.rms_norm_eps = 1e-6
        self.attention_dropout = 0.0
        self.attention_bias = False

        # Optional sub-mechanisms (all default off in 30B-A2B except K-norm
        # which is unconditional in the reference modeling code).
        self.attn_groupnorm = False
        self.attn_elementwise_gate = False
        self.param_sink_number = param_sink_number
        self.torch_dtype = torch.float32
        self.layer_types = None
        self.sliding_window = None
        self._attn_implementation = "eager"

        # rope_parameters dict — populated by OpenPanguOmniConfig.__post_init__
        # in the real model; we set it explicitly for the fake config.
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": 400000.0,
            "partial_rotary_factor": 0.25,
        }


def _copy_params(ours: torch.nn.Module, ref: torch.nn.Module) -> None:
    """Force ours and ref to share identical parameter values, parameter-by-parameter.

    Required for bit-for-bit parity — both modules have the same nn.Module
    structure (qkv_proj, k_layernorm, o_proj, optional param_sink_*) so this
    is a simple state_dict copy.
    """
    ours_state = dict(ours.state_dict())
    ref_state = dict(ref.state_dict())
    common = set(ours_state) & set(ref_state)
    only_ours = set(ours_state) - set(ref_state)
    only_ref = set(ref_state) - set(ours_state)
    assert not only_ours, f"params only in ours: {only_ours}"
    assert not only_ref, f"params only in ref: {only_ref}"
    for k in common:
        assert ours_state[k].shape == ref_state[k].shape, (
            k,
            ours_state[k].shape,
            ref_state[k].shape,
        )
    with torch.no_grad():
        for name, p in ours.named_parameters():
            ref_p = dict(ref.named_parameters())[name]
            ref_p.copy_(p)


def test_repeat_kv_parity() -> None:
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import repeat_kv

    for B, n_kv, S, head_dim, n_rep in [
        (1, 4, 16, 128, 1),
        (2, 4, 64, 128, 5),  # 30B-A2B shape: 4 kv heads * 5 = 20 q heads
        (1, 8, 1024, 64, 4),
    ]:
        x = torch.randn(B, n_kv, S, head_dim)
        y_ours = repeat_kv(x, n_rep)
        y_ref = ref_mod.repeat_kv(x, n_rep)
        assert torch.equal(y_ours, y_ref), f"repeat_kv mismatch B={B} n_kv={n_kv} S={S} n_rep={n_rep}"
    print("  [PASS] repeat_kv parity across 3 shape combos")


def test_eager_attention_forward_parity() -> None:
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import eager_attention_forward

    # Build a minimal stand-in module so eager_attention_forward can read
    # `num_key_value_groups` and `training`. We use a real nn.Module so
    # `.training` works correctly.
    class Stub(torch.nn.Module):
        def __init__(self, n_groups: int) -> None:
            super().__init__()
            self.num_key_value_groups = n_groups

    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(1)
        B, n_heads, n_kv, S, head_dim = 2, 20, 4, 64, 128
        n_groups = n_heads // n_kv
        q = torch.randn(B, n_heads, S, head_dim, dtype=dtype)
        k = torch.randn(B, n_kv, S, head_dim, dtype=dtype)
        v = torch.randn(B, n_kv, S, head_dim, dtype=dtype)
        # Standard causal mask (additive, -inf above diagonal)
        mask = torch.zeros(B, 1, S, S, dtype=dtype)
        mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

        module = Stub(n_groups).eval()

        out_ours, w_ours = eager_attention_forward(module, q, k, v, mask, scaling=head_dim**-0.5, dropout=0.0)
        out_ref, w_ref = ref_mod.eager_attention_forward(module, q, k, v, mask, scaling=head_dim**-0.5, dropout=0.0)
        assert torch.equal(out_ours, out_ref), (
            f"eager attn out mismatch dtype={dtype}: max_diff={(out_ours.float() - out_ref.float()).abs().max()}"
        )
        assert torch.equal(w_ours, w_ref), f"eager attn weights mismatch dtype={dtype}"
    print("  [PASS] eager_attention_forward parity (GQA 20:4, causal mask)")


def test_attention_forward_end_to_end_parity() -> None:
    """Full OpenPanguV2Attention.forward at 30B-A2B shape."""
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import (
        OpenPanguV2Attention,
        OpenPanguV2RotaryEmbedding,
    )

    cfg = _FakeConfig()

    for dtype in (torch.float32, torch.bfloat16):
        # Build fresh pair each dtype so weights match input dtype. Real
        # training/inference always casts the whole module to a single dtype;
        # mixing fp32 weights with bf16 inputs is not a supported config.
        torch.manual_seed(0)
        ours = OpenPanguV2Attention(cfg, layer_idx=0).eval().to(dtype=dtype)
        ref = ref_mod.OpenPanguV2Attention(cfg, layer_idx=0).eval().to(dtype=dtype)
        _copy_params(ours, ref)

        rot_emb = OpenPanguV2RotaryEmbedding(cfg, device=torch.device("cpu"))

        for B, S in [(1, 16), (2, 128)]:
            torch.manual_seed(B * 100 + S)
            x = torch.randn(B, S, cfg.hidden_size, dtype=dtype)
            position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S)
            cos, sin = rot_emb(x, position_ids)

            # No causal mask in this test — we want to isolate the
            # K-norm + partial RoPE + GQA path. Causal mask is exercised
            # by test_attention_forward_with_causal_mask_parity below.
            out_ours, w_ours = ours(x, position_embeddings=(cos, sin), attention_mask=None)
            out_ref, w_ref = ref(x, position_embeddings=(cos, sin), attention_mask=None)
            assert out_ours.shape == out_ref.shape == (B, S, cfg.hidden_size), (
                out_ours.shape,
                out_ref.shape,
            )
            if not torch.equal(out_ours, out_ref):
                max_diff = (out_ours.float() - out_ref.float()).abs().max().item()
                raise AssertionError(
                    f"attention output mismatch dtype={dtype} B={B} S={S}: "
                    f"max_diff={max_diff:.3e}, "
                    f"ours[0,0,:3]={out_ours[0, 0, :3].float().tolist()}, "
                    f"ref[0,0,:3]={out_ref[0, 0, :3].float().tolist()}"
                )
    print("  [PASS] end-to-end attention forward parity (30B-A2B shape, no mask)")


def test_attention_forward_with_causal_mask_parity() -> None:
    """Same as above but with a real causal mask passed through."""
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import (
        OpenPanguV2Attention,
        OpenPanguV2RotaryEmbedding,
    )

    cfg = _FakeConfig()

    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        ours = OpenPanguV2Attention(cfg, layer_idx=0).eval().to(dtype=dtype)
        ref = ref_mod.OpenPanguV2Attention(cfg, layer_idx=0).eval().to(dtype=dtype)
        _copy_params(ours, ref)

        rot_emb = OpenPanguV2RotaryEmbedding(cfg, device=torch.device("cpu"))

        torch.manual_seed(7)
        B, S = 2, 64
        x = torch.randn(B, S, cfg.hidden_size, dtype=dtype)
        position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S)
        cos, sin = rot_emb(x, position_ids)

        # Causal mask: additive, -inf above diagonal, broadcast over batch / heads
        mask = torch.zeros(B, 1, S, S, dtype=dtype)
        mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

        out_ours, _ = ours(x, position_embeddings=(cos, sin), attention_mask=mask)
        out_ref, _ = ref(x, position_embeddings=(cos, sin), attention_mask=mask)
        assert torch.equal(out_ours, out_ref), (
            f"attn with mask dtype={dtype}: max_diff={(out_ours.float() - out_ref.float()).abs().max().item():.3e}"
        )
    print("  [PASS] attention forward with causal mask parity")


def test_attention_with_param_sink_parity() -> None:
    """Optional param_sink branch (off in 30B-A2B but verbatim port must work)."""
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import (
        OpenPanguV2Attention,
        OpenPanguV2RotaryEmbedding,
    )

    cfg = _FakeConfig(param_sink_number=4)
    torch.manual_seed(0)
    ours = OpenPanguV2Attention(cfg, layer_idx=0).eval()
    ref = ref_mod.OpenPanguV2Attention(cfg, layer_idx=0).eval()
    # Initialize param_sink_* explicitly — torch.empty leaves garbage.
    with torch.no_grad():
        for name in ["param_sink_key", "param_sink_value"]:
            sink = torch.randn_like(getattr(ours, name))
            getattr(ours, name).copy_(sink)
            getattr(ref, name).copy_(sink)
    _copy_params(ours, ref)

    rot_emb = OpenPanguV2RotaryEmbedding(cfg, device=torch.device("cpu"))

    torch.manual_seed(11)
    B, S = 2, 32
    x = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S)
    cos, sin = rot_emb(x, position_ids)
    mask = torch.zeros(B, 1, S, S, dtype=torch.float32)
    mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

    out_ours, _ = ours(x, position_embeddings=(cos, sin), attention_mask=mask)
    out_ref, _ = ref(x, position_embeddings=(cos, sin), attention_mask=mask)
    assert torch.equal(out_ours, out_ref), (
        f"param_sink parity: max_diff={(out_ours.float() - out_ref.float()).abs().max().item():.3e}"
    )
    print("  [PASS] attention with param_sink_number=4 parity")


def test_naming_aliases() -> None:
    from veomni.models.transformers import _pangu_common as pc

    assert pc.PanguAttention is pc.OpenPanguV2Attention
    print("  [PASS] PanguAttention is OpenPanguV2Attention")


if __name__ == "__main__":
    print(f"\n{'=' * 72}")
    print("  _pangu_common.pangu_attention <-> Pangu reference parity")
    print(f"{'=' * 72}\n")

    failed = 0
    for fn in (
        test_repeat_kv_parity,
        test_eager_attention_forward_parity,
        test_attention_forward_end_to_end_parity,
        test_attention_forward_with_causal_mask_parity,
        test_attention_with_param_sink_parity,
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
