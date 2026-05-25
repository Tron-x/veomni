"""Bit-for-bit parity test: _pangu_common.pangu_decoder_layer vs Pangu reference.

This is the **integration** parity test of Week 1 — every Day 1-4 module
plays a role in a single end-to-end forward:
- partial RoPE (rotary_emb computed externally, passed via position_embeddings)
- RMSNorm (input_layernorm + pre_mlp_layernorm + K-norm + MHC's internal norm)
- Attention (K-norm + partial RoPE + GQA)
- MoE (routed experts + topk router + 2 shared experts)
- Dense MLP (for layer_idx < first_k_dense_replace)
- MHC (wraps both attention and MLP sub-blocks)

Tests:
1. Dense-MLP layer (layer_idx=0, < first_k_dense_replace=2) — fp32/bf16.
2. MoE layer (layer_idx=2, >= first_k_dense_replace=2) — fp32/bf16.
3. MHC=off variant (use_mhc=False, hidden_states stays at H all the way).
4. use_mla=True must raise NotImplementedError (cooperation with future MLA port).
5. Naming alias.

Run:
    python tests/test_pangu_decoder_layer_parity.py
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


class _DecoderLayerConfig:
    """Scaled-down Pangu config covering all fields the decoder layer touches.

    Hidden = 128 (vs 2560 in 30B-A2B), num_attention_heads = 4 (vs 20),
    n_routed_experts = 8 (vs 384). MoE intermediate = 64 (vs 256).
    Keeps mhc_num_stream = 4 (real) so MHC ops cover the same dim algebra.

    Attention shape sanity:
        head_dim = 128 / 4 = 32
        partial_rotary_factor = 0.25 -> rotary_ndims = 8
        v_head_dim = 32
    """

    def __init__(
        self,
        *,
        use_mhc: bool = True,
        use_mla: bool = False,
        first_k_dense_replace: int = 2,
        sandwich_norm: bool = False,
        block_post_layernorm_idx: list | None = None,
    ) -> None:
        # Sizes
        self.hidden_size = 128
        self.num_attention_heads = 4
        self.num_key_value_heads = 2  # GQA group = 2
        self.head_dim = 32
        self.v_head_dim = 32
        self.intermediate_size = 256  # dense MLP intermediate

        # RoPE
        self.partial_rotary_factor = 0.25  # rotary_ndims = 8
        self.rope_theta = 10000.0
        self.max_position_embeddings = 2048
        self.rope_interleaved = False
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        }

        # Norm
        self.rms_norm_eps = 1e-6

        # Attention
        self.attention_dropout = 0.0
        self.attention_bias = False
        self.attn_groupnorm = False
        self.attn_elementwise_gate = False
        self.param_sink_number = 0
        self.layer_types = None
        self.sliding_window = None
        self._attn_implementation = "eager"
        self.torch_dtype = torch.float32

        # MoE
        self.n_routed_experts = 8
        self.n_shared_experts = 2
        self.num_experts_per_tok = 2  # top_k
        self.topk_group = 1
        self.norm_topk_prob = True
        self.routed_scaling_factor = 2.5
        self.moe_intermediate_size = 64
        self.hidden_act = "silu"
        self._experts_implementation = "eager"

        # MHC
        self.use_mhc = use_mhc
        self.mhc_num_stream = 4
        self.mhc_use_gamma = True
        self.mhc_recur_norm = 20

        # Optional branches
        self.use_mla = use_mla
        self.first_k_dense_replace = first_k_dense_replace
        self.sandwich_norm = sandwich_norm
        self.block_post_layernorm_idx = block_post_layernorm_idx


def _make_pair(cfg, layer_idx: int):
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers._pangu_common import OpenPanguV2DecoderLayer

    torch.manual_seed(0)
    ours = OpenPanguV2DecoderLayer(cfg, layer_idx=layer_idx)
    ref = ref_mod.OpenPanguV2DecoderLayer(cfg, layer_idx=layer_idx)

    # Sync state_dict (cover parameters + buffers including
    # e_score_correction_bias and inv_freq).
    ours_state = dict(ours.state_dict())
    ref_state = dict(ref.state_dict())
    only_ours = set(ours_state) - set(ref_state)
    only_ref = set(ref_state) - set(ours_state)
    assert not only_ours, f"params only in ours: {sorted(only_ours)}"
    assert not only_ref, f"params only in ref: {sorted(only_ref)}"
    for k in ours_state:
        rand = torch.randn_like(ours_state[k]) * 0.05
        ours_state[k].copy_(rand)
        ref_state[k].copy_(rand)
    ours.load_state_dict(ours_state)
    ref.load_state_dict(ref_state)
    return ours, ref


def _make_position_embeddings(cfg, x: torch.Tensor, S: int):
    """Build (cos, sin) for the rotary embedding given input shape."""
    from veomni.models.transformers._pangu_common import OpenPanguV2RotaryEmbedding

    rot_emb = OpenPanguV2RotaryEmbedding(cfg, device=torch.device("cpu"))
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(x.shape[0], S)
    return rot_emb(x, position_ids), position_ids


def test_dense_mlp_layer_parity() -> None:
    """Layer 0 (< first_k_dense_replace=2) uses dense OpenPanguV2MLP."""
    cfg = _DecoderLayerConfig(use_mhc=True)
    ours, ref = _make_pair(cfg, layer_idx=0)

    # Sanity: ours layer 0 should have dense MLP, not SparseMoeBlock.
    from veomni.models.transformers._pangu_common import (
        OpenPanguV2MLP,
        OpenPanguV2SparseMoeBlock,
    )

    assert isinstance(ours.mlp, OpenPanguV2MLP), type(ours.mlp)
    assert not isinstance(ours.mlp, OpenPanguV2SparseMoeBlock)

    for dtype in (torch.float32, torch.bfloat16):
        # bf16 cast needed: attention/MoE Linear layers will require matching
        # input dtype. MHC params are already bf16 (forced by mHCModule.__init__).
        ours_dt = ours.to(dtype=dtype)
        ref_dt = ref.to(dtype=dtype)
        # Re-sync after cast — RMSNorm-fp32-trapped weights stay fp32 but
        # other params get cast; force identical post-cast state.
        with torch.no_grad():
            for name, p in ours_dt.named_parameters():
                dict(ref_dt.named_parameters())[name].copy_(p)
            for name, b in ours_dt.named_buffers():
                dict(ref_dt.named_buffers())[name].copy_(b)

        torch.manual_seed(1)
        B, S = 1, 8
        # Hidden states are at MHC-stream width when use_mhc=True
        feature_dim = cfg.hidden_size * cfg.mhc_num_stream
        x = torch.randn(B, S, feature_dim, dtype=dtype)
        (cos, sin), position_ids = _make_position_embeddings(cfg, x, S)

        out_ours = ours_dt(
            x,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )
        out_ref = ref_dt(
            x,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )
        assert out_ours.shape == out_ref.shape == (B, S, feature_dim)
        if not torch.equal(out_ours, out_ref):
            raise AssertionError(
                f"dense layer dtype={dtype}: max_diff={(out_ours.float() - out_ref.float()).abs().max():.3e}"
            )
    print(f"  [PASS] Dense MLP layer (layer 0, MHC on, n*H={cfg.hidden_size * cfg.mhc_num_stream})")


def test_moe_layer_parity() -> None:
    """Layer 2 (>= first_k_dense_replace=2) uses OpenPanguV2SparseMoeBlock."""
    cfg = _DecoderLayerConfig(use_mhc=True)
    ours, ref = _make_pair(cfg, layer_idx=2)

    from veomni.models.transformers._pangu_common import OpenPanguV2SparseMoeBlock

    assert isinstance(ours.mlp, OpenPanguV2SparseMoeBlock), type(ours.mlp)

    for dtype in (torch.float32, torch.bfloat16):
        ours_dt = ours.to(dtype=dtype)
        ref_dt = ref.to(dtype=dtype)
        with torch.no_grad():
            for name, p in ours_dt.named_parameters():
                dict(ref_dt.named_parameters())[name].copy_(p)
            for name, b in ours_dt.named_buffers():
                dict(ref_dt.named_buffers())[name].copy_(b)

        torch.manual_seed(2)
        B, S = 1, 8
        feature_dim = cfg.hidden_size * cfg.mhc_num_stream
        x = torch.randn(B, S, feature_dim, dtype=dtype)
        (cos, sin), position_ids = _make_position_embeddings(cfg, x, S)

        out_ours = ours_dt(
            x,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )
        out_ref = ref_dt(
            x,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )
        assert out_ours.shape == out_ref.shape == (B, S, feature_dim)
        if not torch.equal(out_ours, out_ref):
            raise AssertionError(
                f"MoE layer dtype={dtype}: max_diff={(out_ours.float() - out_ref.float()).abs().max():.3e}"
            )
    print(f"  [PASS] MoE layer (layer 2, MHC on, {cfg.n_routed_experts} experts + shared)")


def test_mhc_off_parity() -> None:
    """use_mhc=False path — hidden_states stays at H, no MHC ops."""
    cfg = _DecoderLayerConfig(use_mhc=False)
    ours, ref = _make_pair(cfg, layer_idx=2)

    # When use_mhc=False the layer should not have attn_mhc_module / mlp_mhc_module
    assert not hasattr(ours, "attn_mhc_module"), "attn_mhc_module should not exist when use_mhc=False"
    assert not hasattr(ref, "attn_mhc_module")

    torch.manual_seed(3)
    B, S = 1, 8
    # Without MHC, feature_dim is just hidden_size
    x = torch.randn(B, S, cfg.hidden_size)
    (cos, sin), position_ids = _make_position_embeddings(cfg, x, S)

    out_ours = ours(
        x,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )
    out_ref = ref(
        x,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )
    assert out_ours.shape == out_ref.shape == (B, S, cfg.hidden_size)
    assert torch.equal(out_ours, out_ref), f"MHC=off: max_diff={(out_ours.float() - out_ref.float()).abs().max():.3e}"
    print("  [PASS] MHC=off layer (residual stream stays at H)")


def test_use_mla_raises_not_implemented() -> None:
    """use_mla=True must raise NotImplementedError until MLA is ported."""
    cfg = _DecoderLayerConfig(use_mla=True)
    from veomni.models.transformers._pangu_common import OpenPanguV2DecoderLayer

    raised = False
    try:
        OpenPanguV2DecoderLayer(cfg, layer_idx=2)
    except NotImplementedError as e:
        raised = True
        assert "MLA" in str(e) and "30B-A2B" in str(e), f"error message lacks MLA context: {e}"
    assert raised, "use_mla=True did not raise NotImplementedError"
    print("  [PASS] use_mla=True raises NotImplementedError with MLA-context message")


def test_naming_alias() -> None:
    from veomni.models.transformers import _pangu_common as pc

    assert pc.PanguDecoderLayer is pc.OpenPanguV2DecoderLayer
    print("  [PASS] PanguDecoderLayer is OpenPanguV2DecoderLayer")


if __name__ == "__main__":
    print(f"\n{'=' * 72}")
    print("  _pangu_common.pangu_decoder_layer <-> Pangu reference parity")
    print("  (integrates ALL Day 1-4 modules)")
    print(f"{'=' * 72}\n")

    failed = 0
    for fn in (
        test_dense_mlp_layer_parity,
        test_moe_layer_parity,
        test_mhc_off_parity,
        test_use_mla_raises_not_implemented,
        test_naming_alias,
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
