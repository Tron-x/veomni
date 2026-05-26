"""Bit-for-bit parity test: pangu_omni_v2 modeling vs Pangu reference.

This is the **end-to-end** parity test of Week 2 — exercises the full
OpenPanguV2Model and OpenPanguV2ForCausalLM forward pass on a toy config
that is small enough to run on CPU in seconds but exercises every Day 1-5
module composed together:

    embed_tokens -> cat * mhc_num_stream
                 -> N x DecoderLayer (MHC pre/post + attention + MoE/Dense)
                 -> merge_mhc_module.hc_pre
                 -> norm -> [lm_head]

Tests:
1. OpenPanguV2Model.forward — last_hidden_state bit-for-bit parity (fp32/bf16).
2. OpenPanguV2ForCausalLM.forward — logits bit-for-bit parity (fp32/bf16).
3. logits_to_keep slicing — verify the "only compute last token" code path.
4. use_mhc=False variant — bypass the MHC tile/collapse entirely.

Toy config:
- vocab_size = 256
- hidden_size = 128, num_attention_heads = 4, num_key_value_heads = 2
- num_hidden_layers = 4 (layers 0-1 dense, layers 2-3 MoE)
- mhc_num_stream = 4 -> residual stream width 512
- n_routed_experts = 8, num_experts_per_tok = 2

Run:
    python tests/test_pangu_modeling_parity.py
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


def _make_toy_config(use_mhc: bool = True):
    """Build a real PretrainedConfig (OpenPanguOmniConfig) with toy sizes.

    Uses our real config class because PreTrainedModel.post_init() does
    weight init that reads `config.initializer_range` and other fields
    set on PretrainedConfig.
    """
    from veomni.models.transformers.pangu_omni_v2.configuration_pangu_omni_v2 import (
        OpenPanguOmniConfig,
    )

    # We pass num_hidden_layers/swa_layers as kwargs so OpenPanguOmniConfig's
    # post-init derives `layer_types` for us (matches production behavior).
    cfg = OpenPanguOmniConfig(
        num_hidden_layers=4,
        swa_layers=[],  # all layers -> "full_attention"
        sliding_window=None,
    )
    # Sizes
    cfg.vocab_size = 256
    cfg.hidden_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2  # GQA group = 2
    cfg.head_dim = 32
    cfg.v_head_dim = 32
    cfg.intermediate_size = 256  # dense MLP intermediate
    cfg.pad_token_id = None
    cfg.bos_token_id = 0
    cfg.eos_token_id = 1
    cfg.initializer_range = 0.02
    cfg.tie_word_embeddings = False
    cfg.use_cache = False

    # RoPE
    cfg.partial_rotary_factor = 0.25  # rotary_ndims = 8
    cfg.rope_theta = 10000.0
    cfg.max_position_embeddings = 2048
    cfg.rope_interleaved = False
    cfg.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    }

    # Norm
    cfg.rms_norm_eps = 1e-6

    # Attention
    cfg.attention_dropout = 0.0
    cfg.attention_bias = False
    cfg.attn_groupnorm = False
    cfg.attn_elementwise_gate = False
    cfg.param_sink_number = 0
    # NOTE: do NOT set cfg.layer_types = None here. OpenPanguOmniConfig.__init__
    # derived it as ['full_attention'] * num_hidden_layers from swa_layers=[];
    # overriding to None would cause OpenPanguV2DecoderLayer.attention_type
    # to be None, which then KeyErrors in OpenPanguV2Model.forward at
    # `causal_mask_mapping[None]`.
    cfg._attn_implementation = "eager"
    cfg.torch_dtype = torch.float32

    # MoE
    cfg.n_routed_experts = 8
    cfg.n_shared_experts = 2
    cfg.num_experts_per_tok = 2  # top_k
    cfg.topk_group = 1
    cfg.norm_topk_prob = True
    cfg.routed_scaling_factor = 2.5
    cfg.moe_intermediate_size = 64
    cfg.hidden_act = "silu"
    cfg._experts_implementation = "eager"

    # MHC
    cfg.use_mhc = use_mhc
    cfg.mhc_num_stream = 4
    cfg.mhc_use_gamma = True
    cfg.mhc_recur_norm = 20

    # Optional branches
    cfg.use_mla = False
    cfg.first_k_dense_replace = 2
    cfg.sandwich_norm = False
    cfg.block_post_layernorm_idx = None
    return cfg


def _build_pair(cls_name: str, cfg):
    """Instantiate ours + reference, random-init both with identical state."""
    (ref_mod,) = _load_reference_module()
    from veomni.models.transformers.pangu_omni_v2 import modeling_text

    ours_cls = getattr(modeling_text, cls_name)
    ref_cls = getattr(ref_mod, cls_name)

    torch.manual_seed(0)
    ours = ours_cls(cfg)
    ref = ref_cls(cfg)

    # Workaround for upstream Pangu bug: OpenPanguV2Model.forward (line 1089)
    # reads `self.mhc_num_stream`, but __init__ only sets `self.num_stream`.
    # PreTrainedModel does NOT forward unknown attribute access to config,
    # so the input_ids -> inputs_embeds path crashes with AttributeError.
    # Production multimodal inference pre-computes inputs_embeds upstream
    # (vision/audio merging) and skips this branch, masking the bug. Our
    # implementation reads `self.config.mhc_num_stream` directly so it does
    # NOT need this patch, but we patch ref so the parity comparison is on
    # the same computation path. See modeling_text.py docstring
    # "Known upstream quirk".
    if getattr(cfg, "use_mhc", False):
        mhc_n = cfg.mhc_num_stream
        for module in (ours, ref):
            for sub in module.modules():
                if type(sub).__name__ == "OpenPanguV2Model" and getattr(sub, "use_mhc", False):
                    sub.mhc_num_stream = mhc_n

    # Sync state_dict — params + buffers
    ours_state = dict(ours.state_dict())
    ref_state = dict(ref.state_dict())
    only_ours = set(ours_state) - set(ref_state)
    only_ref = set(ref_state) - set(ours_state)
    if only_ours or only_ref:
        raise AssertionError(
            f"state_dict mismatch for {cls_name}:\n"
            f"  only in ours: {sorted(only_ours)}\n"
            f"  only in ref:  {sorted(only_ref)}"
        )
    for k in ours_state:
        rand = torch.randn_like(ours_state[k]) * 0.05
        ours_state[k].copy_(rand)
        ref_state[k].copy_(rand)
    ours.load_state_dict(ours_state)
    ref.load_state_dict(ref_state)
    return ours, ref


def _cast_pair(ours: torch.nn.Module, ref: torch.nn.Module, dtype: torch.dtype):
    """Cast both modules to the target dtype and re-sync (cast can desync)."""
    ours_dt = ours.to(dtype=dtype)
    ref_dt = ref.to(dtype=dtype)
    with torch.no_grad():
        for name, p in ours_dt.named_parameters():
            dict(ref_dt.named_parameters())[name].copy_(p)
        for name, b in ours_dt.named_buffers():
            dict(ref_dt.named_buffers())[name].copy_(b)
    return ours_dt, ref_dt


def test_openpangu_v2_model_forward_parity() -> None:
    """OpenPanguV2Model.forward (last_hidden_state) bit-for-bit parity."""
    cfg = _make_toy_config(use_mhc=True)
    ours, ref = _build_pair("OpenPanguV2Model", cfg)

    for dtype in (torch.float32, torch.bfloat16):
        ours_dt, ref_dt = _cast_pair(ours, ref, dtype)
        ours_dt.eval()
        ref_dt.eval()

        torch.manual_seed(10)
        B, S = 1, 4
        input_ids = torch.randint(0, cfg.vocab_size, (B, S), dtype=torch.long)

        with torch.no_grad():
            out_ours = ours_dt(input_ids=input_ids, use_cache=False)
            out_ref = ref_dt(input_ids=input_ids, use_cache=False)

        h_ours = out_ours.last_hidden_state
        h_ref = out_ref.last_hidden_state
        assert h_ours.shape == h_ref.shape == (B, S, cfg.hidden_size), (
            f"shape mismatch: ours={h_ours.shape} ref={h_ref.shape}"
        )
        assert h_ours.dtype == h_ref.dtype, f"dtype mismatch: ours={h_ours.dtype} ref={h_ref.dtype}"
        if not torch.equal(h_ours, h_ref):
            raise AssertionError(
                f"OpenPanguV2Model dtype={dtype}: max_diff={(h_ours.float() - h_ref.float()).abs().max():.3e}"
            )
    print(f"  [PASS] OpenPanguV2Model.forward parity (MHC on, vocab={cfg.vocab_size})")


def test_openpangu_v2_for_causal_lm_logits_parity() -> None:
    """OpenPanguV2ForCausalLM.forward (logits) bit-for-bit parity."""
    cfg = _make_toy_config(use_mhc=True)
    ours, ref = _build_pair("OpenPanguV2ForCausalLM", cfg)

    for dtype in (torch.float32, torch.bfloat16):
        ours_dt, ref_dt = _cast_pair(ours, ref, dtype)
        ours_dt.eval()
        ref_dt.eval()

        torch.manual_seed(20)
        B, S = 1, 4
        input_ids = torch.randint(0, cfg.vocab_size, (B, S), dtype=torch.long)

        with torch.no_grad():
            out_ours = ours_dt(input_ids=input_ids, use_cache=False)
            out_ref = ref_dt(input_ids=input_ids, use_cache=False)

        l_ours = out_ours.logits
        l_ref = out_ref.logits
        assert l_ours.shape == l_ref.shape == (B, S, cfg.vocab_size), (
            f"shape mismatch: ours={l_ours.shape} ref={l_ref.shape}"
        )
        assert l_ours.dtype == l_ref.dtype, f"dtype mismatch: ours={l_ours.dtype} ref={l_ref.dtype}"
        if not torch.equal(l_ours, l_ref):
            raise AssertionError(
                f"OpenPanguV2ForCausalLM dtype={dtype}: max_diff={(l_ours.float() - l_ref.float()).abs().max():.3e}"
            )
    print(f"  [PASS] OpenPanguV2ForCausalLM.forward logits parity (MHC on, vocab={cfg.vocab_size})")


def test_openpangu_v2_for_causal_lm_logits_to_keep_parity() -> None:
    """Verify `logits_to_keep` slicing matches reference."""
    cfg = _make_toy_config(use_mhc=True)
    ours, ref = _build_pair("OpenPanguV2ForCausalLM", cfg)
    ours_dt, ref_dt = _cast_pair(ours, ref, torch.float32)
    ours_dt.eval()
    ref_dt.eval()

    torch.manual_seed(30)
    B, S = 1, 6
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), dtype=torch.long)

    # Only keep the last token's logits (`logits_to_keep=1`).
    with torch.no_grad():
        out_ours = ours_dt(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        out_ref = ref_dt(input_ids=input_ids, use_cache=False, logits_to_keep=1)

    assert out_ours.logits.shape == out_ref.logits.shape == (B, 1, cfg.vocab_size)
    if not torch.equal(out_ours.logits, out_ref.logits):
        raise AssertionError(
            f"logits_to_keep=1: max_diff={(out_ours.logits.float() - out_ref.logits.float()).abs().max():.3e}"
        )
    print(f"  [PASS] OpenPanguV2ForCausalLM.forward logits_to_keep=1 parity (B=1, S={S})")


def test_openpangu_v2_model_use_mhc_false_parity() -> None:
    """use_mhc=False — text backbone without MHC stream tiling.

    When use_mhc=False, embed_tokens output stays at hidden_size, decoder
    layers operate at hidden_size throughout, and merge_mhc_module is
    never constructed.
    """
    cfg = _make_toy_config(use_mhc=False)
    ours, ref = _build_pair("OpenPanguV2Model", cfg)

    # Sanity: merge_mhc_module should NOT exist when use_mhc=False.
    assert not hasattr(ours, "merge_mhc_module"), "merge_mhc_module should not be constructed when use_mhc=False"

    for dtype in (torch.float32, torch.bfloat16):
        ours_dt, ref_dt = _cast_pair(ours, ref, dtype)
        ours_dt.eval()
        ref_dt.eval()

        torch.manual_seed(40)
        B, S = 1, 4
        input_ids = torch.randint(0, cfg.vocab_size, (B, S), dtype=torch.long)

        with torch.no_grad():
            out_ours = ours_dt(input_ids=input_ids, use_cache=False)
            out_ref = ref_dt(input_ids=input_ids, use_cache=False)

        if not torch.equal(out_ours.last_hidden_state, out_ref.last_hidden_state):
            diff = (out_ours.last_hidden_state.float() - out_ref.last_hidden_state.float()).abs().max()
            raise AssertionError(f"OpenPanguV2Model use_mhc=False dtype={dtype}: max_diff={diff:.3e}")
    print("  [PASS] OpenPanguV2Model use_mhc=False parity")


def test_dispatch_to_text_for_causal_lm() -> None:
    """Verify registry dispatch on 'OpenPanguV2ForCausalLM' returns our class."""
    import veomni.models.transformers.pangu_omni_v2  # noqa: F401  # trigger reg
    from veomni.models.loader import MODELING_REGISTRY
    from veomni.models.transformers.pangu_omni_v2.modeling_text import (
        OpenPanguV2ForCausalLM,
    )

    factory = MODELING_REGISTRY.get("openpangu_omni")
    cls = factory("OpenPanguV2ForCausalLM")
    assert cls is OpenPanguV2ForCausalLM, f"expected ForCausalLM, got {cls.__name__}"
    print("  [PASS] MODELING_REGISTRY dispatches 'OpenPanguV2ForCausalLM' -> our class")


def main() -> None:
    import traceback as _tb

    tests = [
        test_openpangu_v2_model_forward_parity,
        test_openpangu_v2_for_causal_lm_logits_parity,
        test_openpangu_v2_for_causal_lm_logits_to_keep_parity,
        test_openpangu_v2_model_use_mhc_false_parity,
        test_dispatch_to_text_for_causal_lm,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
            _tb.print_exc()
    print()
    print(f"{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
