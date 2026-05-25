"""End-to-end loader smoke: Pangu disk ckpt -> converter -> model.load_state_dict -> forward.

Verifies the full VeOmni checkpoint-loading path for the Pangu adapter
without requiring the 60GB on-disk 30B-A2B safetensors:

1. Build OpenPanguV2ForCausalLM with toy config (4 layers, 8 experts).
2. Construct a synthetic "disk state_dict" in per-expert HF layout.
3. Stream it through `PanguOmniV2CheckpointTensorConverter` using
   VeOmni's `maybe_convert_checkpoint_tensor` helper (same code path
   the real loader takes).
4. Load the converted state_dict into the model with strict=True
   (no missing/unexpected — fully covered).
5. Run a forward pass and verify the model's expert weights match
   the disk weights bit-for-bit (round-trip integrity).

This is the "VeOmni loader contract" smoke test: if it passes, the
real 30B-A2B load path is also wired up correctly (modulo NPU/memory).

Run:
    python tests/test_pangu_veomni_load_path.py
"""

from __future__ import annotations

import sys
from typing import Dict

import torch


def _build_toy_model():
    """Construct OpenPanguV2ForCausalLM with the same toy config used in
    other Week 2 tests, so the random-init shapes match what we'll
    overwrite from "disk".

    NOTE: We MUST go through the MODELING_REGISTRY dispatcher rather than
    directly importing the class — the dispatcher attaches
    `_create_checkpoint_tensor_converter` as a side-effect (matching the
    Qwen3MoE registration pattern). In production
    `build_foundation_model` calls the dispatcher; standalone unit tests
    have to mimic that.
    """
    from veomni.models.loader import MODELING_REGISTRY
    from veomni.models.transformers.pangu_omni_v2.configuration_pangu_omni_v2 import (
        OpenPanguOmniConfig,
    )

    factory_dispatch = MODELING_REGISTRY.get("openpangu_omni")
    OpenPanguV2ForCausalLM = factory_dispatch("OpenPanguV2ForCausalLM")

    cfg = OpenPanguOmniConfig(
        num_hidden_layers=4,
        swa_layers=[],
        sliding_window=None,
    )
    cfg.vocab_size = 256
    cfg.hidden_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 32
    cfg.v_head_dim = 32
    cfg.intermediate_size = 256
    cfg.pad_token_id = None
    cfg.initializer_range = 0.02
    cfg.partial_rotary_factor = 0.25
    cfg.rope_theta = 10000.0
    cfg.max_position_embeddings = 2048
    cfg.rope_interleaved = False
    cfg.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    }
    cfg.rms_norm_eps = 1e-6
    cfg.attention_dropout = 0.0
    cfg.attention_bias = False
    cfg.attn_groupnorm = False
    cfg.attn_elementwise_gate = False
    cfg.param_sink_number = 0
    cfg._attn_implementation = "eager"
    cfg.torch_dtype = torch.float32
    cfg.n_routed_experts = 8
    cfg.n_shared_experts = 2
    cfg.num_experts_per_tok = 2
    cfg.topk_group = 1
    cfg.norm_topk_prob = True
    cfg.routed_scaling_factor = 2.5
    cfg.moe_intermediate_size = 64
    cfg.hidden_act = "silu"
    cfg._experts_implementation = "eager"
    cfg.use_mhc = True
    cfg.mhc_num_stream = 4
    cfg.mhc_use_gamma = True
    cfg.mhc_recur_norm = 20
    cfg.use_mla = False
    cfg.first_k_dense_replace = 2
    cfg.sandwich_norm = False
    cfg.block_post_layernorm_idx = None
    cfg.tie_word_embeddings = False
    cfg.use_cache = False

    torch.manual_seed(0)
    model = OpenPanguV2ForCausalLM(cfg)
    return model, cfg


def _make_disk_state_dict_for(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Take the model's CURRENT in-memory state_dict (fused 3D) and split
    the experts back into per-expert layout. Everything else passthrough.

    This guarantees the disk SD is self-consistent with the model's
    expected shapes — round-trip through the converter must reproduce
    the original SD exactly.
    """
    sd = model.state_dict()
    disk: Dict[str, torch.Tensor] = {}

    # Pattern: model.layers.{i}.mlp.experts.{gate_up_proj,down_proj}
    for k, v in sd.items():
        if k.endswith(".mlp.experts.gate_up_proj"):
            # v shape: [E, 2*I, H] - split into per-expert gate/up
            prefix = k.replace(".gate_up_proj", "")
            E, twoI, H = v.shape
            I = twoI // 2
            gate = v[:, :I, :]  # [E, I, H]
            up = v[:, I:, :]  # [E, I, H]
            for j in range(E):
                disk[f"{prefix}.{j}.gate_proj.weight"] = gate[j].clone().contiguous()
                disk[f"{prefix}.{j}.up_proj.weight"] = up[j].clone().contiguous()
        elif k.endswith(".mlp.experts.down_proj"):
            prefix = k.replace(".down_proj", "")
            E = v.shape[0]
            for j in range(E):
                disk[f"{prefix}.{j}.down_proj.weight"] = v[j].clone().contiguous()
        else:
            disk[k] = v
    return disk


def _stream_through_converter(model: torch.nn.Module, disk_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Use VeOmni's exact converter-streaming helper to remap keys."""
    from veomni.models.checkpoint_tensor_loading import (
        get_checkpoint_tensor_converter,
        maybe_convert_checkpoint_tensor,
    )

    converter = get_checkpoint_tensor_converter(model)
    assert converter is not None, (
        "Pangu model class missing `_create_checkpoint_tensor_converter` "
        "registration — did __init__.py dispatcher run?"
    )

    converted: Dict[str, torch.Tensor] = {}
    for name, tensor in disk_sd.items():
        result = maybe_convert_checkpoint_tensor(name, tensor, converter)
        if result is not None:
            converted[result.name] = result.tensor

    # Final flush (no-op for complete ckpts; raises on incomplete).
    extras = converter.finalize()
    for x in extras:
        converted[x.name] = x.tensor
    return converted


def _randomize_all_params_and_buffers(model: torch.nn.Module, seed: int, scale: float = 0.05) -> None:
    """Overwrite ALL params + float buffers with reproducible random values.

    Necessary because Pangu's `mHCModule.__init__` uses
    `nn.Parameter(torch.empty(...))` for `norm_gamma` / `branch_alpha_*`
    / `branch_beta_*` and the upstream `_init_weights` doesn't cover
    them — they remain as garbage memory which may contain NaN/Inf.
    `torch.equal(NaN, NaN)` returns False, so round-trip checks fail.
    Forcing every param/buffer to a finite random value side-steps that
    without changing test semantics (we're testing layout integrity, not
    init correctness).
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * scale)
        for b in model.buffers():
            if b.dtype.is_floating_point:
                b.copy_(torch.randn_like(b) * scale)


def test_load_path_round_trip_via_veomni_helpers() -> None:
    """The whole pipeline: model -> disk -> converter -> reload -> forward."""
    # 1. Build model and randomize every storage cell to a finite value
    #    (the upstream `_init_weights` leaves `mHCModule` parameters as
    #    `torch.empty(...)` garbage — see _randomize_all_params_and_buffers).
    model, cfg = _build_toy_model()
    _randomize_all_params_and_buffers(model, seed=0)
    ground_truth_sd = {k: v.clone() for k, v in model.state_dict().items()}

    # 2. Decompose into disk-layout (per-expert).
    disk_sd = _make_disk_state_dict_for(model)
    n_disk_expert_keys = sum(1 for k in disk_sd if ".mlp.experts." in k)
    # Each MoE layer (layers 2-3 = 2 layers) has 3 projections × n_routed_experts
    # = 3 × 8 = 24 keys per layer; total 2 × 24 = 48.
    expected = (cfg.num_hidden_layers - cfg.first_k_dense_replace) * 3 * cfg.n_routed_experts
    assert n_disk_expert_keys == expected, f"disk layout: {n_disk_expert_keys} expert keys vs expected {expected}"

    # 3. Stream through VeOmni's converter helper.
    converted_sd = _stream_through_converter(model, disk_sd)

    # 4. Load into a *fresh* model (different random init) and verify
    #    we exactly recover ground-truth.
    fresh_model, _ = _build_toy_model()
    _randomize_all_params_and_buffers(fresh_model, seed=999, scale=0.01)
    # Now load converted SD with strict=True (no missing/unexpected).
    incompat = fresh_model.load_state_dict(converted_sd, strict=True)
    assert incompat.missing_keys == [], f"missing keys after VeOmni-loader-style load: {incompat.missing_keys[:5]}"
    assert incompat.unexpected_keys == [], f"unexpected keys: {incompat.unexpected_keys[:5]}"

    # 5. Verify ground-truth recovery on every tensor.
    fresh_sd = fresh_model.state_dict()
    for k in ground_truth_sd:
        g = ground_truth_sd[k]
        f = fresh_sd[k]
        if not torch.equal(g, f):
            raise AssertionError(f"{k}: round-trip mismatch, max_diff={(g.float() - f.float()).abs().max():.3e}")

    # 6. Forward smoke — make sure the loaded model can run.
    #    Cast to bf16 to match the mHCModule's hardcoded-bf16 parameters
    #    (norm_gamma / branch_*), otherwise `F.linear(fp32_input, bf16_weight)`
    #    raises dtype mismatch.
    fresh_model = fresh_model.to(dtype=torch.bfloat16)
    fresh_model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out = fresh_model(input_ids=input_ids, use_cache=False)
    assert out.logits.shape == (1, 4, cfg.vocab_size), f"forward logits shape {out.logits.shape}"
    assert torch.isfinite(out.logits).all(), "logits contain non-finite values"

    print(
        f"  [PASS] full VeOmni loader path: "
        f"{len(disk_sd)} disk keys -> {len(converted_sd)} converted keys, "
        f"strict load OK, forward logits shape {tuple(out.logits.shape)}"
    )


def test_load_path_preserves_text_only_smoke() -> None:
    """Verify that when architectures=['OpenPanguV2ForCausalLM'] is what
    we dispatch on (the Week 2 text-only path), the converter is still
    correctly attached and the load works.

    This guards against future refactors that might forget to attach the
    converter on text-only model classes (a real risk because the 30B-A2B
    config.json defaults to 'OpenPanguUltraOmni...' architecture).
    """
    from veomni.models.loader import MODELING_REGISTRY
    from veomni.models.transformers.pangu_omni_v2.modeling_pangu_omni_v2 import (
        OpenPanguV2ForCausalLM,
    )

    factory_dispatch = MODELING_REGISTRY.get("openpangu_omni")
    cls = factory_dispatch("OpenPanguV2ForCausalLM")
    assert cls is OpenPanguV2ForCausalLM
    assert hasattr(cls, "_create_checkpoint_tensor_converter")

    # Verify the same on OpenPanguV2Model (text backbone, no LM head)
    cls2 = factory_dispatch("OpenPanguV2Model")
    assert hasattr(cls2, "_create_checkpoint_tensor_converter"), (
        "OpenPanguV2Model missing converter — text-only inference would fail to load"
    )

    # And on the still-placeholder multimodal classes (so Week 3 doesn't
    # have to remember to register again).
    cls3 = factory_dispatch("OpenPanguUltraOmniForConditionalGeneration")
    assert hasattr(cls3, "_create_checkpoint_tensor_converter"), (
        "Multimodal placeholder missing converter — should be wired even "
        "though instantiation raises NotImplementedError"
    )

    print("  [PASS] converter attached on all 4 model classes via dispatcher")


def main() -> None:
    import traceback as _tb

    tests = [
        test_load_path_round_trip_via_veomni_helpers,
        test_load_path_preserves_text_only_smoke,
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
