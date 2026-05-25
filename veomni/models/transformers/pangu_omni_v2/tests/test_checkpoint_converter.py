"""Round-trip test: Pangu Omni v2 checkpoint tensor converter.

Verifies that:
1. Per-expert disk-format keys are correctly merged into fused 3D
   `gate_up_proj` and stacked `down_proj` tensors.
2. Non-expert keys (router gate, shared_experts MLP,
   e_score_correction_bias, dense MLP for layers < first_k_dense_replace)
   are correctly passthrough (converter returns None).
3. `finalize()` flags incomplete checkpoints (missing experts).
4. The converter integrates correctly with VeOmni's loader path by
   verifying the `_create_checkpoint_tensor_converter` staticmethod is
   attached on Pangu model classes.

Run:
    python tests/test_pangu_checkpoint_converter.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import torch


def _build_toy_disk_state_dict(
    num_layers: int = 4,
    first_k_dense: int = 2,
    n_routed_experts: int = 8,
    n_shared_experts: int = 2,
    hidden_size: int = 128,
    moe_intermediate_size: int = 64,
    dense_intermediate_size: int = 256,
    vocab_size: int = 256,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Construct a (disk_sd, expected_fused_sd) pair for round-trip testing.

    Disk SD mirrors Pangu's HF safetensors layout (per-expert keys).
    Expected fused SD is what the converter must produce as its output
    when streamed the disk SD entry-by-entry.

    Returns:
        (disk_sd, expected_fused_sd)
    """
    torch.manual_seed(42)
    H = hidden_size
    I = moe_intermediate_size
    I_dense = dense_intermediate_size
    E = n_routed_experts
    Is = moe_intermediate_size * n_shared_experts  # shared_experts MLP intermediate

    disk_sd: Dict[str, torch.Tensor] = {}
    expected_fused_sd: Dict[str, torch.Tensor] = {}

    # ------- non-MLP keys (passthrough; identical in disk and fused SD) -------
    disk_sd["model.embed_tokens.weight"] = torch.randn(vocab_size, H)
    disk_sd["lm_head.weight"] = torch.randn(vocab_size, H)
    disk_sd["model.norm.weight"] = torch.randn(H)
    for k, v in disk_sd.items():
        expected_fused_sd[k] = v

    # ------- per-layer MLP keys -------
    for layer_idx in range(num_layers):
        is_dense = layer_idx < first_k_dense
        if is_dense:
            # Dense MLP layer: no `.experts.` subscript.
            for proj, shape in [
                ("gate_proj", (I_dense, H)),
                ("up_proj", (I_dense, H)),
                ("down_proj", (H, I_dense)),
            ]:
                k = f"model.layers.{layer_idx}.mlp.{proj}.weight"
                t = torch.randn(*shape)
                disk_sd[k] = t
                expected_fused_sd[k] = t  # passthrough
        else:
            # MoE layer.
            # Router gate weight (passthrough).
            k = f"model.layers.{layer_idx}.mlp.gate.weight"
            t = torch.randn(E, H)
            disk_sd[k] = t
            expected_fused_sd[k] = t

            # e_score_correction_bias (passthrough buffer).
            k = f"model.layers.{layer_idx}.mlp.e_score_correction_bias"
            t = torch.randn(E)
            disk_sd[k] = t
            expected_fused_sd[k] = t

            # Per-expert routed weights (CONVERTED into fused 3D).
            # Build gate/up/down lists per expert; later we'll assemble
            # the expected fused tensors.
            gate_list: List[torch.Tensor] = []
            up_list: List[torch.Tensor] = []
            down_list: List[torch.Tensor] = []
            for j in range(E):
                g = torch.randn(I, H)
                u = torch.randn(I, H)
                d = torch.randn(H, I)
                disk_sd[f"model.layers.{layer_idx}.mlp.experts.{j}.gate_proj.weight"] = g
                disk_sd[f"model.layers.{layer_idx}.mlp.experts.{j}.up_proj.weight"] = u
                disk_sd[f"model.layers.{layer_idx}.mlp.experts.{j}.down_proj.weight"] = d
                gate_list.append(g)
                up_list.append(u)
                down_list.append(d)
            # Expected fused tensors.
            gate_stacked = torch.stack(gate_list, dim=0)  # [E, I, H]
            up_stacked = torch.stack(up_list, dim=0)  # [E, I, H]
            gate_up_fused = torch.cat([gate_stacked, up_stacked], dim=1)  # [E, 2*I, H]
            down_stacked = torch.stack(down_list, dim=0)  # [E, H, I]
            expected_fused_sd[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"] = gate_up_fused
            expected_fused_sd[f"model.layers.{layer_idx}.mlp.experts.down_proj"] = down_stacked

            # Shared experts MLP (passthrough, no `.{k}.` subscript).
            for proj, shape in [
                ("gate_proj", (Is, H)),
                ("up_proj", (Is, H)),
                ("down_proj", (H, Is)),
            ]:
                k = f"model.layers.{layer_idx}.mlp.shared_experts.{proj}.weight"
                t = torch.randn(*shape)
                disk_sd[k] = t
                expected_fused_sd[k] = t

    return disk_sd, expected_fused_sd


def _run_converter(disk_sd: Dict[str, torch.Tensor], n_routed_experts: int) -> Dict[str, torch.Tensor]:
    """Stream `disk_sd` through the converter and collect outputs.

    Simulates the VeOmni loader behavior: for each tensor in disk SD,
    ask `can_handle` -> if True, feed to `convert()` (which may return
    None until enough experts are buffered); if False, the tensor is
    passthrough as-is. Finally call `finalize()`.
    """
    from veomni.models.transformers.pangu_omni_v2.checkpoint_tensor_converter import (
        PanguOmniV2CheckpointTensorConverter,
    )

    converter = PanguOmniV2CheckpointTensorConverter(num_experts=n_routed_experts)

    result_sd: Dict[str, torch.Tensor] = {}
    for name, tensor in disk_sd.items():
        if converter.can_handle(name):
            converted = converter.convert(name, tensor)
            if converted is not None:
                result_sd[converted.name] = converted.tensor
        else:
            result_sd[name] = tensor

    # finalize must not produce extra tensors (no error on complete ckpt)
    extras = converter.finalize()
    assert extras == [], f"finalize() unexpectedly produced {len(extras)} extra tensors"
    return result_sd


def test_round_trip_basic() -> None:
    """End-to-end: disk SD -> converter -> fused SD == expected fused SD."""
    disk_sd, expected_fused_sd = _build_toy_disk_state_dict()
    n_routed_experts = 8

    actual_fused_sd = _run_converter(disk_sd, n_routed_experts=n_routed_experts)

    # Key set equivalence
    only_actual = set(actual_fused_sd) - set(expected_fused_sd)
    only_expected = set(expected_fused_sd) - set(actual_fused_sd)
    assert not only_actual, f"converter produced unexpected keys: {sorted(only_actual)}"
    assert not only_expected, (
        f"converter missing expected keys: {sorted(only_expected)[:5]} (total {len(only_expected)})"
    )

    # Per-tensor bit-for-bit equality
    for k in expected_fused_sd:
        a = actual_fused_sd[k]
        e = expected_fused_sd[k]
        assert a.shape == e.shape, f"{k}: shape mismatch {a.shape} vs {e.shape}"
        if not torch.equal(a, e):
            diff = (a.float() - e.float()).abs().max()
            raise AssertionError(f"{k}: bit-for-bit mismatch, max_diff={diff:.3e}")

    print(
        f"  [PASS] basic round-trip ({len(disk_sd)} disk keys -> "
        f"{len(actual_fused_sd)} fused keys; per-expert merge correct)"
    )


def test_gate_up_layout_gate_first() -> None:
    """Verify gate_up_proj is `cat([gate, up], dim=1)` — gate first, up second.

    This matters because `OpenPanguV2Experts.forward` does
    `gate, up = ...chunk(2, dim=-1)` which returns (first_half, second_half).
    """
    H, I, E = 8, 4, 2
    disk_sd: Dict[str, torch.Tensor] = {}

    # Distinctive sentinel values so we can read off which is which.
    for j in range(E):
        gate = torch.full((I, H), float(j) + 0.1)  # 0.1, 1.1
        up = torch.full((I, H), float(j) + 10.1)  # 10.1, 11.1
        down = torch.full((H, I), float(j) + 100.1)  # 100.1, 101.1
        disk_sd[f"model.layers.2.mlp.experts.{j}.gate_proj.weight"] = gate
        disk_sd[f"model.layers.2.mlp.experts.{j}.up_proj.weight"] = up
        disk_sd[f"model.layers.2.mlp.experts.{j}.down_proj.weight"] = down

    actual = _run_converter(disk_sd, n_routed_experts=E)
    gate_up = actual["model.layers.2.mlp.experts.gate_up_proj"]
    down = actual["model.layers.2.mlp.experts.down_proj"]

    assert gate_up.shape == (E, 2 * I, H), f"shape {gate_up.shape}"
    assert down.shape == (E, H, I), f"shape {down.shape}"
    # gate occupies first I rows along dim=1; up occupies second I rows.
    for j in range(E):
        assert torch.all(gate_up[j, :I, :] == float(j) + 0.1), f"gate sentinel j={j}"
        assert torch.all(gate_up[j, I:, :] == float(j) + 10.1), f"up sentinel j={j}"
        assert torch.all(down[j] == float(j) + 100.1), f"down sentinel j={j}"
    print("  [PASS] gate_up_proj layout (gate first, up second), down_proj order")


def test_passthrough_non_expert_keys() -> None:
    """Non-expert keys must NOT trigger `can_handle()`.

    Verifies the regex doesn't accidentally fire on:
    - shared_experts.{proj}.weight (no `.\\d+.` segment)
    - mlp.gate.weight (router)
    - mlp.e_score_correction_bias (buffer)
    - mlp.{proj}.weight (dense MLP, layer < first_k_dense_replace)
    - embed_tokens / lm_head / norm
    """
    from veomni.models.transformers.pangu_omni_v2.checkpoint_tensor_converter import (
        PanguOmniV2CheckpointTensorConverter,
    )

    converter = PanguOmniV2CheckpointTensorConverter(num_experts=8)

    must_not_match = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.mlp.gate_proj.weight",  # dense MLP
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.2.mlp.gate.weight",  # router
        "model.layers.2.mlp.e_score_correction_bias",
        "model.layers.2.mlp.shared_experts.gate_proj.weight",
        "model.layers.2.mlp.shared_experts.up_proj.weight",
        "model.layers.2.mlp.shared_experts.down_proj.weight",
        "model.layers.2.self_attn.qkv_proj.weight",
        "model.layers.2.input_layernorm.weight",
    ]
    must_match = [
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.2.mlp.experts.7.up_proj.weight",
        "model.layers.2.mlp.experts.3.down_proj.weight",
    ]

    bad: List[str] = []
    for name in must_not_match:
        if converter.can_handle(name):
            bad.append(f"FALSE POSITIVE: {name}")
    for name in must_match:
        if not converter.can_handle(name):
            bad.append(f"FALSE NEGATIVE: {name}")
    if bad:
        raise AssertionError("\n".join(bad))
    print(f"  [PASS] can_handle regex correct ({len(must_match)} match, {len(must_not_match)} skip)")


def test_finalize_detects_incomplete_experts() -> None:
    """finalize() must raise when not all experts arrived."""
    from veomni.models.transformers.pangu_omni_v2.checkpoint_tensor_converter import (
        PanguOmniV2CheckpointTensorConverter,
    )

    converter = PanguOmniV2CheckpointTensorConverter(num_experts=8)
    # Feed only 3 experts' gate_proj (missing 4-7 and other projections)
    for j in range(3):
        converter.convert(
            f"model.layers.2.mlp.experts.{j}.gate_proj.weight",
            torch.randn(4, 8),
        )

    try:
        converter.finalize()
        raise AssertionError("finalize() did NOT raise on incomplete ckpt")
    except RuntimeError as e:
        assert "incomplete checkpoint" in str(e).lower(), f"msg: {e}"
    print("  [PASS] finalize() raises RuntimeError on incomplete ckpt")


def test_registered_on_model_classes() -> None:
    """The factory must be attached as staticmethod on Pangu model classes."""
    import veomni.models.transformers.pangu_omni_v2  # noqa: F401  # trigger reg
    from veomni.models.loader import MODELING_REGISTRY

    # Triggers dispatcher which sets _create_checkpoint_tensor_converter on the class.
    factory_dispatch = MODELING_REGISTRY.get("openpangu_omni")
    cls = factory_dispatch("OpenPanguV2ForCausalLM")

    factory_attr = getattr(cls, "_create_checkpoint_tensor_converter", None)
    assert factory_attr is not None, (
        f"{cls.__name__} missing _create_checkpoint_tensor_converter — VeOmni loader cannot find the converter factory"
    )
    # It should be callable
    assert callable(factory_attr), "_create_checkpoint_tensor_converter not callable"
    print(f"  [PASS] _create_checkpoint_tensor_converter registered on {cls.__name__}")


def test_pangu_30b_a2b_shape_smoke() -> None:
    """Smoke test for actual 30B-A2B shapes — verify converter handles
    the real expert count (384) and intermediate size (256).
    """
    H = 64  # toy hidden_size (real is 2560; CPU memory)
    I = 16  # toy moe_intermediate_size (real is 256)
    E = 384  # REAL n_routed_experts

    disk_sd: Dict[str, torch.Tensor] = {}
    for j in range(E):
        disk_sd[f"model.layers.2.mlp.experts.{j}.gate_proj.weight"] = torch.randn(I, H)
        disk_sd[f"model.layers.2.mlp.experts.{j}.up_proj.weight"] = torch.randn(I, H)
        disk_sd[f"model.layers.2.mlp.experts.{j}.down_proj.weight"] = torch.randn(H, I)

    actual = _run_converter(disk_sd, n_routed_experts=E)
    gate_up = actual["model.layers.2.mlp.experts.gate_up_proj"]
    down = actual["model.layers.2.mlp.experts.down_proj"]
    assert gate_up.shape == (E, 2 * I, H), f"gate_up shape {gate_up.shape}"
    assert down.shape == (E, H, I), f"down shape {down.shape}"
    print(f"  [PASS] 30B-A2B shape smoke: E={E}, gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}")


def main() -> None:
    import traceback as _tb

    tests = [
        test_round_trip_basic,
        test_gate_up_layout_gate_first,
        test_passthrough_non_expert_keys,
        test_finalize_detects_incomplete_experts,
        test_registered_on_model_classes,
        test_pangu_30b_a2b_shape_smoke,
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
