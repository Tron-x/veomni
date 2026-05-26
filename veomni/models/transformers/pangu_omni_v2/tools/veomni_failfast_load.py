"""Step 1: VeOmni fail-fast load of Pangu Omni 30B-A2B.

Goal: try to load the Pangu HF model through VeOmni's build_foundation_model
and collect the exact gaps (registry misses, config mismatches, forward
signature issues, NPU-specific code blockers).

We try TWO modes:
  1. MODELING_BACKEND=hf   - bypass VeOmni patches, use HF AutoModel directly.
     If this works -> at least the model can load; gaps are only in the
     distributed-training glue layer.
     If this fails -> there are pre-existing issues even before VeOmni.

  2. MODELING_BACKEND=veomni - default VeOmni path. This WILL fail because
     model_type is not registered. We document the exact failure mode so the
     adapter knows what to plug in.

We do NOT actually run forward / inference - just try to instantiate the
model on meta device so we surface as many issues as possible without
needing 60 GB RAM / NPU.
"""

import os
import sys
import traceback
from pathlib import Path


MODEL_DIR = os.environ.get("PANGU_MODEL_DIR")
if MODEL_DIR is None:
    raise SystemExit("Set PANGU_MODEL_DIR to the Pangu model directory before running this tool.")


def _section(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}\n")


def probe_mode(backend: str) -> None:
    _section(f"MODE: MODELING_BACKEND={backend}")
    os.environ["MODELING_BACKEND"] = backend

    # Re-import VeOmni modules to make sure the env var takes effect.
    # (env.py reads MODELING_BACKEND on first access)
    for mod in list(sys.modules):
        if mod.startswith("veomni"):
            del sys.modules[mod]

    try:
        from veomni.arguments.arguments_types import OpsImplementationConfig
        from veomni.models.auto import build_config, build_foundation_model
    except Exception as exc:
        print(f"[FATAL] cannot import VeOmni: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    # Probe 1: build_config (lightweight, no model instantiation)
    print(f"--- Probe 1: build_config('{MODEL_DIR}') ---")
    try:
        cfg = build_config(MODEL_DIR)
        print(f"  OK: type={type(cfg).__name__}")
        print(f"      model_type='{getattr(cfg, 'model_type', '?')}'")
        print(f"      architectures={getattr(cfg, 'architectures', '?')}")
        print(f"      hidden_size={getattr(cfg, 'hidden_size', '?')}")
        print(f"      num_hidden_layers={getattr(cfg, 'num_hidden_layers', '?')}")
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=8)
        return

    # Probe 2: get_model_class (this is where the registry lookup happens)
    print("\n--- Probe 2: get_model_class(config) — registry lookup ---")
    try:
        from veomni.models.loader import get_model_class

        model_cls = get_model_class(cfg)
        print(f"  OK: resolved model class = {model_cls}")
        print(f"      module = {model_cls.__module__}")
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=8)
        return

    # Probe 3: build_foundation_model (full load on meta device, no weights)
    print("\n--- Probe 3: build_foundation_model(init_device='meta', no weights) ---")
    try:
        # NPU-friendly minimal config: all ops set to 'eager' / 'npu' fallbacks
        # so VeOmni's NPU validator doesn't bail before we hit real model gaps.
        ops_cfg = OpsImplementationConfig(
            attn_implementation="eager",
            moe_implementation="eager",
            cross_entropy_loss_implementation="eager",
            rms_norm_implementation="eager",
            swiglu_mlp_implementation="eager",
            rotary_pos_emb_implementation="eager",
        )
        model = build_foundation_model(
            config_path=MODEL_DIR,
            weights_path=None,
            torch_dtype="bfloat16",
            init_device="meta",
            ops_implementation=ops_cfg,
        )
        print(f"  OK: instantiated {type(model).__name__}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"      total params: {n_params / 1e9:.2f}B")

        # Probe extras
        has_parallel = hasattr(model, "get_parallel_plan")
        print(f"      has get_parallel_plan: {has_parallel}")
        if has_parallel:
            try:
                plan = model.get_parallel_plan()
                print(f"      parallel plan keys: {list(plan.extra_parallel_plan.keys())}")
            except Exception as e:
                print(f"      get_parallel_plan() raised: {e}")
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=15)


def main() -> None:
    _section("Step 1: VeOmni fail-fast load of Pangu Omni 30B-A2B")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Exists:    {Path(MODEL_DIR).exists()}")

    # Make sure VeOmni picks up our env var freshly
    if "veomni" in sys.modules:
        for mod in list(sys.modules):
            if mod.startswith("veomni"):
                del sys.modules[mod]

    probe_mode("hf")
    probe_mode("veomni")

    _section("Done. See output above for the gap list.")


if __name__ == "__main__":
    main()
