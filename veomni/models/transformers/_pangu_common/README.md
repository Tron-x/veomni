# `_pangu_common/` — Shared library for the Pangu model family

This directory hosts **family-wide** algorithm implementations shared by all
Pangu adapters (`pangu_omni_v2`, future `pangu_omni_v3`, future `pangu_text_v3`,
etc.) inside VeOmni.

The leading underscore (`_pangu_common`) is intentional: it tells the registry
loader **"this is not a `model_type` package"** — the per-model packages
(`pangu_omni_v2/`, ...) import from here, but this directory itself registers
nothing.

## What lives here

| Module | Purpose | Stage |
|---|---|---|
| `pangu_mhc.py` | Multi-Head Computation (Pangu-specific MoE attention preconditioning): 4 streams + gamma scaling + sinkhorn-knopp normalization | TBD (Week 2) |
| `pangu_partial_rope.py` | Partial rotary positional embedding (`qk_rope_dim` / `partial_rotary_factor`); ROPE_INIT_FUNCTIONS-compatible | TBD (Week 1) |
| `pangu_knorm.py` | Kernel-norm operator; registered as a VeOmni OpSlot | TBD (Week 1) |
| `pangu_vision_gated_merger.py` | GatedMerger projector (vs Qwen3-Omni PatchMerger) | TBD (Week 3) |
| `pangu_moe_shared_experts.py` | MoE block with 2 shared experts + N routed experts | TBD (Week 1) |
| `pangu_huanyu_audio.py` | HuanyuAudioEncoder (PyTorch implementation) | TBD (Week 3) |
| `pangu_fbank_extract.py` | Mel filterbank-40 feature extraction; `PANGU_FBANK_BACKEND={so,torchaudio,auto}` env switch | TBD (Week 3) |
| `parallel_plan_templates.py` | Reusable EP/FSDP plan templates that per-model packages extend | TBD (Week 4) |
| `checkpoint_converter_base.py` | Base class for HF safetensors → VeOmni fused/stacked weight converters | TBD (Week 4) |

## Why this directory exists (design rationale)

VeOmni's existing convention is "one `model_type` → one package, no shared
base" (see `qwen3_moe/` vs `qwen3_omni_moe/` for reference — they have
essentially zero shared code). That convention is fine when there are 2-3
unrelated MoE models, but Pangu is a **model family with multiple generations
and variants planned**:

- `pangu_omni_v2` (the first port, this PR): 30B-A2B / 70B-A7B sharing the
  same `model_type`
- `pangu_text_v2` (future): text-only variant with the same MHC / partial RoPE
- `pangu_omni_v3` (future): next-generation Omni with new capabilities

Without `_pangu_common/`, each new variant duplicates ~2000 lines of
MHC + partial RoPE + K-norm + shared-experts MoE + Huanyu audio. With this
directory, the per-model adapter shrinks to **registration + composition**
(~200 lines of glue), and bug fixes / numerical improvements land once
for the whole family.

## What does **not** live here

- **Per-model registration** — in each `pangu_*/` package's `__init__.py`
- **Per-model checkpoint converter** — in each `pangu_*/` package
- **Per-model parallel plan** — in each `pangu_*/` package (inherits from
  `parallel_plan_templates`)
- **patchgen configs** — per-model, in each `pangu_*/` package
- **Anything Qwen-, DeepSeek-, or generic VeOmni** — those stay in their
  own packages

## GPU/NPU compatibility constraint

Every module here MUST be device-agnostic by default:
- No unconditional `import torch_npu` — use `try/except ImportError` or
  `transformers.utils.is_torch_npu_available()`
- No vendor binary dependencies (`.so` files) — provide PyTorch reference
  implementations; optional NPU-optimized paths gated by env vars
- Numerical output MUST match across GPU/NPU within `phase 1` tolerance
  (`max_abs_diff < 1e-3`, see `tools/pangu_oracle_check.py` in
  external AReaL repo)

See `docs/pangu_veomni_adaptation/PHASE1_DESIGN.md` (in the AReaL repo) for
the full design rationale.
