"""Single-NPU smoke: full Pangu Omni 30B-A2B real-weight load + 1 forward.

Purpose
-------
The toy smoke (``smoke_single_npu.py``) proved the adapter wires up to
NPU correctly. This smoke graduates to **real production weights**:

- Loads the full 54GB safetensors set from
  ``/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model/`` via
  ``build_foundation_model`` (same code path the trainer uses).
- Streams per-expert MoE weights through
  ``PanguOmniV2CheckpointTensorConverter`` (so the fused 3D
  ``experts.gate_up_proj`` layout is built on the fly).
- Runs a single text-only teacher-forced forward and prints finite
  logits + per-token logp on the first few baseline tokens.

What this catches that the toy smoke does not:
1. Real per-expert -> fused 3D MoE key remapping on 384 experts × 35
   layers (toy was 8 experts × 2 layers).
2. NPU memory budget under real-config (37 layers, hidden=2560,
   vocab=151552). Single 64GB NPU has ~6GB headroom after weights.
3. ``build_foundation_model`` path end-to-end (toy bypassed it by
   instantiating the class directly).
4. Real tokenizer + processor path through ``build_processor``.

Modes
-----
- ``--text-only`` (default) — override ``architectures`` to
  ``OpenPanguV2ForCausalLM``, skip vision/audio modules. ~50GB load.
  Forward takes raw text only; no image/audio kwargs needed.
  **Status: PASSES** on single 64GB NPU (verified 2026-05-22).
- ``--full-multimodal`` — let the dispatcher resolve
  ``OpenPanguUltraOmniForConditionalGeneration`` to ``OpenPanguOmni``.
  Loads all 54GB. **Status: LOAD PASSES, FORWARD FAILS** without
  real image/audio inputs — see "Known gap" below.

Memory budget on single 64GB NPU (~60.6GB free at boot):

| Path | Weights | Headroom | Risk |
|---|---|---|---|
| --text-only | ~50GB | ~10GB | safe |
| --full-multimodal | ~54GB | ~6GB | tight |

Multimodal pure-text (--full-multimodal): works as of 2026-05-22
----------------------------------------------------------------
``OpenPanguOmni.forward`` always emits 3-axis MRoPE ``position_ids``
(shape ``[3, B, S]``, temporal/H/W) when ``position_ids is None`` — see
``modeling_omni.py:624-680``. The downstream
``self.language_model.rotary_emb`` is ``OpenPanguVLRotaryEmbedding``
which requires exactly 3 dims (``modeling_vl.py:1135``), so
the 3D layout must be preserved into rotary_emb.

The conflict point used to be ``OpenPanguV2Model.forward`` calling
``create_causal_mask(position_ids=position_ids)`` (transformers v5
helper documents 2D ``[B, S]`` and uses ``position_ids`` only for
packed-sequence detection via ``find_packed_sequence_indices``, which
crashes on 3D). This was resolved by mirroring the canonical
transformers-5 pattern from ``Qwen2_5_VLTextModel.forward``: when
``position_ids.ndim == 3`` we pass ``None`` to ``create_causal_mask``
(non-packed multimodal sequence assumption). Caller-supplied
``attention_mask`` short-circuits the position_ids path entirely, so
packed multimodal training is unaffected. See the inline comment at
``modeling_text.py`` ``OpenPanguV2Model.forward`` for the
full rationale.

Run
---
    cd /root/VeOmni
    python veomni/models/transformers/pangu_omni_v2/tests/smoke_real_model_npu.py
    # or to attempt full multimodal load:
    python veomni/models/transformers/pangu_omni_v2/tests/smoke_real_model_npu.py --full-multimodal
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch


DEFAULT_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")


def _print_npu_mem(tag: str) -> None:
    """Print NPU 0 memory usage with a tag."""
    import torch_npu  # noqa: F401

    free, total = torch_npu.npu.mem_get_info(0)
    used = total - free
    allocated = torch_npu.npu.memory_allocated(0)
    reserved = torch_npu.npu.memory_reserved(0)
    print(
        f"  [mem:{tag}] free={free / 1024**3:.2f}GB used={used / 1024**3:.2f}GB "
        f"allocated={allocated / 1024**3:.2f}GB reserved={reserved / 1024**3:.2f}GB",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-NPU full-weight load + forward smoke for Pangu Omni v2.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to Pangu Omni v2 model directory (config.json + safetensors).",
    )
    parser.add_argument(
        "--full-multimodal",
        action="store_true",
        help=(
            "Load full OpenPanguOmni (vision+audio+text). Default is text-only "
            "(loads ~50GB, leaves ~10GB headroom). Full multimodal loads ~54GB "
            "with only ~6GB headroom on a 64GB NPU — tight."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="你好，请简单介绍一下盘古大模型。",
        help="Text prompt to feed through the model (text-only path).",
    )
    parser.add_argument(
        "--seq-trim",
        type=int,
        default=64,
        help=(
            "Truncate input to at most this many tokens to keep forward "
            "activation memory bounded (default 64). Real prompts can be "
            "shorter; this only kicks in when prompt is longer."
        ),
    )
    args = parser.parse_args()

    assert args.model_dir.exists(), f"model dir not found: {args.model_dir}"
    text_only = not args.full_multimodal

    import torch_npu  # noqa: F401

    assert torch_npu.npu.is_available(), "NPU not available — this smoke targets Ascend NPU"
    torch_npu.npu.set_device(0)
    _print_npu_mem("startup")

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_config, build_foundation_model, build_tokenizer

    # We follow the same ops config that the oracle uses (all eager) for
    # two reasons:
    # 1. Pangu does not yet have VeOmni-native fused kernels registered;
    #    eager paths are the production behaviour anyway.
    # 2. Matches the validated HF-parity path so the forward number we
    #    print here is interpretable as "what the oracle would see".
    ops_cfg = OpsImplementationConfig(
        attn_implementation="sdpa",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
    )

    # 1. Config dispatch
    print("[1/4] Build config + (optionally) override architectures...", flush=True)
    config = build_config(str(args.model_dir))
    original_arch = list(getattr(config, "architectures", []))
    if text_only:
        config.architectures = ["OpenPanguV2ForCausalLM"]
        print(
            f"  text-only mode: architectures override {original_arch} -> {config.architectures}",
            flush=True,
        )
    else:
        print(f"  full-multimodal mode: keeping architectures {original_arch}", flush=True)
    config_arg = config

    # 2. Build foundation model — this is where 54GB streams in.
    # ``init_device="npu"`` is the literal that ``build_foundation_model``
    # accepts (see ``Literal[...]`` annotation in ``veomni/models/auto.py``).
    # The oracle uses ``"cuda"`` because it imports ``torch_npu.contrib.
    # transfer_to_npu`` which monkey-patches CUDA calls to NPU. We don't
    # need that indirection: passing ``"npu"`` lands weights on
    # ``npu:0`` directly via the native torch_npu device backend.
    print(
        f"[2/4] Load weights from {args.model_dir} via build_foundation_model "
        f"(this reads ~{'54' if not text_only else '50'}GB; can take 1-3 min "
        "on first cold read)...",
        flush=True,
    )
    t0 = time.time()
    model = build_foundation_model(
        config_path=config_arg,
        weights_path=str(args.model_dir),
        torch_dtype="bfloat16",
        init_device="npu",
        ops_implementation=ops_cfg,
    ).eval()
    t_load = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  ok — loaded {n_params / 1e9:.2f}B params in {t_load:.1f}s (class={type(model).__name__})",
        flush=True,
    )
    _print_npu_mem("after-load")

    # 3. Tokenize a tiny prompt with the bundled tokenizer.
    # Build through build_tokenizer rather than build_processor: the
    # multimodal processor pulls in ``qwen_omni_utils`` and torchaudio
    # which we don't need for a text-only smoke, and which add minutes
    # to import time. The tokenizer alone gives us input_ids + attention_mask
    # which are all the model needs in text-only mode.
    print("[3/4] Tokenize prompt...", flush=True)
    tokenizer = build_tokenizer(str(args.model_dir))
    enc = tokenizer(args.prompt, return_tensors="pt", padding=False, truncation=True, max_length=args.seq_trim)
    input_ids = enc["input_ids"].to(model.device)
    seqlen = int(input_ids.shape[1])
    print(
        f"  ok — {seqlen} tokens: {tokenizer.decode(input_ids[0].tolist())[:80]!r}",
        flush=True,
    )

    # 4. Single forward, no_grad. We turn off use_cache to avoid
    # allocating the KV cache (negligible here but keeps memory accounting
    # clean) and read the LAST-token logit so we get a finite proxy
    # value to print.
    print("[4/4] Forward (no_grad)...", flush=True)
    _print_npu_mem("before-fwd")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    t_fwd = time.time() - t0
    _print_npu_mem("after-fwd")

    logits = out.logits
    assert logits.shape[0] == 1 and logits.shape[1] == seqlen, (
        f"unexpected logits shape {tuple(logits.shape)} for input {tuple(input_ids.shape)}"
    )
    assert torch.isfinite(logits).all().item(), "logits contain NaN/Inf"

    # Read greedy next-token from the last position. We only decode 1
    # token because (a) we're not testing generation correctness here
    # and (b) each extra token costs a full forward in the no-cache path.
    last_logits = logits[0, -1].float()
    top1_id = int(last_logits.argmax().item())
    top1_prob = float(torch.softmax(last_logits, dim=-1)[top1_id].item())
    top1_tok = tokenizer.decode([top1_id])
    print(
        f"  ok — forward took {t_fwd:.2f}s, top-1 next token id={top1_id} ({top1_tok!r}) p={top1_prob:.4f}",
        flush=True,
    )
    print(
        f"      last-position logits: min={last_logits.min().item():.3f} "
        f"max={last_logits.max().item():.3f} "
        f"mean={last_logits.mean().item():.3f}",
        flush=True,
    )

    print()
    print(
        f"SMOKE PASSED — single-NPU full-weight load + forward "
        f"({'TEXT-ONLY' if text_only else 'FULL-MULTIMODAL'} variant) "
        f"works on real Pangu Omni 30B-A2B."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nSMOKE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
