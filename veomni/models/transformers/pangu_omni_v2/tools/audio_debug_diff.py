"""Diagnostic: localize where HF-generate vs HF-teacher-force logps diverge
for audio samples.

Background: Week 3.5 HF self-check (``--mode hf``) reported max_diff
~0.02 between teacher-forced and generate-time logps. That violates
the "HF generate path == HF teacher-forced path on the same model"
invariant that holds for the OCRBench image baseline (max_diff = 0).

This script reproduces the divergence in one Python session (so we
share the same model object, same dtype state, same RNG state, same
kv cache initial conditions) and prints per-token diffs so we can see
which positions diverge and whether it's a single-token spike (likely
attention boundary) or systematic noise (likely kernel mismatch).

Usage:
    python tools/pangu_audio_debug_diff.py --sample-id audio_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow running this script directly: prepend this directory so the
# sibling ``oracle_check.py`` tool can be imported as a top-level module.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_check import (  # noqa: E402
    PANGU_MODEL_DIR,
    build_conversation,
    load_hf_model_and_processor,
    load_jsonl,
    resolve_audio_paths,
    resolve_image_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("/mnt/data_3/models/pangu_audio_oracle/data/audio_demo.jsonl"),
    )
    parser.add_argument("--sample-id", type=str, default="audio_0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model-dir", type=Path, default=PANGU_MODEL_DIR)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from qwen_omni_utils import process_mm_info

    samples = load_jsonl(args.samples)
    sample = next(s for s in samples if str(s["sample_id"]) == args.sample_id)
    samples_root = args.samples.parent.parent

    print(f"[load] loading HF model from {args.model_dir} ...")
    model, processor = load_hf_model_and_processor(args.model_dir)
    print("[load] done")

    # ----- shared input construction -----
    image_paths = resolve_image_paths(sample, samples_root)
    audio_paths = resolve_audio_paths(sample, samples_root)
    conversation = build_conversation(
        image_paths,
        sample["prompt_text"],
        text_only=False,
        audio_paths=audio_paths,
    )
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        audio=audios,
        padding=False,
        return_tensors="pt",
    ).to(model.device)
    prompt_len = int(inputs.input_ids.shape[1])
    print(f"\n[prompt] len={prompt_len}")
    print(f"  audio kwargs in inputs: {[k for k in inputs.keys() if 'audio' in k or 'feature' in k]}")
    for k in inputs.keys():
        v = inputs[k]
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    # =========================================================================
    # Path A: HF generate (= how the baseline JSONL was produced)
    # =========================================================================
    print("\n[A] running model.generate(do_sample=False, output_scores=True) ...")
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
    generated_ids = gen.sequences[0, prompt_len:].tolist()
    logps_A: list[float] = []
    for step_logits, tok_id in zip(gen.scores, generated_ids):
        step_logp = F.log_softmax(step_logits[0].float(), dim=-1)
        logps_A.append(float(step_logp[int(tok_id)].item()))
    tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids)
    print(f"  generated {len(generated_ids)} tokens: {tokens}")

    # =========================================================================
    # Path B: Teacher-forced forward on (prompt + generated_ids)
    #         — same code path as oracle_check.compute_per_token_logps
    # =========================================================================
    print("\n[B] running teacher-forced forward on the same model ...")
    baseline_tensor = torch.tensor([generated_ids], dtype=inputs.input_ids.dtype, device=inputs.input_ids.device)
    full_input_ids = torch.cat([inputs.input_ids, baseline_tensor], dim=1)
    extended_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
    if "attention_mask" in extended_inputs:
        extended_inputs["attention_mask"] = torch.cat(
            [
                extended_inputs["attention_mask"],
                torch.ones_like(baseline_tensor, dtype=extended_inputs["attention_mask"].dtype),
            ],
            dim=1,
        )
    with torch.no_grad():
        out = model(input_ids=full_input_ids, **extended_inputs)
    logits = out.logits[0]
    logps_B: list[float] = []
    for i, tok_id in enumerate(generated_ids):
        logit_pos = prompt_len - 1 + i
        step_logp = F.log_softmax(logits[logit_pos].float(), dim=-1)
        logps_B.append(float(step_logp[int(tok_id)].item()))

    # =========================================================================
    # Path C: Teacher-forced with explicit cache_position=arange(seq_len)
    #         — replicates what generate's first prefill step does
    #         (see modeling_pangu_omni.py L1181-1205 — the rope branch
    #         is gated on ``cache_position is None or [0] == 0``).
    # =========================================================================
    print("\n[C] running teacher-forced WITH cache_position=arange(seq_len) ...")
    seq_len = full_input_ids.shape[1]
    cache_position = torch.arange(seq_len, device=full_input_ids.device)
    with torch.no_grad():
        out2 = model(
            input_ids=full_input_ids,
            cache_position=cache_position,
            **extended_inputs,
        )
    logits2 = out2.logits[0]
    logps_C: list[float] = []
    for i, tok_id in enumerate(generated_ids):
        logit_pos = prompt_len - 1 + i
        step_logp = F.log_softmax(logits2[logit_pos].float(), dim=-1)
        logps_C.append(float(step_logp[int(tok_id)].item()))

    # =========================================================================
    # Path D: Single forward on PROMPT ONLY — no concatenation with
    #         generated_ids. This tells us if the difference comes from
    #         simply having more tokens in the input (which could change
    #         NPU-fusion-attention kernel block partitioning).
    #
    #         Token 0 of path A is generate's prefill logits[-1].
    #         Token 0 of path D is forward(prompt).logits[-1].
    #         These two SHOULD be bit-identical (same input).
    # =========================================================================
    print("\n[D] running forward(prompt_only) — no concatenation ...")
    # Reset any stateful rope_deltas to keep the codepath isolated.
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    if hasattr(model, "rope_deltas"):
        model.rope_deltas = None
    with torch.no_grad():
        out3 = model(**inputs)
    logits3 = out3.logits[0]
    # token 0 of generate is the prediction at position prompt_len-1 of the
    # prompt-only forward
    tok0_logp_D = float(F.log_softmax(logits3[prompt_len - 1].float(), dim=-1)[generated_ids[0]].item())
    print(f"  token 0 logp (path D, prompt-only forward): {tok0_logp_D:.7e}")
    print(f"  token 0 logp (path A, generate prefill):   {logps_A[0]:.7e}")
    print(f"  token 0 logp (path B, full teacher-force): {logps_B[0]:.7e}")
    print(f"  |A - D| = {abs(logps_A[0] - tok0_logp_D):.3e}")
    print(f"  |B - D| = {abs(logps_B[0] - tok0_logp_D):.3e}")
    print("  If |A-D| ~= 0 and |B-D| large → concatenation with generated_ids")
    print("    changes NPU kernel selection, even for causal-protected positions.")
    print("  If |A-D| large → some state in generate path that we can't replicate.")

    # placeholder so the table below still works
    logps_D = logps_C  # noqa: F841  # placeholder slot; kept so the print table layout is stable when more comparison points are added

    # =========================================================================
    # Side-by-side table
    # =========================================================================
    print(
        f"\n{'idx':>4s} {'tok_id':>7s} {'token':<22s} "
        f"{'A=generate':>14s} {'B=tf default':>14s} {'C=tf+cache_pos':>16s} "
        f"{'A-B':>10s} {'A-C':>10s} {'B-C':>10s}"
    )
    print("-" * 130)
    for i, (a, b, c, tid, tok) in enumerate(zip(logps_A, logps_B, logps_C, generated_ids, tokens)):
        tok_disp = tok if len(tok) <= 22 else tok[:19] + "..."
        print(
            f"{i:>4d} {tid:>7d} {tok_disp:<22s} "
            f"{a:>14.7e} {b:>14.7e} {c:>16.7e} "
            f"{a - b:>10.2e} {a - c:>10.2e} {b - c:>10.2e}"
        )

    max_ab = max(abs(a - b) for a, b in zip(logps_A, logps_B))
    max_ac = max(abs(a - c) for a, c in zip(logps_A, logps_C))
    max_bc = max(abs(b - c) for b, c in zip(logps_B, logps_C))
    print(f"\nmax |A-B|={max_ab:.3e}  max |A-C|={max_ac:.3e}  max |B-C|={max_bc:.3e}")
    print("\nInterpretation:")
    print("  |A-C| ~= 0  →  passing cache_position=arange(seq_len) to forward")
    print("                 makes teacher-force match generate's prefill exactly.")
    print("                 Fix: oracle_check.compute_per_token_logps must pass")
    print("                 cache_position=arange(full_seq_len) so the model's")
    print("                 rope-branch (L1181-1205) takes the same code path.")
    print("  |A-B| large but |A-C| ~= 0 → confirms the rope-branch hypothesis")
    print("  |A-C| also large → not the rope branch; need deeper investigation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
