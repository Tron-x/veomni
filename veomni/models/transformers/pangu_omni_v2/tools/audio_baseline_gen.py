"""Pangu Omni audio oracle baseline generator.

Companion to ``tools/pangu_oracle_check.py``. Given a samples JSONL
with audio inputs, generate the HuggingFace reference greedy outputs
and per-token logps and write them to a baseline JSONL with the same
schema as
``/path/to/oracle/results/audio_hf_outputs.jsonl``.

Why we need this (and what it is NOT):

- The Pangu pretraining team's 0518 precision-alignment data set has
  baselines only for image+text (OCRBench, MMMU). They haven't
  published an audio baseline yet, so Week 3.5 has no NPU-end-to-end
  numerical oracle.
- This script produces a **self-generated audio baseline** by running
  the HF reference implementation in the **exact same configuration**
  as ``infer.py:run_hf_one`` (greedy ``do_sample=False``, bf16, sdpa
  attention on NPU), then teacher-forcing those generated tokens back
  through the same model to record per-position logps.
- The result is **NOT an absolute ground-truth oracle** (which would
  require a different framework like Megatron or a CPU fp64 reference).
  It is a **HF-vs-VeOmni reproducibility check**: "VeOmni's
  ``OpenPanguOmni`` audio path produces the same per-token logp as
  HF's ``OpenPanGuOmni`` reference path, given the same input."
- For Pure Path / dispatcher / weight-load correctness this is the
  right yardstick: any divergence between the two means we wired the
  audio tower / audio merger / mrope branch wrong. Numerical noise
  (~1e-6 bf16) is acceptable; orders-of-magnitude divergence (~1e-3+)
  signals a kernel mismatch like the Week 3.6 sdpa-vs-eager drift.

Pipeline (per sample, mirrors ``oracle_check.compute_per_token_logps``):

  1. ``build_conversation(image_paths=[], prompt_text, audio_paths)``
     — Pangu chat-template expands ``<|audio_start|><|audio_pad|>...
     <|audio_end|>`` into the right number of audio pad tokens.
  2. ``qwen_omni_utils.process_mm_info`` extracts the audio bytes.
  3. ``processor(text=..., audio=audios, return_tensors='pt')`` →
     ``input_ids`` + ``input_features`` + ``feature_attention_mask``.
  4. **Stage 1**: ``model.generate(do_sample=False, max_new_tokens=64)``
     → only used to determine the reference token sequence (greedy
     argmax). The per-step logits returned by generate are NOT recorded.
  5. **Stage 2**: single teacher-forced ``model(input_ids=prompt+ref)``
     forward — record per-token logps at the new-token positions. This
     matches the exact forward shape ``oracle_check.py`` uses against
     the VeOmni adapter, so any baseline-vs-VeOmni divergence measures
     pure HF-vs-VeOmni forward differences instead of also folding in
     HF's own generate-vs-teacher-force NPU bf16 noise.
  6. Write one JSONL row per sample.

Why the two-stage approach: NPU ``sdpa_attention_forward`` flips
``is_causal`` based on ``query.shape[2]`` and calls FlashAttentionScore
with different kernel block partitions at different sequence lengths.
On bf16, that produces ~1e-2 systematic differences at the same
logically-causal-protected positions between ``forward(prompt_only)``
and ``forward(prompt + generated_tokens)``. Recording the baseline
with the same shape we'll evaluate at sidesteps that hardware-level
inconsistency entirely — we're effectively measuring "does VeOmni
``forward()`` match HF ``forward()`` on identical inputs", which is
the right question for the adapter.

Usage:

  python tools/pangu_audio_baseline_gen.py \\
      --samples /path/to/audio_oracle/data/audio_demo.jsonl \\
      --out     /path/to/audio_oracle/results/audio_hf_outputs.jsonl \\
      --n-samples 3 \\
      --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


# Allow running this script directly: prepend this directory so the
# sibling ``oracle_check.py`` tool can be imported as a top-level module.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_check import (  # noqa: E402  (after sys.path)
    build_conversation,
    load_hf_model_and_processor,
    load_jsonl,
    resolve_audio_paths,
    resolve_image_paths,
)


def generate_one(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    samples_root: Path,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Greedy-generate the reference completion for one sample and
    return a baseline JSONL row.

    Mirrors ``oracle_check.compute_per_token_logps`` for the input
    construction so the baseline tokens land at exactly the same
    sequence positions when ``--mode hf`` re-teacher-forces them.
    """
    import torch

    image_paths = resolve_image_paths(sample, samples_root)
    audio_paths = resolve_audio_paths(sample, samples_root)
    conversation = build_conversation(
        image_paths,
        sample["prompt_text"],
        text_only=False,
        audio_paths=audio_paths,
    )
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    from qwen_omni_utils import process_mm_info

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

    # =========================================================================
    # Stage 1: greedy generate to determine the reference token sequence.
    #
    # We only use the *token ids* from this stage — the per-step logits
    # returned by ``generate`` are NOT what we record as the baseline,
    # because the generation path on NPU runs through a different
    # attention shape (prefill = 93 tokens, decode = 1 token at a time)
    # than the teacher-forced forward path that ``oracle_check.py`` will
    # use (single forward over 93 + 20 = 113 tokens). On Ascend NPU,
    # ``sdpa_attention.py:sdpa_attention_forward`` flips ``is_causal``
    # based on ``query.shape[2]`` and ``attention_mask`` and calls
    # NPU FlashAttentionScore with different kernel-block partitions
    # at different sequence lengths, producing bf16 outputs that differ
    # by ~1e-2 at the same logically-causal-protected positions. That
    # is hardware-level non-determinism in the reference path itself,
    # not an adapter bug, and we sidestep it by recording the baseline
    # at the same shape we'll evaluate at.
    # =========================================================================
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
        )
    new_token_ids = gen.sequences[0, prompt_len:].tolist()

    # =========================================================================
    # Stage 2: teacher-forced re-record of per-token logps using a SINGLE
    # forward pass over ``prompt + new_token_ids`` — the exact same shape
    # ``oracle_check.compute_per_token_logps`` will use against the
    # VeOmni adapter. This way the baseline and the model-under-test
    # are forwarded through identical kernel paths, and any divergence
    # measures "HF forward vs VeOmni forward at the same shape" — which
    # is what we actually want to certify (and what the OCRBench
    # Week 3.6 image-path oracle also did, getting noise floor 7e-6).
    # =========================================================================
    baseline_tensor = torch.tensor([new_token_ids], dtype=inputs.input_ids.dtype, device=inputs.input_ids.device)
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
    logps = []
    for i, tok_id in enumerate(new_token_ids):
        # logits at position p predict token p+1, so for new tokens at
        # positions [prompt_len, ..., prompt_len+N-1] we read logits at
        # [prompt_len-1, ..., prompt_len+N-2].
        logit_pos = prompt_len - 1 + i
        step_logp = torch.nn.functional.log_softmax(logits[logit_pos].float(), dim=-1)
        logps.append(float(step_logp[int(tok_id)].item()))

    tokenizer = getattr(processor, "tokenizer", processor)
    tokens = tokenizer.convert_ids_to_tokens(new_token_ids)
    prediction = tokenizer.decode(new_token_ids, skip_special_tokens=False)

    sum_logp = float(sum(logps))
    mean_logp = sum_logp / max(len(logps), 1)

    return {
        "sample_id": str(sample["sample_id"]),
        "dataset": sample.get("dataset", ""),
        "prompt_text": sample["prompt_text"],
        # Keep both image_paths and audio_paths so the baseline file
        # round-trips: oracle_check.py reads either, and downstream
        # tooling can tell at a glance which modalities a sample uses.
        "image_paths": sample.get("image_paths", []),
        "audio_paths": sample.get("audio_paths", []),
        "answer": sample.get("answer", ""),
        "prediction": prediction,
        "token_ids": new_token_ids,
        "tokens": tokens,
        "logps": logps,
        "sum_logp": sum_logp,
        "mean_logp": mean_logp,
        "num_tokens": len(new_token_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate HF reference baseline (token_ids + logps) for Pangu Omni audio samples."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Path to samples JSONL (e.g. audio_demo.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the generated baseline JSONL.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ["PANGU_MODEL_DIR"]) if "PANGU_MODEL_DIR" in os.environ else None,
        help="Pangu Omni model directory (config.json + safetensors).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of samples from the JSONL head to process. -1 = all.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help=(
            "Cap on greedy generation length. 64 is enough for the "
            "LlamaFactory demo answers (the longest is ~25 tokens). "
            "Bumping this is essentially free on the NPU at this batch "
            "size; the choice mostly controls how much logp signal "
            "downstream oracle_check has to compare."
        ),
    )
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error("set --model-dir explicitly, or set PANGU_MODEL_DIR")

    print("[config]")
    print(f"  Samples:        {args.samples}")
    print(f"  Out:            {args.out}")
    print(f"  Model dir:      {args.model_dir}")
    print(f"  N samples:      {args.n_samples}")
    print(f"  Max new tokens: {args.max_new_tokens}")

    samples = load_jsonl(args.samples)
    if args.n_samples > 0:
        samples = samples[: args.n_samples]
    if not samples:
        print("[ERROR] empty sample set.")
        return 2

    # Resolve relative media paths against the directory containing
    # ``data/`` — the JSONL convention used by both OCRBench and our
    # audio oracle puts media under ``<root>/data/<modality>_files/``,
    # so ``samples.parent.parent`` is the right anchor.
    samples_root = args.samples.parent.parent
    print(f"  Samples root:   {samples_root}")
    for s in samples:
        print(
            f"    - {s['sample_id']:<22s} "
            f"prompt={s['prompt_text'][:40]!r:<45s} "
            f"audio={s.get('audio_paths', s.get('image_paths', []))}"
        )

    print(f"\n[load] loading HF reference in mode=hf ({args.model_dir})...")
    t0 = time.time()
    model, processor = load_hf_model_and_processor(args.model_dir)
    print(f"[load] done in {time.time() - t0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for i, sample in enumerate(samples):
            t1 = time.time()
            row = generate_one(
                model,
                processor,
                sample,
                samples_root=samples_root,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.time() - t1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"  [{i + 1}/{len(samples)}] {row['sample_id']:<22s} "
                f"n_tokens={row['num_tokens']:<3d} mean_logp={row['mean_logp']:.4f} "
                f"prediction={row['prediction'][:60]!r:<63s} "
                f"({elapsed:.1f}s)"
            )

    print(f"\n[done] wrote {len(samples)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
