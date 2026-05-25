"""Dump intermediate tensors during audio_2 teacher-forced forward.

Stage 4c localization tool. The Week 3.5 audio oracle showed:

    audio_0 (4.03s, 2 toks) | max_diff = 0      bit-identical
    audio_1 (2.90s, 3 toks) | max_diff = 0      bit-identical
    audio_2 (5.86s, 20 toks) | max_diff = 4.6e-2  long-audio divergence

Short audio is bit-identical between HF reference and VeOmni adapter,
which proves the full pipeline (audio_tower forward + merger.proj + LLM
backbone + lm_head) is structurally correct. Long audio diverges. This
script binary-searches the divergence by hooking 4 intermediate tensors
along the forward path and dumping them to disk for each mode (HF /
VeOmni). The companion ``pangu_audio_diff.py`` script then diffs the
two dump dirs and prints which layer is the first to diverge.

The four checkpoint tensors:

1. ``audio_outputs_last_hidden_state``  ← raw audio_tower output
   (after VGG + 16 Conformer blocks, before the audio-tower-internal
   linear projection that drops dim to the LLM hidden size). Captures
   any divergence inside the audio_tower internals (audio mha attention
   kernel, RMSNorm, VGGBlock, etc.).

2. ``audio_features_post_proj``  ← after ``self.audio_tower.proj(...)``
   in ``get_audio_features``. Same as #1 plus the final linear proj
   (audio dim -> ``hidden_size * mhc_num_stream``). Mainly a sanity check
   on the proj layer; if #1 matches but #2 diverges, proj weights got
   mis-loaded.

3. ``inputs_embeds_after_audio_merge``  ← the LLM input embeddings
   after ``inputs_embeds.masked_scatter(audio_mask, audio_features)``.
   Captures any token-id / mask-position mismatch.

4. ``last_hidden_state``  ← LLM final hidden state pre-lm_head, i.e.
   ``outputs.last_hidden_state``. If #3 matches but #4 diverges, the
   LLM backbone forward itself is shape/position-dependent.

Run:
  PY tools/pangu_audio_intermediate_dump.py \\
     --mode hf \\
     --dump-dir /tmp/audio2_dump_hf
  PY tools/pangu_audio_intermediate_dump.py \\
     --mode veomni \\
     --dump-dir /tmp/audio2_dump_veomni
  PY tools/pangu_audio_diff.py \\
     --hf-dir /tmp/audio2_dump_hf \\
     --veomni-dir /tmp/audio2_dump_veomni
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


# Defer heavy imports until after argparse for a snappier --help.
def _build_audio2_inputs(processor, sample: dict[str, Any], samples_root: Path):
    """Mirror ``compute_per_token_logps`` from ``pangu_oracle_check.py``
    so this script feeds the model the SAME tensors as the oracle check.

    Returns ``(model_input_dict, prompt_len)``. The model input has
    ``input_ids`` already extended with the baseline ``new_token_ids``
    (teacher-forced) so that downstream forward sees the SAME 113-token
    sequence shape as the oracle.
    """
    from oracle_check import (
        build_conversation,
        resolve_audio_paths,
        resolve_image_paths,
    )

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
    )
    return inputs


def _install_hooks(model, dump_dir: Path):
    """Install forward hooks on the four checkpoint sites. Returns the
    list of handles so the caller can ``.remove()`` them after the
    forward pass. Tensor dumps happen via closure side-effect into the
    ``captured`` dict (keys = checkpoint names).

    The hook sites are addressed by walking the model attribute tree
    rather than by submodule names so the same code works for both
    HF reference (``model.audio_tower`` directly) and VeOmni adapter
    (``model.model.audio_tower`` — nested through OpenPanguOmniModel).
    """
    import torch

    captured: dict[str, torch.Tensor] = {}

    # Find audio_tower regardless of nesting level
    if hasattr(model, "audio_tower"):
        audio_tower = model.audio_tower
    elif hasattr(model, "model") and hasattr(model.model, "audio_tower"):
        audio_tower = model.model.audio_tower
    else:
        raise RuntimeError("Could not locate audio_tower on model")

    audio_proj = audio_tower.proj

    # Find the LLM backbone (final layer output) so we can hook
    # last_hidden_state. For HF reference: model.model (nested via
    # OpenPanGuOmni's submodule that exposes language_model). For VeOmni:
    # the OpenPanguOmniModel has `.language_model` directly. Try multiple
    # paths defensively.
    if hasattr(model, "language_model"):
        llm_backbone = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        llm_backbone = model.model.language_model
    else:
        # HF reference: model.model IS the language model
        llm_backbone = model.model

    def _hook_audio_tower(module, _inputs, output):
        # Also capture the inputs to audio_tower: input_features tensor
        # and feature_lens kwarg. Use _inputs which is the positional
        # args tuple (input_features, ...) — kwargs aren't visible to
        # forward hooks pre-PyTorch 2.0; we use a pre-hook for those.
        if len(_inputs) > 0 and hasattr(_inputs[0], "detach"):
            captured["audio_tower_input_features"] = _inputs[0].detach().cpu().float()
        # audio_tower returns (BaseModelOutput, audio_output_lengths)
        if isinstance(output, tuple):
            hidden = output[0].last_hidden_state
            if hasattr(output[1], "detach"):
                captured["audio_tower_output_lengths"] = output[1].detach().cpu()
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        else:
            hidden = output
        captured["audio_outputs_last_hidden_state"] = hidden.detach().cpu().float()

    def _pre_hook_audio_tower(module, args, kwargs):
        # Capture feature_lens (audio_feature_lengths) kwarg
        if "feature_lens" in kwargs and hasattr(kwargs["feature_lens"], "detach"):
            captured["audio_tower_feature_lens"] = kwargs["feature_lens"].detach().cpu()
        return None  # don't modify args

    def _hook_audio_proj(module, _inputs, output):
        captured["audio_features_post_proj"] = output.detach().cpu().float()

    def _hook_llm_backbone(module, _inputs, output):
        # LM backbone may return tuple, BaseModelOutputWithPast, etc.
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        captured["last_hidden_state"] = hidden.detach().cpu().float()

    handles = [
        audio_tower.register_forward_pre_hook(_pre_hook_audio_tower, with_kwargs=True),
        audio_tower.register_forward_hook(_hook_audio_tower),
        audio_proj.register_forward_hook(_hook_audio_proj),
        llm_backbone.register_forward_hook(_hook_llm_backbone),
    ]
    return handles, captured


def _dump_captured(captured: dict, dump_dir: Path):
    """Persist captured tensors and a small metadata file."""
    import torch

    dump_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in captured.items():
        torch.save(tensor, dump_dir / f"{name}.pt")
        print(f"  [dump] {name:<40s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    meta = {
        "keys": sorted(captured.keys()),
        "shapes": {k: list(v.shape) for k, v in captured.items()},
        "dtypes": {k: str(v.dtype) for k, v in captured.items()},
    }
    (dump_dir / "_meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("hf", "veomni"), required=True)
    p.add_argument(
        "--samples",
        type=Path,
        default=Path("/mnt/data_3/models/pangu_audio_oracle/data/audio_demo.jsonl"),
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("/mnt/data_3/models/pangu_audio_oracle/results/audio_hf_outputs.jsonl"),
    )
    p.add_argument("--sample-id", default="audio_2", help="Which sample to dump (default audio_2, the divergent one)")
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model"),
    )
    args = p.parse_args()

    # Import the oracle's loaders so we get the SAME model build path
    # as the actual oracle check, ensuring the dumps are apples-to-apples.
    from oracle_check import (
        load_hf_model_and_processor,
        load_veomni_model_and_processor,
    )

    print(f"[load] mode={args.mode} from {args.model_dir}")
    t0 = time.time()
    if args.mode == "hf":
        model, processor = load_hf_model_and_processor(args.model_dir)
    else:
        model, processor = load_veomni_model_and_processor(args.model_dir)
    print(f"[load] done in {time.time() - t0:.1f}s")

    # Pick the requested sample + its baseline new_token_ids
    samples = [json.loads(l) for l in args.samples.read_text().splitlines() if l.strip()]
    sample = next(s for s in samples if s["sample_id"] == args.sample_id)
    sample["_samples_root"] = args.samples.parent.parent
    samples_root = sample["_samples_root"]

    baselines = {}
    for line in args.baseline.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            baselines[rec["sample_id"]] = rec
    baseline = baselines[args.sample_id]
    new_token_ids = baseline["token_ids"]
    print(f"[input] sample={args.sample_id} audio={sample.get('audio_paths')} baseline_n_tokens={len(new_token_ids)}")

    # Build the SAME inputs that compute_per_token_logps does
    inputs = _build_audio2_inputs(processor, sample, samples_root)
    inputs = inputs.to(model.device)

    import torch

    prompt_len = inputs.input_ids.shape[1]
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
    print(
        f"[forward] prompt_len={prompt_len} total_seq_len={full_input_ids.shape[1]} "
        f"input_features.shape={tuple(extended_inputs.get('input_features', torch.tensor([])).shape)}"
    )

    handles, captured = _install_hooks(model, args.dump_dir)
    try:
        with torch.no_grad():
            outputs = model(input_ids=full_input_ids, **extended_inputs)
        # Also dump the final logits at the boundary positions (prompt_len-1 .. -1)
        # so we can sanity-check what the oracle script sees end-to-end.
        logits = outputs.logits[0]  # (T, V)
        boundary_logits = logits[prompt_len - 1 : prompt_len - 1 + len(new_token_ids)]
        captured["boundary_logits"] = boundary_logits.detach().cpu().float()
    finally:
        for h in handles:
            h.remove()

    print(f"[dump] writing intermediates to {args.dump_dir}")
    _dump_captured(captured, args.dump_dir)
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
