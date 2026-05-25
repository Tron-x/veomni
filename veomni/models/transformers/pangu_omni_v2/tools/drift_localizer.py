"""Pangu Omni VL — per-stage drift localizer.

Runs the **same input** through:

- The HF reference model (`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`),
- Our VeOmni adapter (`OpenPanguUltraOmniForConditionalGeneration` → `OpenPanguVL`),

both loaded simultaneously on different NPUs, and prints the
``max_abs_diff`` at each checkpoint along the forward pipeline:

    1. text embed (input_ids → embed_tokens)         — text-only, no vision
    2. vision tower output (pixel_values → visual.forward, pre-projection)
    3. vision projection output (vision_projection)
    4. inputs_embeds after masked_scatter merge      — pre-language-model
    5. position_ids (3D mrope)                       — integer, must match exactly
    6. last_hidden_state                             — post-language-model
    7. logits                                        — post-lm_head

Where the diff first **leaps** is where adapter drift originates.

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0,1 python tools/pangu_drift_localizer.py \\
        --n-samples 1 --hf-device npu:0 --veomni-device npu:1

Requires 2 NPUs with ≥65 GB HBM each.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


PANGU_INFER_DIR = Path("/mnt/data_3/models/pangu/test_hf_percision.0518.parallel")
PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
DEFAULT_SAMPLES = PANGU_INFER_DIR / "data" / "ocrbench.jsonl"
DEFAULT_BASELINE = PANGU_INFER_DIR / "results" / "ocrbench_hf_outputs.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def setup_npu() -> None:
    """Match infer.py / oracle_check.py NPU setup so all paths align."""
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401


def load_hf(model_dir: Path, device: str, attn_implementation: str = "eager") -> tuple[Any, Any]:
    """Load reference HF model on the specified device.

    The production ``config.json`` carries
    ``_attn_selection = "npu_fusion_attention"``, which HF's loader
    would otherwise honor (selecting the NPU fused attention kernel).
    For bit-for-bit drift localization we explicitly force
    ``attn_implementation="eager"`` on BOTH sides so the kernel-level
    differences are eliminated; that lets us isolate model-code drift
    from kernel-numeric drift.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation=attn_implementation,
        )
        .eval()
        .to(device)
    )
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    return model, processor


def load_veomni(model_dir: Path, device: str) -> tuple[Any, Any]:
    """Load our adapter model on the specified device."""
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model, build_processor

    ops_cfg = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
    )
    model = build_foundation_model(
        config_path=str(model_dir),
        weights_path=str(model_dir),
        torch_dtype="bfloat16",
        init_device=device,
        ops_implementation=ops_cfg,
    ).eval()
    processor = build_processor(str(model_dir))
    return model, processor


def prepare_inputs(processor, sample: dict, device: str):
    """Replicate oracle_check.py:compute_per_token_logps input prep, but
    return the inputs dict ready to feed through the model."""
    from qwen_omni_utils import process_mm_info

    def _resolve(s):
        out = []
        for p in s.get("image_paths", []):
            path = Path(p)
            if not path.is_absolute():
                path = PANGU_INFER_DIR / path
            out.append(str(path))
        return out

    image_paths = _resolve(sample)
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "image": p} for p in image_paths]
            + [{"type": "text", "text": sample["prompt_text"]}],
        }
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        audio=audios,
        padding=False,
        return_tensors="pt",
    ).to(device)
    return inputs


def diff(a, b, name: str) -> str:
    """Compute max_abs / mean_abs / shape report between two tensors,
    moving both to CPU + float32 for fair comparison."""

    if a is None and b is None:
        return f"  {name:<35s}  both None"
    if a is None or b is None:
        return f"  {name:<35s}  one is None (a={a is not None}, b={b is not None})"
    a_cpu = a.detach().cpu().float()
    b_cpu = b.detach().cpu().float()
    if a_cpu.shape != b_cpu.shape:
        return f"  {name:<35s}  SHAPE MISMATCH a={tuple(a_cpu.shape)} b={tuple(b_cpu.shape)}"
    d = (a_cpu - b_cpu).abs()
    return f"  {name:<35s}  shape={tuple(a_cpu.shape)}  max={d.max().item():.3e}  mean={d.mean().item():.3e}"


def install_hooks(model, label: str, captured: dict) -> list:
    """Hook the key forward checkpoints on either the ref or ours model.

    Structural disclaimer: the reference's production class is
    ``OpenPanGuOmni`` (from ``modeling_pangu_omni.py``, **flat**:
    ``self.model`` = text backbone, ``self.visual`` = vision tower
    top-level, ``self.audio_tower`` = audio top-level, ``self.lm_head``
    top-level). Our adapter's class is ``OpenPanguVL`` (Week 3.4.d port
    of ``modeling_openpangu_vl.py:OpenPanguVL``, **nested**:
    ``self.model`` = ``OpenPanguVLModel`` which itself has ``.visual``
    + ``.language_model``, plus top-level ``self.lm_head``).

    This means we have to dispatch the visual / language_model lookups
    differently between the two — but the SAME ``nn.Module`` types are
    used internally (``OpenPanguVisionTransformerPretrainedModel`` /
    ``OpenPanguVLTextModel``), so the captured tensor SHAPES align and
    we can do a per-element diff at each stage.

    Captures:

    - text_embed_pre_merge: ``self.get_input_embeddings()(input_ids)``
      output, post MHC ``repeat``, pre ``masked_scatter``.
    - vision_tower_out: ``visual(pixel_values, grid_thw)`` raw output
      (pre-projection).
    - vision_projection_out: ``visual.vision_projection(visual_out)``.
    - last_hidden_state: text backbone output (the thing fed into
      ``lm_head``).
    - logits: ``lm_head`` output.
    """
    handles = []

    # Resolve visual / text_backbone according to which side we're hooking.
    # Both sides have `model.lm_head`. Visual + text differ.
    is_ours = type(model).__module__.startswith("veomni.")

    if is_ours:
        # Nested: model.model.visual, model.model.language_model
        visual = model.model.visual
        text_backbone = model.model.language_model
        embed_tokens = model.model.language_model.embed_tokens
    else:
        # Flat: model.visual, model.model
        visual = model.visual
        text_backbone = model.model  # OpenPanguVLTextModel directly
        embed_tokens = model.model.embed_tokens
    lm_head = model.lm_head

    def cap_visual(_mod, _inp, out):
        captured[f"{label}:vision_tower_out"] = out.detach()

    handles.append(visual.register_forward_hook(cap_visual))

    def cap_vp(_mod, _inp, out):
        captured[f"{label}:vision_projection_out"] = out.detach()

    handles.append(visual.vision_projection.register_forward_hook(cap_vp))

    def cap_embed(_mod, _inp, out):
        # Only capture the FIRST embed call (the one off input_ids in
        # forward). MHC repeat happens after this in the parent
        # forward — we'll see it land in the language_model inputs_embeds.
        if f"{label}:text_embed" not in captured:
            captured[f"{label}:text_embed"] = out.detach()

    handles.append(embed_tokens.register_forward_hook(cap_embed))

    def cap_text_backbone(_mod, inp, out):
        # Save the inputs_embeds that goes INTO the text backbone — this
        # is post-MHC-repeat post-masked_scatter, the final fused
        # embedding the text layers consume.
        # `inp` is a tuple; we want the keyword `inputs_embeds` which
        # PyTorch positional/kwarg-merges before passing. The pre-hook
        # has access to (args, kwargs); register a separate pre-hook
        # for that.
        lhs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        captured[f"{label}:last_hidden_state"] = lhs.detach()

    handles.append(text_backbone.register_forward_hook(cap_text_backbone))

    def cap_text_backbone_pre(_mod, args, kwargs):
        # Capture inputs_embeds + position_ids going INTO the text backbone.
        ie = kwargs.get("inputs_embeds")
        pi = kwargs.get("position_ids")
        am = kwargs.get("attention_mask")
        if ie is not None:
            captured[f"{label}:inputs_embeds_to_backbone"] = ie.detach()
        if pi is not None:
            captured[f"{label}:position_ids_to_backbone"] = pi.detach()
        if am is not None and hasattr(am, "detach"):
            captured[f"{label}:attn_mask_to_backbone"] = am.detach()
        return None

    handles.append(text_backbone.register_forward_pre_hook(cap_text_backbone_pre, with_kwargs=True))

    # Per-layer captures (post-layer output) — used for binary-search drift
    # localization across the 37-layer text backbone.
    layers = text_backbone.layers
    for i, layer in enumerate(layers):

        def _make_cap(idx):
            def _cap(_mod, _inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                captured[f"{label}:layer_{idx:02d}_out"] = t.detach()

            return _cap

        handles.append(layer.register_forward_hook(_make_cap(i)))

    def cap_logits(_mod, _inp, out):
        captured[f"{label}:logits"] = out.detach()

    handles.append(lm_head.register_forward_hook(cap_logits))

    return handles


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--hf-device", type=str, default="npu:0")
    parser.add_argument("--veomni-device", type=str, default="npu:1")
    parser.add_argument("--model-dir", type=Path, default=PANGU_MODEL_DIR)
    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print("  Pangu Omni VL — per-stage drift localizer")
    print(f"{'=' * 72}")
    print(f"  hf device:    {args.hf_device}")
    print(f"  veomni dev:   {args.veomni_device}")
    print(f"  model dir:    {args.model_dir}")
    print()

    setup_npu()

    samples = load_jsonl(args.samples)
    baseline = {str(r["sample_id"]): r for r in load_jsonl(args.baseline)}
    picked = [s for s in samples if str(s["sample_id"]) in baseline][: args.n_samples]
    print(f"  Picked {len(picked)} samples: {[s['sample_id'] for s in picked]}")
    print()

    print("[load] HF reference ...")
    import time

    t0 = time.time()
    hf_model, hf_proc = load_hf(args.model_dir, args.hf_device)
    print(f"  done in {time.time() - t0:.1f}s")

    print("[load] VeOmni adapter ...")
    t0 = time.time()
    veomni_model, veomni_proc = load_veomni(args.model_dir, args.veomni_device)
    print(f"  done in {time.time() - t0:.1f}s")

    captured = {}
    hf_handles = install_hooks(hf_model, "hf", captured)
    ours_handles = install_hooks(veomni_model, "ours", captured)

    try:
        for sample in picked:
            sid = str(sample["sample_id"])
            bl = baseline[sid]
            print(f"\n{'=' * 72}")
            print(f"  Sample: {sid}  (n_tokens={bl['num_tokens']})")
            print(f"{'=' * 72}")

            # Prepare inputs separately for each side (different devices)
            print("  preparing inputs (hf side) ...")
            hf_inputs = prepare_inputs(hf_proc, sample, args.hf_device)
            print("  preparing inputs (veomni side) ...")
            veomni_inputs = prepare_inputs(veomni_proc, sample, args.veomni_device)

            # Confirm input parity at the start — both processors should give
            # identical tokenization
            input_ids_match = torch.equal(hf_inputs.input_ids.cpu(), veomni_inputs.input_ids.cpu())
            print(f"  input_ids match between sides: {input_ids_match}")
            if not input_ids_match:
                print("  [WARN] input_ids differ — processor mismatch")
                print(f"    hf shape: {tuple(hf_inputs.input_ids.shape)}")
                print(f"    ours shape: {tuple(veomni_inputs.input_ids.shape)}")
                continue

            # Compare pixel_values
            pv_match = torch.allclose(
                hf_inputs.pixel_values.cpu().float(),
                veomni_inputs.pixel_values.cpu().float(),
                atol=1e-6,
                rtol=0.0,
            )
            print(f"  pixel_values match (atol=1e-6): {pv_match}")

            # Teacher-forced forward — append baseline tokens to input_ids
            baseline_tokens = torch.tensor(
                [bl["token_ids"]],
                dtype=hf_inputs.input_ids.dtype,
                device=hf_inputs.input_ids.device,
            )
            hf_full = torch.cat([hf_inputs.input_ids, baseline_tokens], dim=1)
            veomni_full = torch.cat(
                [veomni_inputs.input_ids, baseline_tokens.to(args.veomni_device)],
                dim=1,
            )
            attn_pad = lambda inp, side, _bt=baseline_tokens: torch.cat(  # noqa: E731, B023
                [
                    inp["attention_mask"],
                    torch.ones_like(_bt.to(side), dtype=inp["attention_mask"].dtype),
                ],
                dim=1,
            )
            hf_extra = {k: v for k, v in hf_inputs.items() if k != "input_ids"}
            hf_extra["attention_mask"] = attn_pad(hf_inputs, args.hf_device)
            veomni_extra = {k: v for k, v in veomni_inputs.items() if k != "input_ids"}
            veomni_extra["attention_mask"] = attn_pad(veomni_inputs, args.veomni_device)

            captured.clear()
            print("  forward (hf) ...")
            with torch.no_grad():
                hf_out = hf_model(input_ids=hf_full, **hf_extra)
            print("  forward (veomni) ...")
            with torch.no_grad():
                ours_out = veomni_model(input_ids=veomni_full, **veomni_extra)

            print()
            print("  Per-stage diff (max_abs, mean_abs):")
            print()
            stages = [
                "text_embed",
                "vision_tower_out",
                "vision_projection_out",
                "inputs_embeds_to_backbone",
                "position_ids_to_backbone",
                "attn_mask_to_backbone",
            ]
            # Add every text-backbone layer
            n_layers = sum(1 for k in captured if k.startswith("hf:layer_") and k.endswith("_out"))
            for i in range(n_layers):
                stages.append(f"layer_{i:02d}_out")
            stages.extend(["last_hidden_state", "logits"])
            for stage in stages:
                print(
                    diff(
                        captured.get(f"hf:{stage}"),
                        captured.get(f"ours:{stage}"),
                        stage,
                    )
                )
            print()
            # Also report final logits at the baseline-token positions
            prompt_len = int(hf_inputs.input_ids.shape[1])
            n_baseline = baseline_tokens.shape[1]
            hf_logits_b = hf_out.logits[0, prompt_len - 1 : prompt_len - 1 + n_baseline]
            ours_logits_b = ours_out.logits[0, prompt_len - 1 : prompt_len - 1 + n_baseline]
            print(
                diff(
                    hf_logits_b,
                    ours_logits_b,
                    "logits_at_baseline_positions",
                )
            )
    finally:
        for h in hf_handles + ours_handles:
            h.remove()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
