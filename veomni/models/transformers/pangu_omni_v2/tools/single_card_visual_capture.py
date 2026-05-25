"""Single-card visual tower capture for multi-card drift localization.

Loads ``OpenPanguOmni`` on a single NPU with ``init_device=npu`` —
no FSDP, no EP, no mixed-precision policy. For each OCRBench sample,
captures ``model.visual`` output via a forward hook, writes it as a
``.pt`` file alongside any matching 8-card capture
(``ocrbench_{i}_world8.pt`` produced by
``multi_card_multimodal_parity_runner.py`` with
``MM_PARITY_CAPTURE=1``), then element-wise diffs the two.

Validated 2026-05-23 on real Pangu Omni 30B-A2B with OCRBench:
all three sampled visual_output tensors had identical max diff
3.125e-2 = 1 bf16 ULP at magnitude ~30, mean diff ~3e-4 in the
(64-84, 3584) tensors. This is the floor of FSDP2 + bf16 across a
26-deep visual tower; see the "Multi-card multimodal" section of
``../README.md`` for how this propagates into per-sample logp drift
via softmax sensitivity.

Usage:
    python veomni/models/transformers/pangu_omni_v2/tools/single_card_visual_capture.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from veomni.models.transformers.pangu_omni_v2.tools.oracle_check import (
    build_conversation,
    resolve_audio_paths,
    resolve_image_paths,
)


def main() -> int:
    model_dir = "/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model"
    # Default to the OCRBench (image) corpus; allow override to audio_demo.jsonl
    # (or any other oracle corpus) via env so the script doubles as the
    # audio-side capture for multi-card parity bisection.
    samples_path = os.environ.get(
        "PARITY_SAMPLES",
        "/mnt/data_3/models/pangu/test_hf_percision.0518.parallel/data/ocrbench.jsonl",
    )
    baseline_path = os.environ.get(
        "PARITY_BASELINE",
        "/mnt/data_3/models/pangu/test_hf_percision.0518.parallel/results/ocrbench_hf_outputs.jsonl",
    )
    out_dir = Path(os.environ.get("PARITY_CAPTURE_DIR", "/tmp/pangu_multimodal_parity/captured"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(os.environ.get("PARITY_N_SAMPLES", "3"))

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model, build_processor

    # Default to ``eager`` MoE for max-precision single-card capture, but
    # allow override to ``fused_npu`` so we can compare the audio path
    # against the 8-card runner under the same MoE kernel (the 8-card
    # config requires ``fused_npu`` for ep_size>1, so this lets us tell
    # apart "fused_npu vs eager MoE topk drift" from "true multi-card drift").
    moe_impl = os.environ.get("PARITY_MOE_IMPL", "eager")
    print(f"Loading model (single-card, no FSDP, moe_impl={moe_impl})...")
    ops_impl = OpsImplementationConfig(
        attn_implementation="sdpa",
        moe_implementation=moe_impl,
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    model = build_foundation_model(
        config_path=model_dir,
        weights_path=model_dir,
        torch_dtype="bfloat16",
        init_device="npu",
        attn_implementation="sdpa",
        ops_implementation=ops_impl,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  device: {device}  dtype: {next(model.parameters()).dtype}")

    processor = build_processor(model_dir)

    samples = []
    with open(samples_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    baselines = {}
    with open(baseline_path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                baselines[row["sample_id"]] = row

    n_done = 0
    for sample in samples:
        if n_done >= n_samples:
            break
        sid = sample["sample_id"]
        if sid not in baselines:
            continue
        print(f"\n=== {sid} ===")

        samples_root = Path(samples_path).parent.parent
        image_paths = resolve_image_paths(sample, samples_root)
        audio_paths = resolve_audio_paths(sample, samples_root)
        conversation = build_conversation(image_paths, sample["prompt_text"], text_only=False, audio_paths=audio_paths)
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        from qwen_omni_utils import process_mm_info

        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            padding=False,
            return_tensors="pt",
        ).to(device)
        print(f"  input_ids shape: {tuple(inputs.input_ids.shape)}")
        if "image_grid_thw" in inputs:
            print(f"  image_grid_thw: {inputs['image_grid_thw'].tolist()}")
        if "pixel_values" in inputs:
            print(f"  pixel_values shape: {tuple(inputs['pixel_values'].shape)}")
        if "input_features" in inputs:
            print(f"  input_features shape: {tuple(inputs['input_features'].shape)}")

        bl = baselines[sid]
        baseline_ids = torch.tensor([bl["token_ids"]], dtype=inputs.input_ids.dtype, device=device)
        full_input_ids = torch.cat([inputs.input_ids, baseline_ids], dim=1)
        attention_mask = torch.ones_like(full_input_ids)

        forward_kwargs = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}
        forward_kwargs["input_ids"] = full_input_ids
        forward_kwargs["attention_mask"] = attention_mask

        if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = None

        captured: dict[str, torch.Tensor] = {}

        def _to_cpu_float(t):
            if hasattr(t, "to_local"):
                t = t.to_local()
            return t.detach().float().cpu().clone()

        # ruff B023: ``captured`` is captured by reference so the hook
        # writes to the dict we read later — intentional, not a leak.
        def _hook_visual(_mod, _inputs, output, _captured=captured):
            out = output
            if isinstance(out, tuple):
                out = out[0]
            _captured["visual_output"] = _to_cpu_float(out)

        inner = model.model
        hs = [inner.visual.register_forward_hook(_hook_visual)]

        # Deep capture: mirror the probe points from
        # ``multi_card_multimodal_parity_runner.py`` so the resulting captures
        # diff key-by-key for drift bisection.
        if os.environ.get("MM_PARITY_DEEP_CAPTURE"):
            v = inner.visual

            # Hooks below pin ``captured`` via the ``_captured=captured`` default
            # argument trick to defeat ruff B023 (the outer for-sample loop
            # rebinds ``captured`` each iteration).
            def _hk_patch(_mod, _inputs, output, _key="post_patch_embed", _captured=captured):
                _captured[_key] = _to_cpu_float(output)

            def _make_block_hook(_layer_num, _captured=captured):
                _key = f"post_block_{_layer_num:02d}"

                def _hk(_mod, _inputs, output, _captured=_captured, _key=_key):
                    out = output[0] if isinstance(output, tuple) else output
                    _captured[_key] = _to_cpu_float(out)

                return _hk

            def _hk_merger(_mod, _inputs, output, _key="post_merger", _captured=captured):
                out = output[0] if isinstance(output, tuple) else output
                _captured[_key] = _to_cpu_float(out)

            hs.append(v.patch_embed.register_forward_hook(_hk_patch))
            n_blocks = len(v.blocks)
            probe_idxs = sorted({0, n_blocks // 4, n_blocks // 2, (3 * n_blocks) // 4, n_blocks - 1})
            for li in probe_idxs:
                hs.append(v.blocks[li].register_forward_hook(_make_block_hook(li)))
            if hasattr(v, "merger"):
                if isinstance(v.merger, torch.nn.ModuleList):
                    for mi, m in enumerate(v.merger):
                        hs.append(
                            m.register_forward_hook(
                                lambda _m, _i, _o, _k=f"post_merger_{mi}", _c=captured: _c.update(
                                    {_k: _to_cpu_float(_o[0] if isinstance(_o, tuple) else _o)}
                                )
                            )
                        )
                else:
                    hs.append(v.merger.register_forward_hook(_hk_merger))
            captured["__probe_idxs__"] = torch.tensor(probe_idxs)

        # FINE capture: bisect inside block 0 (mirrors runner FINE path).
        if os.environ.get("MM_PARITY_FINE_CAPTURE"):
            b0 = inner.visual.blocks[0]

            def _hk_pre_b0(_mod, args, kwargs, _key="pre_block_00", _captured=captured):
                _hs = args[0] if args else kwargs.get("hidden_states")
                _captured[_key] = _to_cpu_float(_hs)

            def _hk_n1(_mod, _inputs, output, _key="b00_post_norm1", _captured=captured):
                _captured[_key] = _to_cpu_float(output)

            def _hk_attn(_mod, _inputs, output, _key="b00_post_attn", _captured=captured):
                out = output[0] if isinstance(output, tuple) else output
                _captured[_key] = _to_cpu_float(out)

            def _hk_n2(_mod, _inputs, output, _key="b00_post_norm2", _captured=captured):
                _captured[_key] = _to_cpu_float(output)

            def _hk_mlp(_mod, _inputs, output, _key="b00_post_mlp", _captured=captured):
                out = output[0] if isinstance(output, tuple) else output
                _captured[_key] = _to_cpu_float(out)

            hs.append(b0.register_forward_pre_hook(_hk_pre_b0, with_kwargs=True))
            hs.append(b0.norm1.register_forward_hook(_hk_n1))
            hs.append(b0.attn.register_forward_hook(_hk_attn))
            hs.append(b0.norm2.register_forward_hook(_hk_n2))
            hs.append(b0.mlp.register_forward_hook(_hk_mlp))

        # Audio deep capture: mirrors the audio probe points in
        # ``multi_card_multimodal_parity_runner.py`` for diff-by-key bisection.
        if os.environ.get("MM_PARITY_AUDIO_CAPTURE") and hasattr(inner, "audio_tower"):
            a = inner.audio_tower

            def _hk_audio_in(_mod, args, kwargs, _key="audio_input", _captured=captured):
                inp = args[0] if args else kwargs.get("input_features")
                if inp is not None:
                    _captured[_key] = _to_cpu_float(inp)

            def _hk_audio_lin_before(_mod, _inputs, output, _key="audio_lin_before", _captured=captured):
                _captured[_key] = _to_cpu_float(output)

            def _make_audio_layer_hook(_layer_num, _captured=captured):
                _key = f"audio_layer_{_layer_num:02d}"

                def _hk(_mod, _inputs, output, _captured=_captured, _key=_key):
                    out = output[0] if isinstance(output, tuple) else output
                    _captured[_key] = _to_cpu_float(out)

                return _hk

            def _hk_audio_lin_after(_mod, _inputs, output, _key="audio_lin_after", _captured=captured):
                _captured[_key] = _to_cpu_float(output)

            def _hk_audio_proj(_mod, _inputs, output, _key="audio_proj_out", _captured=captured):
                _captured[_key] = _to_cpu_float(output)

            def _hk_audio_tower_out(_mod, _inputs, output, _key="audio_tower_out", _captured=captured):
                if isinstance(output, tuple) and len(output) >= 1:
                    bmo = output[0]
                    if hasattr(bmo, "last_hidden_state"):
                        _captured[_key] = _to_cpu_float(bmo.last_hidden_state)

            hs.append(a.register_forward_pre_hook(_hk_audio_in, with_kwargs=True))
            if hasattr(a, "linear_before_attn"):
                hs.append(a.linear_before_attn.register_forward_hook(_hk_audio_lin_before))
            if hasattr(a, "layers") and len(a.layers) > 0:
                n_alayers = len(a.layers)
                audio_probe_idxs = sorted({0, n_alayers // 4, n_alayers // 2, (3 * n_alayers) // 4, n_alayers - 1})
                for li in audio_probe_idxs:
                    hs.append(a.layers[li].register_forward_hook(_make_audio_layer_hook(li)))
                captured["__audio_probe_idxs__"] = torch.tensor(audio_probe_idxs)
            if hasattr(a, "linear_after_attn"):
                hs.append(a.linear_after_attn.register_forward_hook(_hk_audio_lin_after))
            if hasattr(a, "proj"):
                hs.append(a.proj.register_forward_hook(_hk_audio_proj))
            hs.append(a.register_forward_hook(_hk_audio_tower_out))

        with torch.no_grad():
            outputs = model(**forward_kwargs)
        for h in hs:
            h.remove()

        logits = outputs.logits[0]
        prompt_len = int(inputs.input_ids.shape[1])
        first_logits = logits[prompt_len - 1].float()
        bl_first_id = int(bl["token_ids"][0])
        bl_first_logp = F.log_softmax(first_logits, dim=-1)[bl_first_id].item()
        bl_first_logit = first_logits[bl_first_id].item()
        top5 = torch.topk(first_logits, k=5)
        top5_tokens = [processor.tokenizer.decode([tid]) for tid in top5.indices.tolist()]
        top5_logps = F.log_softmax(first_logits, dim=-1)[top5.indices].tolist()
        top5_logits = top5.values.tolist()
        print(f"  first-token logit={bl_first_logit:+.4f} logp={bl_first_logp:+.4f}")
        print(f"  baseline first-token logp={bl['logps'][0]:+.4f}")
        print(f"  diff={abs(bl_first_logp - bl['logps'][0]):.4e}")
        print("  top5:")
        for t, lt, lp in zip(top5_tokens, top5_logits, top5_logps):
            print(f"    {t!r}: logit={lt:+.4f} logp={lp:+.4f}")

        out_path = out_dir / f"{sid}_world1.pt"
        torch.save(captured, out_path)

        primary_key = "audio_tower_out" if "audio_tower_out" in captured else "visual_output"
        primary = captured.get(primary_key)
        if primary is not None:
            print(f"  {primary_key}: shape={tuple(primary.shape)} norm={primary.norm().item():.6e}")
        print(f"  -> {out_path}")

        # If we have the 8-card capture, diff it now
        c8_path = out_dir / f"{sid}_world8.pt"
        if c8_path.exists() and primary is not None:
            c8 = torch.load(c8_path).get(primary_key)
            if c8 is not None and c8.shape == primary.shape:
                diff = (c8 - primary).abs()
                rel = diff / (primary.abs() + 1e-8)
                print(
                    f"  vs world8: max={diff.max().item():.4e} mean={diff.mean().item():.4e} max_rel={rel.max().item():.4e}"
                )
            elif c8 is not None:
                print(f"  shape mismatch: world1 {primary.shape} vs world8 {c8.shape}")

        n_done += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
