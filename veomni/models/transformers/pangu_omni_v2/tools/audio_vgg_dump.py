"""Dump VGG sublayer outputs for Pangu audio sample.

This is a narrow diagnostic for the audio_2 precision issue:

- audio_0: HF and VeOmni match bit-for-bit at every audio_tower layer.
- audio_2: inputs match, but first divergence appears at `vgg_out`.

This script loads either HF or VeOmni, builds the same audio input as the
oracle, runs only `audio_tower.run_VGG_layers(...)`, and hooks every sublayer
inside `audio_tower.conv_layers[*].layers[*]` to find the first VGG op that
diverges.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _find_audio_tower(model):
    if hasattr(model, "audio_tower"):
        return model.audio_tower
    if hasattr(model, "model") and hasattr(model.model, "audio_tower"):
        return model.model.audio_tower
    raise RuntimeError("no audio_tower")


def _build_inputs(processor, sample, samples_root):
    from oracle_check import (
        build_conversation,
        resolve_audio_paths,
        resolve_image_paths,
    )
    from qwen_omni_utils import process_mm_info

    image_paths = resolve_image_paths(sample, samples_root)
    audio_paths = resolve_audio_paths(sample, samples_root)
    conversation = build_conversation(image_paths, sample["prompt_text"], text_only=False, audio_paths=audio_paths)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    return processor(
        text=text,
        images=images,
        videos=videos,
        audio=audios,
        padding=False,
        return_tensors="pt",
    )


def _install_vgg_hooks(audio_tower):
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().cpu().float()

        return hook

    for block_idx, block in enumerate(audio_tower.conv_layers):
        for layer_idx, layer in enumerate(block.layers):
            name = f"vgg_block{block_idx}_layer{layer_idx}_{layer.__class__.__name__}"
            handles.append(layer.register_forward_hook(make_hook(name)))
    return handles, captured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hf", "veomni"), required=True)
    parser.add_argument("--sample-id", default="audio_2")
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model"),
    )
    args = parser.parse_args()

    from oracle_check import (
        load_hf_model_and_processor,
        load_veomni_model_and_processor,
    )

    print(f"[load] mode={args.mode}")
    t0 = time.time()
    if args.mode == "hf":
        model, processor = load_hf_model_and_processor(args.model_dir)
    else:
        model, processor = load_veomni_model_and_processor(args.model_dir)
    print(f"[load] done in {time.time() - t0:.1f}s")

    samples_path = Path("/mnt/data_3/models/pangu_audio_oracle/data/audio_demo.jsonl")
    sample = next(
        json.loads(line)
        for line in samples_path.read_text().splitlines()
        if line.strip() and json.loads(line)["sample_id"] == args.sample_id
    )
    inputs = _build_inputs(processor, sample, samples_path.parent.parent).to(model.device)
    audio_tower = _find_audio_tower(model)

    feature_attention_mask = inputs.get("feature_attention_mask")
    input_features = inputs["input_features"]
    if feature_attention_mask is not None:
        feature_lens = torch.sum(feature_attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    else:
        feature_lens = inputs.get("audio_feature_lengths")

    if args.mode == "hf":
        audio_dtype = torch.bfloat16
    else:
        audio_dtype = next(audio_tower.parameters()).dtype
    input_features = input_features.to(dtype=audio_dtype, device=model.device)
    feature_lens = feature_lens.to(model.device)

    cu_in_seqlen = torch.cat(
        (
            torch.zeros(1, device=feature_lens.device, dtype=torch.int32),
            feature_lens.cumsum(0),
        )
    ).to(torch.int32)
    hidden_states = input_features.transpose(-1, -2).unsqueeze(0).contiguous()

    handles, captured = _install_vgg_hooks(audio_tower)
    try:
        x_list = []
        for i in range(feature_lens.shape[0]):
            x_list.append(audio_tower.run_VGG_layers(hidden_states[:, cu_in_seqlen[i] : cu_in_seqlen[i + 1], :]))
        captured["vgg_final"] = torch.cat(x_list, dim=1).detach().cpu().float()
    finally:
        for handle in handles:
            handle.remove()

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in captured.items():
        torch.save(tensor, args.dump_dir / f"{name}.pt")
        print(f"  [dump] {name:<40s} shape={tuple(tensor.shape)}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
