"""Probe NPU tensor formats at the first audio VGG Conv2d.

The current audio_2 precision investigation has narrowed the first mismatch to
`audio_tower.conv_layers.0.layers.0` (Conv2d). Standalone NPU Conv2d with the
same disk weight and same dumped input exactly matches the HF reference, while
the Conv2d inside the VeOmni-loaded model differs by two bf16 elements.

This script prints the tensor value/stride/format seen by the actual first
Conv2d in either HF or VeOmni mode. The goal is to determine whether VeOmni's
full-model load path gives the Conv2d a different input layout/internal format
even though the values are identical.
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


def _fmt(t: torch.Tensor) -> str:
    parts = [
        f"shape={tuple(t.shape)}",
        f"dtype={t.dtype}",
        f"device={t.device}",
        f"stride={tuple(t.stride())}",
        f"contig={t.is_contiguous()}",
    ]
    if t.device.type == "npu":
        try:
            import torch_npu

            parts.append(f"npu_format={torch_npu.get_npu_format(t)}")
        except Exception as exc:  # pragma: no cover - diagnostic
            parts.append(f"npu_format=<err {type(exc).__name__}: {exc}>")
    return " ".join(parts)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hf", "veomni"), required=True)
    parser.add_argument("--sample-id", default="audio_2")
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
    conv0 = audio_tower.conv_layers[0].layers[0]
    print("[conv0 weight]", _fmt(conv0.weight))
    print("[conv0 bias]", _fmt(conv0.bias))
    print(
        "[conv0 weight probes]",
        "w_sum=",
        float(conv0.weight.detach().cpu().float().sum()),
        "b_sum=",
        float(conv0.bias.detach().cpu().float().sum()),
    )

    seen = {}

    def pre_hook(_module, conv_inputs):
        x = conv_inputs[0]
        seen["conv_input"] = x.detach()
        print("[conv0 input]", _fmt(x))
        if x.numel() > 0:
            print(
                "[conv0 input sample]",
                "x[0,0,0]=",
                float(x.flatten()[0].detach().cpu().float()),
                "x_sum=",
                float(x.detach().cpu().float().sum()),
            )

    def post_hook(_module, _conv_inputs, output):
        print("[conv0 output]", _fmt(output))
        # The two coordinates that differ in audio_2.
        if output.ndim == 3 and output.shape[0] > 54:
            print(
                "[conv0 output probes]",
                "c32_t396_m8=",
                float(output[32, 396, 8].detach().cpu().float()),
                "c54_t319_m25=",
                float(output[54, 319, 25].detach().cpu().float()),
            )

    h1 = conv0.register_forward_pre_hook(pre_hook)
    h2 = conv0.register_forward_hook(post_hook)
    try:
        feature_attention_mask = inputs.get("feature_attention_mask")
        input_features = inputs["input_features"]
        print("[raw input_features]", _fmt(input_features))
        if feature_attention_mask is not None:
            feature_lens = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            feature_lens = inputs.get("audio_feature_lengths")
        print("[packed input_features before dtype]", _fmt(input_features))
        if args.mode == "hf":
            audio_dtype = torch.bfloat16
        else:
            audio_dtype = next(audio_tower.parameters()).dtype
        input_features = input_features.to(dtype=audio_dtype)
        print("[packed input_features after dtype]", _fmt(input_features))
        print("[feature_lens]", feature_lens.detach().cpu().tolist(), _fmt(feature_lens))

        with torch.no_grad():
            audio_outputs, audio_output_lengths = audio_tower(
                input_features,
                feature_lens=feature_lens,
            )
        # Run a standalone conv2d in the same process with cloned weights/input.
        # This detects whether torch_npu dispatch changes based on module/parameter
        # object state versus raw tensor values.
        x = seen["conv_input"]
        with torch.no_grad():
            y_func = torch.nn.functional.conv2d(
                x,
                conv0.weight.detach().clone(),
                conv0.bias.detach().clone(),
                stride=conv0.stride,
                padding=conv0.padding,
                dilation=conv0.dilation,
                groups=conv0.groups,
            )
        print("[functional conv output]", _fmt(y_func))
        print(
            "[functional conv probes]",
            "c32_t396_m8=",
            float(y_func[32, 396, 8].detach().cpu().float()),
            "c54_t319_m25=",
            float(y_func[54, 319, 25].detach().cpu().float()),
        )
        print("[audio_output_lengths]", audio_output_lengths.detach().cpu().tolist())
        print(
            "[audio final]",
            _fmt(audio_outputs.last_hidden_state),
            "sum=",
            float(audio_outputs.last_hidden_state.detach().cpu().float().sum()),
        )
    finally:
        h1.remove()
        h2.remove()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
