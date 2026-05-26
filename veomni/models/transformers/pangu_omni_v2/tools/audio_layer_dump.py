"""Layer-by-layer audio_tower dump for audio_2 localization.

Hooks checkpoints inside HuanyuAudioEncoder.forward:
  vgg_out, linear_before_attn_out, layer_{i}_out (i=0..N-1),
  linear_after_attn_out, ln_post_out, final_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def _run_audio_tower_with_checkpoints(audio_tower, input_features, feature_lens):
    """Reimplement HuanyuAudioEncoder.forward with intermediate dumps."""
    import torch

    captured: dict[str, torch.Tensor] = {}
    feature_lens = feature_lens.to(input_features.device)

    cu_inSeqlen = torch.cat(
        (
            torch.zeros(1, device=feature_lens.device, dtype=torch.int32),
            feature_lens.cumsum(0),
        )
    ).to(torch.int32)
    hidden_states = input_features.transpose(-1, -2).unsqueeze(0).contiguous()
    convOut_seq_lens = torch.zeros_like(feature_lens)
    x_list = []
    for i in range(feature_lens.shape[0]):
        x_list.append(audio_tower.run_VGG_layers(hidden_states[:, cu_inSeqlen[i] : cu_inSeqlen[i + 1], :]))
        convOut_seq_lens[i] = x_list[i].shape[1]
    cu_Seqlen_ConvOut = torch.cat(
        (
            torch.zeros(1, device=feature_lens.device, dtype=torch.int32),
            convOut_seq_lens.cumsum(0),
        )
    ).to(torch.int32)
    hidden_states = torch.cat(x_list, dim=1).transpose(-2, -3).contiguous().view(cu_Seqlen_ConvOut[-1], -1)
    captured["vgg_out"] = hidden_states.detach().cpu().float()

    hidden_states = audio_tower.linear_before_attn(hidden_states)
    captured["linear_before_attn_out"] = hidden_states.detach().cpu().float()

    position_ids = audio_tower.calc_ids(convOut_seq_lens)
    rotary_pos_emb = audio_tower.select_cos_sin(position_ids)

    for idx, encoder_layer in enumerate(audio_tower.layers):
        layer_outputs = encoder_layer(
            hidden_states,
            cu_Seqlen_ConvOut,
            audio_tower.layer_dw_conv_mask[idx],
            rotary_pos_emb,
            idx,
        )
        hidden_states = layer_outputs[0]
        captured[f"layer_{idx}_out"] = hidden_states.detach().cpu().float()

    hidden_states = audio_tower.linear_after_attn(hidden_states)
    captured["linear_after_attn_out"] = hidden_states.detach().cpu().float()

    hidden_states = audio_tower.ln_post(hidden_states)
    captured["ln_post_out"] = hidden_states.detach().cpu().float()

    if audio_tower.avg_pooler is not None:
        hidden_states = hidden_states.transpose(-1, -2)
        pooled_list = []
        for i in range(feature_lens.shape[0]):
            x = hidden_states[:, cu_Seqlen_ConvOut[i] : cu_Seqlen_ConvOut[i + 1]]
            dtype = x.dtype
            pooled = audio_tower.avg_pooler(x.float())
            pooled_list.append(pooled.to(dtype))
        hidden_states = torch.cat(pooled_list, dim=-1).transpose(-1, -2)
        convOut_seq_lens = convOut_seq_lens // 2

    captured["final_out"] = hidden_states.detach().cpu().float()
    return captured, convOut_seq_lens


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("hf", "veomni"), required=True)
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ["PANGU_MODEL_DIR"]) if "PANGU_MODEL_DIR" in os.environ else None,
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=Path(os.environ["PANGU_AUDIO_SAMPLES"]) if "PANGU_AUDIO_SAMPLES" in os.environ else None,
    )
    p.add_argument("--sample-id", default="audio_2")
    args = p.parse_args()
    if args.model_dir is None or args.samples is None:
        p.error("set --model-dir/--samples explicitly, or set PANGU_MODEL_DIR and PANGU_AUDIO_SAMPLES")

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

    samples_path = args.samples
    sample = next(
        json.loads(l)
        for l in samples_path.read_text().splitlines()
        if l.strip() and json.loads(l)["sample_id"] == args.sample_id
    )
    samples_root = samples_path.parent.parent
    inputs = _build_inputs(processor, sample, samples_root).to(model.device)

    # get_audio_features-style repack
    audio_tower = _find_audio_tower(model)
    feature_attention_mask = inputs.get("feature_attention_mask")
    input_features = inputs["input_features"]
    if feature_attention_mask is not None:
        feature_lens = torch.sum(feature_attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    else:
        feature_lens = inputs.get("audio_feature_lengths")

    if args.mode == "veomni":
        audio_dtype = next(audio_tower.parameters()).dtype
    else:
        audio_dtype = torch.bfloat16
    input_features = input_features.to(dtype=audio_dtype, device=model.device)
    feature_lens = feature_lens.to(device=model.device)

    print(f"[forward] input_features={tuple(input_features.shape)} feature_lens={feature_lens.tolist()}")
    captured, out_lens = _run_audio_tower_with_checkpoints(audio_tower, input_features, feature_lens)
    print(f"[forward] out_lens={out_lens.tolist()}")

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in captured.items():
        torch.save(tensor, args.dump_dir / f"{name}.pt")
        print(f"  [dump] {name:<28s} shape={tuple(tensor.shape)}")
    meta = {"keys": sorted(captured.keys()), "out_lens": out_lens.tolist()}
    (args.dump_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
