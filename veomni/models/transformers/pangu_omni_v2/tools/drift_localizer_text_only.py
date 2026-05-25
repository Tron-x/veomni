"""Text-only per-layer drift localizer for Pangu Omni v2.

Companion to ``tools/drift_localizer.py`` (multimodal-focused). This one
specifically isolates the **fused_npu MoE vs eager MoE** drift exposed by
the multi-card parity check. It runs the SAME text-only inputs through:

* HF reference (``OpenPanGuOmni`` via ``AutoModelForCausalLM
  +trust_remote_code=True``) on ``npu:0`` — uses the upstream eager
  Pangu MoE forward (verbatim ``for expert_idx in expert_hit: ...``
  loop). Attention is forced to ``eager`` on both sides so the only
  algorithmic difference is the MoE kernel.
* VeOmni adapter (``OpenPanguV2ForCausalLM`` via the text-only
  architecture override; same path the 8-card EP=8 SFT smoke uses)
  on ``npu:1`` — uses ``moe_implementation=fused_npu`` →
  ``veomni/ops/kernels/moe/npu_group_gemm.py::npu_fused_moe_forward``,
  which is built from the same ``torch_npu.npu_grouped_matmul`` +
  ``npu_moe_token_permute`` / ``unpermute`` primitives that
  ``torchtitan-npu/torchtitan_npu/converters/kernels/gmm.py`` uses for
  DeepSeek-V4 on NPU.

Per-layer post-decoder-block ``hidden_states`` are captured on each
side and diffed in float32 on CPU. The 30B-A2B config has
``first_k_dense_replace=2`` and ``num_hidden_layers=37``, so:

* layers 0, 1 — dense SwiGLU MLP. With ``attn_implementation=eager``
  on BOTH sides these should be **bit-identical** (max-diff ≈ 0).
  If they aren't, the drift includes a non-MoE source.
* layers 2–36 — MoE layers (eager loop vs fused group-GEMM). The
  shape of the curve (single big leap then flat vs linear accumulation
  through all 35 MoE layers) tells us the modal failure mode of the
  fused kernel.

Usage::

    ASCEND_RT_VISIBLE_DEVICES=0,1 \\
        python veomni/models/transformers/pangu_omni_v2/tools/drift_localizer_text_only.py \\
        --samples /tmp/pangu_text_only_oracle/samples.jsonl \\
        --n-samples 1 \\
        --hf-device npu:0 --veomni-device npu:1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
TEXT_ONLY_VIEW_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_text_only_view")


def _setup_npu() -> None:
    """Mirror ``oracle_check.setup_npu_if_available``."""
    from transformers.utils import is_torch_npu_available

    if is_torch_npu_available() and "910" in torch.npu.get_device_name():
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401


def load_hf(model_dir: Path, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[hf]     loading on {device} (eager attn, default eager MoE) ...")
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation="eager",
        )
        .eval()
        .to(device)
    )
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    print(f"[hf]     class = {type(model).__name__}")
    return model, processor


def load_veomni(model_dir: Path, device: str, moe_impl: str) -> tuple[Any, Any]:
    """Load OpenPanguV2ForCausalLM (text-only) with ``moe_impl`` routed
    expert kernel; everything else eager so the diff is pinned to MoE."""
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_config, build_foundation_model, build_processor

    config = build_config(str(model_dir))
    config.architectures = ["OpenPanguV2ForCausalLM"]
    ops = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation=moe_impl,
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    print(f"[veomni] loading on {device} (eager attn, moe={moe_impl}) ...")
    model = build_foundation_model(
        config_path=config,
        weights_path=str(model_dir),
        torch_dtype="bfloat16",
        init_device=device,
        ops_implementation=ops,
    ).eval()
    processor = build_processor(str(model_dir))
    # Match oracle_check.py: restore NPU internal format so kernel
    # selection matches HF reference.
    try:
        if hasattr(torch, "npu") and hasattr(torch.npu, "config"):
            torch.npu.config.allow_internal_format = True
    except Exception:
        pass
    print(f"[veomni] class = {type(model).__name__}")
    return model, processor


def install_layer_hooks(model: Any, label: str, captured: dict) -> list:
    """Register a forward hook on every decoder layer + final norm + lm_head.

    Both ``OpenPanGuOmni`` (HF) and ``OpenPanguV2ForCausalLM`` (VeOmni
    text-only override) expose the text backbone at ``model.model``, so
    the access pattern is identical. Layer index encodes the absolute
    position in the 37-layer backbone.
    """
    handles: list = []
    text_backbone = model.model  # OpenPanguV2Model / OpenPanGuOmni text backbone
    layers = text_backbone.layers
    n = len(layers)
    for i, layer in enumerate(layers):

        def _make_cap(idx: int):
            def _cap(_mod, _inp, out):
                # Decoder layer output: either tensor or (tensor, *aux).
                hs = out[0] if isinstance(out, tuple) else out
                captured[f"{label}:layer_{idx:02d}"] = hs.detach()

            return _cap

        handles.append(layer.register_forward_hook(_make_cap(i)))

    # Final norm
    def _cap_norm(_mod, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured[f"{label}:final_norm"] = hs.detach()

    if hasattr(text_backbone, "norm"):
        handles.append(text_backbone.norm.register_forward_hook(_cap_norm))

    # lm_head logits
    def _cap_logits(_mod, _inp, out):
        captured[f"{label}:logits"] = out.detach()

    handles.append(model.lm_head.register_forward_hook(_cap_logits))
    print(f"[{label}] installed hooks on {n} layers + final_norm + lm_head")
    return handles


def diff(a: torch.Tensor | None, b: torch.Tensor | None, name: str) -> tuple[str, float, float]:
    if a is None or b is None:
        return (f"  {name:<22s}  missing", -1.0, -1.0)
    a_cpu = a.detach().to("cpu", dtype=torch.float32)
    b_cpu = b.detach().to("cpu", dtype=torch.float32)
    if a_cpu.shape != b_cpu.shape:
        return (f"  {name:<22s}  SHAPE MISMATCH a={tuple(a_cpu.shape)} b={tuple(b_cpu.shape)}", -1, -1)
    d = (a_cpu - b_cpu).abs()
    mx = d.max().item()
    mn = d.mean().item()
    return (f"  {name:<22s}  max={mx:.3e}  mean={mn:.3e}", mx, mn)


def prepare_text_inputs(processor: Any, prompt_text: str, device: str) -> dict:
    """Same chat-template + tokenize path as oracle_check.py text-only."""
    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    chat_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat_text, padding=False, return_tensors="pt").to(device)
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=PANGU_MODEL_DIR)
    parser.add_argument("--veomni-model-dir", type=Path, default=TEXT_ONLY_VIEW_DIR)
    parser.add_argument("--samples", type=Path, default=Path("/tmp/pangu_text_only_oracle/samples.jsonl"))
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--hf-device", default="npu:0")
    parser.add_argument("--veomni-device", default="npu:1")
    parser.add_argument("--moe-impl", default="fused_npu", choices=["eager", "fused_npu"])
    args = parser.parse_args()

    _setup_npu()

    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    picked = samples[: args.n_samples]
    if not picked:
        print(f"[ERROR] no samples in {args.samples}", file=sys.stderr)
        return 2

    print(f"\n{'=' * 72}")
    print(f"  Text-only drift localizer — HF eager MoE  vs  VeOmni {args.moe_impl} MoE")
    print(f"{'=' * 72}")
    print(f"  HF model dir     = {args.model_dir}")
    print(f"  VeOmni model dir = {args.veomni_model_dir}  (text-only view)")
    print(f"  Samples          = {args.samples} (picked {len(picked)})")
    print(f"  HF device        = {args.hf_device}")
    print(f"  VeOmni device    = {args.veomni_device}")
    print()

    hf_model, hf_proc = load_hf(args.model_dir, args.hf_device)
    veomni_model, veomni_proc = load_veomni(args.veomni_model_dir, args.veomni_device, args.moe_impl)

    for s in picked:
        sid = s["sample_id"]
        print(f"\n--- {sid} ---")
        prompt = s["prompt_text"]

        # Prepare input_ids on both sides; verify they match (token-level
        # parity) before running the forwards.
        hf_inputs = prepare_text_inputs(hf_proc, prompt, args.hf_device)
        veo_inputs = prepare_text_inputs(veomni_proc, prompt, args.veomni_device)
        hf_ids = hf_inputs.input_ids.detach().cpu()
        veo_ids = veo_inputs.input_ids.detach().cpu()
        if not torch.equal(hf_ids, veo_ids):
            print(
                f"   [WARN] input_ids differ between HF and VeOmni processor — "
                f"shapes {tuple(hf_ids.shape)} vs {tuple(veo_ids.shape)}"
            )

        captured: dict[str, torch.Tensor] = {}
        hf_handles = install_layer_hooks(hf_model, "hf", captured)
        veo_handles = install_layer_hooks(veomni_model, "ours", captured)
        try:
            print("   running HF forward ...")
            with torch.no_grad():
                hf_model(**hf_inputs)
            print("   running VeOmni forward ...")
            with torch.no_grad():
                veomni_model(**veo_inputs)
        finally:
            for h in hf_handles + veo_handles:
                h.remove()

        # Print per-layer diff. n is the number of decoder layers, read
        # off the hf side (HF + VeOmni both have 37 layers in the
        # 30B-A2B text backbone).
        n_layers = sum(1 for k in captured if k.startswith("hf:layer_"))
        print()
        print(f"   per-layer hidden_states diff (HF eager  vs  VeOmni {args.moe_impl}):")
        print(f"   {'layer':<10s} {'max abs':>14s} {'mean abs':>14s} {'note':<22s}")
        print(f"   {'-' * 65}")
        prev_max = 0.0
        for i in range(n_layers):
            key = f"layer_{i:02d}"
            a = captured.get(f"hf:{key}")
            b = captured.get(f"ours:{key}")
            line, mx, mn = diff(a, b, key)
            note = "dense" if i < 2 else "MoE"
            # Annotate the first big leap (>5× previous layer).
            if i >= 2 and prev_max > 1e-6 and mx > 5 * prev_max:
                note += "  <-- LEAP"
            print(f"   {key:<10s} {mx:>14.3e} {mn:>14.3e}  {note}")
            prev_max = mx

        for tail_key in ("final_norm", "logits"):
            a = captured.get(f"hf:{tail_key}")
            b = captured.get(f"ours:{tail_key}")
            line, mx, mn = diff(a, b, tail_key)
            print(f"   {tail_key:<10s} {mx:>14.3e} {mn:>14.3e}")

        # Free large captures before the next sample.
        captured.clear()

    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
