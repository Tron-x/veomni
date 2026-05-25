"""Pangu Omni VeOmni adapter — per-token logp oracle checker.

Phase 1 yardstick: for each weekly milestone we run this script against the
current adapter implementation and compare per-token log-probabilities against
a pre-recorded HuggingFace baseline.

Reference function: `infer.py:run_hf_one` from the Pangu HF inference repo.
Our oracle baseline lives at:
  /mnt/data_3/models/pangu/test_hf_percision.0518.parallel/results/ocrbench_hf_outputs.jsonl

Each line in that file contains:
  sample_id, prompt_text, image_paths, prediction, token_ids, tokens, logps,
  sum_logp, mean_logp, num_tokens

The oracle was generated with greedy decoding (do_sample=False) using HF
AutoModelForCausalLM + trust_remote_code=True. We verified bit-for-bit
reproducibility on transformers 5.0.0 (sample 0 mean_logp=-0.24750558946288947
matches exactly).

## Modes

This checker supports two modes:

  --mode hf         Load model via HF AutoModelForCausalLM (sanity check;
                    should give zero diff against baseline).
  --mode veomni     Load model via VeOmni build_foundation_model.
                    Routes through the Pangu adapter (`pangu_omni_v2/`),
                    landing on `OpenPanguVL` (Week 3.4.d) for multimodal
                    inputs or `OpenPanguV2ForCausalLM` (Week 2) when
                    `--text-only` is passed.

## Usage

  # Smoke (mode=hf, 3 samples, should print ALL ZEROS):
  python tools/pangu_oracle_check.py --mode hf --n-samples 3

  # Text-only sanity (mode=veomni, bypasses multimodal merge):
  python tools/pangu_oracle_check.py --mode veomni --text-only --n-samples 3

  # Multimodal parity (mode=veomni, image+text — Week 3.6 acceptance gate):
  python tools/pangu_oracle_check.py --mode veomni --n-samples 3

## Notes on `--text-only`

Until Week 3.4.d (OpenPanguVL full forward) was implemented, the
multimodal path crashed because the dispatcher resolved to a
`NotImplementedError` stub; `--text-only` was the only way to use
`--mode veomni`. As of Week 3.4.d, multimodal works through
`OpenPanguVL`, and `--text-only` is now just a sanity-check option
(text-only logp does NOT match the OCRBench multimodal baseline; use
a text-only baseline if you need clean parity in that mode).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PANGU_INFER_DIR = Path("/mnt/data_3/models/pangu/test_hf_percision.0518.parallel")
PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
DEFAULT_BASELINE = PANGU_INFER_DIR / "results" / "ocrbench_hf_outputs.jsonl"
DEFAULT_SAMPLES = PANGU_INFER_DIR / "data" / "ocrbench.jsonl"


@dataclass
class SampleResult:
    sample_id: str
    n_tokens: int
    max_abs_logp_diff: float
    mean_abs_logp_diff: float
    sum_logp_diff: float
    tokens_equal: bool
    passed: bool


@dataclass
class AggregateResult:
    n_samples: int
    n_passed: int
    mean_of_max_diffs: float
    mean_of_mean_diffs: float
    worst_max_diff: float
    worst_sample_id: str
    tolerance_max: float
    tolerance_mean: float

    def summary(self) -> str:
        return (
            f"\n{'=' * 72}\n"
            f"  Aggregate ({self.n_samples} samples)\n"
            f"{'=' * 72}\n"
            f"  Pass:           {self.n_passed}/{self.n_samples}\n"
            f"  Mean of max-diffs:  {self.mean_of_max_diffs:.3e}\n"
            f"  Mean of mean-diffs: {self.mean_of_mean_diffs:.3e}\n"
            f"  Worst sample:   {self.worst_sample_id} (max-diff {self.worst_max_diff:.3e})\n"
            f"  Tolerance:      max={self.tolerance_max}, mean={self.tolerance_mean}\n"
            f"  Verdict:        {'PASS' if self.n_passed == self.n_samples else 'FAIL'}\n"
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_baseline_map(baseline_path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(baseline_path)
    return {str(r["sample_id"]): r for r in rows}


def _resolve_mm_paths(sample: dict[str, Any], samples_root: Path, key: str) -> list[str]:
    """Resolve relative paths under a sample's media field (`image_paths` /
    `audio_paths` / `video_paths`) against ``samples_root``. Absolute
    paths pass through unchanged. Missing key → empty list.

    Keeps the resolution rule consistent across all modalities: the
    OCRBench JSONL ships ``image_paths`` like ``data/ocrbench_images/0.png``
    that are relative to the directory the JSONL itself lives in. Our
    audio oracle JSONL follows the same convention with
    ``audio_paths: ["data/audio_demo_files/1.mp3"]``.
    """
    out = []
    for p in sample.get(key, []):
        path = Path(p)
        if not path.is_absolute():
            path = samples_root / path
        out.append(str(path))
    return out


def resolve_image_paths(sample: dict[str, Any], samples_root: Path) -> list[str]:
    """Match infer.py:resolve_image_paths — relative paths under PANGU_INFER_DIR."""
    return _resolve_mm_paths(sample, samples_root, "image_paths")


def resolve_audio_paths(sample: dict[str, Any], samples_root: Path) -> list[str]:
    """Companion to ``resolve_image_paths`` for the audio path. JSONL
    samples carry ``audio_paths: [...]``; see ``audio_demo.jsonl`` for
    the schema."""
    return _resolve_mm_paths(sample, samples_root, "audio_paths")


def build_conversation(
    image_paths: list[str],
    prompt_text: str,
    text_only: bool = False,
    audio_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Match infer.py:build_conversation exactly + Week 3.5 audio extension.

    The Pangu chat_template handles three media types — image, audio,
    video — via content entries shaped like
    ``{"type": "image", "image": path}`` etc. The template expands them
    to ``<|vision_start|>...<|vision_end|>`` / ``<|audio_start|>...<|audio_end|>``
    placeholders that the processor then replaces with the right number
    of pad tokens computed from ``get_image_features`` /
    ``get_audio_output_length``.

    When ``text_only=True``, all media is silently dropped from the
    conversation content. Used by the Week 2 unlock path: the Pangu VL /
    UltraOmni multimodal classes were initially NotImplementedError stubs,
    so VeOmni loaded ``OpenPanguV2ForCausalLM`` (text backbone only) and
    couldn't accept image inputs. Today this flag is mainly a
    sanity-check option (Week 2.4 baseline lookups; modality-mixed
    parity tests use the multimodal path).
    """
    content: list[dict[str, Any]] = []
    if not text_only:
        content.extend({"type": "image", "image": p} for p in image_paths)
        if audio_paths:
            content.extend({"type": "audio", "audio": p} for p in audio_paths)
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def setup_npu_if_available() -> None:
    """Match infer.py NPU setup so we run on the same device path."""
    import torch
    from transformers.utils import is_torch_npu_available

    if is_torch_npu_available() and "910" in torch.npu.get_device_name():
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def load_hf_model_and_processor(model_dir: Path) -> tuple[Any, Any]:
    """Mode=hf: vanilla HF AutoModel + trust_remote_code."""
    setup_npu_if_available()
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    return model, processor


def load_veomni_model_and_processor(model_dir: Path, text_only: bool = False) -> tuple[Any, Any]:
    """Mode=veomni: load through VeOmni adapter (requires pangu_omni_v2 registered).

    Args:
        model_dir: Path to Pangu Omni v2 model directory (config.json +
            safetensors).
        text_only: When True, override `config.architectures` to
            `["OpenPanguV2ForCausalLM"]` so the dispatcher returns the
            text-only Week 2 model class instead of the multimodal
            class. The safetensors will have unexpected keys for
            vision/audio modules — VeOmni's loader is `strict=False`
            by default so those are warned-and-skipped.

            When False (default), the dispatcher resolves
            ``architectures[0]`` (= ``OpenPanguUltraOmniForConditionalGeneration``
            on the production 30B-A2B config) through to
            ``OpenPanguVL`` (Week 3.4.d), which runs the full
            vision+text multimodal merge. This is the Week 3.6
            acceptance path.
    """
    setup_npu_if_available()
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_config, build_foundation_model, build_processor

    # HF's `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)` on
    # NPU auto-selects `_attn_implementation="sdpa"` (verified by introspection
    # of `m.config.text_config._attn_implementation` on the 30B-A2B production
    # checkpoint). To run a faithful adapter-vs-reference oracle, VeOmni must
    # use the SAME kernel — forcing eager here causes a ~2e-3 drift purely from
    # SDPA-vs-eager numerical asymmetry on bf16 (verified with
    # `tools/pangu_drift_localizer.py`: both-eager gives bit-for-bit max=0,
    # eager-vs-sdpa gives ~1e-3 at layer 0 and ~2e-3 at logp). The other ops
    # (moe / rms_norm / etc.) follow the same logic — leave them at the HF
    # native eager path so we don't introduce VeOmni-specific kernels into
    # the oracle parity test.
    # Defaults match the gold-standard HF-parity oracle: eager kernels +
    # SDPA attention. The single env-var overrides let us probe whether a
    # specific kernel is responsible for a multi-card drift without
    # forking the whole script. (Used by the Phase-1 multi-card parity
    # decomposition.)
    #
    # PANGU_ORACLE_MOE_IMPL — defaults to "eager"; set to "fused_npu" to
    #   route the routed-expert forward through `npu_fused_moe_forward`
    #   (the same kernel the 8-card EP=8 SFT smoke uses), keeping
    #   everything else eager. Lets us split "kernel drift" from
    #   "EP/FSDP communication drift" in the multi-card parity check.
    moe_impl = os.environ.get("PANGU_ORACLE_MOE_IMPL", "eager")
    ops_cfg = OpsImplementationConfig(
        attn_implementation="sdpa",
        moe_implementation=moe_impl,
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
    )

    if text_only:
        # Load config separately so we can override `architectures`
        # before dispatch. Otherwise the dispatcher reads
        # `OpenPanguUltraOmniForConditionalGeneration` from
        # config.architectures[0] and lands on the NotImplementedError
        # stub.
        config = build_config(str(model_dir))
        original_arch = list(getattr(config, "architectures", []))
        config.architectures = ["OpenPanguV2ForCausalLM"]
        print(f"[veomni text-only] overriding architectures: {original_arch} -> {config.architectures}")
        config_arg: Any = config
    else:
        config_arg = str(model_dir)

    model = build_foundation_model(
        config_path=config_arg,
        weights_path=str(model_dir),
        torch_dtype="bfloat16",
        init_device="cuda",
        ops_implementation=ops_cfg,
    ).eval()
    processor = build_processor(str(model_dir))

    # Restore HF's default NPU conv kernel selection.
    #
    # VeOmni's `veomni/utils/device.py` sets `torch.npu.config.allow_internal_format =
    # False` at import time so that FSDP / weight-sharding does not have to chase
    # NPU-internal layouts (NZ / 5HD). HF's `AutoModelForCausalLM.from_pretrained`
    # path leaves it at the NPU default (True) which lets `Conv2d` pick the
    # internal-format kernel. The two kernels accumulate bf16 in different
    # orders; for the Pangu Omni audio tower this manifests as ULP-level diffs
    # in the very first VGG Conv2d that get amplified through 24 Conformer
    # layers + MoE LLM to ~4.6e-2 final logp (see /tmp/probe_allow_internal_format.py).
    #
    # For the oracle parity test we want to match HF reference exactly, so
    # restore True here. RL training runs that need FSDP layout safety should
    # leave the flag at False and accept the audio-path bf16 noise above (or
    # write a per-op scope around audio_tower).
    if os.environ.get("PANGU_ORACLE_NPU_INTERNAL_FMT", "1") != "0":
        try:
            import torch as _torch

            if hasattr(_torch, "npu") and hasattr(_torch.npu, "config"):
                _torch.npu.config.allow_internal_format = True
                print(
                    "[veomni] restored torch.npu.config.allow_internal_format=True "
                    "(VeOmni device.py disables it at import time; HF reference "
                    "leaves it True, so the oracle path matches HF kernel selection)"
                )
        except Exception as _exc:  # pragma: no cover - diagnostic
            print(f"[veomni] could not restore allow_internal_format: {_exc!r}")

    return model, processor


# ---------------------------------------------------------------------------
# Forward + per-token logp computation (teacher forcing on baseline tokens)
# ---------------------------------------------------------------------------


def compute_per_token_logps(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    baseline_token_ids: list[int],
    text_only: bool = False,
) -> list[float]:
    """Teacher-forced forward pass.

    Constructs the prompt the same way as infer.py:run_hf_one, then runs a
    single forward pass with the input being (prompt_tokens + baseline_tokens).
    Returns per-position log-prob for each baseline token.

    This is fundamentally different from `model.generate()` — we don't let
    the model choose tokens, we force the baseline's chosen tokens and ask
    "what logp did the model assign to this token?". This isolates numerical
    differences in the forward pass from compounding decoding divergence.
    """
    import torch
    import torch.nn.functional as F

    samples_root = sample.get("_samples_root", PANGU_INFER_DIR)
    if not isinstance(samples_root, Path):
        samples_root = Path(samples_root)
    image_paths = resolve_image_paths(sample, samples_root) if not text_only else []
    audio_paths = resolve_audio_paths(sample, samples_root) if not text_only else []
    conversation = build_conversation(
        image_paths,
        sample["prompt_text"],
        text_only=text_only,
        audio_paths=audio_paths,
    )
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    if text_only:
        # Skip multimodal feature extraction entirely — the text-only
        # backbone doesn't accept pixel_values / image_grid_thw kwargs
        # and the qwen_omni_utils dependency isn't needed for text path.
        inputs = processor(text=text, padding=False, return_tensors="pt").to(model.device)
    else:
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
    baseline_tensor = torch.tensor([baseline_token_ids], dtype=inputs.input_ids.dtype, device=inputs.input_ids.device)
    full_input_ids = torch.cat([inputs.input_ids, baseline_tensor], dim=1)

    extended_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
    if "attention_mask" in extended_inputs:
        new_mask = torch.cat(
            [
                extended_inputs["attention_mask"],
                torch.ones_like(baseline_tensor, dtype=extended_inputs["attention_mask"].dtype),
            ],
            dim=1,
        )
        extended_inputs["attention_mask"] = new_mask

    with torch.no_grad():
        outputs = model(input_ids=full_input_ids, **extended_inputs)

    # logits at position p predict token p+1, so for baseline_tokens at positions
    # [prompt_len, prompt_len+1, ..., prompt_len+N-1], we read logits at
    # [prompt_len-1, prompt_len, ..., prompt_len+N-2].
    logits = outputs.logits[0]
    logps = []
    for i, tok_id in enumerate(baseline_token_ids):
        logit_pos = prompt_len - 1 + i
        logp = F.log_softmax(logits[logit_pos].float(), dim=-1)[int(tok_id)]
        logps.append(float(logp.detach().cpu()))
    return logps


# ---------------------------------------------------------------------------
# Per-sample + aggregate comparison
# ---------------------------------------------------------------------------


def compare_one(
    sample: dict[str, Any],
    baseline_row: dict[str, Any],
    model_logps: list[float],
    tolerance_max: float,
) -> SampleResult:
    baseline_logps = baseline_row["logps"]
    baseline_tokens = baseline_row["token_ids"]  # noqa: F841  # kept for debug/inspection while we iterate on diff thresholds

    if len(model_logps) != len(baseline_logps):
        return SampleResult(
            sample_id=str(sample["sample_id"]),
            n_tokens=len(baseline_logps),
            max_abs_logp_diff=math.inf,
            mean_abs_logp_diff=math.inf,
            sum_logp_diff=math.inf,
            tokens_equal=False,
            passed=False,
        )

    diffs = [abs(a - b) for a, b in zip(model_logps, baseline_logps)]
    max_diff = max(diffs) if diffs else 0.0
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    sum_diff = sum(model_logps) - sum(baseline_logps)

    return SampleResult(
        sample_id=str(sample["sample_id"]),
        n_tokens=len(baseline_logps),
        max_abs_logp_diff=max_diff,
        mean_abs_logp_diff=mean_diff,
        sum_logp_diff=sum_diff,
        tokens_equal=True,
        passed=max_diff < tolerance_max,
    )


def print_sample(sr: SampleResult) -> None:
    mark = "PASS" if sr.passed else "FAIL"
    print(
        f"  [{mark}] {sr.sample_id:<22s} "
        f"n_toks={sr.n_tokens:<4d} "
        f"max={sr.max_abs_logp_diff:.3e}  "
        f"mean={sr.mean_abs_logp_diff:.3e}  "
        f"sum_diff={sr.sum_logp_diff:+.3e}"
    )


def aggregate(results: list[SampleResult], tol_max: float, tol_mean: float) -> AggregateResult:
    if not results:
        raise ValueError("no results to aggregate")
    n_pass = sum(1 for r in results if r.passed)
    worst = max(results, key=lambda r: r.max_abs_logp_diff)
    return AggregateResult(
        n_samples=len(results),
        n_passed=n_pass,
        mean_of_max_diffs=sum(r.max_abs_logp_diff for r in results) / len(results),
        mean_of_mean_diffs=sum(r.mean_abs_logp_diff for r in results) / len(results),
        worst_max_diff=worst.max_abs_logp_diff,
        worst_sample_id=worst.sample_id,
        tolerance_max=tol_max,
        tolerance_mean=tol_mean,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Pangu Omni VeOmni adapter — per-token logp oracle checker")
    parser.add_argument("--mode", choices=("hf", "veomni"), default="hf")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Drop image_paths from prompts and (in veomni mode) load "
            "`OpenPanguV2ForCausalLM` instead of the multimodal class. "
            "Required while multimodal classes are NotImplementedError "
            "stubs. NOTE: per-token logp will NOT match the multimodal "
            "OCRBench baseline — see the docstring of `build_conversation`."
        ),
    )
    parser.add_argument("--model-dir", type=Path, default=PANGU_MODEL_DIR)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--n-samples", type=int, default=3, help="how many samples to check")
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="explicit sample_id list (overrides --n-samples ordering)",
    )
    parser.add_argument("--tolerance-max", type=float, default=1e-3)
    parser.add_argument("--tolerance-mean", type=float, default=1e-4)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional JSON report path (e.g. results/oracle_check_week2.json)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print("  Pangu Omni VeOmni adapter — logp oracle check")
    print(f"{'=' * 72}")
    print(f"  Mode:      {args.mode}")
    print(f"  Text-only: {args.text_only}")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Samples:   {args.samples}")
    print(f"  Baseline:  {args.baseline}")
    print(f"  Tolerance: max={args.tolerance_max}, mean={args.tolerance_mean}")
    print()

    samples = load_jsonl(args.samples)
    baseline = load_baseline_map(args.baseline)

    # Resolve relative media paths (image_paths / audio_paths) against
    # the directory the samples JSONL lives in. OCRBench's
    # ``data/ocrbench_images/0.png`` resolves under
    # ``/mnt/data_3/models/pangu/test_hf_percision.0518.parallel/``;
    # our audio oracle's ``data/audio_demo_files/1.mp3`` resolves under
    # ``/mnt/data_3/models/pangu_audio_oracle/``. Stash the inferred
    # root on each sample so ``compute_per_token_logps`` (which sees
    # only one sample at a time) can use it without an extra parameter.
    samples_root = args.samples.parent.parent if args.samples != DEFAULT_SAMPLES else PANGU_INFER_DIR
    for s in samples:
        s["_samples_root"] = samples_root

    if args.sample_ids:
        wanted = set(args.sample_ids)
        picked = [s for s in samples if str(s["sample_id"]) in wanted]
    else:
        # Pick first N samples that have baseline entries.
        picked = [s for s in samples if str(s["sample_id"]) in baseline][: args.n_samples]

    if not picked:
        print("[ERROR] no samples to check. Either samples file is empty or no overlap with baseline.")
        return 2

    print(f"  Picked {len(picked)} samples")
    for s in picked:
        bl = baseline[str(s["sample_id"])]
        print(f"    - {s['sample_id']:<22s} (baseline n_tokens={bl['num_tokens']}, mean_logp={bl['mean_logp']:.4f})")

    # Load model
    print(f"\n[load] loading model in mode={args.mode}...")
    t0 = time.time()
    if args.mode == "hf":
        model, processor = load_hf_model_and_processor(args.model_dir)
    else:
        model, processor = load_veomni_model_and_processor(args.model_dir, text_only=args.text_only)
    print(f"[load] done in {time.time() - t0:.1f}s")

    # Run per-sample comparison
    print("\n[check] running teacher-forced forward + diff:")
    results: list[SampleResult] = []
    for s in picked:
        sid = str(s["sample_id"])
        bl = baseline[sid]
        t0 = time.time()
        try:
            model_logps = compute_per_token_logps(model, processor, s, bl["token_ids"], text_only=args.text_only)
        except Exception as exc:
            print(f"  [ERROR] sample {sid}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            results.append(
                SampleResult(
                    sample_id=sid,
                    n_tokens=bl["num_tokens"],
                    max_abs_logp_diff=math.inf,
                    mean_abs_logp_diff=math.inf,
                    sum_logp_diff=math.inf,
                    tokens_equal=False,
                    passed=False,
                )
            )
            continue
        sr = compare_one(s, bl, model_logps, args.tolerance_max)
        results.append(sr)
        print_sample(sr)
        print(f"           (took {time.time() - t0:.1f}s)")
        # Per-token table when env var is set: helps see whether logp
        # divergence is concentrated in a few tokens (likely a kernel
        # boundary on some specific position) or spread uniformly
        # (likely accumulation noise across the whole sequence).
        if os.environ.get("PANGU_ORACLE_VERBOSE_PER_TOKEN", "0") == "1":
            tokenizer = getattr(processor, "tokenizer", processor)
            tok_strs = tokenizer.convert_ids_to_tokens(bl["token_ids"])
            print(f"        {'idx':>4s} {'tok_id':>7s} {'token':<20s} {'baseline':>14s} {'model':>14s} {'diff':>10s}")
            for i, (tok_id, tok_str, b_lp, m_lp) in enumerate(
                zip(bl["token_ids"], tok_strs, bl["logps"], model_logps)
            ):
                tok_disp = tok_str if len(tok_str) <= 20 else tok_str[:17] + "..."
                diff = m_lp - b_lp
                print(f"        {i:>4d} {tok_id:>7d} {tok_disp:<20s} {b_lp:>14.6e} {m_lp:>14.6e} {diff:>10.3e}")

    agg = aggregate(results, args.tolerance_max, args.tolerance_mean)
    print(agg.summary())

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            json.dump(
                {
                    "mode": args.mode,
                    "model_dir": str(args.model_dir),
                    "samples": str(args.samples),
                    "baseline": str(args.baseline),
                    "tolerance_max": args.tolerance_max,
                    "tolerance_mean": args.tolerance_mean,
                    "per_sample": [vars(r) for r in results],
                    "aggregate": vars(agg),
                },
                fh,
                indent=2,
            )
        print(f"[out] report written to {args.out}")

    return 0 if agg.n_passed == agg.n_samples else 1


if __name__ == "__main__":
    sys.exit(main())
