"""Multi-card multimodal forward-parity runner for Pangu Omni v2.

Companion to ``multi_card_parity_runner.py`` (text-only). The multi-card
text-only runner already proved that FSDP2 + EP=8 + ``fused_npu`` MoE is
**bit-identical** to single-card ``fused_npu`` MoE on per-token logp
(IEEE 754 float64 match across 128 prompt tokens) — so the production
distributed wiring is numerically correct for the language backbone.

This runner is the multimodal extension. Goals:

1. Verify the L2b MRoPE × ``create_causal_mask`` fix (2026-05-22) plus
   the ``OpenPanguOmni.get_parallel_plan`` (= ``model.language_model``
   FQN prefix) make the full multimodal model load and run forward
   correctly across 8 NPUs with EP=8 + FSDP2.
2. Confirm OCRBench-style image+text inputs go through the production
   stack (FSDP all-gather, EP all-to-all on the language MoE, dense
   FSDP2 sharding on visual + audio_tower) without crashing.
3. Diff per-token logps against the OCRBench HF baseline (which was
   generated single-process with HF reference modeling).

Status (2026-05-23): with the four multimodal-multi-card fixes
(MRoPE × ``create_causal_mask`` reconciliation, multimodal
``get_parallel_plan`` with ``model.language_model`` prefix, MoE
OpSlot re-export on ``OpenPanguOmni`` / ``OpenPanguVL``, and the
corrected ``_no_split_modules`` in ``modeling_vl.py``),
the runner produces the **correct OCR top-1 token** on every
sample tested (e.g. ``'Centre'``, ``'Friend'``, ``'Chain'`` for
the three OCRBench prompts). 2/3 PASS at the strict 5e-2 per-token
logp tolerance; the remaining 1/3 sits at max-diff 1.12e-1 / mean
5.62e-2, dominated by accumulated FSDP2 all-gather noise across
the 26-block visual tower + visual→language merger. This is a
>100× improvement over the original 14 logp-unit drift (visual
stream not reaching language tokens). See README "Multi-card
multimodal" section for the full audit trail.

Reuses the production setup pipeline (``BaseTrainer._setup`` /
``_build_model`` / ``_build_parallelized_model``) so the forward path
is exactly what ``tasks/train_vlm.py`` would use, plus the same
multimodal preprocessing that ``tools/oracle_check.py`` uses for HF
parity.

Launch::

    PARITY_SAMPLES=/path/to/oracle/data/ocrbench.jsonl \\
    PARITY_BASELINE=/path/to/oracle/results/ocrbench_hf_outputs.jsonl \\
    PARITY_OUT=outputs/pangu_multimodal_parity/report_8card.json \\
    PARITY_TOLERANCE=5e-2 \\
    PARITY_N_SAMPLES=3 \\
    torchrun --nnodes=1 --nproc_per_node=8 --master_port=29512 \\
        veomni/models/transformers/pangu_omni_v2/tools/multi_card_multimodal_parity_runner.py \\
        configs/multimodal/pangu_omni_v2/local_smoke/pangu_real_8card_multimodal.yaml

Output (rank 0):

  per-sample max-abs / mean-abs logp diffs vs the HF baseline,
  plus an aggregate verdict (PASS / FAIL with respect to PARITY_TOLERANCE).

The 5e-2 tolerance is calibrated against the text-only ``fused_npu`` vs
HF eager drift (~1.2e-1 max in single-card). Multimodal samples pass
through the same MoE so the same drift applies; vision / audio paths
are deterministic so they don't add additional drift. If we see drift
significantly larger than 1e-1 here, that points at a multimodal-specific
wiring bug in this runner or in the multimodal forward path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from veomni.arguments import VeOmniArguments, parse_args

# Reuse the multimodal helpers that already exist in oracle_check so the
# preprocessing path is bit-identical to the single-card HF parity check
# (chat template, image resolution, processor invocation). Absolute import
# (rather than relative) so this file can be invoked directly via torchrun
# — relative imports require a parent package on sys.path which torchrun
# does not provide.
from veomni.models.transformers.pangu_omni_v2.tools.oracle_check import (
    build_conversation,
    resolve_audio_paths,
    resolve_image_paths,
)
from veomni.trainer.base import BaseTrainer


class MultimodalParityRunner:
    """Composition wrapper around ``BaseTrainer`` for forward-only
    multi-card multimodal parity.

    Mirrors the structure of ``multi_card_parity_runner.ParityRunner``
    (text-only) — manual ``BaseTrainer.__new__`` + selective helper
    invocations — but adds multimodal input preparation in
    ``parity_check`` (image + audio loading, processor invocation that
    yields ``pixel_values`` / ``image_grid_thw`` / etc.).

    The model class loaded here is ``OpenPanguOmni`` (= visual +
    audio_tower + language_model). Its ``get_parallel_plan()`` (added
    2026-05-22 alongside the L2b fix) returns
    ``parallel_plan.get_parallel_plan(prefix="model.language_model")``,
    which is what makes EP work for the multimodal class.
    """

    def __init__(self, args):
        self.args: VeOmniArguments = args
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args
        self.base._setup()
        self.base._build_model()
        self.base._freeze_model_module()
        self.base._build_parallelized_model()
        self.base.model.eval()
        self.model = self.base.model
        self.device = self.base.device

    @staticmethod
    def _load_jsonl(path: str) -> list[dict]:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    @staticmethod
    def _dump_weight_norms(model) -> None:
        """Print weight norms for a few representative parameters on
        rank 0. Used to debug multi-card multimodal weight loading —
        verified 2026-05-22 that norms match single-card baseline,
        confirming the multimodal drift comes from the forward path
        (FSDP2 wrap on visual / audio_tower) rather than from weight
        loading. Triggered by ``MM_PARITY_DUMP_WEIGHTS=1`` in env.
        """
        if not hasattr(model, "model"):
            return
        inner = model.model
        for fqn in [
            "visual.patch_embed.proj.weight",
            "visual.blocks.0.attn.qkv.weight",
            "language_model.embed_tokens.weight",
            "language_model.layers.2.mlp.gate.weight",
        ]:
            obj = inner
            for part in fqn.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is None:
                print(f"  [diag] {fqn:<60s}  not found")
                continue
            local = obj.to_local() if hasattr(obj, "to_local") else obj
            if local.is_meta:
                print(f"  [diag] {fqn:<60s}  is_meta=True (NOT LOADED)")
            else:
                print(
                    f"  [diag] {fqn:<60s}  shape={tuple(local.shape)} "
                    f"dtype={local.dtype} norm={local.float().norm().item():.4e}"
                )
        print()

    def parity_check(
        self,
        samples_path: str,
        baseline_path: str,
        out_path: str,
        tolerance: float,
        n_samples: int,
    ) -> int:
        from veomni.models.auto import build_processor

        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1
        device = self.device

        # The multimodal processor wraps the tokenizer + image preprocessor +
        # audio preprocessor; identical to what oracle_check uses, so token
        # IDs and pixel-tensor floats are bit-identical to the HF parity
        # baseline (image preprocessing is deterministic).
        model_dir = self.args.model.model_path
        processor = build_processor(str(model_dir))

        samples = self._load_jsonl(samples_path)
        baseline = {row["sample_id"]: row for row in self._load_jsonl(baseline_path)}

        # Resolve relative media paths against the dataset root inferred
        # from the samples JSONL. Absolute paths are preserved.
        samples_root = Path(samples_path).parent.parent

        if rank == 0:
            print(f"\n[mm-parity] world_size = {world_size}  device = {device}")
            print(f"[mm-parity] samples       = {samples_path}")
            print(f"[mm-parity] baseline      = {baseline_path}")
            print(f"[mm-parity] samples_root  = {samples_root}")
            print(f"[mm-parity] tolerance     = max abs logp diff < {tolerance}")
            print(f"[mm-parity] n_samples     = {n_samples}")
            print()

        # Optional weight-load diagnostic — set ``MM_PARITY_DUMP_WEIGHTS=1``
        # in env to print vision / audio / language tower weight norms
        # on rank 0. Useful for distinguishing weight-load bugs (norm 0
        # / is_meta=True) from forward-path bugs (norms match
        # single-card but logits differ). Off by default to keep
        # parity output clean.
        if rank == 0 and os.environ.get("MM_PARITY_DUMP_WEIGHTS"):
            self._dump_weight_norms(self.model)

        # Diagnostic: enumerate FSDP wrap granularity under each tower so we
        # can verify (a) ``_no_split_modules`` actually matches the layer
        # class names produced by the model (a misspelling silently coarsens
        # the wrap) and (b) which towers got per-layer wrap vs. tower-level
        # wrap. Triggered by ``MM_PARITY_DUMP_FSDP=1`` (off by default to
        # keep parity output clean).
        if rank == 0 and os.environ.get("MM_PARITY_DUMP_FSDP"):
            from torch.distributed.fsdp import FSDPModule

            inner = self.model.model if hasattr(self.model, "model") else self.model
            for tower_name in ("visual", "audio_tower", "language_model"):
                tower = getattr(inner, tower_name, None)
                if tower is None:
                    continue
                # Count FSDP-wrapped descendants by class
                from collections import Counter

                counts: Counter[str] = Counter()
                for sub in tower.modules():
                    if isinstance(sub, FSDPModule):
                        counts[type(sub).__name__] += 1
                print(f"[fsdp-probe] {tower_name}: FSDP-wrapped modules = {dict(counts)}")
                # Also show child class name distribution for top-level "layers" if present
                layers = getattr(tower, "layers", None) or getattr(tower, "blocks", None)
                if layers is not None and len(layers) > 0:
                    layer_class = type(layers[0]).__name__
                    n_fsdp_layers = sum(1 for li in layers if isinstance(li, FSDPModule))
                    print(
                        f"[fsdp-probe] {tower_name}.layers: {len(layers)} × {layer_class}, "
                        f"{n_fsdp_layers} FSDP-wrapped"
                    )

        results: list[dict] = []
        n_pass = 0
        # Run only the first ``n_samples`` matching prompts (OCRBench has
        # ~1k samples; we don't need them all to validate the wiring).
        used = 0
        for sample in samples:
            if used >= n_samples:
                break
            sid = sample["sample_id"]
            if sid not in baseline:
                continue
            bl = baseline[sid]

            # Build the same conversation HF saw, with image entries
            # resolved to absolute paths (the chat_template handles the
            # ``<|vision_start|>...<|vision_end|>`` token expansion).
            image_paths = resolve_image_paths(sample, samples_root)
            audio_paths = resolve_audio_paths(sample, samples_root)
            conversation = build_conversation(
                image_paths,
                sample["prompt_text"],
                text_only=False,
                audio_paths=audio_paths,
            )
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

            # Build multimodal model inputs. Lazy import keeps qwen_omni_utils
            # cost out of the no-MM path and matches oracle_check's pattern.
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

            prompt_len = int(inputs.input_ids.shape[1])
            baseline_ids = torch.tensor([bl["token_ids"]], dtype=inputs.input_ids.dtype, device=device)
            full_input_ids = torch.cat([inputs.input_ids, baseline_ids], dim=1)
            attention_mask = torch.ones_like(full_input_ids)

            # Replace input_ids with the teacher-forced full sequence;
            # keep all other multimodal kwargs as-is. (pixel_values /
            # image_grid_thw etc. only encode the prompt portion; the
            # appended baseline tokens are pure text and need no extra
            # multimodal context.)
            forward_kwargs = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}
            forward_kwargs["input_ids"] = full_input_ids
            forward_kwargs["attention_mask"] = attention_mask

            # Reset rope_deltas cache between samples — OpenPanguOmni
            # caches it on the first prefill and the cached value is wrong
            # for a different sample's input. Single-card oracle works
            # because past_key_values is None each call so the prefill
            # branch is re-entered, but multi-card init has subtle
            # interactions where the cache leaks. Defensive reset matches
            # what infer.py does between samples.
            if hasattr(self.model, "model") and hasattr(self.model.model, "rope_deltas"):
                self.model.model.rope_deltas = None

            # Optional: capture intermediate tensors via forward hooks for
            # per-sample drift localization. Set ``MM_PARITY_CAPTURE=1`` in
            # env to enable. Writes a numpy file per sample with
            # ``image_embeds`` (vision tower output) and
            # ``inputs_embeds_post_scatter`` (after masked_scatter). Used
            # to diff single-card vs multi-card on the same sample.
            captured: dict[str, torch.Tensor] = {}
            hooks = []
            if rank == 0 and os.environ.get("MM_PARITY_CAPTURE"):
                inner = self.model.model if hasattr(self.model, "model") else self.model

                def _to_cpu_float(t):
                    if hasattr(t, "to_local"):
                        t = t.to_local()
                    return t.detach().float().cpu().clone()

                # ruff B023: ``captured`` is captured by reference so the
                # hook writes to the dict we read later — intentional.
                def _hook_visual(_mod, _inputs, output, _captured=captured):
                    out = output
                    if isinstance(out, tuple):
                        out = out[0]
                    _captured["visual_output"] = _to_cpu_float(out)

                if hasattr(inner, "visual"):
                    hooks.append(inner.visual.register_forward_hook(_hook_visual))

                    # Deep capture: probe multiple landmarks inside the vision
                    # tower to bisect the residual 1 bf16 ULP drift after the
                    # mp-ignore hook eliminates the cos/sin downcast. Same probe
                    # points are mirrored in
                    # ``single_card_visual_capture.py`` so the dicts diff key-by-key.
                    if os.environ.get("MM_PARITY_DEEP_CAPTURE"):
                        v = inner.visual

                        # All hooks defined here capture ``captured`` via the
                        # ``_captured=captured`` default argument to defeat ruff B023
                        # (the outer for-sample loop rebinds ``captured`` each
                        # iteration; default-arg binding pins the current dict).
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

                        hooks.append(v.patch_embed.register_forward_hook(_hk_patch))
                        n_blocks = len(v.blocks)
                        # Pick 0, mid, second-to-last, last so we can localize.
                        probe_idxs = sorted({0, n_blocks // 4, n_blocks // 2, (3 * n_blocks) // 4, n_blocks - 1})
                        for li in probe_idxs:
                            hooks.append(v.blocks[li].register_forward_hook(_make_block_hook(li)))
                        if hasattr(v, "merger"):
                            # Merger may be ModuleList for non-gated variant; hook each.
                            if isinstance(v.merger, torch.nn.ModuleList):
                                for mi, m in enumerate(v.merger):
                                    hooks.append(
                                        m.register_forward_hook(
                                            lambda _m, _i, _o, _k=f"post_merger_{mi}", _c=captured: _c.update(
                                                {_k: _to_cpu_float(_o[0] if isinstance(_o, tuple) else _o)}
                                            )
                                        )
                                    )
                            else:
                                hooks.append(v.merger.register_forward_hook(_hk_merger))
                        captured["__probe_idxs__"] = torch.tensor(probe_idxs)

                    # Audio deep capture: bisect ``model.audio_tower`` (VGG conv →
                    # linear_before_attn → 24 × ConformerEncoderLayerBlock →
                    # linear_after_attn → ln_post → proj) to localize the
                    # multi-card-vs-single-card drift observed on audio_0 / audio_1
                    # in audio_demo.jsonl. Use ``MM_PARITY_AUDIO_CAPTURE=1``.
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
                            # HuanyuAudioEncoder returns (BaseModelOutput, lengths)
                            if isinstance(output, tuple) and len(output) >= 1:
                                bmo = output[0]
                                if hasattr(bmo, "last_hidden_state"):
                                    _captured[_key] = _to_cpu_float(bmo.last_hidden_state)

                        hooks.append(a.register_forward_pre_hook(_hk_audio_in, with_kwargs=True))
                        if hasattr(a, "linear_before_attn"):
                            hooks.append(a.linear_before_attn.register_forward_hook(_hk_audio_lin_before))
                        if hasattr(a, "layers") and len(a.layers) > 0:
                            n_alayers = len(a.layers)
                            audio_probe_idxs = sorted(
                                {0, n_alayers // 4, n_alayers // 2, (3 * n_alayers) // 4, n_alayers - 1}
                            )
                            for li in audio_probe_idxs:
                                hooks.append(a.layers[li].register_forward_hook(_make_audio_layer_hook(li)))
                            captured["__audio_probe_idxs__"] = torch.tensor(audio_probe_idxs)
                        if hasattr(a, "linear_after_attn"):
                            hooks.append(a.linear_after_attn.register_forward_hook(_hk_audio_lin_after))
                        if hasattr(a, "proj"):
                            hooks.append(a.proj.register_forward_hook(_hk_audio_proj))
                        hooks.append(a.register_forward_hook(_hk_audio_tower_out))

                    # FINE capture: bisect inside block 0 (norm1 / attn / norm2 / mlp).
                    # Used during the original drift bisection that traced the
                    # second drift source to ``b00_post_norm1`` (fp32 storage
                    # under the previous mp-ignore hook). Left in place — kept
                    # behind ``MM_PARITY_FINE_CAPTURE`` — for future regressions.
                    if os.environ.get("MM_PARITY_FINE_CAPTURE"):
                        b0 = inner.visual.blocks[0]

                        def _hk_pre_b0(_mod, args, kwargs, _key="pre_block_00", _captured=captured):
                            hs = args[0] if args else kwargs.get("hidden_states")
                            _captured[_key] = _to_cpu_float(hs)

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

                        hooks.append(b0.register_forward_pre_hook(_hk_pre_b0, with_kwargs=True))
                        hooks.append(b0.norm1.register_forward_hook(_hk_n1))
                        hooks.append(b0.attn.register_forward_hook(_hk_attn))
                        hooks.append(b0.norm2.register_forward_hook(_hk_n2))
                        hooks.append(b0.mlp.register_forward_hook(_hk_mlp))

            with torch.no_grad():
                outputs = self.model(**forward_kwargs)
            logits = outputs.logits[0]  # [T, V]

            for h in hooks:
                h.remove()

            if rank == 0 and captured:
                cap_dir = Path(os.environ.get("PARITY_CAPTURE_DIR", "outputs/pangu_multimodal_parity/captured"))
                cap_dir.mkdir(parents=True, exist_ok=True)
                fname = cap_dir / f"{sid}_world{world_size}.pt"
                torch.save(captured, fname)
                vo = captured.get("visual_output")
                if vo is not None:
                    print(
                        f"    [capture] visual_output: shape={tuple(vo.shape)} norm={vo.norm().item():.6e} → {fname}"
                    )

            logps: list[float] = []
            for i, tok_id in enumerate(bl["token_ids"]):
                logit_pos = prompt_len - 1 + i
                lp = F.log_softmax(logits[logit_pos].float(), dim=-1)[int(tok_id)]
                logps.append(float(lp.detach().cpu()))

            # Debug: capture top-5 tokens + raw logit value at the first
            # generation position. Logits provide a noise-floor measure
            # that's softmax-temperature independent, so we can tell
            # whether per-sample logp drift comes from logit drift or
            # from softmax sensitivity at low-confidence regions.
            if rank == 0:
                first_logits = logits[prompt_len - 1].float()
                top5 = torch.topk(first_logits, k=5)
                top5_lps = F.log_softmax(first_logits, dim=-1)[top5.indices]
                top5_ids = top5.indices.tolist()
                top5_logps = top5_lps.tolist()
                top5_logits = top5.values.tolist()
                top5_tokens = [processor.tokenizer.decode([tid]) for tid in top5_ids]
                top5_str = ", ".join(
                    f"{t!r}(logit={lt:+.4f},logp={lp:+.4f})" for t, lt, lp in zip(top5_tokens, top5_logits, top5_logps)
                )
                # Also capture the raw logit at the baseline's first token,
                # so we can compute logit-level drift in post-processing.
                bl_first_id = int(bl["token_ids"][0])
                bl_first_logit = float(first_logits[bl_first_id].item())
                bl_first_lp = float(F.log_softmax(first_logits, dim=-1)[bl_first_id].item())
                # Image grid info — sample 0 was an outlier (image
                # 104×27 → tiny merged grid), capturing this helps
                # explain prompt-specific drift.
                grid_str = ""
                if "image_grid_thw" in inputs:
                    grid_str = f"  grid_thw={inputs['image_grid_thw'].tolist()}"
                print(f"    top5: {top5_str}{grid_str}")
                print(
                    f"    baseline_first_token: id={bl_first_id} logit={bl_first_logit:+.4f} logp={bl_first_lp:+.4f}"
                )

            if rank == 0:
                diffs = [abs(a - b) for a, b in zip(logps, bl["logps"])]
                max_diff = max(diffs) if diffs else 0.0
                mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
                passed = max_diff < tolerance
                n_pass += int(passed)
                tag = "PASS" if passed else "FAIL"
                # truncate the prompt for compact log lines
                prompt_preview = sample["prompt_text"].replace("\n", " ")[:50]
                print(
                    f"  [{tag}] {sid:<14s} "
                    f"n_toks={len(bl['token_ids']):<3d} "
                    f"max={max_diff:.3e}  mean={mean_diff:.3e}  "
                    f"prompt={prompt_preview!r}"
                )
                results.append(
                    {
                        "sample_id": sid,
                        "num_tokens": len(bl["token_ids"]),
                        "num_images": len(image_paths),
                        "num_audios": len(audio_paths),
                        "max_abs_logp_diff": max_diff,
                        "mean_abs_logp_diff": mean_diff,
                        "passed": passed,
                        "veomni_logps": logps,
                        "baseline_logps": bl["logps"],
                    }
                )

            used += 1

        if rank == 0:
            agg = {
                "world_size": world_size,
                "tolerance_max": tolerance,
                "n_samples": len(results),
                "n_passed": n_pass,
                "mean_of_max_diffs": (sum(r["max_abs_logp_diff"] for r in results) / len(results) if results else 0.0),
                "worst_max_diff": (max((r["max_abs_logp_diff"] for r in results), default=0.0)),
                "verdict": "PASS" if n_pass == len(results) and results else "FAIL",
            }
            print(f"\n{'=' * 78}")
            print(f"  Aggregate ({agg['n_samples']} samples, world_size={world_size})")
            print(f"{'=' * 78}")
            print(f"  Pass:               {agg['n_passed']}/{agg['n_samples']}")
            print(f"  Mean of max-diffs:  {agg['mean_of_max_diffs']:.3e}")
            print(f"  Worst max-diff:     {agg['worst_max_diff']:.3e}")
            print(f"  Tolerance:          max={tolerance}")
            print(f"  Verdict:            {agg['verdict']}\n")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps({"per_sample": results, "aggregate": agg}, indent=2))
            print(f"[mm-parity] report written to {out_path}")
            return 0 if agg["verdict"] == "PASS" else 1
        return 0


if __name__ == "__main__":
    samples_path = os.environ.get("PARITY_SAMPLES")
    baseline_path = os.environ.get("PARITY_BASELINE")
    out_path = os.environ.get("PARITY_OUT", "outputs/pangu_multimodal_parity_report.json")
    tolerance = float(os.environ.get("PARITY_TOLERANCE", "5e-2"))
    n_samples = int(os.environ.get("PARITY_N_SAMPLES", "3"))
    if not samples_path or not baseline_path:
        print(
            "[ERROR] set PARITY_SAMPLES and PARITY_BASELINE in env. See module docstring for example.",
            file=sys.stderr,
        )
        sys.exit(2)

    args = parse_args(VeOmniArguments)
    runner = MultimodalParityRunner(args)
    sys.exit(
        runner.parity_check(
            samples_path=samples_path,
            baseline_path=baseline_path,
            out_path=out_path,
            tolerance=tolerance,
            n_samples=n_samples,
        )
    )
