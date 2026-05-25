"""Multi-card parity runner for Pangu Omni v2 text backbone.

Reuses the production ``TextTrainer`` setup pipeline (``_setup``,
``_build_model``, ``_build_parallelized_model``) so the forward path
through FSDP2 + EP + ``fused_npu`` MoE is exactly the same as
``tasks/train_text.py`` uses for training. Skips dataloader / optimizer
/ scheduler / callbacks (none are needed for forward-only parity).

For each prompt in ``PARITY_SAMPLES`` we run a teacher-forced forward
on ``[prompt; baseline_token_ids]`` and read the per-token
log-probabilities the multi-card model assigns to each baseline token.
On rank 0 these are diffed against ``PARITY_BASELINE`` (typically the
HF text-only oracle produced by ``bootstrap_text_only_baseline.py``).

Designed to be launched the same way ``tasks/train_text.py`` is:

    torchrun --nnodes=1 --nproc_per_node=8 --master_port=29508 \\
        veomni/models/transformers/pangu_omni_v2/tools/multi_card_parity_runner.py \\
        configs/text/pangu_real_8card.yaml \\
        --train.gradient_checkpointing.enable=false

with the parity I/O paths passed via env so they don't pollute the
``VeOmniArguments`` schema:

    PARITY_SAMPLES=/tmp/pangu_text_only_oracle/samples.jsonl
    PARITY_BASELINE=/tmp/pangu_text_only_oracle/baseline.jsonl
    PARITY_OUT=/tmp/pangu_text_only_oracle/report_8card.json
    PARITY_TOLERANCE=5e-2    # optional; default 5e-2 (bf16 + EP + fused MoE)

Expected verdicts:

* Single-card HF parity (``oracle_check.py --mode veomni --text-only``)
  already proved that ``moe_implementation=eager`` matches HF bit-for-bit
  on the text-only path (see ``b3``). So a residual >> 5e-2 here would
  point at FSDP2+EP wiring or the ``fused_npu`` MoE kernel itself; a
  residual close to 0 confirms the whole production stack agrees.
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
from veomni.trainer.base import BaseTrainer


class ParityRunner:
    """Wraps ``BaseTrainer`` to build only the model + parallel plan,
    then runs a forward-only parity loop instead of training.

    Mirrors ``TextTrainer.__init__``'s composition pattern (private
    helpers called explicitly on ``self.base`` via
    ``BaseTrainer.__new__``) — see ``veomni/trainer/text_trainer.py``.
    The difference here: we skip ``_build_model_assets``,
    ``_build_dataset``, ``_build_collate_fn``, ``_build_dataloader``,
    ``_build_optimizer``, ``_build_lr_scheduler``,
    ``_build_training_context``, and ``_init_callbacks`` because none of
    them affect the forward graph we care about.
    """

    def __init__(self, args):
        self.args: VeOmniArguments = args
        # Manual BaseTrainer construction — same trick TextTrainer uses
        # so we can run setup helpers individually.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args
        # _setup: init distributed, parallel_state (incl. ep mesh from
        # extra_parallel_sizes), seeding, etc.
        self.base._setup()
        # _build_model: build_foundation_model(init_device=meta) for
        # FSDP2. Leaves params on meta; weights are streamed in by FSDP
        # later inside _build_parallelized_model.
        self.base._build_model()
        # Print trainable params count; harmless in eval mode.
        self.base._freeze_model_module()
        # FSDP2 wrap + EP plan + (optionally) gradient checkpointing
        # wrap. After this the model is on-device, params are sharded,
        # and weights have been streamed in from disk.
        self.base._build_parallelized_model()
        self.base.model.eval()
        # Expose what the parity loop reads.
        self.model = self.base.model
        self.device = self.base.device

    def parity_check(self, samples_path: str, baseline_path: str, out_path: str, tolerance: float) -> int:
        from transformers import AutoProcessor

        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1
        device = self.device

        model_dir = self.args.model.model_path
        # AutoProcessor handles the Pangu chat_template + Chinese
        # tokenizer; same path the bootstrap script and oracle_check use,
        # so token IDs are guaranteed to match across {HF, 1-card
        # VeOmni, this 8-card runner}.
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

        samples = _load_jsonl(samples_path)
        baseline = {row["sample_id"]: row for row in _load_jsonl(baseline_path)}

        if rank == 0:
            print(f"\n[parity] world_size={world_size}  device={device}")
            print(f"[parity] samples   = {samples_path} ({len(samples)} rows)")
            print(f"[parity] baseline  = {baseline_path}")
            print(f"[parity] tolerance = max abs logp diff < {tolerance}")
            print()

        results: list[dict] = []
        n_pass = 0
        for sample in samples:
            sid = sample["sample_id"]
            if sid not in baseline:
                if rank == 0:
                    print(f"  [SKIP] {sid}: no baseline entry")
                continue
            bl = baseline[sid]
            # All ranks tokenize the same prompt deterministically;
            # avoids broadcasting ragged-length tensors.
            conversation = [{"role": "user", "content": [{"type": "text", "text": sample["prompt_text"]}]}]
            chat_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=chat_text, padding=False, return_tensors="pt").to(device)
            prompt_len = int(inputs.input_ids.shape[1])
            baseline_ids = torch.tensor([bl["token_ids"]], dtype=inputs.input_ids.dtype, device=device)
            full_input_ids = torch.cat([inputs.input_ids, baseline_ids], dim=1)
            attention_mask = torch.ones_like(full_input_ids)

            with torch.no_grad():
                outputs = self.model(input_ids=full_input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # [T, V] — replicated across DP ranks (no SP)

            logps: list[float] = []
            for i, tok_id in enumerate(bl["token_ids"]):
                # See oracle_check.compute_per_token_logps for the +/-1 offset:
                # logits at position p predict token p+1, so the position that
                # predicts baseline_token_ids[i] is (prompt_len - 1 + i).
                logit_pos = prompt_len - 1 + i
                lp = F.log_softmax(logits[logit_pos].float(), dim=-1)[int(tok_id)]
                logps.append(float(lp.detach().cpu()))

            # Only rank 0 diffs/prints/dumps; the forward already ran on all
            # ranks (DP/FSDP requirement) but the result is replicated.
            if rank == 0:
                diffs = [abs(a - b) for a, b in zip(logps, bl["logps"])]
                max_diff = max(diffs) if diffs else 0.0
                mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
                passed = max_diff < tolerance
                n_pass += int(passed)
                tag = "PASS" if passed else "FAIL"
                print(
                    f"  [{tag}] {sid:<22s} n_toks={len(bl['token_ids']):<3d} max={max_diff:.3e}  mean={mean_diff:.3e}"
                )
                results.append(
                    {
                        "sample_id": sid,
                        "num_tokens": len(bl["token_ids"]),
                        "max_abs_logp_diff": max_diff,
                        "mean_abs_logp_diff": mean_diff,
                        "passed": passed,
                        "veomni_logps": logps,
                        "baseline_logps": bl["logps"],
                    }
                )

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
            print(f"\n{'=' * 72}")
            print(f"  Aggregate ({agg['n_samples']} samples, world_size={world_size})")
            print(f"{'=' * 72}")
            print(f"  Pass:               {agg['n_passed']}/{agg['n_samples']}")
            print(f"  Mean of max-diffs:  {agg['mean_of_max_diffs']:.3e}")
            print(f"  Worst max-diff:     {agg['worst_max_diff']:.3e}")
            print(f"  Tolerance:          max={tolerance}")
            print(f"  Verdict:            {agg['verdict']}\n")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps({"per_sample": results, "aggregate": agg}, indent=2))
            print(f"[parity] report written to {out_path}")
            return 0 if agg["verdict"] == "PASS" else 1
        return 0


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    samples_path = os.environ.get("PARITY_SAMPLES")
    baseline_path = os.environ.get("PARITY_BASELINE")
    out_path = os.environ.get("PARITY_OUT", "/tmp/parity_report.json")
    tolerance = float(os.environ.get("PARITY_TOLERANCE", "5e-2"))
    if not samples_path or not baseline_path:
        print(
            "[ERROR] set PARITY_SAMPLES and PARITY_BASELINE in env. See module docstring for example.",
            file=sys.stderr,
        )
        sys.exit(2)

    args = parse_args(VeOmniArguments)
    runner = ParityRunner(args)
    sys.exit(
        runner.parity_check(
            samples_path=samples_path,
            baseline_path=baseline_path,
            out_path=out_path,
            tolerance=tolerance,
        )
    )
