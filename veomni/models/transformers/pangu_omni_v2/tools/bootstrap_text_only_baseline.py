"""Generate a text-only HF baseline JSONL for Pangu Omni v2 oracle parity.

The existing OCRBench baseline at::

    /path/to/oracle/results/ocrbench_hf_outputs.jsonl

was generated from MULTIMODAL inputs (image + text) and lands on the
multimodal model class. It cannot be used to validate the **text-only**
``OpenPanguV2ForCausalLM`` path that ``configs/text/pangu_real_8card.yaml``
uses for the multi-card EP SFT smoke.

This script bootstraps a small text-only baseline by:

  1. Loading the full 30B-A2B model via HF ``AutoModelForCausalLM`` +
     ``trust_remote_code=True`` (the same path ``oracle_check.py
     --mode hf`` uses).
  2. For each prompt:
     a) Greedy-decode N continuation tokens (``do_sample=False``).
     b) Teacher-force forward on ``[prompt; greedy_tokens]`` and read the
        per-position log-probability of each greedy token. This matches
        what ``compute_per_token_logps`` does, so a follow-up
        ``oracle_check.py --mode hf --text-only`` against this baseline
        gives bit-identical zero diff (sanity check).
  3. Writing two JSONL files in the schema ``oracle_check.py`` expects:
     - ``samples.jsonl``: ``{sample_id, prompt_text}``
     - ``baseline.jsonl``: ``{sample_id, prompt_text, token_ids, tokens,
                              logps, sum_logp, mean_logp, num_tokens}``

Usage::

    python veomni/models/transformers/pangu_omni_v2/tools/bootstrap_text_only_baseline.py \\
        --out-dir /tmp/pangu_text_only_oracle \\
        --max-new-tokens 32

After this finishes, run::

    python veomni/models/transformers/pangu_omni_v2/tools/oracle_check.py \\
        --mode hf --text-only \\
        --samples /tmp/pangu_text_only_oracle/samples.jsonl \\
        --baseline /tmp/pangu_text_only_oracle/baseline.jsonl
        # expected: max-diff = 0 (sanity)

    python veomni/models/transformers/pangu_omni_v2/tools/oracle_check.py \\
        --mode veomni --text-only \\
        --samples /tmp/pangu_text_only_oracle/samples.jsonl \\
        --baseline /tmp/pangu_text_only_oracle/baseline.jsonl
        # expected: max-diff < 1e-3 (Week 3.6-style tolerance)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# Four mid-length Chinese prompts. Mix of:
#   t_short:   ~20 tokens — common knowledge query
#   t_train:   matches the style of tests/toy_data/pangu_toy/train.jsonl
#              (used by the 8-card SFT smoke training data)
#   t_code:    technical / English-mixed (covers BPE merges into Latin sub-tokens)
#   t_long:    longer multi-sentence prompt — covers position embedding rollout
#              past the short-context regime that t_short / t_train hit
PROMPTS = [
    {"sample_id": "txt_short", "prompt_text": "请简要介绍一下大语言模型的训练流程。"},
    {
        "sample_id": "txt_train",
        "prompt_text": ("盘古-Omni 是华为推出的多模态大模型，请说明它在文本、视觉和音频三种模态上的能力分别有哪些。"),
    },
    {
        "sample_id": "txt_code",
        "prompt_text": (
            "Mixture-of-Experts (MoE) 模型为什么需要 expert parallelism？请解释 EP=8 和 ep_outside=false 的含义。"
        ),
    },
    {
        "sample_id": "txt_long",
        "prompt_text": (
            "在分布式训练 30B 参数规模的语言模型时，"
            "FSDP2 与张量并行 (TP) 各自的优缺点是什么？"
            "如果只有 8 张昇腾 NPU 卡，应该如何组合 EP、FSDP、SP 三种并行策略？"
        ),
    },
]


def build_text_only_conversation(prompt_text: str) -> list[dict]:
    """Pangu chat-template format — same shape as
    ``oracle_check.build_conversation(text_only=True)``."""
    return [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ["PANGU_MODEL_DIR"]) if "PANGU_MODEL_DIR" in os.environ else None,
        help="HF model dir (real 30B-A2B; trust_remote_code=True).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/pangu_text_only_oracle"),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Greedy-decode this many continuation tokens per prompt.",
    )
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error("set --model-dir explicitly, or set PANGU_MODEL_DIR")

    # Defer torch / HF imports so `--help` is fast.
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoProcessor

    # `oracle_check.setup_npu_if_available()` — kept inline so this script
    # has no cross-file deps when run standalone.
    from transformers.utils import is_torch_npu_available

    if is_torch_npu_available() and "910" in torch.npu.get_device_name():
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] loading HF model from {args.model_dir} ...")
    t0 = time.time()
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(args.model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True)
    print(f"[*] load done in {time.time() - t0:.1f}s — class={type(model).__name__}")
    tokenizer = getattr(processor, "tokenizer", processor)

    # Samples + baseline file handles
    samples_path = args.out_dir / "samples.jsonl"
    baseline_path = args.out_dir / "baseline.jsonl"
    print(f"[*] writing samples → {samples_path}")
    print(f"[*] writing baseline → {baseline_path}")

    samples_fh = samples_path.open("w", encoding="utf-8")
    baseline_fh = baseline_path.open("w", encoding="utf-8")

    for prompt_spec in PROMPTS:
        sid = prompt_spec["sample_id"]
        prompt_text = prompt_spec["prompt_text"]
        print(f"\n--- {sid} ---")

        conversation = build_text_only_conversation(prompt_text)
        chat_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # Text-only path: skip multimodal extraction.
        inputs = processor(text=chat_text, padding=False, return_tensors="pt").to(model.device)
        prompt_len = int(inputs.input_ids.shape[1])
        print(f"   prompt_len = {prompt_len}")

        # --- 1) Greedy decode N tokens. The HF model exposes the standard
        # `.generate()` API; the Pangu config has the multimodal extras but
        # for text-only inputs `generate` reduces to a regular causal-LM
        # generate. We avoid the multimodal sampling kwargs entirely.
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy — same convention as the OCRBench baseline
                num_beams=1,
                temperature=1.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_ids = gen_out[0].tolist()
        baseline_token_ids = full_ids[prompt_len:]
        # Strip trailing eos tokens — they're decoded once but adding their
        # logp under teacher forcing wastes a position and noises the
        # mean_logp metric.
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and baseline_token_ids and baseline_token_ids[-1] == eos_id:
            baseline_token_ids = baseline_token_ids[:-1]
        if not baseline_token_ids:
            print(f"   [WARN] empty continuation for {sid}; skipping")
            continue
        print(f"   greedy   = {baseline_token_ids[:8]}... ({len(baseline_token_ids)} toks)")
        print(f"   decoded  = {tokenizer.decode(baseline_token_ids)[:80]!r}")

        # --- 2) Teacher-forced forward on [prompt + greedy] and read the
        # per-position log-probability of each greedy token. Same logic as
        # `oracle_check.compute_per_token_logps` to keep formats identical.
        baseline_tensor = torch.tensor(
            [baseline_token_ids], dtype=inputs.input_ids.dtype, device=inputs.input_ids.device
        )
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
        with torch.no_grad():
            outputs = model(input_ids=full_input_ids, **extended_inputs)
        logits = outputs.logits[0]  # [T, V]
        logps: list[float] = []
        for i, tok_id in enumerate(baseline_token_ids):
            logit_pos = prompt_len - 1 + i
            logp = F.log_softmax(logits[logit_pos].float(), dim=-1)[int(tok_id)]
            logps.append(float(logp.detach().cpu()))

        token_strs = tokenizer.convert_ids_to_tokens(baseline_token_ids)
        sum_logp = sum(logps)
        mean_logp = sum_logp / len(logps) if logps else 0.0

        # Sample line is the lean ``{sample_id, prompt_text}`` form that
        # ``oracle_check.main()`` expects from a ``--samples`` JSONL.
        samples_fh.write(json.dumps({"sample_id": sid, "prompt_text": prompt_text}, ensure_ascii=False) + "\n")
        # Baseline line matches the OCRBench schema fields the
        # ``compare_one`` + ``print_sample`` path reads.
        baseline_fh.write(
            json.dumps(
                {
                    "sample_id": sid,
                    "prompt_text": prompt_text,
                    "token_ids": baseline_token_ids,
                    "tokens": token_strs,
                    "logps": logps,
                    "sum_logp": sum_logp,
                    "mean_logp": mean_logp,
                    "num_tokens": len(baseline_token_ids),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        baseline_fh.flush()
        samples_fh.flush()
        print(f"   sum_logp = {sum_logp:.4f}, mean_logp = {mean_logp:.4f}")

    samples_fh.close()
    baseline_fh.close()
    print(f"\n[OK] wrote {samples_path} + {baseline_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
