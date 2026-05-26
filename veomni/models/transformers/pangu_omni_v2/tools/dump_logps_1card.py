"""Single-process per-token logp dumper for Pangu Omni v2 text-only path.

Reuses ``oracle_check.load_veomni_model_and_processor`` +
``compute_per_token_logps`` (the same machinery that gave us
``report_1card_fusednpu.json``), and additionally writes the full
``[num_samples][num_tokens]`` float64 per-token logp array to JSON so it
can be element-wise diffed against ``report_8card_nomp.json``'s
``veomni_logps``.

The diff against the 8-card runner answers a stricter question than
"is the aggregate `max_abs_logp_diff` the same double" (already shown):
**are the per-token logp arrays themselves bit-identical between
1-NPU fused_npu and 8-NPU EP=8 fused_npu?**

Usage::

    PANGU_ORACLE_MOE_IMPL=fused_npu \\
      python veomni/models/transformers/pangu_omni_v2/tools/dump_logps_1card.py \\
      --out /tmp/pangu_text_only_oracle/logps_1card_fusednpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ["PANGU_MODEL_DIR"]) if "PANGU_MODEL_DIR" in os.environ else None,
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("outputs/pangu_text_only_oracle/samples.jsonl"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/pangu_text_only_oracle/baseline.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/pangu_text_only_oracle/logps_1card_fusednpu.json"),
    )
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error("set --model-dir explicitly, or set PANGU_MODEL_DIR")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle_check import (
        compute_per_token_logps,
        load_jsonl,
        load_veomni_model_and_processor,
    )

    samples = load_jsonl(args.samples)
    baseline = {row["sample_id"]: row for row in load_jsonl(args.baseline)}

    print("[*] loading VeOmni text-only (env PANGU_ORACLE_MOE_IMPL controls MoE kernel)")
    model, processor = load_veomni_model_and_processor(args.model_dir, text_only=True)

    results: list[dict] = []
    for s in samples:
        sid = s["sample_id"]
        if sid not in baseline:
            print(f"   [SKIP] {sid}: no baseline entry")
            continue
        bl = baseline[sid]
        logps = compute_per_token_logps(model, processor, s, bl["token_ids"], text_only=True)
        results.append({"sample_id": sid, "veomni_logps": logps, "baseline_logps": bl["logps"]})
        print(f"   {sid:<14s} n_toks={len(logps)} done")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"per_sample": results}, indent=2))
    print(f"\n[OK] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
