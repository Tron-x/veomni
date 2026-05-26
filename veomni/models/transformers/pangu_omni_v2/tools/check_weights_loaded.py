"""Pangu Omni v2 — weight load inventory checker (single NPU, no FSDP).

Answers the question: when we load the real 30B-A2B safetensors via
``OpenPanguV2ForCausalLM`` (the text-only architecture override used by
``configs/text/pangu_real_8card.yaml``), what actually lands in memory?

Specifically reports:
    1. **In-memory params/buffers**: every tensor on
       ``OpenPanguV2ForCausalLM`` after ``build_foundation_model``,
       grouped by top-level submodule prefix.
    2. **On-disk safetensors keys**: every key in
       ``model.safetensors.index.json``, grouped by the same prefix.
    3. **Coverage**: which in-memory params got filled from on-disk keys
       (via direct match or via the ``checkpoint_tensor_converter``
       fused 3D collapse), and which on-disk keys were skipped (expected
       for vision/audio when the target is text-only).
    4. **Sanity check**: confirm no in-memory tensor is still on the
       ``meta`` device after load (would mean silently-missing weights).

Run:

    python veomni/models/transformers/pangu_omni_v2/tools/check_weights_loaded.py \\
        --model-dir /path/to/pangu_text_only_view

The view dir is the symlink-view created in step-1 of the 8-card SFT
smoke: real Pangu weights + a ``config.json`` rewritten with
``architectures=["OpenPanguV2ForCausalLM"]``. See
``configs/text/pangu_real_8card.yaml`` header comments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


os.environ.setdefault("MODELING_BACKEND", "veomni")


def _topgroup(fqn: str) -> str:
    """Return the high-level subtree this FQN belongs to.

    Buckets the 30B-A2B parameter tree into a handful of human-readable
    groups so the inventory report is short. Anything below
    ``model.layers.X.YYY`` collapses into the per-decoder-layer group
    ``"layers.YYY"``.
    """
    parts = fqn.split(".")
    if not parts:
        return fqn
    head = parts[0]
    if head in {"vision_model", "audio_tower", "vision_tower", "language_model"}:
        return head
    if head == "lm_head":
        return "lm_head"
    if head == "model":
        if len(parts) >= 2 and parts[1] == "layers" and len(parts) >= 4:
            return f"model.layers.*.{parts[3]}"
        if len(parts) >= 2:
            return f"model.{parts[1]}"
    return head


def _summarize(name: str, counts: dict[str, int]) -> None:
    print(f"\n--- {name} ---")
    total = sum(counts.values())
    for k in sorted(counts.keys()):
        print(f"  {k:<35s}  {counts[k]:>6d}")
    print(f"  {'TOTAL':<35s}  {total:>6d}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ["PANGU_MODEL_DIR"]) if "PANGU_MODEL_DIR" in os.environ else None,
        help="Directory with config.json + model.safetensors.index.json + per-shard .safetensors",
    )
    parser.add_argument(
        "--device",
        default="npu",
        choices=["cpu", "npu", "cuda"],
        help="Where to materialize the loaded model (cpu = no NPU needed)",
    )
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error("set --model-dir explicitly, or set PANGU_MODEL_DIR")

    print(f"[*] model_dir = {args.model_dir}")
    print(f"[*] device    = {args.device}")

    # --- on-disk inventory --------------------------------------------------
    index_path = args.model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"[ERR] no safetensors index at {index_path}", file=sys.stderr)
        return 2
    with index_path.open() as fh:
        weight_map = json.load(fh)["weight_map"]
    disk_keys = sorted(weight_map.keys())
    disk_groups: dict[str, int] = defaultdict(int)
    for k in disk_keys:
        disk_groups[_topgroup(k)] += 1
    _summarize(f"On-disk safetensors keys ({len(disk_keys)})", dict(disk_groups))

    # --- build model via VeOmni --------------------------------------------
    # Import inside main() so the script can `--help` without pulling in
    # torch_npu on the smoke node.
    import veomni.models.transformers.pangu_omni_v2  # noqa: F401 — registry
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model

    ops = OpsImplementationConfig(
        attn_implementation="sdpa",
        # 1-card path → eager is correct + cheap; for EP inventory we'd flip
        # to fused_npu but this script intentionally exercises the same
        # eager loader as oracle_check.py.
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    print("[*] build_foundation_model() — streaming safetensors...")
    model = build_foundation_model(
        config_path=str(args.model_dir),
        weights_path=str(args.model_dir),
        init_device=args.device,
        ops_implementation=ops,
    )
    print(f"[*] model class = {type(model).__name__}")

    # --- in-memory inventory -----------------------------------------------
    param_groups: dict[str, int] = defaultdict(int)
    buffer_groups: dict[str, int] = defaultdict(int)
    meta_tensors: list[str] = []
    total_params = 0
    in_memory_keys: set[str] = set()
    for name, p in model.named_parameters():
        param_groups[_topgroup(name)] += 1
        total_params += p.numel()
        in_memory_keys.add(name)
        if p.device.type == "meta":
            meta_tensors.append(name)
    for name, b in model.named_buffers():
        buffer_groups[_topgroup(name)] += 1
        in_memory_keys.add(name)
        if b.device.type == "meta":
            meta_tensors.append(name)
    _summarize(f"In-memory parameters ({sum(param_groups.values())})", dict(param_groups))
    _summarize(f"In-memory buffers ({sum(buffer_groups.values())})", dict(buffer_groups))
    print(f"[*] total parameter elements = {total_params / 1e9:.2f} B")

    # --- coverage ----------------------------------------------------------
    # The checkpoint converter fuses per-expert keys
    # (e.g. ``model.layers.X.mlp.experts.{0..E-1}.gate_proj.weight``) into
    # ``model.layers.X.mlp.experts.gate_up_proj`` (single fused 3D tensor).
    # So matching is "is there ANY on-disk key that maps to this in-memory
    # key under the converter rules". The simplest faithful check: bucket
    # both sides by the converter's "destination" name (strip
    # ``.experts.<int>.`` → ``.experts.`` and join gate/up → gate_up).
    def _normalize_for_match(disk_key: str) -> str:
        """Map an on-disk key into the in-memory namespace the model exposes."""
        # Fused expert: drop ".experts.<int>." → ".experts.", and merge
        # gate_proj / up_proj into gate_up_proj.
        import re

        k = re.sub(r"\.experts\.(\d+)\.", ".experts.", disk_key)
        if k.endswith(".gate_proj.weight") and ".experts." in k:
            k = k.replace(".gate_proj.weight", ".gate_up_proj")
        elif k.endswith(".up_proj.weight") and ".experts." in k:
            k = k.replace(".up_proj.weight", ".gate_up_proj")
        elif k.endswith(".down_proj.weight") and ".experts." in k:
            k = k.replace(".down_proj.weight", ".down_proj")
        return k

    disk_to_mem: dict[str, set[str]] = defaultdict(set)  # in-memory name -> set of disk keys mapping there
    for dk in disk_keys:
        disk_to_mem[_normalize_for_match(dk)].add(dk)
    on_disk_normed: set[str] = set(disk_to_mem.keys())
    covered = in_memory_keys & on_disk_normed
    in_mem_unfilled = in_memory_keys - on_disk_normed
    disk_only = on_disk_normed - in_memory_keys
    print(
        f"\n[*] coverage: {len(covered)} / {len(in_memory_keys)} in-memory tensors "
        f"have a matching on-disk key after converter normalization"
    )
    if in_mem_unfilled:
        print("\n[!] in-memory tensors WITHOUT a matching on-disk key (would be random-init):")
        for k in sorted(in_mem_unfilled)[:30]:
            print(f"     {k}")
        if len(in_mem_unfilled) > 30:
            print(f"     ... and {len(in_mem_unfilled) - 30} more")
    if disk_only:
        # Group disk-only keys by top-group for a readable summary —
        # we EXPECT a large chunk here (vision_tower / audio_tower etc.
        # that aren't part of OpenPanguV2ForCausalLM).
        skipped_groups: dict[str, int] = defaultdict(int)
        # Re-bucket by original key topgroup (not normalized), so the
        # report mentions "vision_tower" etc. clearly.
        for normalized_key in disk_only:
            for orig in disk_to_mem[normalized_key]:
                skipped_groups[_topgroup(orig)] += 1
        _summarize(
            f"On-disk keys SKIPPED by text-only architecture ({sum(skipped_groups.values())})",
            dict(skipped_groups),
        )

    # --- meta sanity -------------------------------------------------------
    if meta_tensors:
        print(f"\n[!!] {len(meta_tensors)} tensor(s) still on meta device after load:")
        for k in meta_tensors[:20]:
            print(f"      {k}")
        return 1

    print("\n[OK] all in-memory tensors materialized off meta device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
