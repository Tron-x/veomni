"""Offline diff between two ``pangu_audio_intermediate_dump.py`` runs.

Companion to ``pangu_audio_intermediate_dump.py``. After dumping HF and
VeOmni intermediate tensors to two separate dirs, this script loads them
side-by-side and reports max-abs-diff per checkpoint.

Reading the output (in increasing-depth order of the forward pass):

  1. audio_outputs_last_hidden_state — audio_tower raw output, before
     proj. If this is the FIRST checkpoint to diverge, the bug is
     inside the audio_tower (VGG / Conformer / norm / mha kernel).
  2. audio_features_post_proj — same as #1 after audio_tower.proj.
     If only this diverges (and #1 matches), the proj layer's weights
     loaded differently.
  3. last_hidden_state — LLM backbone final hidden state. If the
     audio_features matched but this diverges, the LLM backbone's
     forward path is divergent (likely position_ids / rope / cache
     handling at long input shapes).
  4. boundary_logits — the lm_head outputs at the positions that
     drive per-token logp (prompt_len-1 .. prompt_len-1+n_new). This
     is the closest pre-softmax view of what the oracle sees.

Run:
  PY tools/pangu_audio_diff.py \\
     --hf-dir /tmp/audio2_dump_hf \\
     --veomni-dir /tmp/audio2_dump_veomni
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHECKPOINT_ORDER = [
    "audio_tower_input_features",  # input to audio_tower
    "audio_tower_feature_lens",  # feature_lens kwarg to audio_tower
    "audio_tower_output_lengths",  # output lengths returned from audio_tower
    "audio_outputs_last_hidden_state",  # raw audio_tower output (pre-proj)
    "audio_features_post_proj",  # after audio_tower.proj
    "last_hidden_state",  # LLM final hidden
    "boundary_logits",  # lm_head boundary logits
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--veomni-dir", type=Path, required=True)
    args = p.parse_args()

    import torch

    print(
        f"{'checkpoint':<42s} {'hf_shape':>22s} {'vo_shape':>22s} {'max_abs':>12s} {'mean_abs':>12s} {'verdict':>8s}"
    )
    print("-" * 130)

    for name in CHECKPOINT_ORDER:
        hf_path = args.hf_dir / f"{name}.pt"
        vo_path = args.veomni_dir / f"{name}.pt"
        if not hf_path.exists() or not vo_path.exists():
            print(f"{name:<42s}   MISSING: hf={hf_path.exists()} vo={vo_path.exists()}")
            continue
        hf = torch.load(hf_path, map_location="cpu")
        vo = torch.load(vo_path, map_location="cpu")
        if hf.shape != vo.shape:
            print(f"{name:<42s} {str(tuple(hf.shape)):>22s} {str(tuple(vo.shape)):>22s}   SHAPE MISMATCH")
            continue
        diff = (hf.float() - vo.float()).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        verdict = "MATCH" if max_d < 1e-5 else "DIFF"
        print(
            f"{name:<42s} {str(tuple(hf.shape)):>22s} {str(tuple(vo.shape)):>22s} "
            f"{max_d:>12.4e} {mean_d:>12.4e} {verdict:>8s}"
        )

    # Cross-reference with the meta if both exist
    hf_meta = args.hf_dir / "_meta.json"
    vo_meta = args.veomni_dir / "_meta.json"
    if hf_meta.exists() and vo_meta.exists():
        print()
        print("[meta] hf keys:", json.loads(hf_meta.read_text())["keys"])
        print("[meta] vo keys:", json.loads(vo_meta.read_text())["keys"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
