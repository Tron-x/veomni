"""Build the Pangu Omni v2 multimodal SFT toy corpus.

Output: ``tests/toy_data/pangu_mm_toy/train.jsonl``.

The corpus is built from on-disk oracle samples (already copied to
``/mnt/data_3/...``) so the smoke run doesn't depend on network or
HuggingFace dataset hubs. Four sample categories — kept small (4 of
each, 16 total) on purpose, since the SFT smoke only runs 5 steps:

1. ``image+text`` — OCRBench: image + "what is written" + answer string.
2. ``audio+text`` — audio_demo: audio clip + "What's that sound?" + answer.
3. ``image+audio+text`` — pair one OCR image with one audio clip,
   concatenate the two prompts, and concatenate the two answers.
   This exercises both encoders in a single forward (FSDP wrap order
   matters here — see ``_no_split_modules`` change on 2026-05-24).
4. ``text+text`` — plain Q+A (no media) so we exercise the
   ``dummy_forward`` path on the vision and audio towers when no
   media tokens are present in the batch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
# Data lives in a sibling dir so the dataset loader's
# ``sorted(os.listdir(train_path))`` only sees .jsonl files.
TRAIN_JSONL = HERE.parent / "pangu_mm_toy" / "train.jsonl"

OCRBENCH_ROOT = Path("/mnt/data_3/models/pangu/test_hf_percision.0518.parallel")
OCRBENCH_JSONL = OCRBENCH_ROOT / "data" / "ocrbench.jsonl"
AUDIO_ROOT = Path("/mnt/data_3/models/pangu_audio_oracle")
AUDIO_JSONL = AUDIO_ROOT / "data" / "audio_demo.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def resolve_paths(items: list[str], root: Path, key: str) -> list[str]:
    out: list[str] = []
    for rel in items:
        p = root / rel
        if not p.exists():
            print(f"WARN: missing {key} {p}", file=sys.stderr)
            continue
        out.append(str(p))
    return out


def main() -> int:
    ocr_samples = read_jsonl(OCRBENCH_JSONL)[:8]
    audio_samples = read_jsonl(AUDIO_JSONL)

    rows: list[dict] = []

    # 4 image+text samples.
    for s in ocr_samples[:4]:
        rows.append(
            {
                "source": "pangu_mm_sft_v1",
                "conversations": [
                    {"from": "human", "value": f"<image>{s['prompt_text']}"},
                    {"from": "gpt", "value": str(s["answer"])},
                ],
                "images": resolve_paths(s["image_paths"], OCRBENCH_ROOT, "image"),
                "audios": [],
                "videos": [],
            }
        )

    # 3 audio+text samples (only 3 oracle audio samples exist).
    for s in audio_samples:
        rows.append(
            {
                "source": "pangu_mm_sft_v1",
                "conversations": [
                    {"from": "human", "value": f"<audio>{s['prompt_text']}"},
                    {"from": "gpt", "value": str(s["answer"])},
                ],
                "images": [],
                "audios": resolve_paths(s["audio_paths"], AUDIO_ROOT, "audio"),
                "videos": [],
            }
        )

    # 3 image+audio+text samples — pair OCR image i with audio i%3.
    for i, s in enumerate(ocr_samples[4:7]):
        audio_s = audio_samples[i % len(audio_samples)]
        rows.append(
            {
                "source": "pangu_mm_sft_v1",
                "conversations": [
                    {
                        "from": "human",
                        "value": (f"<image><audio>{s['prompt_text']} Also, {audio_s['prompt_text'].lower()}"),
                    },
                    {
                        "from": "gpt",
                        "value": f"{s['answer']} And: {audio_s['answer']}",
                    },
                ],
                "images": resolve_paths(s["image_paths"], OCRBENCH_ROOT, "image"),
                "audios": resolve_paths(audio_s["audio_paths"], AUDIO_ROOT, "audio"),
                "videos": [],
            }
        )

    # 4 text-only samples — exercise the no-media branch (vision /
    # audio towers go through dummy_forward to stay in the FSDP
    # compute graph).
    text_qa = [
        ("盘古大模型是华为推出的什么类型的模型？", "盘古大模型是华为推出的大语言模型系列。"),
        ("VeOmni 是什么？", "VeOmni 是字节跳动 Seed 团队开源的分布式多模态训练框架。"),
        ("MoE 是什么的简称？", "Mixture of Experts，一种稀疏激活的模型架构。"),
        ("FSDP 是什么？", "Fully Sharded Data Parallel，PyTorch 的分布式参数分片训练方案。"),
    ]
    for q, a in text_qa:
        rows.append(
            {
                "source": "pangu_mm_sft_v1",
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": a},
                ],
                "images": [],
                "audios": [],
                "videos": [],
            }
        )

    with TRAIN_JSONL.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples to {TRAIN_JSONL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
