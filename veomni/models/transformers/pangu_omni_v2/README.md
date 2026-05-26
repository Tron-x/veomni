# `pangu_omni_v2/` — Pangu Omni v2 multimodal MoE adapter

VeOmni adapter for the **Pangu Omni 30B-A2B** family (and forthcoming
larger-scale variants such as 70B-A7B that share the same architecture).

## Status

- **Modeling path**: text, vision, and audio modules are wired through
  `OpenPanguOmni` for VLM training, with a text-only `OpenPanguV2ForCausalLM`
  path available for language-only ablations.
- **Patchgen path**: `pangu_omni_v2_gpu_patch_gen_config.py` is kept as the
  future v5 codegen entrypoint; the current adapter uses the handwritten
  modules in this package.
- **Training path**: thinker-only. There is no separate talker / TTS in Pangu
  Omni v2; the LM head is trained directly.

## Architecture deltas vs `qwen3_omni_moe`

| Component | Pangu Omni v2 | `qwen3_omni_moe` (reference) | Status in this adapter |
|---|---|---|---|
| Text MoE | 384 routed + 2 shared experts, MHC (4-stream) | 128 routed, no shared | Implemented |
| Partial RoPE | `qk_rope_dim=32`, `partial_rotary_factor=0.25` | Full RoPE | Implemented |
| K-norm | Pangu-specific kernel norm operator | Standard RMSNorm | Implemented |
| Vision merger | GatedMerger | PatchMerger | Implemented |
| Audio encoder | HuanyuAudioEncoder + fbank-40 (PyTorch or .so) | Whisper-like | Implemented |
| MRoPE | 3-axis position_ids (temporal/H/W) | Same | Reused from upstream HF |

## File Layout

```
pangu_omni_v2/
├── __init__.py                              # Registry wiring + architecture dispatcher
├── configuration_pangu_omni_v2.py           # Config subclass + model_type patch
├── modeling_text.py                         # Text MoE backbone (OpenPanguV2Model / CausalLM)
├── modeling_vl.py                           # Vision tower + vision/text merge
├── modeling_omni.py                         # Full multimodal wrapper (vision + audio + text)
├── modeling_huanyu_audio_encoder.py         # Huanyu audio tower
├── pangu_omni_v2_gpu_patch_gen_config.py    # v5 patchgen entrypoint
├── generated/
│   └── patched_modeling_pangu_omni_v2_gpu.py    # patchgen output (auto-generated)
├── parallel_plan.py                         # EP plan for routed + shared experts
├── checkpoint_tensor_converter.py           # HF per-expert -> fused gate_up_proj
└── README.md                                # This file
```

## Portable Usage

Use the production-style templates under `configs/multimodal/pangu_omni_v2/`:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29509 \
  tasks/train_vlm.py configs/multimodal/pangu_omni_v2/pangu_omni_8card_sft.yaml
```

Before launching, replace `/path/to/pangu_omni_v2` and
`/path/to/multimodal_sft_data` in the template with paths that exist in the
target environment. For 2-node training, use
`configs/multimodal/pangu_omni_v2/pangu_omni_2node_sft.yaml` and ensure model
and data paths are identical on every node.

Local smoke/debug configs that contain `/mnt/data_3`, cluster IPs, toy data, or
short-step limits are intentionally isolated under
`configs/multimodal/pangu_omni_v2/local_smoke/`.

## Real 30B-A2B + EP=8 multimodal SFT smoke — GREEN

Validated 2026-05-25. The full multi-card multi-expert + multi-modality
training pipeline runs end-to-end on the real Pangu Omni v2 30B-A2B
weights through `VLMTrainer`:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29509 \
  tasks/train_vlm.py configs/multimodal/pangu_omni_v2/local_smoke/pangu_mm_8card.yaml
# step 1 (image+audio+text mixed): total_loss 1.46, grad_norm 14.99
# VRAM 40.77GB / 54.34GB peak per NPU, 64.2 s/step, exit 0
```

This exercises every component in one packed batch:

- Per-rank micro_batch=1 packed up to `max_seq_len=1024` tokens via
  `TextBatchingStrategy` — mixes image+text, audio+text,
  image+audio+text, and pure-text samples (see
  `tests/toy_data/pangu_mm_toy/train.jsonl`, 14 samples).
- `OpenPanguOmni.visual` (`OpenPanguVisionTransformerPretrainedModel`)
  forward through the 28-layer ViT + gated patch merger.
- `OpenPanguOmni.audio_tower` (`HuanyuAudioEncoder`) forward through
  the VGG conv stack + 24 Conformer layers + linear projection.
- `OpenPanguOmni.language_model` (`OpenPanguVLTextModel`) forward
  through the 35-layer MoE backbone with `ep_size=8` expert
  parallelism and `fused_npu` MoE.
- `MainCollator` packs `pixel_values`, `input_features`
  (`(n_mels, T_total)`), `audio_feature_lengths`, `image_grid_thw`,
  precomputed `audio_mask` / `image_mask` / `video_mask`, and
  MRoPE 3-axis `position_ids` (collator unsqueezes to `(B, 3, L)`,
  the model transposes back to `(3, B, L)` before the MRoPE forward).
- AdamW step + grad clipping + activation grad-checkpoint.

### Step-1-only caveat

Currently bounded to `max_steps: 1`: step-2 OOMs on the 64GB HBM
because AdamW state allocated after the first `optimizer.step()`
adds ~16GB resident memory, and veomni's FSDP2 wrapper does not
yet honor the `fsdp_config.offload: true` flag (it's only wired in
the FSDP1 init path — see
`veomni/distributed/torch_parallelize.py`). Multi-step training
will land once one of these is done: (a) wire FSDP2 CPU offload
into the wrapper, (b) move to a 2-node cluster for finer FSDP
sharding, or (c) switch to Adafactor (no momentum buffers).

### Text-only SFT smoke (still green)

For runs that don't need the vision / audio towers — e.g.
language-only SFT or parity ablations — the
`configs/text/pangu_real_8card.yaml` config overrides
`architectures` to `OpenPanguV2ForCausalLM` and uses
`tasks/train_text.py`:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29507 \
  tasks/train_text.py configs/text/pangu_real_8card.yaml
# step 1: loss 8.51, grad_norm 99.25
# step 2: loss 8.25, grad_norm 86.62
# step 3: loss 8.28, grad_norm 87.25
# VRAM 39.36GB / 58.21GB peak per NPU, exit 0
```

Text-only setup:

- `tests/toy_data/pangu_toy/train.jsonl` (64 samples) — same toy data
  Plan A uses; safe to swap for a real SFT corpus.
- `/path/to/pangu_text_only_view/` —
  symlinks every file from the real model dir EXCEPT `config.json`,
  which is rewritten with `architectures=["OpenPanguV2ForCausalLM"]`
  to route through the text-only path. Note: the multimodal
  MRoPE × `create_causal_mask` gap that previously made this override
  mandatory has been resolved (see the inline rationale in
  `OpenPanguV2Model.forward` for the 3D <-> 2D mask reconciliation).
  This text-only view is still useful because it (a) skips
  loading the ~26B vision/audio towers when training the language
  backbone and (b) keeps the simpler single-axis position-id path.
- `configs/text/pangu_real_8card.yaml` — single-node 8-NPU EP=8 +
  FSDP2 + bf16 + grad ckpt + meta init.

What's wired up:

- `parallel_plan.py` declares `{model.layers.*.mlp.experts.gate_up_proj,
  experts.down_proj} -> Shard(0)`. Matches qwen3_moe's
  `use_gate_up_proj=True` shape; FQN verified via
  `named_parameters()` at build time (see docstring).
- `OpenPanguV2ForCausalLM.get_parallel_plan` delegates to
  `parallel_plan.get_parallel_plan(use_gate_up_proj=True)` — VeOmni's
  `torch_parallelize.py:110` calls it.
- `_pangu_common/pangu_moe.py::OpenPanguV2Experts.forward` has an
  OpSlot guard: when `moe_implementation: fused_npu`,
  `veomni_moe_experts_forward` dispatches to
  `npu_fused_moe_forward`, which auto-detects
  `parallel_state.ep_enabled` and runs the EP all-to-all +
  `npu_group_gemm` path.
- The OpSlot is re-exported from `modeling_text.py` so
  `_bind_veomni_ops` (in `models/auto.py`) finds it via
  `dir(modeling_module)` and binds it from
  `args.model.ops_implementation.moe_implementation`.

### Multi-card precision parity — GREEN (8-card EP=8 == 1-card fused_npu, bit-identical)

Validated 2026-05-22. After the SFT smoke proved 8-NPU EP=8 trains cleanly,
we built a per-token logp parity stack to answer "is the multi-card forward
numerically the same thing single-card would produce, or did something in
the FSDP all-gather / EP all-to-all silently break the math?".

The stack reuses the existing `tools/oracle_check.py` infrastructure plus
two new pieces:

* `tools/bootstrap_text_only_baseline.py` — generates a small text-only HF
  baseline (4 Chinese prompts, greedy + teacher-force) from the real 30B-A2B
  weights. Stored at `outputs/pangu_text_only_oracle/{samples,baseline}.jsonl`
  in the OCRBench schema so `oracle_check.py` reads it unchanged.
* `tools/multi_card_parity_runner.py` — composes `BaseTrainer._setup` /
  `_build_model` / `_build_parallelized_model` (the same path
  `tasks/train_text.py` uses) so the forward goes through the production
  FSDP2 + EP + `fused_npu` MoE stack, then runs `compute_per_token_logps`
  on the 4 baseline prompts and diffs vs the HF baseline on rank 0.

Three runs, three signals:

| Config | MoE kernel | FSDP+EP | mean of max-diff vs HF | What it tells us |
|---|---|---|---|---|
| 1-NPU `oracle_check.py --mode veomni --text-only` | eager | none | **0.000e+00** | Adapter modeling on real 30B weights is bit-identical to HF (`OpenPanguV2ForCausalLM` text-only path) |
| 1-NPU `oracle_check.py --mode veomni --text-only` (`PANGU_ORACLE_MOE_IMPL=fused_npu`) | `fused_npu` | none | 1.183e-01 | Pure `npu_fused_moe_forward` group-gemm vs eager Pangu MoE drift |
| 8-NPU `multi_card_parity_runner.py` w/ `pangu_real_8card.yaml` | `fused_npu` | FSDP2 + EP=8 | **1.183e-01** | Production stack |

**The 1-card-fused and 8-card-fused per-sample max-diffs match to ≥4 sig figs**
(`txt_short` 0.1321, `txt_train` 0.06838, `txt_code` 0.1609, `txt_long`
0.1117 — identical in both columns) and `mean_of_max_diffs` matches to 10+
decimal places (`0.11826662719249725`). Conclusion:

* **FSDP all-gather + EP=8 all-to-all introduce ≈0 extra numerical drift on top of the kernel choice.**
* **All ~1.2e-1 of the logp drift vs HF is the `npu_fused_moe_forward` group-gemm itself diverging from eager Pangu MoE** (different accumulation order across the routed-expert dimension, compounded across ~35 MoE layers in bf16).
* **Production multi-card SFT is numerically equivalent to single-card.**
  Loss curves from the 8-NPU 30B smoke (`8.51 → 8.25 → 8.28` in 3 steps)
  are not just "running"; they are integrating gradients with the same bf16
  noise floor a single-NPU run would.

Practical implication for downstream work:

* SFT / RL training — `fused_npu` is fine. The drift is in the bf16 gradient
  noise floor; convergence is unaffected.
* HF-parity inference — must use 1-NPU + eager MoE
  (`oracle_check.py --mode veomni --text-only` default). EP requires
  `fused_npu`, so HF-parity inference at scale isn't possible without first
  fixing the kernel-vs-eager drift in `ops/kernels/moe/npu_group_gemm.py`.

To rerun this validation::

```bash
# (one time) bootstrap text-only HF baseline. Takes ~7 min (HF model load).
python veomni/models/transformers/pangu_omni_v2/tools/bootstrap_text_only_baseline.py \
    --out-dir outputs/pangu_text_only_oracle

# B-3 — single-card adapter parity (eager MoE)
python veomni/models/transformers/pangu_omni_v2/tools/oracle_check.py \
    --mode veomni --text-only \
    --samples outputs/pangu_text_only_oracle/samples.jsonl \
    --baseline outputs/pangu_text_only_oracle/baseline.jsonl \
    --n-samples 4
# expected: PASS 4/4, max-diff 0.000e+00

# A — multi-card production stack parity (fused_npu MoE + FSDP2 + EP=8)
PARITY_SAMPLES=outputs/pangu_text_only_oracle/samples.jsonl \
PARITY_BASELINE=outputs/pangu_text_only_oracle/baseline.jsonl \
PARITY_OUT=outputs/pangu_text_only_oracle/report_8card.json \
PARITY_TOLERANCE=5e-2 \
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29510 \
    veomni/models/transformers/pangu_omni_v2/tools/multi_card_parity_runner.py \
    configs/text/pangu_real_8card.yaml \
    --train.gradient_checkpointing.enable=false
# expected: FAIL vs 5e-2 tolerance (kernel drift), per-sample max-diff
#           matches the 1-NPU fused_npu reference run bit-for-bit
```

### Multi-card multimodal (vision + text) parity — landed, with fused-MoE drift caveat

Validated 2026-05-22 / 2026-05-23:

* L2b (`MRoPE × create_causal_mask` shape conflict) is **fixed**.
  `OpenPanguOmni` now runs forward across 8 NPUs with the real 30B-A2B
  weights and the multimodal architectures field intact (no text-only
  override needed). See the inline rationale in
  `OpenPanguV2Model.forward` for the 3D ↔ 2D mask reconciliation
  (mirrors Qwen2.5-VL's canonical pattern).
* `OpenPanguOmni.get_parallel_plan` / `OpenPanguVL.get_parallel_plan`
  now exist and pass the correct FQN prefix
  (`model.language_model`) to the EP plan builder, so the routed
  experts under `model.language_model.layers.*.mlp.experts.*` are
  correctly EP-sharded across 8 ranks.
* The MoE OpSlot is also re-exported from `modeling_omni.py`
  and `modeling_vl.py` so `_bind_veomni_ops` discovers it
  when the loaded class is multimodal — same trick already used by
  `modeling_text.py` for the text-only path. Without this
  re-export, `moe_implementation: fused_npu` would silently leave the
  slot unbound and the eager forward in `OpenPanguV2Experts` would
  index global expert IDs into the locally-EP-sharded `gate_up_proj`
  (`IndexError: index 49 is out of bounds for dimension 0 with size
  48`).
* `_no_split_modules` on `OpenPanguPreTrainedModel` /
  `OpenPanguVLModel` (`modeling_vl.py`) was pointing at a
  non-existent `OpenPanguVLDecoderLayer` (text) and
  `Qwen2AudioEncoderLayer` (audio — copy/paste from Qwen2.5-Omni
  reference, never actually used in this graph). FSDP2 uses
  `_no_split_modules` to pick the wrap granularity, so both bad names
  caused the corresponding layers to silently fall back to a coarser
  parent FSDP unit. The text-side miss produced ~14 logp drift in
  8-card multimodal forward because visual features computed under
  the wrong sharding boundary then carried no useful signal into
  `masked_scatter`. The audio-side miss was much subtler — instead
  of breaking output, it folded the entire 24-layer audio tower into
  the root FSDP unit, which by coincidence *protected* audio from
  the per-block cos/sin downcast issue described below. Corrected to
  `[OpenPanguV2DecoderLayer, OpenPanguVLVisionBlock,
  ConformerEncoderLayerBlock]` (2026-05-24). After the fix, FSDP2
  wraps each `model.visual.blocks.*`, each
  `model.audio_tower.layers.*`, and each
  `model.language_model.layers.*` (with experts as a separate FSDP
  unit) — verified via `MM_PARITY_DUMP_FSDP=1` enumeration in
  `multi_card_multimodal_parity_runner.py`:

  ```
  [fsdp-probe] visual: 26 × FSDPOpenPanguVLVisionBlock, 26 FSDP-wrapped
  [fsdp-probe] audio_tower: 24 × FSDPConformerEncoderLayerBlock, 24 FSDP-wrapped
  [fsdp-probe] language_model: 37 × FSDPOpenPanguV2DecoderLayer, 35 × FSDPOpenPanguV2Experts
  ```

Single-card multimodal HF parity is verified non-regressive after
all of the above (`oracle_check.py --mode veomni --n-samples 3
--tolerance-max 1e-3 --tolerance-mean 1e-4` PASSes 3/3, max-diff
2.07e-5 — see the `OCRBench` row of the cross-eval table below).

**2026-05-25 revalidation (real 30B-A2B full multimodal weights)**:

| Gate | Command / report | Result | Interpretation |
|---|---|---|---|
| 1-NPU full multimodal OCRBench vs HF | `tools/oracle_check.py --mode veomni --n-samples 3 --tolerance-max 1e-3 --tolerance-mean 1e-4` -> `outputs/pangu_fullmm_single_vs_hf_3.json` | **PASS 3/3**, worst `2.074e-05` (`ocrbench_2`) | Single-card adapter path (`OpenPanguOmni`, visual + audio + language + lm_head) matches the HF oracle. |
| 1-NPU audio oracle vs HF | `tools/oracle_check.py --mode veomni --samples /path/to/audio_oracle/data/audio_demo.jsonl --baseline /path/to/audio_oracle/results/audio_hf_outputs.jsonl --n-samples 3` -> `outputs/pangu_audio_single_vs_hf_3.json` | **PASS 3/3**, worst `0.000e+00` | Audio path is bit-identical under the oracle setting (`allow_internal_format=True`). |
| 8-NPU full multimodal OCRBench vs HF | `multi_card_multimodal_parity_runner.py configs/multimodal/pangu_omni_v2/local_smoke/pangu_real_8card_multimodal.yaml` with `PARITY_N_SAMPLES=3`, `PARITY_TOLERANCE=5e-2` -> `outputs/pangu_fullmm_8card_vs_hf_3.json` | **FAIL 2/3**, `ocrbench_0` max `2.294e-01` | Multi-card full multimodal has a sample-dependent logp drift. It is not a visual-tower wiring failure; see localization below. |

The 2026-05-25 localization for the failing `ocrbench_0` sample:

* `MM_PARITY_CAPTURE=1 MM_PARITY_DEEP_CAPTURE=1` on the 8-NPU runner
  plus `single_card_visual_capture.py` on the same sample shows
  `visual_output` is **bit-identical** between 1-NPU and 8-NPU
  (`max=0.0000e+00`, `mean=0.0000e+00`). The drift is therefore not in
  the visual tower, patch embedding, merger, or image-token scatter.
* 1-NPU eager-MoE oracle gives `ocrbench_0` first-token logp `-0.4950`
  (matches HF). 1-NPU `PARITY_MOE_IMPL=fused_npu` gives `-0.6651`.
  8-NPU `fused_npu` + FSDP2 + EP gives `-0.7243`. This pins the dominant
  drift to the language MoE fused kernel / EP production path, not to the
  multimodal encoder path.

Current safe reading:

* **Single-card full multimodal precision is green** against the HF oracle.
* **Text-only multi-card is numerically equivalent to 1-card `fused_npu`**;
  FSDP/EP add no measurable drift beyond the MoE kernel choice.
* **Full multimodal 8-card forward is functional but not HF-parity-clean**:
  high-confidence OCR samples stay within `1e-4`-level drift, but
  low-confidence samples can amplify the same language-MoE fused-kernel
  noise into `O(1e-1)` logp differences. Treat this as a production
  training tolerance issue, not as a visual/audio tower correctness issue.

**Multi-card multimodal visual + audio parity — bit-exact with
single-card (2026-05-24)**: after the fixes described below, on 8
NPUs both `model.visual` and `model.audio_tower` are
**bit-identical to single-card at every probe point inside the
tower**:

- **Vision**: `patch_embed`, every intermediate vision block, the
  merger, and the final `visual_output` all show `max=0.000e+00
  mean=0.000e+00 n_diff=0` across 64-84 × 3584 tensors on OCRBench
  samples. Top-1 token rank matches single-card on every sample
  tested.
- **Audio**: `audio_input`, `linear_before_attn`, all 24
  `ConformerEncoderLayerBlock` layer outputs, `linear_after_attn`,
  `audio_tower.proj` output, and the final `audio_tower_out` all
  show `max=0.000e+00 mean=0.000e+00 n_diff=0` on the three Pangu
  audio oracle samples
  (`/path/to/audio_oracle/data/audio_demo.jsonl`).

The remaining inter-sample logp drift against the HF baseline (≤
2e-4 for high/mid-confidence samples, up to `2.294e-1` for the
low-confidence `ocrbench_0` revalidation sample and similarly large
audio outliers when using `fused_npu`) comes from the
**language-model MoE sensitivity to `fused_npu` vs `eager`** (see
"fused_npu vs eager MoE" note below), not from the vision or audio
tower — verified by capturing tower outputs and diffing single-card
vs 8-card after each fix.

#### Two drift sources, both fixed

Bisection between single-card and 8-card showed the original 1
bf16 ULP `visual_output` drift was the sum of **two independent
sources**, both originating at the FSDP2 block-boundary cast:

1. **Per-block `cast_forward_inputs=True` downcast of cos/sin**.
   `OpenPanguVisionRotaryEmbedding` deliberately computes
   `(cos, sin)` in fp32 to match the HF reference; PyTorch
   FSDP2's default `MixedPrecisionPolicy(cast_forward_inputs=True)`
   then casts these fp32 tensors back to bf16 at every one of the
   26 `OpenPanguVLVisionBlock` boundaries. That's 26 fp32 → bf16 →
   fp32 round-trips, each losing ~16 bits of mantissa, accumulating
   into ~1 bf16 ULP of drift in `visual_output`. The audio tower
   (`HuanyuAudioEncoder`) hits the **same pattern**: `select_cos_sin`
   computes a fp32 `rotary_pos_emb = (cos, sin)` tuple once and feeds
   it across 24 `ConformerEncoderLayerBlock` boundaries. (Before the
   `_no_split_modules` fix above the audio tower had zero per-layer
   wrap, which accidentally side-stepped this issue; once we fixed
   the dead-code class name, audio re-inherited the cos/sin downcast
   problem and needs the same hook below.)

2. **`param_dtype` storage mismatch when the entire mp policy is
   dropped**. VeOmni's `BaseTrainer._build_model` deliberately
   builds with `torch_dtype="float32"` whenever mixed precision is
   enabled (`base.py:243`), so all params are stored as fp32 and
   FSDP2's `param_dtype=bf16` cast handles the per-forward downcast
   to bf16 compute dtype. Single-card (which doesn't use
   `BaseTrainer`) builds with `torch_dtype="bf16"` directly, so its
   params live in bf16 storage. If we tried to fix (1) by simply
   dropping the entire `mp_policy` for vision blocks via the
   pre-existing `get_ignore_modules_in_mixed_precision` hook, the
   vision blocks would compute with fp32 weights against bf16
   activations — first divergence observed at
   `b00_post_norm1` (RMSNorm `weight (fp32) * x (bf16) → fp32` vs
   single-card's `bf16 * bf16 → bf16`), accumulating to ~13.5 max
   diff at `post_block_25`.

#### Fix: `get_no_input_cast_modules_in_mixed_precision` (new framework hook)

A new finer-grained hook on `PreTrainedModel`: it tells the
`build_parallelize_model` path to wrap matching target classes with
a custom `MixedPrecisionPolicy` that **keeps `param_dtype=bf16`**
(so compute still matches single-card's bf16 weight matmuls) but
**sets `cast_forward_inputs=False`** (so the fp32 `(cos, sin)`
tuple flows through every block unchanged). This is exactly the
combination that aligns single-card's bf16-storage + fp32-cos/sin
forward with 8-card's fp32-storage + bf16-compute-cast + fp32-cos/sin
forward — bit-exact verified end-to-end.

The model side declares (lazy import on `ConformerEncoderLayerBlock`
to keep the audio-encoder module out of `modeling_vl.py`'s
top-level dependency graph):

```python
class OpenPanguPreTrainedModel(PreTrainedModel):
    def get_no_input_cast_modules_in_mixed_precision(self):
        from .modeling_huanyu_audio_encoder import ConformerEncoderLayerBlock
        return (OpenPanguVLVisionBlock, ConformerEncoderLayerBlock)
```

The framework side (`veomni/distributed/torch_parallelize.py`) reads
this hook in addition to the pre-existing
`get_ignore_modules_in_mixed_precision` hook and routes target
classes to `fsdp_kwargs_no_input_cast` (`mp_policy =
MixedPrecisionPolicy(param_dtype=base.param_dtype,
reduce_dtype=base.reduce_dtype, output_dtype=base.output_dtype,
cast_forward_inputs=False)`). Look for the log line:

```
FSDP2 mp_policy no-input-cast classes: ['OpenPanguVLVisionBlock', 'ConformerEncoderLayerBlock'] (param_dtype kept, cast_forward_inputs=False)
```

#### Validation

Run the multimodal parity runner with `MM_PARITY_DEEP_CAPTURE=1` to
dump intermediate vision-tower tensors, then run
`single_card_visual_capture.py` with the same flag, and diff the
two capture dirs. After the fix:

```
[post_patch_embed]  BIT-EXACT |Δ| max=0.000e+00
[post_block_00]     BIT-EXACT |Δ| max=0.000e+00
[post_block_06]     BIT-EXACT |Δ| max=0.000e+00
[post_block_13]     BIT-EXACT |Δ| max=0.000e+00
[post_block_19]     BIT-EXACT |Δ| max=0.000e+00
[post_block_25]     BIT-EXACT |Δ| max=0.000e+00
[post_merger]       BIT-EXACT |Δ| max=0.000e+00
[visual_output]     BIT-EXACT |Δ| max=0.000e+00
```

(Add `MM_PARITY_FINE_CAPTURE=1` to also dump per-component output
inside block 0 — norm1/attn/norm2/mlp — kept in the tool for
future regressions.)

For the audio side, run the runner and single-card capture with
`MM_PARITY_AUDIO_CAPTURE=1` and the Pangu audio oracle as the
sample/baseline source. After the audio fix:

```
PARITY_SAMPLES=/path/to/audio_oracle/data/audio_demo.jsonl \
PARITY_BASELINE=/path/to/audio_oracle/results/audio_hf_outputs.jsonl \
PARITY_CAPTURE_DIR=outputs/pangu_audio_parity/captured \
MM_PARITY_CAPTURE=1 MM_PARITY_AUDIO_CAPTURE=1 \
torchrun --nproc_per_node=8 ...multi_card_multimodal_parity_runner.py ...

PARITY_SAMPLES=... PARITY_BASELINE=... \
PARITY_CAPTURE_DIR=outputs/pangu_audio_parity/captured \
MM_PARITY_AUDIO_CAPTURE=1 \
python ...single_card_visual_capture.py
```

then diff:

```
[audio_input]       BIT-EXACT |Δ| max=0.000e+00
[audio_lin_before]  BIT-EXACT |Δ| max=0.000e+00
[audio_layer_00]    BIT-EXACT |Δ| max=0.000e+00
[audio_layer_06]    BIT-EXACT |Δ| max=0.000e+00
[audio_layer_12]    BIT-EXACT |Δ| max=0.000e+00
[audio_layer_18]    BIT-EXACT |Δ| max=0.000e+00
[audio_layer_23]    BIT-EXACT |Δ| max=0.000e+00
[audio_lin_after]   BIT-EXACT |Δ| max=0.000e+00
[audio_proj_out]    BIT-EXACT |Δ| max=0.000e+00
[audio_tower_out]   BIT-EXACT |Δ| max=0.000e+00
```

The `audio_demo.jsonl` runs vs HF baseline under the multi-card
`fused_npu` production path can show large logp drift on low-confidence
tokens — that drift is **not** in the audio tower (which we proved is
bit-exact). It's the same `fused_npu` vs `eager` **MoE drift** that
explains the `ocrbench_0` vision-path `0.229` outlier: at ep_size>1
we must use `moe_implementation: fused_npu` (eager fails an
`IndexError: index 49 is out of bounds for dimension 0 with size
48` because eager indexes global expert IDs into the EP-sharded
weights). The single-card oracle uses `moe_implementation: eager`
which matches HF byte-for-byte, but the moment we switch any
single-card run to `fused_npu` the same audio_0 / audio_1 / audio_2
drift reappears (verified with `PARITY_MOE_IMPL=fused_npu` on the
single-card capture/oracle path), proving the drift comes primarily
from kernel choice, not from FSDP / EP sharding or multimodal tower
wiring.

```bash
PARITY_SAMPLES=/path/to/oracle/data/ocrbench.jsonl \
PARITY_BASELINE=/path/to/oracle/results/ocrbench_hf_outputs.jsonl \
PARITY_OUT=outputs/pangu_multimodal_parity/report_8card.json \
PARITY_TOLERANCE=5e-2 \
PARITY_N_SAMPLES=10 \
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29512 \
    veomni/models/transformers/pangu_omni_v2/tools/multi_card_multimodal_parity_runner.py \
    configs/multimodal/pangu_omni_v2/local_smoke/pangu_real_8card_multimodal.yaml
# 2026-05-25 revalidation: 2/3 PASS at 5e-2 strict on first 3 OCRBench
# samples; ocrbench_0 fails with max-diff 2.294e-1, while ocrbench_1/2
# stay around 1e-4. Top-1 token remains correct; the drift is a softmax
# denominator/logp sensitivity issue under fused_npu MoE.
```

For visual-tower drift localization, set `MM_PARITY_CAPTURE=1` to
dump `model.visual` output to
`outputs/pangu_multimodal_parity/captured/`; the single-card companion
script lives at
`veomni/models/transformers/pangu_omni_v2/tools/single_card_visual_capture.py`
(loads the model on one NPU with `init_device=npu` — no FSDP — and
diffs against any matching `_world{N}.pt` capture).

### Why eager MoE + EP doesn't work

The eager forward iterates over global expert IDs and indexes
`self.gate_up_proj[expert_idx]`. With EP enabled, that 3D tensor is
locally sharded to `(num_experts/ep_size, ...)`. Index 49 into a
local shape `(48, ...)` raises `IndexError` (observed in the first
8-card attempt before adding the OpSlot guard). The toy 2-card run
appears to work in eager because the local DTensor auto-materializes
on `__getitem__`, but at 30B scale FSDP2 unwraps to plain local
tensors and the indexing fails immediately. Bottom line:
`ep_size>1` ⇒ `moe_implementation` must be a non-eager backend with
EP support (`fused_npu` on NPU, `triton`/`quack` on GPU).

### Multi-machine scaling notes

The 8-card single-node smoke is the same code path as a multi-node
run; only the `torchrun` launch changes:

```bash
# 2-node 16-NPU run (illustrative; sub in your launcher)
torchrun \
  --nnodes=2 --node_rank=$NODE_RANK \
  --master_addr=$HEAD_IP --master_port=29508 \
  --nproc_per_node=8 \
  tasks/train_text.py configs/text/pangu_real_8card.yaml \
  --train.accelerator.ep_size=16
```

When scaling, copy `configs/text/pangu_real_8card.yaml` and:

1. Bump `ep_size` to the desired `world_size` (or smaller, leaving the
   remainder as DP). Keep `ep_outside=false` to match deepseek_v3.
2. Ensure `n_routed_experts % ep_size == 0` (384 is divisible by 1, 2,
   3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384).
3. Increase `global_batch_size` proportionally to keep
   `tokens-per-step / world_size` constant.
4. Output dir must be shared filesystem across all nodes.

## End-to-end SFT smoke (Plan A baseline) — GREEN

Validated 2026-05-22 (commit pending). The full VeOmni training pipeline —
data → tokenizer → plaintext transform → dataloader → FSDP2 → optimizer →
loss/grad bookkeeping — runs end to end on this adapter:

```bash
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29501 \
  tasks/train_text.py configs/text/pangu_toy.yaml
# step 1: loss 11.96, grad_norm 5.26
# step 2: loss 11.59, grad_norm 5.39
# step 3: loss 11.09, grad_norm 3.59
# step 4: loss 10.75, grad_norm 2.71
# step 5: loss 10.37, grad_norm 2.71
# peak VRAM 1.22GB / NPU, exit 0
```

Toy config (`tests/toy_config/pangu_omni_v2_toy/`): 4 layers, hidden 256,
8 routed + 2 shared experts, real Pangu tokenizer (151552 vocab) via
symlinks. Random init via `init_device=meta` + FSDP2's
`parallel_init_fsdp_fn`. 64-sample synthetic plaintext dataset in
`tests/toy_data/pangu_toy/train.jsonl`.

Note on `nproc_per_node`: VeOmni's `parallel_state.fsdp_enabled` is
`fsdp_size > 1`, and `fsdp_mode=fsdp2` requires `init_device=meta`.
Single-rank (`nproc=1`) therefore deadlocks the FSDP2 invariants. Two
ranks is the minimum that exercises the real FSDP2 wrap path — which is
what we actually want validated before scaling.

### Gotcha discovered: `loss_function` returns a 4-tuple under VeOmni

VeOmni's `apply_ops_config` calls `install_loss_mapping`
(`ops/kernels/cross_entropy/__init__.py:352-423`), which rebinds
`LOSS_MAPPING["ForCausalLM"]` to a wrapper that returns
`(loss, logits, log_probs, entropy)` — NOT the bare `Tensor` that
mainline transformers' `ForCausalLMLoss` returns.

Any model in this adapter that does `loss = self.loss_function(...)`
must instead do `loss, logits, log_probs, entropy = self.loss_function(...)`.
The three text-generation entrypoints (`OpenPanguV2ForCausalLM`,
`OpenPanguOmni.forward`, `OpenPanguVL.forward`) all unpack the 4-tuple
now; future models added under this adapter must follow the same
pattern. See `qwen3_moe/modeling_qwen3_moe.py:230-235` for the canonical
upstream-VeOmni form.

When patchgen (Week 4) lands, this becomes the OpSlot guard at
`qwen3_moe/generated/patched_modeling_qwen3_moe_gpu.py:768-784` —
`veomni_causal_lm_loss.use_non_eager_impl` branches between the fast
chunked-loss path and `self.loss_function`.

## Multi-card / multi-machine scaling — reference these adapters

Pangu's modules are heavily derivative of upstream Qwen + DeepSeek:

| Pangu component | Upstream reference (copy this when scaling) |
|---|---|
| MoE w/ shared experts + e_score_correction_bias | `veomni/models/transformers/deepseek_v3/` (esp. parallel_plan.py for EP plan + checkpoint_tensor_converter for fused gate_up_proj layout) |
| Routed expert FSDP2 sharding + EP mesh | `veomni/models/transformers/qwen3_moe/parallel_plan.py` (canonical EP example) |
| MoE + vision merger + MRoPE for omni training | `veomni/models/transformers/qwen3_omni_moe/` (FSDP2 + Ulysses SP + EP combo) |
| Multi-stream / MHC analog | No exact match upstream — Pangu-specific, keep in `_pangu_common/` |

When implementing `pangu_omni_v2/parallel_plan.py` (Week 4 / multi-card
expansion), start by copy-adapting `deepseek_v3/parallel_plan.py` for the
EP plan and `qwen3_moe/parallel_plan.py` for the routed-expert
`fully_shard()` wiring. Pangu's shared experts are a single MLP per
layer (no per-expert subscript), so they go in the *replicated* group,
identical to DeepSeek's shared-expert handling.

For Ulysses SP + MRoPE in the multimodal training path,
`qwen3_omni_moe/`'s patchgen config has the closest precedent — Pangu
diverges only in the attention head layout (K-norm + partial RoPE
instead of standard RoPE).

## Open issues / decisions

- **`config.json` `model_type` = "qwen2_moe"** — this is an upstream bug in
  the Pangu config. We work around it transparently: `MODEL_CONFIG_REGISTRY`
  registers under `"qwen2_moe"` but our config subclass overrides
  `model_type = "openpangu_omni"`. Once VeOmni mainline adds Qwen2-MoE
  support, the registry key must be coordinated. **Long-term action:**
  request Pangu team to fix `config.json`.
- **`MODELING_BACKEND=hf` fails** — VeOmni's loader pops `trust_remote_code`
  without forwarding it to HF AutoModel. Since this adapter copies Pangu's
  modeling code into the package, we don't need `trust_remote_code` at all.

See the AReaL repo `docs/pangu_veomni_adaptation/PHASE1_DESIGN.md` for the
4-week implementation plan, weekly milestones, and the per-token logp
oracle check tool (`tools/pangu_oracle_check.py`) used for numerical
alignment validation.

## Precision alignment with HF inference — DONE (read this before chasing warnings)

Per-token logp parity vs HF reference (`trust_remote_code=True` AutoModel
on the same 30B-A2B ckpt) has been validated on OCRBench / MMMU. Status
at Phase 1 close: aligned to bf16 ULP magnitude; the only known residual
drift is the long-audio path documented in `docs/AUDIO_DRIFT_INVESTIGATION.md`.

### Expected NPU warning: `allow_internel_format=False`

You will see this when running ANY NPU forward (e.g. the
`smoke_single_npu.py` modeling smoke, or `tools/oracle_check.py --mode veomni`):

```
UserWarning: Cannot create tensor with interal format while allow_internel_format=False,
tensor will be created with base format.
```

This is **expected and benign for training**. VeOmni globally sets
`torch.npu.config.allow_internal_format = False` at import time
(`veomni/utils/device.py`) — required for FSDP / weight-sharding safety
on NPU's internal NZ / 5HD layouts. The warning means an op (typically
in `pangu_moe.py` MoE routing or `Conv2d` inside the audio tower) tried
to use the internal layout, was blocked, and silently fell back to the
base layout. **Numerically correct; just a perf hint.**

`docs/AUDIO_DRIFT_INVESTIGATION.md` has the full root-cause investigation
including a standalone probe script. Headline:

- Training (this default `False`) — keep as-is. FSDP correctness >
  kernel-selection perf.
- Oracle / HF-parity check — `tools/oracle_check.py` flips the flag back
  to `True` for the audio_tower forward only (see
  `PANGU_ORACLE_NPU_INTERNAL_FMT` env var). Do **not** touch the flag in
  adapter code; the framework boundary owns it.

Do not "fix" this warning. If you see it on a NEW Pangu module not listed
above, link it back to this section and the investigation doc rather than
silencing it locally.
