# Week 3 Drift Investigation & Multimodal Acceptance Closure

**Status**: Closed (2026-05-21)
**Owner**: pangu/veomni adapter
**Verdict**: Week 3.6 multimodal oracle PASS 3/3 (noise floor)

## TL;DR

Week 3.6 NPU oracle smoke test initially showed `max_logp_diff ≈ 2e-3` on
the production 30B-A2B image+text path — 2/3 samples above our 1e-3
tolerance gate. Investigation traced the drift not to any Week 1–3 port
(text backbone / vision tower / audio encoder / mrope / multimodal merge)
but to a **single configuration mismatch**: the HF reference loaded with
`AutoModelForCausalLM.from_pretrained(trust_remote_code=True)` on NPU
auto-selects `_attn_implementation="sdpa"`, while our oracle script
hard-coded VeOmni to `attn_implementation="eager"`. Different attention
kernels in bf16 give ~2e-3 numerical asymmetry purely from algebraic
re-association — no code bug involved.

After aligning the oracle script to `attn_implementation="sdpa"` (matching
HF default), parity dropped to **noise floor (max=2.07e-5, mean=7.3e-6)**,
3/3 PASS, both text **and image** paths verified end-to-end.

## Symptoms

`tools/pangu_oracle_check.py --mode veomni --n-samples 3` against
`/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model` initially produced:

```
[PASS] ocrbench_0   n_toks=2   max=7.5e-4   mean=4.2e-4
[FAIL] ocrbench_1   n_toks=2   max=2.3e-3   mean=1.6e-3
[FAIL] ocrbench_2   n_toks=2   max=1.5e-3   mean=9.8e-4
Verdict: FAIL (1/3 within 1e-3 tolerance)
```

HF baseline self-reload noise floor sits at ~2e-5. The adapter contributed
~100× more drift than expected. Code review of Week 1–3 ports found no
numerical bugs (all toy parity tests at max_diff=0.0).

## Investigation Tool: `tools/pangu_drift_localizer.py`

Built a side-by-side localizer that:
1. Loads HF reference (`OpenPanGuOmni`) and VeOmni adapter (`OpenPanguVL`)
   on two NPUs in parallel.
2. Installs forward hooks at every stage:
   - Vision tower output (`vision_tower_out`)
   - Vision projection output (`vision_projection_out`)
   - Token embedding output (`inputs_embeds_to_backbone`)
   - **Per-layer text backbone output** (`layer_00_out` ... `layer_36_out`)
   - Final hidden state (`last_hidden_state`)
   - Logits (`logits`)
3. Compares activations stage-by-stage with `max_diff(a-b).abs()`.

### Structural Pitfall

The HF reference (`OpenPanGuOmni` from `modeling_pangu_omni.py`) has a
**flat** structure:
```python
model.visual            # vision tower at top level
model.audio_tower       # audio encoder at top level
model.model             # text backbone (OpenPanguVLTextModel)
model.lm_head
```

Our adapter (`OpenPanguVL` from `modeling_openpangu_vl.py`) has a
**nested** structure:
```python
model.model.visual              # vision tower nested under model
model.model.language_model      # text backbone nested under model
model.lm_head
```

The localizer's `install_hooks` does a runtime dispatch by checking the
module path (`type(model).__module__.startswith("veomni.")`) and resolves
`visual` / `text_backbone` / `embed_tokens` paths correctly for each
flavor.

## Root Cause

With matched attention kernels (`eager × eager`), the localizer reports:

```
vision_tower_out:         max_diff = 0.000e+00
vision_projection_out:    max_diff = 0.000e+00
inputs_embeds_to_backbone:max_diff = 0.000e+00
layer_00_out:             max_diff = 0.000e+00
layer_01_out:             max_diff = 0.000e+00
...
layer_36_out:             max_diff = 0.000e+00
last_hidden_state:        max_diff = 0.000e+00
logits:                   max_diff = 0.000e+00
```

**Bit-for-bit** at every stage of the multimodal forward pass. This
confirmed the ports are correct.

With mismatched kernels (`eager × sdpa` — HF on default sdpa, VeOmni
forced to eager), drift appears at `layer_00_out`:

```
vision_tower_out:         max_diff = 0.000e+00
vision_projection_out:    max_diff = 0.000e+00
inputs_embeds_to_backbone:max_diff = 0.000e+00
layer_00_out:             max_diff ≈ 1.1e-3   ← drift starts here
layer_01_out:             max_diff ≈ 1.4e-3
...
layer_36_out:             max_diff ≈ 2.0e-3
logits:                   max_diff ≈ 2.1e-3
```

Drift originates at the **first** attention computation and amplifies
mildly through subsequent layers — exactly the signature of an attention
kernel mismatch in bf16 (SDPA fuses softmax + matmul with different
floating-point accumulation order than naive eager).

### Verification of HF Default

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -c "
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    '/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model',
    trust_remote_code=True, torch_dtype='auto',
).eval().to('npu:0')
print(m.config.text_config._attn_implementation)
print(m.config.vision_config._attn_implementation)
print(type(m.model.layers[0].self_attn).__name__)
"
# Output:
# sdpa
# sdpa
# OpenPanguV2Attention
```

The production `config.json` does **not** force `_attn_implementation`;
HF's default selector picks `sdpa` on NPU (since `flash_attention_2` is
not available and SDPA is the next-best fused option).

## Fix

`tools/pangu_oracle_check.py:223`:

```python
ops_cfg = OpsImplementationConfig(
    attn_implementation="sdpa",       # was "eager" — caused 2e-3 drift
    moe_implementation="eager",
    cross_entropy_loss_implementation="eager",
    rms_norm_implementation="eager",
    swiglu_mlp_implementation="eager",
    rotary_pos_emb_implementation="eager",
)
```

The other ops stay on `eager` — `eager × eager` for non-attention ops is
already bit-for-bit (no fused alternative on NPU that diverges); only
attention has the SDPA-vs-eager fork that matters.

## Result

`tools/pangu_oracle_check.py --mode veomni --n-samples 3` (full
multimodal image+text path, no `--text-only`):

```
[PASS] ocrbench_0   n_toks=2   max=4.768e-07   mean=2.384e-07
[PASS] ocrbench_1   n_toks=2   max=7.152e-07   mean=3.576e-07
[PASS] ocrbench_2   n_toks=2   max=2.074e-05   mean=1.037e-05
Pass:               3/3
Mean of max-diffs:  7.310e-06
Worst sample:       ocrbench_2 (max-diff 2.074e-05)
Tolerance:          max=0.001, mean=0.0001
Verdict:            PASS
```

Empirical verification that the run is genuinely multimodal (not silently
falling back to text-only):

```
model class:      OpenPanguUltraOmniForConditionalGeneration
model module:     veomni.models.transformers.pangu_omni_v2.modeling_pangu_omni_v2
has visual tower: True
pixel_values:     torch.Size([256, 1176]) bfloat16
image_grid_thw:   [[1, 8, 32]]
inputs keys:      [input_ids, attention_mask, pixel_values, image_grid_thw]
```

## Invariants Established

Three invariants now baked into the oracle protocol:

### Invariant 1: Same-kernel comparison

Any VeOmni-vs-HF NPU parity test **must** load both sides with the same
`attn_implementation`. HF's default on NPU is `sdpa`; VeOmni must follow.
This applies recursively to any future op (FlashAttention-3, Triton MoE,
quantized linear) — if either side is on a fused kernel, the other must
match. The localizer is the truth source: any non-zero diff at a stage
where both kernels are aligned is a real code bug; non-zero diff where
kernels differ is expected (and is not a regression).

### Invariant 2: Vision tower + projection are bit-for-bit

`vision_tower_out` and `vision_projection_out` produce `max_diff=0.0`
on identical inputs. Future regressions in `veomni/models/transformers/pangu_omni_v2/_pangu_vision/`
or the projection layer should be caught immediately by the localizer —
no fused-kernel ambiguity to hide behind in the vision path.

### Invariant 3: Multimodal merge is bit-for-bit

`inputs_embeds_to_backbone` (post `OpenPanguVLModel.forward` image-token
substitution) and the mrope `position_ids` produce `max_diff=0.0` on
identical inputs. This locks down the Week 3.4.c / 3.4.d ports.

## What This Means for `forge` Integration

When `forge` adapts Pangu to its training pipeline, the oracle protocol
required to certify the adapter's numerical correctness is:

1. **Same `attn_implementation` on both sides** — load HF reference and
   forge-adapted model with matching kernels. If forge uses
   FlashAttention-3 on H100, the HF reference must also be loaded with
   FlashAttention-3 (or both forced to eager for a clean baseline; eager
   is the cheapest gold standard).
2. **Stage-by-stage diff via localizer** — `pangu_drift_localizer.py` is
   the reusable tool. Any new model (Pangu Omni next, Omni Plus next next)
   gets a localizer instance with the same hook taxonomy.
3. **Tolerance gates** — `max_diff < 5e-5` for noise-floor parity (HF
   self-reload baseline ~2e-5; adapter adds ≤2× headroom). `max_diff < 1e-3`
   only when kernels intentionally diverge (e.g., production deploys with
   `sdpa` but parity test cannot match because forge uses something else).

## Files Touched

| File | Change |
|------|--------|
| `tools/pangu_oracle_check.py` | `attn_implementation: "eager"` → `"sdpa"` + comment explaining why |
| `tools/pangu_drift_localizer.py` | Added `attn_implementation` arg to `load_hf` (defaults to `"eager"` for gold-standard parity), per-layer hooks for all 37 text backbone layers, dual-shape support for flat HF / nested VeOmni structures |
| `veomni/models/transformers/pangu_omni_v2/modeling_pangu_omni_v2.py` | (Week 3.4.d) `OpenPanguUltraOmniForConditionalGeneration` and `OpenPanguVLForConditionalGeneration` now inherit `_get_open_pangu_vl_class()` (= `OpenPanguVL`), so the production dispatcher routes to the multimodal class |

## Remaining Work

Week 3.5 (audio path wrap): `OpenPanGuOmni(Qwen2_5OmniThinker)` wrap +
audio merger + generation glue. Needed before Pangu Ultra Omni audio
inputs are oracle-checked. The drift-localizer infrastructure built here
directly transfers — audio path just adds two more hook taps
(`audio_encoder_out`, `audio_projection_out`) and one more bit-for-bit
invariant.
