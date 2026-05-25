# Week 3.5 Audio Drift Investigation & Closure

**Status**: Closed (2026-05-22)
**Owner**: pangu/veomni adapter (audio path)
**Verdict**: Week 3.5 NPU audio oracle PASS 3/3 (bit-identical, max=0.000e+00)

## TL;DR

Week 3.5 NPU audio acceptance initially showed:

- `audio_0` (4.03s) — `max_diff=0.000e+00` bit-identical PASS
- `audio_1` (2.90s) — `max_diff=0.000e+00` bit-identical PASS
- `audio_2` (5.86s) — `max_diff ≈ 4.6e-2` FAIL (vs `1e-3` tolerance)

Two short audios bit-identical proved the **adapter port itself is byte-exact**.
audio_2 (the longest sample, 197 mel frames vs 134 / 96 for audio_1 / audio_0)
diverged through a chain of small bf16 ULP differences accumulated over 24
Conformer layers + a 27.8B-parameter MoE LLM.

Root cause was traced to a **single NPU configuration flag** flipped at
VeOmni import time:

```python
# veomni/utils/device.py:31-32
if IS_NPU_AVAILABLE:
    torch.npu.config.allow_internal_format = False
```

NPU defaults this flag to `True`. With `True`, `Conv2d` is allowed to select
the NPU-internal optimized layout (NZ / 5HD); with `False`, it must use the
explicit NCHW path. Both kernels are mathematically correct, but they
**accumulate bf16 in different orders** — for the audio tower's very first
VGG `Conv2d` this manifests as ULP-level diffs on 2 / N output elements,
which then amplify through:

```
VGG Conv stack → Conformer × 24 layers → audio_merger.proj
              → MoE LLM (×24) → final logits
```

to ~4.6e-2 final log-probability diff on 5+s audios. Short audios sit below
the ULP boundary and stay bit-identical.

Fix landed in `tools/pangu_oracle_check.py` (oracle restores the HF default
after VeOmni load). Re-run gives **3/3 PASS bit-identical
(`max_diff = 0.000e+00`)** for all three audios.

## Symptoms

`tools/pangu_oracle_check.py --mode veomni --n-samples 3` against the audio
oracle JSONLs produced:

```
[PASS] audio_0   n_toks= 2   max=0.000e+00   mean=0.000e+00   sum_diff=+0.000e+00
[PASS] audio_1   n_toks= 3   max=0.000e+00   mean=0.000e+00   sum_diff=+0.000e+00
[FAIL] audio_2   n_toks=20   max=4.6e-02     mean=2.0e-02     sum_diff=+3.9e-01
Verdict: FAIL (2/3 within 1e-3 tolerance)
```

Audio path was already functionally proven (short audios bit-identical) so
the question was specifically: **why does audio_2 diverge while audio_0 and
audio_1 are exact?**

## Investigation Chain (bisect-style)

### 1. Rule out NPU fused attention kernel boundary

Hypothesis (carried over from Week 3.6 image-path drift): NPU
`npu_fusion_attention` partitions blocks differently based on sequence
length, so longer audios might cross a kernel boundary.

Test: force `attn_implementation="eager"` for **both** HF reference and
VeOmni audio tower via a monkey-patched `NPU_ATTN_INFR=False` flag
(`tools/_audio_debug_force_eager.py`).

Result: divergence **got worse** (4.6e-2 → 7.2e-2). Eager path is *more*
sensitive than fused, not less. **Not** an attention kernel boundary.

### 2. Localize to audio_tower forward (vs LLM backbone)

`tools/pangu_audio_intermediate_dump.py` registered forward hooks on:

- `audio_tower` input (`input_features`, `feature_lens`)
- `audio_tower` output (`last_hidden_state`, `output_lengths`)
- `audio_merger.proj` output (post-projection audio features fed into LLM)
- LLM first decoder layer input (`hidden_states`)

Result for audio_2:

```
audio_tower_input_features        | shape=[80, 591]  | max_abs_diff=0.000e+00 (bit-identical)
audio_tower_feature_lens          | shape=[1]        | bit-identical
audio_tower_output_last_hidden    | shape=[197, 4096]| max_abs_diff=1.56e-2  ← divergence starts here
audio_merger_proj_output          | shape=[197, 4096]| max_abs_diff=2.91e-2
llm_decoder_layer0_input          | shape=[1, N, 4096]| max_abs_diff=3.14e-2
```

Divergence appears *inside* `audio_tower.forward`, despite identical inputs.

### 3. Confirm audio_tower weights are bit-identical

`tools/_audio_weight_compare.py` loaded `HuanyuAudioEncoder.state_dict()`
from both HF and VeOmni models and compared every parameter byte-for-byte.

Result: **all 370M audio_tower params bit-identical**. Loading is not the
issue.

### 4. Layer-by-layer descent inside audio_tower

`tools/pangu_audio_layer_dump.py` hooked every Conv2d / Linear inside the
VGG stack + Conformer encoder, dumped per-layer output for both runs:

```
vgg_block0_layer0_Conv2d    | max_abs_diff=1.95e-3  ← FIRST divergence
vgg_block0_layer0_ReLU      | max_abs_diff=1.95e-3
vgg_block0_layer1_Conv2d    | max_abs_diff=3.91e-3
vgg_block0_layer1_MaxPool2d | max_abs_diff=3.91e-3
...
conformer_layer0_attention  | max_abs_diff=6.25e-3
...
conformer_layer23_output    | max_abs_diff=1.56e-2  → matches step 2
```

The **very first Conv2d** is where divergence starts.

### 5. Tensor format probe — inputs are bit-identical

`tools/pangu_audio_conv_format_probe.py` captured at the first VGG Conv2d:

- `input` tensor: shape `[1, 80, 591, 3]`, dtype `bf16`, NPU format `2 (ND)`,
  strides identical, values bit-identical between HF and VeOmni
- `weight` tensor: shape `[32, 1, 3, 3]`, dtype `bf16`, NPU format `2`, values
  bit-identical
- `bias` tensor: shape `[32]`, dtype `bf16`, values bit-identical

Yet the **output diverges**. So the same kernel, given the same inputs, is
producing different outputs depending on… *something else in the process*.

### 6. Isolate to a process-global side effect

`tools/pangu_audio_conv_format_probe.py` extended:

- Run `torch.nn.functional.conv2d(input, weight, bias, ...)` standalone
  (no VeOmni model) → output matches **HF dump bit-identically**.
- Run same `torch.nn.functional.conv2d` *after* loading the VeOmni model
  (parameters explicitly reassigned from disk to be bit-identical) → output
  matches **VeOmni dump bit-identically**.

So merely *constructing the VeOmni model* (or one of its imports) globally
flips the NPU `Conv2d` kernel selection.

### 7. Bisect VeOmni's import chain

`tmp/probe_conv_global.py` then `tmp/probe_conv_stubbed.py` (which stubs out
`veomni/__init__.py` and imports submodules directly by file path):

```
import nothing               → matches HF  (vs_hf=0, vs_vo=1.95e-3)
import veomni.utils.logging  → matches HF
import veomni.utils.env      → matches HF
import veomni.utils.import_utils → matches HF
import veomni.distributed.parallel_state → FLIPS (matches VeOmni)
import veomni.distributed.sequence_parallel → FLIPS
import veomni.ops.kernels.attention → FLIPS
import veomni.ops.liger      → FLIPS
```

`parallel_state.py` line 28 imports `..utils.device`, which has:

```python
# veomni/utils/device.py:28-32
IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = is_torch_npu_available()

if IS_NPU_AVAILABLE:
    torch.npu.config.allow_internal_format = False
```

### 8. Direct verification: flip the flag, reproduce / undo the divergence

`tools/_probe_allow_internal_format.py` (no VeOmni imports, pure
`torch_npu`):

```
[default (True)]  conv2d output matches HF dump bit-identically
[after False]     conv2d output matches VeOmni dump bit-identically
[back to True]    conv2d output matches HF dump bit-identically
```

The flag flip is **completely reversible** and accounts for **every bit** of
the divergence at the first Conv2d layer. Causation confirmed.

## Root Cause

VeOmni's `utils/device.py` sets `torch.npu.config.allow_internal_format =
False` at import time for safe FSDP / weight-sharding. This is the correct
choice for training (NZ / 5HD format weights have shape gotchas under
sharding). It changes NPU `Conv2d` kernel selection compared to HF's
default, which leaves the flag at `True` and lets the optimized
internal-layout kernel run.

Both kernels are correct. They differ only in bf16 accumulation order. For
audio inputs short enough that the cumulative ULP errors stay below 1e-3,
the difference is invisible. For audio_2's 197-frame mel sequence, 2 ULP
errors at the first conv multiply through 24 Conformer layers + MoE LLM to
4.6e-2 at the final logp.

## Fix

`tools/pangu_oracle_check.py::load_veomni_model()` now restores the flag
after model build (controlled by `PANGU_ORACLE_NPU_INTERNAL_FMT` env var,
default = restore):

```python
if os.environ.get("PANGU_ORACLE_NPU_INTERNAL_FMT", "1") != "0":
    import torch as _torch
    if hasattr(_torch, "npu") and hasattr(_torch.npu, "config"):
        _torch.npu.config.allow_internal_format = True
```

Result on the full audio oracle (3 samples, real 30B-A2B):

```
[PASS] audio_0   n_toks= 2   max=0.000e+00   mean=0.000e+00
[PASS] audio_1   n_toks= 3   max=0.000e+00   mean=0.000e+00
[PASS] audio_2   n_toks=20   max=0.000e+00   mean=0.000e+00
Verdict: PASS (3/3 bit-identical)
```

## RL Training Impact

This fix is **only** for the oracle parity test. Real RL training should
**keep** `allow_internal_format = False` (VeOmni default) because FSDP and
NPU weight sharding rely on it.

Practical implication for production RL on long-audio rollouts:

- VeOmni-trained actor logp will differ from a HF-reference inference logp
  on the same long audio by roughly the bf16 ULP magnitude we measured
  (single-token diff up to ~5e-2 on the worst tokens of multi-second clips,
  empirically; short audios stay bit-identical).
- This is **not a bug**, it's an NPU `Conv2d` kernel-selection difference
  baked into VeOmni's safe-FSDP default.
- RL is stochastic to begin with and uses sampling, not argmax; advantage
  computation already absorbs much larger noise. The kernel-selection diff
  is harmless for training, just noticeable for any oracle-style "VeOmni
  inference == HF inference" comparison.
- If a future use case ever needs HF-identical inference under VeOmni (e.g.
  reproducing a reference logp on long audio for evaluation), wrap the
  audio_tower forward in a context manager that sets
  `allow_internal_format = True` and restores on exit.

## Three Invariants (after Week 3.5 closure)

1. **Audio tower port is byte-exact**: short audios bit-identical without any
   flag fiddling. Long-audio divergence comes entirely from the NPU
   conv-kernel selection difference, not from any computation in
   `_pangu_audio/`.

2. **Oracle parity ≠ training inference**: oracle restores
   `allow_internal_format=True` to match HF reference; training/Forge rollout
   leaves it at `False` for FSDP safety. The two paths are intentionally
   different and both correct.

3. **No NPU-implementation-detail leaks into adapter code**: the flag is
   restored at the framework boundary (oracle script), not in adapter code,
   so `_pangu_audio/` and `pangu_omni_v2/` are environment-agnostic.

## Forge Integration Protocol

When wrapping VeOmni's audio path inside a Forge trainer / rollout actor:

- Default behavior (training) inherits VeOmni's `False` setting — no action
  needed.
- If running a parity / reproducibility check from inside the Forge actor,
  set `allow_internal_format = True` before the audio forward pass and
  restore after. Same env-var convention as oracle:
  `PANGU_ORACLE_NPU_INTERNAL_FMT=1`.
- Document the bf16 kernel-selection difference in actor docstrings so
  downstream RL engineers don't chase phantom divergence when comparing
  VeOmni actor logp to HF reference logp on long audio samples.

## Related Artifacts

- `tools/pangu_oracle_check.py` — oracle entry point with fix applied
  (search `allow_internal_format`)
- `tools/pangu_audio_baseline_gen.py` — HF baseline generator
- `tools/pangu_audio_intermediate_dump.py` — audio_tower / merger / LLM
  layer-input dumper
- `tools/pangu_audio_layer_dump.py` — per-Conv2d / per-Conformer-layer
  dumper inside audio_tower
- `tools/pangu_audio_conv_format_probe.py` — NPU tensor format + standalone
  functional-conv2d isolation probe
- `/mnt/data_3/models/pangu_audio_oracle/data/audio_demo.jsonl` — 3 audio
  samples (Glass / Cough / mr-quilter spoken passage)
- `/mnt/data_3/models/pangu_audio_oracle/results/audio_hf_outputs.jsonl` —
  HF reference baseline (per-token logps)
