# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""v5 patchgen config for Pangu Omni v2 — PHASE 1 PLACEHOLDER.

This file declares the patch intent that VeOmni's `patchgen` codegen
reads to produce `generated/patched_modeling_pangu_omni_v2_gpu.py`.

Run:
    python -m veomni.patchgen.run_codegen \\
        veomni.models.transformers.pangu_omni_v2.pangu_omni_v2_gpu_patch_gen_config \\
        -o veomni/models/transformers/pangu_omni_v2/generated --diff

PHASE 1 STATUS: empty spec — fully filled in across Week 1-4. See
`docs/pangu_veomni_adaptation/PHASE1_DESIGN.md` (in the AReaL repo) for
the detailed implementation order.
"""

from __future__ import annotations

# Placeholder: the real patchgen config will define:
# - PatchConfig with source_module pointing to the in-package modeling
#   (since Pangu is not in transformers mainline, source_module is local)
# - @config.override_method / @config.replace_class decorators for:
#   * RMSNorm -> K-norm
#   * RotaryEmbedding -> partial RoPE
#   * Standard MoE block -> MoE + shared experts + MHC
#   * Standard attention -> partial-RoPE attention
#   * Vision merger -> GatedMerger
#   * Audio encoder -> HuanyuAudioEncoder
#
# For now we intentionally do not call patchgen — generated/ stays empty
# until Week 1 implementation lands.
