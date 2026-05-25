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

"""Pangu-family shared building blocks.

This is **not** a `model_type` package — it is a library of reusable layer
implementations shared by all current and future Pangu adapter packages
(`pangu_omni_v2/`, future `pangu_omni_v3/`, `pangu_*/`, ...).

## Naming policy: upstream-verbatim + family alias

Each symbol is exported under **two names**:

1. The **upstream-verbatim name** (e.g. `OpenPanguV2RMSNorm`) — matches the
   Pangu reference modeling files (`/mnt/data_3/models/pangu/configs/`)
   1:1. Use this for: patchgen rewrite rules, oracle log grep, diffing
   against the reference implementation, any code that mirrors the
   upstream class hierarchy.

2. The **family-neutral alias** (e.g. `PanguRMSNorm = OpenPanguV2RMSNorm`)
   — expresses the "shared across all Pangu variants" semantics.
   Use this in adapter code that wants to be agnostic about which Pangu
   generation it targets (e.g. future v3 / Embedded variants reusing the
   same primitive).

The two names are the same Python object — `is` check is True, no runtime
cost. Pick whichever conveys intent better at the call site.

## Module organization

Each module is a verbatim port of the corresponding piece in the upstream
Pangu reference modeling files, with vendor-only decorators stripped and
OpSlot dispatch points wired so VeOmni's NPU/GPU kernel swap mechanism
stays in control.

Parity tests under `tests/test_pangu_*_parity.py` enforce bit-for-bit
numerical equivalence vs the reference on the CPU.
"""

from .pangu_attention import (
    OpenPanguV2Attention,
    eager_attention_forward,
    repeat_kv,
)
from .pangu_decoder_layer import OpenPanguV2DecoderLayer
from .pangu_mhc import mHCModule
from .pangu_moe import (
    OpenPanguV2Experts,
    OpenPanguV2MLP,
    OpenPanguV2SparseMoeBlock,
    OpenPanguV2TopkRouter,
)
from .pangu_partial_rope import (
    OpenPanguV2RotaryEmbedding,
    apply_partial_rotary_pos_emb,
    apply_rotary_pos_emb,
    compute_default_rope_parameters,
    rotate_half,
)
from .pangu_rms_norm import OpenPanguV2RMSNorm


# Family-neutral aliases ---------------------------------------------------
# RMSNorm — used for input_layernorm / pre_mlp_layernorm AND for K-norm
# (per-head RMSNorm on key projection before RoPE). K-norm has no separate
# algorithmic primitive; it is this module instantiated with
# hidden_size=head_dim and wired into attention.forward.
PanguRMSNorm = OpenPanguV2RMSNorm

# Partial RoPE
PanguRotaryEmbedding = OpenPanguV2RotaryEmbedding
apply_pangu_partial_rope = apply_partial_rotary_pos_emb

# Attention
PanguAttention = OpenPanguV2Attention

# MoE
PanguMLP = OpenPanguV2MLP
PanguExperts = OpenPanguV2Experts
PanguTopkRouter = OpenPanguV2TopkRouter
PanguSparseMoeBlock = OpenPanguV2SparseMoeBlock

# Multi-Head Computation. Note: upstream uses lowercase-m `mHCModule` so
# the state_dict key `model.layers.X.{attn,mlp}_mhc_module.*` matches —
# the verbatim name is intentionally not PEP-8.
PanguMHCModule = mHCModule

# Decoder layer
PanguDecoderLayer = OpenPanguV2DecoderLayer

__all__ = [
    # Upstream-verbatim names (canonical)
    "OpenPanguV2Attention",
    "OpenPanguV2DecoderLayer",
    "OpenPanguV2Experts",
    "OpenPanguV2MLP",
    "OpenPanguV2RMSNorm",
    "OpenPanguV2RotaryEmbedding",
    "OpenPanguV2SparseMoeBlock",
    "OpenPanguV2TopkRouter",
    "apply_partial_rotary_pos_emb",
    "apply_rotary_pos_emb",
    "compute_default_rope_parameters",
    "eager_attention_forward",
    "mHCModule",
    "repeat_kv",
    "rotate_half",
    # Family-neutral aliases
    "PanguAttention",
    "PanguDecoderLayer",
    "PanguExperts",
    "PanguMHCModule",
    "PanguMLP",
    "PanguRMSNorm",
    "PanguRotaryEmbedding",
    "PanguSparseMoeBlock",
    "PanguTopkRouter",
    "apply_pangu_partial_rope",
]
