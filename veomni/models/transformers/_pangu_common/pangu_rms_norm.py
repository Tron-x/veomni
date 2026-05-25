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

"""RMSNorm for the Pangu model family.

Pangu uses standard T5-style RMSNorm everywhere it has a normalization:

- `input_layernorm` / `pre_mlp_layernorm` on the residual stream (dim = hidden_size)
- `k_layernorm` on the key projection of each attention layer (dim = head_dim,
  applied per-head before RoPE). This is what Pangu's config calls "K-norm"
  via `use_k_norm=True`. It is the same algorithmic primitive as q/k-norm in
  mainline Gemma2 / Qwen3 — Pangu just applies it to K only.
- `groupnorm` (optional) and several small projection norms (q_a / kv_a /
  shared experts) — all the same RMSNorm class with different `hidden_size`.

This module ports `OpenPanguV2RMSNorm` from the Pangu reference
(`modeling_openpangu_v2.py:226-244`) verbatim. No K-norm-specific code lives
here; K-norm is just **where** the attention module decides to instantiate
this class with `hidden_size=head_dim`. See
`_pangu_common/pangu_attention.py` (Week 1 Day 2) for the wiring.

## OpSlot dispatch

VeOmni's `rms_norm_implementation` OpSlot can swap the forward to NPU /
liger / triton implementations. The port preserves the dtype-preserving
contract (input dtype in -> input dtype out, intermediate float32) that
those backends expect, so swap-in is value-equivalent.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OpenPanguV2RMSNorm(nn.Module):
    """RMSNorm with weight, equivalent to T5LayerNorm.

    Verbatim port of `OpenPanguV2RMSNorm` from the Pangu reference
    (`modeling_openpangu_v2.py:226-244`). Class name is preserved
    verbatim so:
    - grep / blame / oracle log comparison with the reference stays trivial
    - `patchgen` rewrite rules can match upstream symbol names directly
    - any future Pangu modeling subclass that imports the upstream class by
      name (e.g. through HF auto-modules) keeps working unchanged

    The `@use_kernel_forward_from_hub` decorator on the reference is
    dropped — VeOmni handles kernel swap via
    `OpsImplementationConfig.rms_norm_implementation` instead.

    A family-neutral alias `PanguRMSNorm` is exported from
    `_pangu_common.__init__` for adapter code that wants to express
    "shared across all Pangu variants" semantics rather than V2-specific.

    Args:
        hidden_size: last-dim size to normalize over (e.g. `hidden_size` for
            residual-stream norms, `head_dim` for per-head K-norm).
        eps: variance epsilon. Pangu config sets `rms_norm_eps` per layer.

    Forward contract:
        - Input dtype is preserved (float32 intermediate, cast back at end).
        - Last-dim normalization: `x / sqrt(mean(x**2) + eps) * weight`.
        - No bias term.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


__all__ = ["OpenPanguV2RMSNorm"]
