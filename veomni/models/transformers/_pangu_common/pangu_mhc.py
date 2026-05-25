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

"""Multi-Head Computation (MHC) for the Pangu model family.

## What is MHC

MHC is a Pangu-specific mechanism that maintains the residual stream in a
**multi-stream** representation `(B, S, n*H)` (with `n = num_stream = 4`
and `H = hidden_size = 2560` for the 30B-A2B model — so the true residual
stream width is `4 * 2560 = 10240`). Each decoder layer wraps its
sub-block (attention or MLP) with a pair of MHC operations:

- `hc_pre`: project `(B, S, n*H)` -> `(B, S, H)` so attention/MLP can
  process at the standard width. Computes `h_post` and `h_res` side
  outputs needed for the matching `hc_post`.
- `hc_post`: combine the sub-block output `(B, S, H)` with the residual
  `(B, S, n*H)` and the cached side outputs to produce the new
  `(B, S, n*H)` residual.

Each decoder layer instantiates **two** `mHCModule` instances (one for
the attention sub-block, one for the MLP sub-block) — exactly what
the Pangu 30B-A2B state_dict shows: 37 layers × 2 modules × 8 weights
= 592 MHC weight tensors.

## Algorithm details

`hc_pre` (verbatim port of `mHCModule.hc_pre`, ref line 98-115):

1. RMSNorm-like normalization: `rsqrt = 1/sqrt(mean(x^2) + norm_eps)`.
2. (If `mhc_use_gamma`) multiply by `norm_gamma` then project through
   `phi`: `weight = phi(x * rsqrt * norm_gamma.unsqueeze(0))`.
   Else: `weight = phi(x) * rsqrt`.
3. Split `weight` via `hc_split_sinkhorn_torch` into:
   - `h_pre`  shape `(B, S, n)` — sigmoid-gated stream coefficients
   - `h_post` shape `(B, S, n)` — sigmoid-gated, for hc_post combination
   - `h_res`  shape `(B, S, n, n)` — Sinkhorn-Knopps doubly-stochastic
     mixing matrix, for hc_post residual combination
4. Reduce x along the stream axis: `y = sum(h_pre.unsqueeze(-1) *
   x.unflatten(-1, (n, H)), dim=-2)` -> `(B, S, H)`.

`hc_post` (verbatim port, ref line 117-136):

1. `y = h_post.unsqueeze(-1) * x.unsqueeze(-2)` (broadcast: `(B, S, n, H)`)
2. `+ sum(h_res.unsqueeze(-1) * residual.unflatten(-1, (n, H)).unsqueeze(-2),
   dim=-3)` (broadcast: `(B, S, n, H)`)
3. Reshape back to `(B, S, n*H)`.

`sinkhorn_knopps` (ref line 154-163):

Doubly-stochastic normalization: softmax along last dim, then
iteratively normalize columns then rows for `sinkhorn_iters - 1` times.
Used to ensure the residual mixing matrix `h_res` has consistent
row/column sums.

## Config fields consumed

- `mhc_num_stream` (4 in 30B-A2B): stream count `n`
- `hidden_size` (2560): per-stream hidden width `H`
- `mhc_use_gamma` (True): toggle `norm_gamma` projection prefix
- `mhc_recur_norm` (20): Sinkhorn-Knopps iteration count
- `rms_norm_eps` (1e-6): epsilon for the input RMSNorm

## Parameter dtype

All `mHCModule` parameters (branch_alpha_*, branch_beta_*, norm_gamma,
phi.weight) are **bfloat16** by upstream design — this is hardcoded in
the constructor, not driven by `config.torch_dtype`. The verbatim port
preserves this.

## merge_layer_only_pre option

Pangu's `mHCModule.__init__` accepts a `merge_layer_only_pre=False`
flag. When True, the module operates as a one-way reducer (only `hc_pre`
is meaningful; `hc_post` is identity). Off for all decoder layers in
30B-A2B; preserved for future variants.

## Naming policy

Class name is kept verbatim `mHCModule` (note lowercase `m`) so the
upstream state_dict key `model.layers.X.{attn,mlp}_mhc_module.*` works
1:1. A family-neutral alias `PanguMHCModule = mHCModule` is exported
from `_pangu_common.__init__` (camel-case for readability at adapter
call sites).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class mHCModule(nn.Module):
    """Multi-Head Computation module (Pangu-specific).

    Verbatim port of `mHCModule` from the Pangu reference
    (`modeling_openpangu_v2.py:64-163`). See module docstring for the
    algorithm overview and config consumption.

    A family-neutral alias `PanguMHCModule = mHCModule` is exported
    from `_pangu_common.__init__`.
    """

    def __init__(self, config, merge_layer_only_pre: bool = False) -> None:
        super().__init__()
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.merge_layer_only_pre = merge_layer_only_pre

        if not self.merge_layer_only_pre:
            phi_output_hidden_size = (self.num_stream + 2) * self.num_stream
            self.branch_alpha_post = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
            self.branch_alpha_res = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
            self.branch_beta_post = nn.Parameter(torch.empty(self.num_stream, dtype=torch.bfloat16))
            self.branch_beta_res = nn.Parameter(torch.empty(self.num_stream * self.num_stream, dtype=torch.bfloat16))
        else:
            phi_output_hidden_size = self.num_stream

        self.branch_alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
        self.branch_beta_pre = nn.Parameter(torch.empty(self.num_stream, dtype=torch.bfloat16))
        self.phi = nn.Linear(
            self.hidden_size * self.num_stream,
            phi_output_hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.mhc_use_gamma = config.mhc_use_gamma
        self.hc_eps = 1e-6
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        if self.mhc_use_gamma:
            self.norm_gamma = nn.Parameter(torch.empty(self.hidden_size * self.num_stream, dtype=torch.bfloat16))

    def hc_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Project (B, S, n*H) -> (B, S, H) and compute side outputs.

        Returns:
            (y, h_post, h_res):
            - y: (B, S, H) the compressed stream
            - h_post: (B, S, n) or None (None if merge_layer_only_pre)
            - h_res: (B, S, n, n) or None (None if merge_layer_only_pre)
        """
        dtype = x.dtype
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        if self.mhc_use_gamma:
            weight = self.phi(x * rsqrt * self.norm_gamma.unsqueeze(0))
        else:
            weight = self.phi(x) * rsqrt

        # (B,S,n), (B,S,n), (B,S,n,n)
        h_pre, h_post, h_res = self.hc_split_sinkhorn_torch(weight)

        # (B, S, H) — reduce x across the stream axis
        y = torch.sum(
            h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(self.num_stream, -1)),
            dim=-2,
        )
        return y.to(dtype), h_post, h_res

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor | None,
        h_res: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine sub-block output with residual to produce new (B, S, n*H).

        Args:
            x: (B, S, H) — sub-block output (attention/MLP)
            residual: (B, S, n*H) — original input before hc_pre
            h_post: (B, S, n) — from hc_pre
            h_res: (B, S, n, n) — from hc_pre

        Returns:
            (B, S, n*H) — new residual stream
        """
        if self.merge_layer_only_pre:
            return x

        y = h_post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            h_res.unsqueeze(-1) * residual.unflatten(dim=-1, sizes=(self.num_stream, -1)).unsqueeze(-2),
            dim=-3,
        )
        return y.view(residual.shape).type_as(x)

    def hc_split_sinkhorn_torch(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.merge_layer_only_pre:
            h_pre, h_post, h_res = weight.split(
                [self.num_stream, self.num_stream, self.num_stream * self.num_stream],
                dim=-1,
            )
            h_post = 2 * torch.sigmoid(h_post * self.branch_alpha_post + self.branch_beta_post)
            h_res = h_res.unflatten(-1, (self.num_stream, self.num_stream))
            h_res = h_res * self.branch_alpha_res + self.branch_beta_res.view(self.num_stream, self.num_stream)
            h_res = self.sinkhorn_knopps(h_res, self.mhc_recur_norm, self.hc_eps)
        else:
            h_pre = weight
            h_post = None
            h_res = None
        h_pre = torch.sigmoid(h_pre * self.branch_alpha_pre + self.branch_beta_pre)
        return h_pre, h_post, h_res

    def sinkhorn_knopps(self, h_res: torch.Tensor, sinkhorn_iters: int, eps: float) -> torch.Tensor:
        """Doubly-stochastic normalization via Sinkhorn-Knopps iteration.

        Verbatim port of `sinkhorn_knopps` (ref line 154-163).

        Note the asymmetric iteration count: the first iteration is half
        (softmax + col-norm only); the remaining `sinkhorn_iters - 1`
        iterations do row-norm then col-norm. Total normalization ops =
        `1 + 2 * (sinkhorn_iters - 1)`.
        """
        h_res = h_res.softmax(-1) + eps
        col_sum = h_res.sum(-2, keepdim=True)
        h_res = h_res / (col_sum + eps)
        for _ in range(sinkhorn_iters - 1):
            row_sum = h_res.sum(-1, keepdim=True)
            h_res = h_res / (row_sum + eps)
            col_sum = h_res.sum(-2, keepdim=True)
            h_res = h_res / (col_sum + eps)
        return h_res


__all__ = ["mHCModule"]
