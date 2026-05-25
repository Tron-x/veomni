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

"""MoE building blocks for the Pangu model family.

## Components

Verbatim port of four pieces from the Pangu reference modeling file
(`modeling_openpangu_v2.py`):

1. `OpenPanguV2MLP` (line 314-327) — single SwiGLU MLP. Used for:
   - Dense layers (layer_idx < first_k_dense_replace), with
     `intermediate_size = config.intermediate_size` (6144 in 30B-A2B).
   - Shared experts inside SparseMoeBlock, with
     `intermediate_size = config.moe_intermediate_size * config.n_shared_experts`
     (256 * 2 = 512 in 30B-A2B).
2. `OpenPanguV2Experts` (line 798-834) — routed experts stored as fused
   3D Parameters. **The on-disk ckpt format is per-expert** (one MLP per
   expert), so the adapter package needs a checkpoint converter that
   fuses the per-expert weights into these 3D tensors at load time. See
   `pangu_omni_v2/checkpoint_tensor_converter.py` (Week 2).
3. `OpenPanguV2TopkRouter` (line 837-848) — linear gate over hidden states
   producing per-expert router logits. Forward always promotes to float32
   for numerical stability of the downstream sigmoid/topk.
4. `OpenPanguV2SparseMoeBlock` (line 851-905) — the full MoE forward:
   group-aware top-k routing with sigmoid scores and additive
   `e_score_correction_bias`, optional norm-topk-prob, plus a residual
   shared-experts branch.

## Pangu-specific design notes

- **Fused-3D routed experts** (`gate_up_proj` shape `(E, 2*I, H)`,
  `down_proj` shape `(E, H, I)`). This in-memory layout is what the
  modeling code expects; the on-disk safetensors index has per-expert
  keys (`.experts.{0..E-1}.{gate,up,down}_proj.weight`). Loading
  requires fusion at load time.
- **Bias correction** (`e_score_correction_bias`) is a `register_buffer`,
  not a `Parameter` — it is not trained, but is in `state_dict`. The
  upstream initializes it to zeros and updates it through a side-channel
  mechanism (typically expert-load balancing); this port preserves the
  zero init and the buffer semantics. The ckpt provides a per-layer
  value under key `model.layers.X.mlp.e_score_correction_bias`.
- **Group routing** (`n_group=1`, `topk_group=1`) in 30B-A2B effectively
  degenerates to "topk over all routed experts", but the group-aware
  code path is preserved verbatim for any future variant that uses
  non-trivial groupings (e.g. DeepSeek-V3-style 8 groups).
- **Shared experts** are stored as a **single** MLP with
  `intermediate_size = moe_intermediate_size * n_shared_experts` — not
  as `n_shared_experts` separate MLP instances. This is mathematically
  equivalent (down_proj is linear) and is the layout in the upstream
  state_dict (one set of `shared_experts.{gate,up,down}_proj.weight`
  per MoE layer).
- **Residual add of shared experts**: the forward signature is
  `output = routed(x) + shared(residual)` (not `routed(residual) + shared(x)`).
  `residual` here is the input pre-view; both branches see the same
  (B, S, H) tensor.

## Naming policy

Class names are kept verbatim (`OpenPanguV2MLP`, `OpenPanguV2Experts`,
`OpenPanguV2TopkRouter`, `OpenPanguV2SparseMoeBlock`). Family-neutral
aliases (`PanguMLP`, `PanguExperts`, `PanguTopkRouter`, `PanguSparseMoeBlock`)
are exported from `_pangu_common.__init__`.

## OpSlot dispatch

The expert dispatch / fused-MoE optimization happens through VeOmni's
`moe_implementation` OpSlot. The verbatim implementation here is the
"eager" fallback (loop over hit experts), which matches the reference
bit-for-bit and is the baseline for OpSlot kernel swap-in (fused_triton /
fused_npu / grouped_gemm / etc.).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ....ops.dispatch import OpSlot


# Module-level OpSlot for the routed-expert forward. Bound by
# ``apply_ops_config`` (in ``veomni/ops/__init__.py``) based on the
# yaml's ``model.ops_implementation.moe_implementation`` field. When a
# non-eager kernel is bound (e.g. ``"npu"`` for the NPU group-gemm with
# auto-EP-dispatch), ``OpenPanguV2Experts.forward`` short-circuits to
# the kernel — see the OpSlot guard in ``OpenPanguV2Experts.forward``.
#
# Naming convention matches qwen3_5_moe / deepseek_v3 — same
# ``("moe_experts", "standard")`` slot, same adapter signature (pulls
# ``num_experts``, ``gate_up_proj``, ``down_proj`` off ``self``). The
# adapter is registered at ``veomni/ops/kernels/moe/__init__.py``.
#
# ``OpSlot`` is a module global, so all four Pangu MoE module instances
# (layers 2..N, each with its own ``OpenPanguV2Experts``) share the same
# slot. Rebinding the slot would warn (intentional).
veomni_moe_experts_forward = OpSlot("moe_experts", "standard")


# ---------------------------------------------------------------------------
# Dense / shared-expert MLP
# ---------------------------------------------------------------------------


class OpenPanguV2MLP(nn.Module):
    """SwiGLU MLP. Used for dense layers and shared experts.

    Verbatim port of `OpenPanguV2MLP` from the Pangu reference
    (`modeling_openpangu_v2.py:314-327`).

    Args:
        config: needs `hidden_size`, `intermediate_size`, `hidden_act`.
        intermediate_size: override for `config.intermediate_size`. Set by
            `OpenPanguV2SparseMoeBlock` to
            `config.moe_intermediate_size * config.n_shared_experts` to build
            the shared-experts branch as a single MLP with combined width.
    """

    def __init__(self, config, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Routed experts (fused 3D Parameter layout)
# ---------------------------------------------------------------------------


class OpenPanguV2Experts(nn.Module):
    """Collection of expert weights stored as fused 3D Parameters.

    Verbatim port of `OpenPanguV2Experts` from the Pangu reference
    (`modeling_openpangu_v2.py:798-834`).

    Layout:
        gate_up_proj: (num_experts, 2 * intermediate_dim, hidden_dim)
        down_proj:    (num_experts, hidden_dim, intermediate_dim)

    On-disk ckpt has per-expert weights — a checkpoint converter fuses
    them into these 3D Parameters at load time.

    Forward implementation:
        - Compute one-hot expert mask, find hit experts (those activated
          by at least one token).
        - For each hit expert, gather the tokens routed to it, run the
          SwiGLU MLP, scale by `top_k_weights`, scatter-add into output.

    This is the "eager" reference implementation. Fused / grouped-gemm
    kernels (triton / NPU) hook in here via VeOmni's `moe_implementation`
    OpSlot.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # OpSlot guard: dispatch to the fused MoE kernel when bound (e.g.
        # ``moe_implementation: "npu"`` -> NPU group-gemm with built-in
        # EP all-to-all dispatch via ``npu_fused_moe_forward``; on GPU,
        # ``"triton"`` / ``"quack"`` register here too).
        #
        # This guard is REQUIRED for any ``ep_size > 1`` run: the eager
        # loop below indexes global expert indices into ``self.gate_up_proj``,
        # which assumes the full (E, 2I, H) shape. With EP enabled, the
        # ParallelPlan replaces those tensors with DTensors whose local
        # shape is (E/ep_size, 2I, H), and indexing a global ID outside
        # the local shard raises ``IndexError`` (observed 2026-05-22 on
        # real 30B-A2B + 8-card EP). The fused kernels, by contrast,
        # perform the all-to-all in ``preprocess`` / ``alltoall_dispatch``
        # and only group-gemm over local experts — see
        # ``veomni/ops/kernels/moe/npu_group_gemm.py::npu_ep_fused_moe_forward``.
        if veomni_moe_experts_forward.use_non_eager_impl:
            return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)

        # hidden_states: [tokens, hidden_dim]
        # top_k_index, top_k_weights: [tokens, top_k]
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            )  # [tokens, top_k, num_experts]
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, tokens]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


# ---------------------------------------------------------------------------
# Top-k router
# ---------------------------------------------------------------------------


class OpenPanguV2TopkRouter(nn.Module):
    """Linear router producing per-expert logits (in float32).

    Verbatim port of `OpenPanguV2TopkRouter` from the Pangu reference
    (`modeling_openpangu_v2.py:837-848`).

    Forward always promotes to float32 for the linear op — this is the
    reference behavior and is preserved bit-for-bit. The subsequent
    sigmoid / topk in `OpenPanguV2SparseMoeBlock.route_tokens_to_experts`
    consumes these float32 logits directly.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


# ---------------------------------------------------------------------------
# Sparse MoE block (routed + shared experts)
# ---------------------------------------------------------------------------


class OpenPanguV2SparseMoeBlock(nn.Module):
    """MoE block with shared experts and group-aware top-k routing.

    Verbatim port of `OpenPanguV2SparseMoeBlock` from the Pangu reference
    (`modeling_openpangu_v2.py:851-905`).

    Forward signature:
        output = experts(view(hidden)) + shared_experts(residual)

    where `residual = hidden_states` (the input). Both branches see the
    same `[B, S, H]` tensor; the routed branch flattens-then-restores via
    `.view(-1, H)` + `.view(*orig_shape)`.

    Routing details (`route_tokens_to_experts`):
        - sigmoid the router logits to per-expert scores in (0, 1).
        - Add `e_score_correction_bias` for choice (load-balancing bias).
        - Group routing: split scores into `n_group` groups, score each
          group by sum-of-top-2, pick `topk_group` groups, mask out the rest.
        - Top-k over the masked scores (per-token, `top_k = num_experts_per_tok`).
        - Gather the sigmoid scores (not the biased ones) as weights.
        - Optionally normalize weights to sum to 1 (`norm_topk_prob`).
        - Scale by `routed_scaling_factor` (e.g. 2.5 in 30B-A2B).

    The `e_score_correction_bias` buffer is initialized to zeros and is
    loaded from the ckpt key `model.layers.X.mlp.e_score_correction_bias`.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.experts = OpenPanguV2Experts(config)
        self.gate = OpenPanguV2TopkRouter(config)
        # Shared experts: single MLP with intermediate width
        # = moe_intermediate_size * n_shared_experts. Mathematically equivalent
        # to n_shared_experts separate MLPs (down_proj is linear), and matches
        # the on-disk state_dict layout (one set of {gate,up,down}_proj.weight).
        self.shared_experts = OpenPanguV2MLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = 1  # upstream hardcodes this to 1; group_size = n_routed_experts.
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        # Buffer (not Parameter): not trained, loaded from ckpt under
        # `model.layers.X.mlp.e_score_correction_bias`.
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        # NPU emits "Cannot create tensor with interal format while
        # allow_internel_format=False" here on the first call — expected
        # and benign. VeOmni globally disables NPU internal formats for
        # FSDP weight-sharding correctness; this op falls back to base
        # layout (numerically identical, slight perf hint). See
        # ``pangu_omni_v2/README.md#expected-npu-warning-allow_internel_formatfalse``
        # and ``docs/AUDIO_DRIFT_INVESTIGATION.md`` for the full story.
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        # Residual through shared experts (Pangu-specific).
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


__all__ = [
    "OpenPanguV2Experts",
    "OpenPanguV2MLP",
    "OpenPanguV2SparseMoeBlock",
    "OpenPanguV2TopkRouter",
]
