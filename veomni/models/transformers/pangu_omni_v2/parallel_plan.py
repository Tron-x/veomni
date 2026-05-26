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

"""Expert-parallel plan for Pangu Omni v2.

## What this declares

A single dict mapping FQN-glob patterns to ``Shard(0)`` over the
expert axis. VeOmni's ``ParallelPlan`` (see
``veomni/distributed/parallel_plan.py``) consumes the dict to:

1. Replace matching ``nn.Parameter`` tensors with ``DTensor`` sharded
   along the EP device mesh.
2. Auto-exempt the parent module (e.g. ``model.layers.*.mlp.experts``)
   from the FSDP2 ``fully_shard`` wrap, so its tensors are sharded by
   EP only and not double-sharded by FSDP.

Routed-expert weights ``experts.gate_up_proj`` ``(E, 2I, H)`` and
``experts.down_proj`` ``(E, H, I)`` are sharded along the expert axis.
``shared_experts`` is a single MLP (Pangu rolls ``n_shared_experts``
into a wider single MLP — see ``_pangu_common/pangu_moe.py:281-287``)
and stays replicated across EP ranks under default FSDP2 sharding.

## FQN Verification

Built ``OpenPanguV2ForCausalLM`` from
``tests/toy_config/pangu_omni_v2_toy`` and walked
``named_parameters()`` — only two FQN patterns carry expert-axis
parameters:

    model.layers.{i}.mlp.experts.gate_up_proj   (E, 2*I, H)
    model.layers.{i}.mlp.experts.down_proj      (E, H, I)

Dense layers (i < first_k_dense_replace) use ``mlp.{gate,up,down}_proj``
without the ``.experts.`` prefix and are correctly NOT matched by the
wildcard ``model.layers.*.mlp.experts.*``. ``mlp.gate.weight``
(router) is also outside the pattern.

## Why this matches qwen3_moe verbatim

Pangu's ``OpenPanguV2Experts`` is a structural twin of Qwen3-MoE's
``Qwen3MoeExperts`` post v5-fusion: both store routed weights as fused
3D ``Parameter`` tensors ``(E, ..., ...)`` and forward via per-expert
slicing. The EP shard map therefore copies
``qwen3_moe/parallel_plan.py:get_parallel_plan(use_gate_up_proj=True)``
1:1. (qwen3_moe's ``use_gate_up_proj=False`` branch is the pre-v5
non-fused path, which Pangu does not need — our converter always
produces the fused 3D layout.)

## DeepSeek-V3 has separate ``gate_proj`` and ``up_proj``

By contrast, ``deepseek_v3/parallel_plan.py`` shards three FQN patterns
(``gate_proj``, ``up_proj``, ``down_proj``) because its
``PatchDeepseekV3NaiveMoe`` keeps the gate/up matrices separate. Pangu
fuses them at load time via
``checkpoint_tensor_converter.py``, so we land on the qwen3_moe
2-pattern shape, not the deepseek 3-pattern.

## Multimodal Classes

``OpenPanguOmni`` (audio+vision+text top-level) and
``OpenPanguVL`` wrap the V2 text backbone under
``model.language_model.*`` (verified via ``named_parameters()``;
expert FQNs land at ``model.language_model.layers.{i}.mlp.experts.*``,
shape ``(384, 512, 2560)`` for ``gate_up_proj`` on the real 30B-A2B
checkpoint). They share the same EP layout as the text-only path —
just under a different FQN prefix — so we expose ``prefix`` as a
parameter rather than duplicating ``get_parallel_plan`` per class.
The respective ``OpenPanguOmni.get_parallel_plan`` /
``OpenPanguVL.get_parallel_plan`` methods (defined alongside their
class) call ``get_parallel_plan(prefix="model.language_model")`` to
land on the right keys.

The vision tower (``model.visual``) and audio tower
(``model.audio_tower``) are dense and stay under default FSDP2
sharding — no extra EP planning is needed for them.
"""

from __future__ import annotations

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan(
    use_gate_up_proj: bool = True,
    prefix: str = "model",
) -> ParallelPlan:
    """Return the EP plan for Pangu Omni v2.

    Args:
        use_gate_up_proj: Keep ``True`` for any production Pangu config
            — the on-disk per-expert ckpt is always fused into 3D
            ``gate_up_proj`` by
            ``checkpoint_tensor_converter.create_pangu_omni_v2_checkpoint_tensor_converter``.
            The ``False`` branch is kept for API parity with qwen3_moe
            in case a future Pangu variant ships a non-fused checkpoint.
        prefix: FQN prefix in front of ``layers.*.mlp.experts.*``.
            Use ``"model"`` (default) for ``OpenPanguV2ForCausalLM``
            (text-only), and ``"model.language_model"`` for
            ``OpenPanguVL`` / ``OpenPanguOmni`` (multimodal wrappers
            where the text backbone is one sub-module among
            ``visual`` / ``audio_tower`` / ``language_model``).

    Returns:
        ``ParallelPlan`` whose ``extra_parallel_plan["ep"]`` maps
        ``experts.gate_up_proj`` / ``experts.down_proj`` (or the
        un-fused ``gate_proj`` / ``up_proj`` / ``down_proj``) to
        ``Shard(0)``.
    """
    if use_gate_up_proj:
        ep_plan = {
            f"{prefix}.layers.*.mlp.experts.gate_up_proj": Shard(0),
            f"{prefix}.layers.*.mlp.experts.down_proj": Shard(0),
        }
    else:
        ep_plan = {
            f"{prefix}.layers.*.mlp.experts.gate_proj": Shard(0),
            f"{prefix}.layers.*.mlp.experts.up_proj": Shard(0),
            f"{prefix}.layers.*.mlp.experts.down_proj": Shard(0),
        }
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


__all__ = ["get_parallel_plan"]
