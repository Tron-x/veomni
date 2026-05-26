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

"""Runtime checkpoint tensor converter for Pangu Omni v2.

Converts the Pangu HF safetensors per-expert layout into the in-memory
fused 3D `OpenPanguV2Experts` layout at load time, mirroring the
`Qwen3MoeCheckpointTensorConverter` pattern.

## On-disk format (Pangu HF safetensors)

    model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]   (j in [0..E-1])
    model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
    model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]
    model.layers.{i}.mlp.gate.weight                    [E, H]   (router weights)
    model.layers.{i}.mlp.e_score_correction_bias        [E]      (router bias buffer)
    model.layers.{i}.mlp.shared_experts.gate_proj.weight  [I_s, H]
    model.layers.{i}.mlp.shared_experts.up_proj.weight    [I_s, H]
    model.layers.{i}.mlp.shared_experts.down_proj.weight  [H, I_s]

    (For dense layers — `layer_idx < first_k_dense_replace = 2`:)
    model.layers.{i}.mlp.gate_proj.weight  [I, H]
    model.layers.{i}.mlp.up_proj.weight    [I, H]
    model.layers.{i}.mlp.down_proj.weight  [H, I]

## In-memory format (our `OpenPanguV2Experts`)

    model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]   (gate || up concatenated)
    model.layers.{i}.mlp.experts.down_proj     [E, H, I]
    (everything else: passthrough — names match disk format)

## Conversion semantics

This converter fires ONLY on per-expert keys (regex match). All other
keys (router gate, shared_experts MLP, e_score_correction_bias, dense
MLP for layers < first_k_dense_replace) are passthrough by virtue of
`can_handle()` returning False on them — the loader streams them through
unchanged.

The gate_up_proj fusion uses `cat([gate, up], dim=1)` — gate first, up
second. This matches `OpenPanguV2Experts.forward`:

    gate, up = nn.functional.linear(state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)

where `.chunk(2, dim=-1)` returns (first_half, second_half), so first
half is gate.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch

from ....utils import logging
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


logger = logging.get_logger(__name__)

# Matches per-expert split keys like: model.layers.0.mlp.experts.3.gate_proj.weight
# Does NOT match shared_experts.{gate_proj,up_proj,down_proj}.weight (no `.\d+.`
# segment in between — they passthrough as-is).
_EXPERT_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class PanguOmniV2CheckpointTensorConverter:
    """Stack & merge Pangu per-expert ckpt keys into fused 3D layout.

    Buffers per-expert tensors as they stream from safetensors files,
    emits the merged tensors once all experts for a given (layer,
    projection) are collected.

    Args:
        num_experts: Number of routed experts per MoE layer
            (`config.n_routed_experts`, 384 for Pangu Omni v2 30B-A2B).

    The class mirrors `Qwen3MoeCheckpointTensorConverter` exactly because
    the per-expert -> fused 3D conversion is identical between the two
    models (only the surrounding layout differs: Pangu has shared_experts
    + e_score_correction_bias, Qwen3-MoE has neither, but those are
    passthrough and handled by `can_handle()` returning False).
    """

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        # {(prefix, proj_name): {expert_id: tensor}}
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        # {prefix: {proj_name: stacked_tensor}} for gate/up merge waiting
        self._stacked_buffer: Dict[str, Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(_EXPERT_PATTERN.match(name))

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        match = _EXPERT_PATTERN.match(name)
        if not match:
            return None

        prefix, expert_id_str, proj_name = match.groups()
        expert_id = int(expert_id_str)
        buf_key = (prefix, proj_name)

        if buf_key not in self._expert_buffer:
            self._expert_buffer[buf_key] = {}
        self._expert_buffer[buf_key][expert_id] = tensor

        # Check if all experts collected for this (prefix, proj)
        if len(self._expert_buffer[buf_key]) < self.num_experts:
            return None

        # Stack all experts: [E, I, H] for gate_proj/up_proj or [E, H, I]
        # for down_proj. The dict-comprehension uses expert_id as index so
        # ordering is canonical (matches `gate_up_proj[expert_idx]` access
        # in `OpenPanguV2Experts.forward`).
        stacked = torch.stack([self._expert_buffer[buf_key][i] for i in range(self.num_experts)])
        del self._expert_buffer[buf_key]

        if proj_name == "down_proj":
            return ConvertedCheckpointTensor(f"{prefix}.experts.down_proj", stacked)

        # gate_proj or up_proj — buffer until the other arrives, then concat.
        if prefix not in self._stacked_buffer:
            self._stacked_buffer[prefix] = {}
        self._stacked_buffer[prefix][proj_name] = stacked

        if "gate_proj" in self._stacked_buffer[prefix] and "up_proj" in self._stacked_buffer[prefix]:
            gate = self._stacked_buffer[prefix].pop("gate_proj")
            up = self._stacked_buffer[prefix].pop("up_proj")
            if not self._stacked_buffer[prefix]:
                del self._stacked_buffer[prefix]
            # gate first, up second — matches `OpenPanguV2Experts.forward`
            # which does `gate, up = ...chunk(2, dim=-1)`.
            merged = torch.cat([gate, up], dim=1)  # [E, 2*I, H]
            return ConvertedCheckpointTensor(f"{prefix}.experts.gate_up_proj", merged)

        return None

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        """Validate that all buffers were flushed.

        Raises RuntimeError if any buffers remain unflushed — incomplete
        expert tensors cannot be merged into valid fused format and
        indicate a corrupted or incomplete checkpoint.
        """
        errors: List[str] = []
        if self._expert_buffer:
            unflushed = {k: len(v) for k, v in self._expert_buffer.items()}
            errors.append(
                f"unflushed per-expert buffer (incomplete experts, expected {self.num_experts}): {unflushed}"
            )
        if self._stacked_buffer:
            unflushed = {k: list(v.keys()) for k, v in self._stacked_buffer.items()}
            errors.append(f"unflushed stacked buffer (missing gate/up pair): {unflushed}")
        if errors:
            raise RuntimeError(
                "Pangu Omni v2 checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors)
            )
        return []


def create_pangu_omni_v2_checkpoint_tensor_converter(model):
    """Factory function — registered on Pangu model classes.

    VeOmni's `checkpoint_tensor_loading` machinery calls this with the
    constructed model and uses the returned converter to remap per-expert
    keys as safetensors stream in.
    """
    return PanguOmniV2CheckpointTensorConverter(
        num_experts=model.config.n_routed_experts,
    )


__all__ = [
    "PanguOmniV2CheckpointTensorConverter",
    "create_pangu_omni_v2_checkpoint_tensor_converter",
]
