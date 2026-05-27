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

"""Registry wiring for Pangu Omni v2.

This module ONLY registers the Pangu adapter with VeOmni's three registries:
- MODEL_CONFIG_REGISTRY[<raw config.json model_type>] -> our config subclass
  that exposes the canonical `openpangu_omni` model_type to downstream lookups.
- MODELING_REGISTRY[<canonical model_type>] -> dispatcher that picks the right
  modeling class based on `config.architectures[0]`.
- MODEL_PROCESSOR_REGISTRY[<HF processor class name>] -> optional processor patch.

## On Pangu checkpoint `model_type` variants

Some Pangu config.json files set `model_type = "qwen2_moe"`, which is
incorrect. Newer training snapshots may instead use
`model_type = "openpangu_v2_omni"`. Both must route to this adapter.
Coordination implications:

1. We MUST register `MODEL_CONFIG_REGISTRY["qwen2_moe"]` because VeOmni's
   `get_model_config` looks up the registry by `config.model_type` BEFORE
   any of our code runs. The same applies to the newer
   `openpangu_v2_omni` spelling. Our config subclass then rewrites
   `model_type` to the canonical `openpangu_omni`.
2. Once `model_type` is rewritten, `MODELING_REGISTRY[cfg.model_type]`
   resolves to `MODELING_REGISTRY["openpangu_omni"]` — that's our entry,
   no conflict with future VeOmni Qwen2-MoE support.
3. Long-term, upstream configs should converge on one stable model_type
   so compatibility aliases can eventually be removed.
"""

from __future__ import annotations

from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


# ---------------------------------------------------------------------------
# Config registration (keys = on-disk Pangu model_type variants)
# ---------------------------------------------------------------------------


@MODEL_CONFIG_REGISTRY.register("qwen2_moe")
@MODEL_CONFIG_REGISTRY.register("openpangu_v2_omni")
def register_pangu_omni_v2_config():
    """Return the Pangu Omni v2 config subclass.

    The config subclass returned here rewrites all supported on-disk Pangu
    model_type spellings to `openpangu_omni`, isolating downstream lookups.
    If VeOmni adds a real Qwen2-MoE adapter, this dispatcher will need to
    branch on `config.architectures[0]` for the `qwen2_moe` compatibility key.
    """
    from .configuration_pangu_omni_v2 import OpenPanguOmniConfig, apply_veomni_pangu_omni_v2_patch

    apply_veomni_pangu_omni_v2_patch()
    return OpenPanguOmniConfig


# ---------------------------------------------------------------------------
# Modeling registration (key = canonical model_type "openpangu_omni")
# ---------------------------------------------------------------------------


@MODELING_REGISTRY.register("openpangu_omni")
@MODELING_REGISTRY.register("openpangu_v2_omni")
def register_pangu_omni_v2_modeling(architecture: str):
    """Dispatch on `architectures[0]` to the right concrete class.

    Architecture strings observed in Pangu Omni v2 (30B-A2B) `config.json`:
    - "OpenPanguUltraOmniForConditionalGeneration" — top-level multimodal
    - "OpenPanguVLForConditionalGeneration" — VL-only variant
    - "OpenPanguV2ForCausalLM" — text-only causal LM head
    - "OpenPanguV2Model" — text-only backbone

    The order of `if`-checks below matters: substring matches like
    `"OpenPanguV2Model" in architecture` would also fire on
    `"OpenPanguV2ModelForX"`, so the more-specific ForCausalLM /
    Ultra / VL checks come first.
    """
    import importlib

    from .checkpoint_tensor_converter import (
        create_pangu_omni_v2_checkpoint_tensor_converter,
    )

    # Import order matters — see "On circular imports" below.
    #
    # We MUST import ``modeling_text`` BEFORE
    # ``modeling_omni``. Going the other way around triggers
    # this 3-way cycle:
    #
    #   modeling_omni (top-level)
    #     L92  from .modeling_vl import OpenPanguVL
    #   modeling_vl (top-level)
    #     L1368  _OpenPanguV2Model = _get_openpangu_v2_model_cls()
    #     ──────> from .modeling_text import OpenPanguV2Model
    #   modeling_text (top-level)
    #     L426  class OpenPanguVLForConditionalGeneration(
    #                _get_open_pangu_vl_class()):   ← class-def-time call
    #     ──────> from .modeling_vl import OpenPanguVL
    #              ╰──> vl is partially-loaded (we entered at L1368, the
    #                   class def for OpenPanguVL lives at line ~600 BEFORE
    #                   L1368 in source order but we're already past it
    #                   on the call stack — actually OpenPanguVL is
    #                   defined LATER in the file, so partial module
    #                   doesn't yet have the symbol). ImportError.
    #
    # By importing ``modeling_text`` first, we trigger the
    # same chain in a way that terminates: text starts loading, hits L426,
    # imports vl, vl hits L1368, imports text (which is partial but
    # already past L132 where OpenPanguV2Model is defined — so the
    # import succeeds), vl finishes loading, v2's L426 class def
    # completes. After that ``modeling_omni`` can be loaded
    # cleanly because vl is fully resolved in ``sys.modules``.
    #
    # CRITICAL: we use ``importlib.import_module`` (and not a ``from
    # .modeling_omni import OpenPanguOmni``) for the second
    # module on purpose. Ruff's ``I001`` import-sort rule reorders
    # consecutive ``from`` imports inside a function body
    # alphabetically — and ``modeling_omni`` sorts BEFORE
    # ``modeling_text`` (``o`` < ``p``). Every time
    # ``make style`` ran it silently put the imports back in the
    # crashing order. ``importlib.import_module`` is not pattern-
    # matched by I001 so it stays put, preserving the required
    # "text first, omni second" runtime ordering.
    from .modeling_text import (
        OpenPanguUltraOmniForConditionalGeneration,
        OpenPanguV2ForCausalLM,
        OpenPanguV2Model,
        OpenPanguVLForConditionalGeneration,
    )

    OpenPanguOmni = importlib.import_module(  # noqa: F811
        ".modeling_omni", package=__package__
    ).OpenPanguOmni

    # Attach the ckpt converter on each class so VeOmni's
    # `checkpoint_tensor_loading` machinery can remap per-expert keys to
    # the in-memory fused 3D layout during `from_pretrained`. Idempotent:
    # repeated registration calls (e.g. dispatcher invoked twice) just
    # re-assign the same staticmethod.
    for model_cls in (
        OpenPanguV2ForCausalLM,
        OpenPanguV2Model,
        OpenPanguVLForConditionalGeneration,
        OpenPanguUltraOmniForConditionalGeneration,
        OpenPanguOmni,
    ):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_pangu_omni_v2_checkpoint_tensor_converter)

    if "OpenPanguUltraOmni" in architecture:
        # Route the production 30B-A2B architecture string to the
        # audio-aware `OpenPanguOmni` (vision + audio + text). The
        # compatibility alias `OpenPanguUltraOmniForConditionalGeneration` (which
        # inherits `OpenPanguVL` for legacy import stability) is NOT
        # used here — see its docstring + `_get_open_pangu_omni_class`
        # in `modeling_text.py` for why the resolution must be
        # dispatcher-time rather than class-def-time (3-way circular
        # import otherwise).
        return OpenPanguOmni
    if "OpenPanguVL" in architecture:
        return OpenPanguVLForConditionalGeneration
    if "OpenPanguV2ForCausalLM" in architecture:
        return OpenPanguV2ForCausalLM
    if "OpenPanguV2Model" in architecture or architecture == "OpenPanguV2Model":
        return OpenPanguV2Model
    # Default to the top-level multimodal class — matches what Pangu Omni v2
    # ships today. Any unknown OpenPangu architecture will land here.
    return OpenPanguOmni


# Also register the canonical model_type so callers that already have a
# config object with `model_type = "openpangu_omni"` (e.g. after the Pangu
# team fixes their config.json upstream) hit the same path. Both keys
# resolve to the same class graph.
@MODELING_REGISTRY.register("qwen2_moe")
def register_pangu_omni_v2_modeling_qwen2_moe_alias(architecture: str):
    """Alias for the buggy `qwen2_moe` model_type — dispatches on architecture.

    If `architectures[0]` is not OpenPangu*, raise to prevent collision with
    a future real Qwen2-MoE adapter — that adapter should explicitly handle
    its own architectures, not fall through to us.
    """
    if "OpenPangu" not in (architecture or ""):
        raise RuntimeError(
            f"MODELING_REGISTRY['qwen2_moe'] is currently owned by the Pangu Omni "
            f"v2 adapter (which works around a config.json bug). Got "
            f"architecture={architecture!r}, which is not an OpenPangu variant. "
            f"If you need real Qwen2-MoE support, add a VeOmni adapter for it and "
            f"coordinate the registry key with veomni/models/transformers/pangu_omni_v2/."
        )
    return register_pangu_omni_v2_modeling(architecture)


# ---------------------------------------------------------------------------
# Processor registration.
# ---------------------------------------------------------------------------


# Pangu's processor class is typically named based on the model variant.
# Leaving this commented out keeps the registry untouched until a local
# processor patch is needed.
#
# @MODEL_PROCESSOR_REGISTRY.register("OpenPanguOmniProcessor")
# def register_pangu_omni_v2_processor():
#     from .processing_pangu_omni_v2 import OpenPanguOmniProcessor, apply_veomni_pangu_omni_v2_patch
#     apply_veomni_pangu_omni_v2_patch()
#     return OpenPanguOmniProcessor


# ---------------------------------------------------------------------------
# Data-transform registration (runs at adapter import time)
# ---------------------------------------------------------------------------
#
# The data-transform registry is consulted at training start (via
# ``VLMTrainer._build_data_transform`` -> ``build_data_transform``).
# Because that lookup happens long after this ``__init__.py`` is
# imported (the trainer is constructed in ``tasks/train_vlm.py``
# main()), we can safely import + register at module load — no
# circular-import risk through the data subsystem.
#
# The Pangu data transform itself is intentionally placed in
# ``pangu_omni_v2/data_transform.py`` instead of being shoved into
# ``veomni/data/data_transform.py`` next to the Qwen one so that
# adapter-local changes (e.g. tokenizer fork, system-message format)
# stay inside the adapter's blast radius.
from .data_transform import register_pangu_omni_data_transform


register_pangu_omni_data_transform()
