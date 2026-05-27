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

"""Configuration for Pangu Omni v2 (30B-A2B and larger-scale variants).

Two responsibilities live here:

1. **`OpenPanguOmniConfig`** — the top-level multimodal config. Nests
   `vision_config` / `text_config` / `audio_config` sub-configs (matching
   the upstream `configuration_openpangu_omni.py`), plus carries
   multimodal-only top-level fields (`image_token_id`, `audio_token_id`,
   etc.). The on-disk `config.json` has `model_type="qwen2_moe"` (a Pangu
   bug); we override it to the canonical `"openpangu_omni"` so VeOmni's
   `MODELING_REGISTRY` dispatches to the Pangu adapter.

2. **Sub-config classes** — `OpenPanguOmniVisionConfig`,
   `OpenPanguOmniTextConfig`, `OpenPanguOmniAudioConfig`. Each is a thin
   `PretrainedConfig` subclass with `base_config_key` set; the only one
   carrying derivation logic is `OpenPanguOmniTextConfig` which inherits
   the same `swa_layers -> layer_types` rule that `OpenPanguV2Config`
   uses upstream.

## Backwards Compatibility With Flat Text Configs

The text-only path can build `OpenPanguV2ForCausalLM` from a flat
`OpenPanguOmniConfig(num_hidden_layers=37, hidden_size=2560,
...)` where every text field is a top-level kwarg (no `text_config`
nesting). We preserve this:

- Flat production load (`text_config` absent from `config.json`, all
  text fields top-level): `__init__` snapshots user kwargs into
  `user_text_seed`, builds `self.text_config` from that, and ALSO leaves
  the top-level attributes intact (already set by
  `super().__init__(**kwargs)`).
- Multimodal explicit nesting
  (`OpenPanguOmniConfig(text_config={...}, vision_config={...},
  audio_config={...})`): `__init__` builds sub-configs from the dicts
  AND mirrors text fields to top-level (reference behavior at
  `configuration_openpangu_omni.py:215-220`).

The two paths are interchangeable — downstream code can read either
`cfg.hidden_size` (top-level) or `cfg.text_config.hidden_size` (nested)
and get the same value. This matters because:

- `OpenPanguV2Model.__init__` reads top-level fields.
- Reference multimodal modeling reads `cfg.vision_config.out_hidden_size`
  / `cfg.text_config.hidden_size`.

## Why we don't inherit OpenPanguV2Config for text_config

The reference defines `OpenPanguOmniTextConfig(OpenPanguV2Config)`. We
don't ship a separate `OpenPanguV2Config` class (`OpenPanguOmniConfig`
also supports that flat role). Instead, the text sub-config is a
peer of `OpenPanguOmniConfig` that holds the same `_derive_layer_types`
logic; downstream modeling that wants the OpenPanguV2 text backbone gets
it via `OpenPanguOmniConfig.text_config`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from transformers import PretrainedConfig


_CANONICAL_MODEL_TYPE = "openpangu_omni"


def _derive_layer_types_from_swa_layers(cfg: PretrainedConfig) -> None:
    """Populate `layer_types` from `swa_layers` when not explicitly set.

    Verbatim port of `OpenPanguV2Config.__init__` lines 157-163.

    The 30B-A2B config has `swa_layers=[]` and no `layer_types` key,
    which means every layer is `"full_attention"`. A model variant that
    ships SWA layers would have `swa_layers=[2, 5, 8, ...]` and would
    land on `"sliding_attention"` at those indices.

    Extracted as a free function so both `OpenPanguOmniConfig` (flat
    path) and `OpenPanguOmniTextConfig` (nested path) can
    use it without duplicating the rule.
    """
    if (
        getattr(cfg, "layer_types", None) is None
        and getattr(cfg, "swa_layers", None) is not None
        and getattr(cfg, "num_hidden_layers", None) is not None
    ):
        swa_layers = cfg.swa_layers
        sliding_window = getattr(cfg, "sliding_window", None)
        cfg.layer_types = [
            "sliding_attention" if sliding_window is not None and i in swa_layers else "full_attention"
            for i in range(cfg.num_hidden_layers)
        ]


class OpenPanguOmniVisionConfig(PretrainedConfig):
    """Pangu Omni v2 vision-tower config (ViT-style encoder).

    Field set declared verbatim from
    ``configuration_openpangu_omni.OpenPanguOmniVisionConfig`` (upstream).
    The explicit declaration is important for two reasons:

    1. **Default-driven aliasing**: The production ``config.json`` carries
       ``in_chans: 3`` (not ``in_channels: 3``). Because our ``__init__``
       receives ``in_chans`` as an unknown kwarg, the base
       ``PretrainedConfig`` stores it on ``self.in_chans`` — but the
       reference's vision tower reads ``config.in_channels``. Declaring
       ``in_channels=3`` as a constructor default ensures
       ``self.in_channels`` is always populated even when JSON only
       supplies ``in_chans``. This matches upstream behavior exactly
       (upstream silently drops ``in_chans`` for the same reason).

    2. **Round-trippable serialization**: ``PretrainedConfig.to_dict``
       only emits attributes set via ``__init__`` parameters or
       ``self.foo = ...`` assignments inside ``__init__``. With explicit
       declarations our config is a faithful round-trip of the upstream
       schema.

    Optional fields not in upstream (``output_dim``, ``rms_norm_eps``,
    ``mm_unit_vision_select_layer``, etc.) are accepted via ``**kwargs``
    and stashed on ``self`` by the base class so the vision tower's
    ``getattr(config, ...)`` calls work.
    """

    model_type = "openpangu_omni_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 32,
        hidden_size: int = 3584,
        hidden_act: str = "silu",
        intermediate_size: int = 3420,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        tokens_per_second: int = 4,
        window_size: int = 112,
        out_hidden_size: int = 3584,
        fullatt_block_indexes: list | None = None,
        initializer_range: float = 0.02,
        use_gatedmerger: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes if fullatt_block_indexes is not None else [7, 15, 23, 31]
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range
        self.use_gatedmerger = use_gatedmerger


class OpenPanguOmniAudioConfig(PretrainedConfig):
    """Pangu Omni v2 audio-tower config (Conformer-style "Huanyu" encoder).

    Field set declared verbatim from
    ``configuration_openpangu_omni.OpenPanguOmniAudioConfig`` (upstream).
    See :class:`OpenPanguOmniVisionConfig` for the rationale on explicit
    declarations vs. ``**kwargs`` opacity.
    """

    model_type = "openpangu_omni_audio"
    base_config_key = "audio_config"

    def __init__(
        self,
        d_model: int = 1280,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        encoder_layers: int = 32,
        num_mel_bins: int = 128,
        max_source_positions: int = 1500,
        scale_embedding: bool = False,
        activation_function: str = "gelu",
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.scale_embedding = scale_embedding
        self.activation_function = activation_function
        self.initializer_range = initializer_range


class OpenPanguOmniTextConfig(PretrainedConfig):
    """Pangu Omni v2 text-backbone config (OpenPanguV2 MoE).

    Holds the same fields as the flat-Week-2 `OpenPanguOmniConfig` — this
    is the nested counterpart used by `OpenPanguVLTextModel(OpenPanguV2Model)`
    when multimodal modeling passes `config.text_config` down.

    Reference: `OpenPanguOmniTextConfig(OpenPanguV2Config)` from
    `configuration_openpangu_omni.py:62`.
    """

    model_type = "openpangu_omni_text"
    base_config_key = "text_config"

    def __init__(self, topk_group: int = 1, **kwargs: Any) -> None:
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.topk_group = topk_group
        _derive_layer_types_from_swa_layers(self)


class OpenPanguOmniConfigPatch:
    """Marker base class kept for compatibility with existing imports."""


class OpenPanguOmniConfig(PretrainedConfig, OpenPanguOmniConfigPatch):
    """Pangu Omni v2 top-level config.

    Invariants enforced:

    - `model_type == "openpangu_omni"` — ensures `MODELING_REGISTRY[cfg.model_type]`
      dispatches to the Pangu adapter rather than to a future Qwen2-MoE loader.

    - **Sub-config nesting** — `self.vision_config` /
      `self.audio_config` / `self.text_config` are always populated as
      `PretrainedConfig` instances (never raw dicts), even if the input
      `config.json` only carries flat top-level fields.

    - **Top-level mirror of text fields** — `self.hidden_size`,
      `self.num_hidden_layers`, etc. are accessible at top-level too,
      preserving the `OpenPanguV2ForCausalLM`-on-flat-config path.

    - `layer_types` is derived from `swa_layers` when not explicitly set
      (same as `OpenPanguV2Config` upstream).
    """

    model_type = _CANONICAL_MODEL_TYPE
    sub_configs: ClassVar[dict[str, Any]] = {
        "vision_config": OpenPanguOmniVisionConfig,
        "text_config": OpenPanguOmniTextConfig,
        "audio_config": OpenPanguOmniAudioConfig,
    }
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(self, **kwargs: Any) -> None:
        # Strip the buggy model_type from disk so the parent doesn't see it.
        kwargs.pop("model_type", None)

        # Pop sub-configs out of kwargs so the rest can be treated as
        # top-level (text + multimodal-only) fields. Sub-configs may be:
        # - dict (production load from JSON)
        # - PretrainedConfig instance (test construction)
        # - None (flat path; sub-config absent from JSON)
        vision_cfg = kwargs.pop("vision_config", None)
        audio_cfg = kwargs.pop("audio_config", None)
        text_cfg = kwargs.pop("text_config", None)

        # Snapshot the remaining user-provided kwargs BEFORE super().__init__
        # consumes them. This snapshot is the seed for text_config when
        # the caller didn't supply an explicit one (flat compatibility path).
        user_text_seed = dict(kwargs)

        super().__init__(**kwargs)

        # Build sub-configs. _make_sub accepts dict / instance / None.
        self.vision_config = self._make_sub(self.sub_configs["vision_config"], vision_cfg)
        self.audio_config = self._make_sub(self.sub_configs["audio_config"], audio_cfg)

        if text_cfg is None:
            # Synthesize text_config from the flat kwargs the
            # user passed. user_text_seed contains exactly what they
            # intended for the text backbone.
            self.text_config = self.sub_configs["text_config"](**user_text_seed)
        else:
            if isinstance(text_cfg, dict) and "topk_group" not in text_cfg and "topk_group" in user_text_seed:
                text_cfg = {**text_cfg, "topk_group": user_text_seed["topk_group"]}
            self.text_config = self._make_sub(self.sub_configs["text_config"], text_cfg)
            # If text_config was a dict, ALSO mirror its fields to top-level
            # for downstream code that reads `cfg.hidden_size` flat.
            # Matches reference behavior at
            # `configuration_openpangu_omni.py:215-220`.
            if isinstance(text_cfg, dict):
                for key, value in text_cfg.items():
                    setattr(self, key, value)

        # Operates on top-level attributes (mirrors what
        # OpenPanguV2Config did upstream).
        if not hasattr(self, "topk_group"):
            self.topk_group = self.text_config.topk_group
        _derive_layer_types_from_swa_layers(self)

    @staticmethod
    def _make_sub(cls: type, val: Any) -> PretrainedConfig:
        """Instantiate a sub-config from None / dict / PretrainedConfig."""
        if val is None:
            return cls()
        if isinstance(val, dict):
            return cls(**val)
        return val

    # Kept for callers that explicitly invoked the
    # helper as a method. Delegates to the free function.
    def _derive_layer_types_from_swa_layers(self) -> None:
        _derive_layer_types_from_swa_layers(self)


def apply_veomni_pangu_omni_v2_patch() -> None:
    """Apply runtime patches needed before model construction.

    Hook for any runtime monkey-patches that must run after `transformers`
    is imported but before the model class is instantiated. Today it is a
    no-op.
    """
    return


__all__ = [
    "OpenPanguOmniConfig",
    "OpenPanguOmniTextConfig",
    "OpenPanguOmniVisionConfig",
    "OpenPanguOmniAudioConfig",
    "OpenPanguOmniConfigPatch",
    "apply_veomni_pangu_omni_v2_patch",
]
