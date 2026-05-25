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

"""Structural smoke tests against the real Pangu Omni 30B-A2B `config.json`
(Week 3.6 build-path validation; no weight load, no forward).

What this catches:

- Production `config.json` field names (e.g. ``in_chans`` vs
  ``in_channels``) being silently dropped or mis-defaulted.
- Dispatcher routing for the on-disk ``architectures[0]`` =
  ``OpenPanguUltraOmniForConditionalGeneration`` landing on our
  ``OpenPanguVL`` graph (Week 3.4.d).
- Sub-config nesting (``text_config`` / ``vision_config`` /
  ``audio_config``) populating correctly.
- ``OpenPanguVL.__init__`` constructing the full class graph
  (vision tower + text backbone + lm_head + vision_projection) on
  ``meta`` device.
- Top-line parameter accounting matching expected 30B-A2B sizing.

What this does NOT catch (and explicitly cannot, without NPU):

- Forward numerical correctness (covered by the toy-scale
  ``test_pangu_vl_model_parity.py`` instead, which runs on CPU).
- Actual weight loading via ``from_pretrained`` (covered by the NPU
  oracle smoke in ``tools/pangu_oracle_check.py --mode veomni``,
  which requires an NPU box).

Cost: ~2-3 s per test (config parse + meta-device construction).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


PANGU_MODEL_DIR = Path("/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model")
pytestmark = pytest.mark.skipif(
    not PANGU_MODEL_DIR.exists(),
    reason="Production Pangu model dir not mounted; skipping real-config smoke.",
)


def _load_real_config():
    from veomni.models.auto import build_config

    config = build_config(str(PANGU_MODEL_DIR))
    # Force eager attention so the build doesn't try to import flash_attn.
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    config.audio_config._attn_implementation = "eager"
    return config


def test_real_config_loads_with_canonical_model_type():
    """Production ``config.json`` has ``model_type='qwen2_moe'`` (Pangu bug)
    but our config subclass rewrites it to the canonical
    ``'openpangu_omni'`` so ``MODELING_REGISTRY`` dispatches to the Pangu
    adapter, not the future Qwen2-MoE loader."""
    config = _load_real_config()
    assert config.model_type == "openpangu_omni"
    assert config.architectures == ["OpenPanguUltraOmniForConditionalGeneration"]


def test_real_config_subconfigs_populated():
    """Sub-configs must all be present and reachable through both nested
    and top-level mirror paths (Week 3.1 BC contract)."""
    config = _load_real_config()
    assert config.vision_config is not None
    assert config.text_config is not None
    assert config.audio_config is not None
    # Multimodal token ids on top-level
    for tok in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "audio_start_token_id",
        "audio_end_token_id",
        "audio_token_id",
    ):
        assert hasattr(config, tok), f"missing top-level {tok}"


def test_real_vision_config_in_channels_alias():
    """The on-disk ``vision_config`` carries ``in_chans`` (not
    ``in_channels``); the reference's vision tower reads
    ``config.in_channels``. Our config class declares ``in_channels=3``
    as a constructor default, mirroring upstream's silent drop of
    ``in_chans``. This test pins the alias behavior."""
    config = _load_real_config()
    vc = config.vision_config
    # Both must be present and equal to 3.
    assert getattr(vc, "in_channels", None) == 3
    assert getattr(vc, "in_chans", None) == 3


def test_real_text_config_mrope_section_matches_qk_rope_dim():
    """Production ``rope_scaling.mrope_section`` must satisfy
    ``sum(section) * 2 == qk_rope_dim`` for the rope split inside
    ``OpenPanguVLRotaryEmbedding`` to align correctly. Production:
    section=[6,5,5], sum=16, ×2=32, qk_rope_dim=32. This test asserts
    the invariant so a future config change can't silently break
    rotation."""
    config = _load_real_config()
    tc = config.text_config
    rope_scaling = tc.rope_scaling
    section = rope_scaling["mrope_section"]
    assert sum(section) * 2 == tc.qk_rope_dim, (
        f"mrope_section {section} (sum={sum(section)}) does not match "
        f"qk_rope_dim={tc.qk_rope_dim} (expected {tc.qk_rope_dim // 2})"
    )


def test_real_config_dispatcher_routes_to_omni():
    """The production ``architectures[0]`` =
    ``OpenPanguUltraOmniForConditionalGeneration`` must dispatch through
    ``MODELING_REGISTRY['openpangu_omni']`` to our Week 3.5
    ``OpenPanguOmni`` (audio-aware multimodal CausalLM).

    Week 3.5 swap notes: before Week 3.5 the dispatcher returned
    ``OpenPanguVL`` (image-only). Week 3.5 widened the class graph to
    ``OpenPanguOmni`` which adds the audio tower; ``OpenPanguVL`` and
    ``OpenPanguOmni`` are sibling subclasses of
    ``OpenPanguPreTrainedModel``. The dispatcher now returns the
    audio-aware variant so an ``input_features`` kwarg on the forward
    will Just Work. Image-only and text-only invocations remain
    bit-for-bit identical (the audio branch is a no-op when
    ``input_features=None``).
    """
    # Pre-load ``modeling_pangu_omni_v2`` to unwind the
    # ``modeling_openpangu_vl ↔ modeling_pangu_omni_v2`` import cycle
    # before the dispatcher (which transitively loads
    # ``modeling_openpangu_omni``) fires. Standalone-run safety; see
    # the same ritual in ``tests/test_pangu_omni_model_parity.py::_ours_modules``.
    from veomni.models.transformers.pangu_omni_v2 import (
        modeling_pangu_omni_v2,  # noqa: F401 — bootstrap order
        register_pangu_omni_v2_modeling,
    )

    cls = register_pangu_omni_v2_modeling("OpenPanguUltraOmniForConditionalGeneration")
    from veomni.models.transformers.pangu_omni_v2.modeling_openpangu_omni import (
        OpenPanguOmni,
    )

    # The dispatcher returns the audio-aware ``OpenPanguOmni`` class
    # **directly** (not the legacy ``OpenPanguUltraOmniForConditionalGeneration``
    # subclass-of-``OpenPanguVL`` stub). See
    # ``modeling_pangu_omni_v2.OpenPanguUltraOmniForConditionalGeneration``
    # docstring + ``__init__.register_pangu_omni_v2_modeling`` for the
    # circular-import reasoning behind the indirection.
    assert cls is OpenPanguOmni or issubclass(cls, OpenPanguOmni), (
        f"Dispatcher returned {cls.__name__} which is not (a subclass of) OpenPanguOmni"
    )


@pytest.mark.slow
def test_real_config_open_pangu_omni_constructs_on_meta_device():
    """**The big build-path test.** Construct the full ``OpenPanguOmni``
    graph (vision + audio + text) on ``meta`` device using the real
    30B-A2B config — verifies that every sub-module accepts the
    production field set without raising ``AttributeError`` or shape
    errors.

    Param count assertions (Week 3.5 — adds audio_tower vs Week 3.6
    image-only baseline):

    - Total: ~29.0–29.3 B (vision 527M + language_model 27.78B +
      lm_head 388M + audio_tower ~370M).
    - Vision tower: in the 500-600M range (depth=26, hidden=1280,
      gated merger output=10240).
    - lm_head: ``vocab_size × hidden_size`` = 151552 × 2560 ≈ 388 M.
    - audio_tower: in the 250-450M range (depth=24, d_model=768,
      MoE-free conformer + audio_tower.proj to hidden_size *
      mhc_num_stream).

    Marked ``slow`` because the meta-device construction still walks
    through every ``nn.Linear`` / ``nn.LayerNorm`` etc. for 37 text
    layers + 26 vision blocks + 24 audio conformer layers; ~5-8 s in
    practice.
    """
    config = _load_real_config()
    # Build through the dispatcher rather than instantiating the
    # ``OpenPanguUltraOmniForConditionalGeneration`` stub directly: the
    # stub inherits from ``OpenPanguVL`` (no audio tower) to keep
    # ``modeling_pangu_omni_v2`` free of the 3-way circular import.
    # Production code paths (``VeOmni.from_pretrained`` /
    # ``build_model``) always go through ``MODELING_REGISTRY`` →
    # ``register_pangu_omni_v2_modeling`` → ``OpenPanguOmni``, so
    # mirroring that here is the right way to exercise the
    # audio-aware class graph.
    from veomni.models.transformers.pangu_omni_v2 import (
        modeling_pangu_omni_v2,  # noqa: F401 — bootstrap order
        register_pangu_omni_v2_modeling,
    )

    omni_cls = register_pangu_omni_v2_modeling("OpenPanguUltraOmniForConditionalGeneration")

    with torch.device("meta"):
        model = omni_cls(config)

    # Top-line param accounting
    total = sum(p.numel() for p in model.parameters())
    visual = sum(p.numel() for p in model.model.visual.parameters())
    lm = sum(p.numel() for p in model.model.language_model.parameters())
    head = model.lm_head.weight.numel()
    audio = sum(p.numel() for p in model.model.audio_tower.parameters())
    assert 28e9 < total < 31e9, f"unexpected total: {total / 1e9:.2f}B"
    assert 4e8 < visual < 7e8, f"unexpected vision tower: {visual / 1e6:.0f}M"
    assert 25e9 < lm < 30e9, f"unexpected language_model: {lm / 1e9:.2f}B"
    assert 3.5e8 < head < 4.5e8, f"unexpected lm_head: {head / 1e6:.0f}M"
    assert 2e8 < audio < 5e8, f"unexpected audio_tower: {audio / 1e6:.0f}M"
    # lm_head must project to vocab_size × hidden_size
    assert tuple(model.lm_head.weight.shape) == (
        config.text_config.vocab_size,
        config.text_config.hidden_size,
    )

    # Class graph identity — both VL pieces AND the audio tower
    from veomni.models.transformers.pangu_omni_v2.modeling_huanyu_audio_encoder import (
        HuanyuAudioEncoder,
    )
    from veomni.models.transformers.pangu_omni_v2.modeling_openpangu_vl import (
        OpenPanguVisionTransformerPretrainedModel,
        OpenPanguVLTextModel,
    )

    assert isinstance(model.model.visual, OpenPanguVisionTransformerPretrainedModel)
    assert isinstance(model.model.language_model, OpenPanguVLTextModel)
    assert isinstance(model.model.audio_tower, HuanyuAudioEncoder)
    # audio_tower.proj output dim matches hidden_size * mhc_num_stream
    expected_proj_out = config.text_config.hidden_size * (config.mhc_num_stream if config.use_mhc else 1)
    assert model.model.audio_tower.proj.weight.shape == (
        expected_proj_out,
        config.audio_config.d_model,
    )


def test_real_config_checkpoint_conversion_mapping_includes_audio_tower():
    """Verify the production checkpoint's ``audio_tower.*`` keys map
    cleanly through ``OpenPanguOmni._checkpoint_conversion_mapping`` to
    our nested ``model.audio_tower.*`` parameter names. Without the
    ``^audio_tower → model.audio_tower`` rule, those keys would either
    be silently dropped (no rule matches) or incorrectly rewritten to
    ``model.language_model.audio_tower.*`` (collision with the
    text-backbone exclusion).
    """
    # Same bootstrap-order ritual; see
    # ``test_real_config_dispatcher_routes_to_omni``.
    from veomni.models.transformers.pangu_omni_v2 import (
        modeling_pangu_omni_v2,  # noqa: F401 — bootstrap order
    )
    from veomni.models.transformers.pangu_omni_v2.modeling_openpangu_omni import (
        OpenPanguOmni,
    )

    mapping = OpenPanguOmni._checkpoint_conversion_mapping
    assert "^audio_tower" in mapping
    assert mapping["^audio_tower"] == "model.audio_tower"
    # The text-backbone exclusion must include audio_tower so it doesn't
    # accidentally re-route audio params through `language_model`.
    text_rule_key = next(k for k in mapping if k.startswith(r"^model(?!"))
    assert "audio_tower" in text_rule_key, f"text backbone exclusion rule {text_rule_key!r} missing audio_tower"
