"""Pangu Omni v2 data transform — per-sample SFT preprocessor.

Registered into ``veomni.data.data_transform.DATA_TRANSFORM_REGISTRY``
under the key ``"openpangu_omni"`` from
``pangu_omni_v2/__init__.py``. Called once per raw sample by
``VLMTrainer._build_data_transform`` -> ``build_data_transform`` to
produce model-ready ``input_ids`` / ``attention_mask`` /
``position_ids`` / ``pixel_values`` / ``input_features`` /
``audio_feature_lengths`` / ``labels`` / multimodal masks.

Mirrors ``veomni.data.data_transform.process_sample_qwen_omni`` (the
Qwen2.5-Omni / Qwen3-Omni-MoE transform) with three Pangu-specific
adjustments:

1. **System message**: Pangu's chat template auto-emits an empty
   ``<|message_start|>系统：<|message_end|>`` block when no system
   message is provided, so we do *not* prepend a synthetic system
   message the way Qwen does. (Qwen's transform hardcodes the
   "You are Qwen, a virtual human..." string because Qwen's
   default-system check matches against that exact text. Pangu has
   no such requirement.)

2. **Role tokens**: Pangu's chat template uses the Chinese role
   markers ``用户`` (5618) and ``助手`` (55407) instead of Qwen's
   ``user`` / ``assistant``. The label-masking algorithm is the
   same (slice between assistant and next user) but the tokens
   queried in the vocab differ.

3. **Sample input shape**: Pangu samples in OCRBench-style oracle
   format have ``image_paths`` / ``audio_paths`` as direct keys
   (not nested under ``conversations``). For SFT we expect the
   richer ``conversations`` schema; ``conv_preprocess`` from
   ``veomni.data.multimodal.conv_preprocess`` is reused unchanged
   to normalize either layout into the per-turn ``[role, [(type,
   payload), ...]]`` tuple format the rest of the function expects.

The transform is a *closure-free* function so worker processes can
pickle it across the dataloader fork boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

import torch

from veomni.utils.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX


if TYPE_CHECKING:
    from transformers import ProcessorMixin


def _get_pangu_omni_token_ids(processor: "ProcessorMixin") -> tuple[int, int, int]:
    """Resolve Pangu Omni multimodal pad-token IDs from the processor's
    tokenizer vocab. Pangu uses fixed token names (no fallback aliases
    like Qwen's ``<|IMAGE|>`` legacy form), so this is the only valid
    name set:

      - ``<|image_pad|>``  -> 148909
      - ``<|video_pad|>``  -> 148910
      - ``<|audio_pad|>``  -> 148911

    Raised as ``ValueError`` rather than ``KeyError`` so the error
    surfaces in VeOmni's data-worker logs with a useful message rather
    than a bare traceback.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    vocab = tokenizer.get_vocab()
    missing = []
    image_token_id = vocab.get("<|image_pad|>")
    if image_token_id is None:
        missing.append("<|image_pad|>")
    video_token_id = vocab.get("<|video_pad|>")
    if video_token_id is None:
        missing.append("<|video_pad|>")
    audio_token_id = vocab.get("<|audio_pad|>")
    if audio_token_id is None:
        missing.append("<|audio_pad|>")
    if missing:
        raise ValueError(
            f"Pangu Omni data transform: tokens {missing!r} not found in tokenizer vocab. "
            "Expected Pangu Omni v2 tokenizer at "
            "/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model (or equivalent)."
        )
    return image_token_id, video_token_id, audio_token_id


def _get_pangu_omni_role_token_ids(processor: "ProcessorMixin") -> tuple[int, int]:
    """Resolve ``用户`` (user) and ``助手`` (assistant) role token IDs.
    Used to slice assistant turns for label masking. The Pangu chat
    template emits these as standalone tokens between
    ``<|message_start|>`` and ``：``.

    IMPORTANT: Pangu's ``SophonTokenizerFast.get_vocab()`` does not
    expose the BPE-merged Chinese surface forms in its vocab dict,
    even though those exact IDs decode back to the strings. So we
    encode the literal strings instead — which goes through the
    full tokenizer pipeline and returns the canonical single-token
    ID (5618 / 55407 on the production tokenizer). A sanity check
    asserts single-token encoding because the labels-masking logic
    assumes one role token per assistant turn.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    user_ids = tokenizer.encode("用户", add_special_tokens=False)
    assistant_ids = tokenizer.encode("助手", add_special_tokens=False)
    if len(user_ids) != 1 or len(assistant_ids) != 1:
        raise ValueError(
            "Pangu Omni data transform: '用户' / '助手' did not encode to single tokens. "
            f"Got user={user_ids!r}, assistant={assistant_ids!r}. Tokenizer is "
            f"{type(tokenizer).__name__}; expected SophonTokenizerFast (Pangu Omni v2)."
        )
    return user_ids[0], assistant_ids[0]


def process_sample_openpangu_omni(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: Callable,
    **kwargs,
) -> list[Dict[str, torch.Tensor]]:
    """Pangu Omni v2 per-sample data transform.

    See module docstring for the diff against Qwen2.5-Omni's transform.
    Output shape matches what
    ``veomni.data.data_collator.MainCollator`` expects when
    ``VLMTrainer._build_collate_fn`` populates
    ``data_collate_info={"audio_feature_lengths": (0, False, None,
    None), "input_features": (0, True, 0, 1), "audio_mask": (-1, False,
    0, 1)}`` (Phase 4: same as the Qwen omni branch — see
    ``vlm_trainer.py:_build_collate_fn``).
    """
    from veomni.data.multimodal import conv_preprocess
    from veomni.data.multimodal.audio_utils import fetch_audios
    from veomni.data.multimodal.image_utils import fetch_images
    from veomni.data.multimodal.video_utils import fetch_videos

    image_token_id, video_token_id, audio_token_id = _get_pangu_omni_token_ids(processor)

    # Normalize input layout (oracle-style ``image_paths`` vs SFT
    # ``conversations`` field) into the canonical per-turn tuple form.
    source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")
    conversations = sample["conversations"] if ("conversations" in sample and len(sample["conversations"])) else sample
    conversations = conv_preprocess(source, conversations, **kwargs)

    # Build the messages list. Pangu has no synthetic system message —
    # the chat template handles the empty-system case on its own.
    input_conversations: list[dict[str, Any]] = []
    for conversation in conversations:
        contents = []
        for message in conversation[1:]:
            contents.append({"type": message[0], message[0]: message[1]})
        input_conversations.append(
            {
                "role": conversation[0],
                "content": contents,
            }
        )
    text = processor.apply_chat_template(input_conversations, tokenize=False)

    # Media loading. ``fetch_videos`` also returns the video-track
    # audios so we can route them through the audio tower together
    # with standalone audio samples — same convention as
    # ``process_sample_qwen_omni``.
    images = sample.get("images", []) or []
    if images:
        images = fetch_images(images, **kwargs)

    videos = sample.get("videos", []) or []
    video_audios: list[Any] = []
    if videos:
        videos, video_audios = fetch_videos(videos, **kwargs)

    audios_raw = sample.get("audios", []) or []
    if audios_raw:
        audio_audios = fetch_audios(audios_raw, **kwargs)
    else:
        audio_audios = []

    # Re-thread audios in the same order the chat template visits
    # video / audio contents — the processor relies on positional
    # correspondence between ``audios`` and the audio tokens in the
    # templated text.
    video_audios_iter = iter(video_audios)
    audio_audios_iter = iter(audio_audios)
    audios: list[Any] = []
    for item in input_conversations:
        for content in item["content"]:
            if content["type"] == "video":
                audios.append(next(video_audios_iter))
            elif content["type"] == "audio":
                audios.append(next(audio_audios_iter))

    # Pangu's processor uses ``audio=`` (singular) — not the Qwen-Omni
    # ``audios=`` plural — and it crashes when handed an empty list
    # for any of ``images`` / ``videos`` / ``audio`` (HF's
    # ``make_batched_videos`` does ``videos[0]`` unconditionally; the
    # image / audio processors are similarly intolerant of ``[]``).
    # Map ``[]`` -> ``None`` per modality so the processor's
    # ``if X is not None:`` branches short-circuit cleanly. See
    # ``processor_openpangu_omni.py:OpenPanguOmniProcessor.__call__``.
    model_inputs = processor(
        text=text,
        audio=audios if audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = model_inputs.data
    input_features = model_inputs.pop("input_features", None)
    feature_attention_mask = model_inputs.pop("feature_attention_mask", None)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        valid_mask = audio_feature_lengths != 0
        # Re-pack ``(B, n_mels, T_padded)`` -> ``(n_mels, T_unpadded_total)``
        # exactly the way ``OpenPanguOmniModel.get_audio_features``
        # expects it when ``feature_attention_mask`` is None at call
        # time (model already sees flat features + per-audio lengths).
        #
        # NOTE on the trailing ``.permute(1, 0)``: Pangu's audio_tower
        # forward does ``input_features.transpose(-1, -2).unsqueeze(0)``
        # to land in ``(1, T_total, n_mels)`` and then slices on dim 1
        # using ``feature_lens`` cumsums. So the INPUT to audio_tower
        # must be ``(n_mels, T_total)`` — same convention as the
        # ``feature_attention_mask is not None`` branch inside
        # ``modeling_openpangu_omni.py:get_audio_features``. Qwen2.5-Omni
        # uses the opposite ``(T_total, n_mels)`` layout (its audio
        # encoder transposes differently), which is why the upstream
        # ``process_sample_qwen_omni`` transform omits this final
        # ``.permute(1, 0)``.
        input_features = (
            input_features[valid_mask].permute(0, 2, 1)[feature_attention_mask[valid_mask].bool()].permute(1, 0)
        )
        model_inputs["input_features"] = input_features
        model_inputs["audio_feature_lengths"] = audio_feature_lengths
    else:
        audio_feature_lengths = None

    # Replace the raw multimodal token IDs with VeOmni sentinels —
    # ``get_position_id_func`` matches against the sentinels (see
    # ``modeling_openpangu_omni.py:get_position_id_func``). Sentinels
    # are later zeroed out before the forward (the model fills them
    # via ``masked_scatter`` from the encoder outputs).
    input_ids = model_inputs["input_ids"].squeeze(0)
    image_mask = input_ids == image_token_id
    video_mask = input_ids == video_token_id
    audio_mask = input_ids == audio_token_id
    input_ids[image_mask] = IMAGE_INPUT_INDEX
    input_ids[video_mask] = VIDEO_INPUT_INDEX
    input_ids[audio_mask] = AUDIO_INPUT_INDEX

    # Per-token rope ids (MRoPE: 3 axes for image/video, audio TF axis).
    model_inputs["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs.get("image_grid_thw", None),
        video_grid_thw=model_inputs.get("video_grid_thw", None),
        attention_mask=model_inputs["attention_mask"],
        audio_seqlens=audio_feature_lengths,
    )["position_ids"]
    model_inputs["position_ids"] = model_inputs["position_ids"].clone()

    # Multimodal masks for the collator and the forward path.
    model_inputs["image_mask"] = image_mask
    model_inputs["video_mask"] = video_mask
    model_inputs["audio_mask"] = audio_mask
    # Zero-out sentinels in the actual input_ids — model expects
    # placeholder zeros at multimodal positions; embeddings come from
    # the encoders via masked_scatter.
    input_ids[image_mask | video_mask | audio_mask] = 0
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)

    # Label masking — keep only assistant-turn content tokens, mask
    # everything else (system, user, role markers, ``：``) with
    # ``IGNORE_INDEX`` so cross-entropy ignores them. Logic mirrors
    # ``process_sample_qwen_omni`` line by line: the only delta is
    # the Pangu-specific role-token IDs.
    user_token_id, assistant_token_id = _get_pangu_omni_role_token_ids(processor)
    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    user_positions = torch.where(input_ids == user_token_id)[0].tolist()
    assistant_positions = torch.where(input_ids == assistant_token_id)[0].tolist()
    # Sentinel for "no more user turns" so the inner loop terminates
    # cleanly when the last turn is an assistant turn (= eval-style
    # samples). +1 so the slice ``assis_i + 2 : user_start_index[user_i] - 1``
    # extends to the end of input_ids.
    user_positions.append(len(input_ids) + 1)
    user_i = 0
    for assis_i in assistant_positions:
        while user_positions[user_i] < assis_i:
            user_i += 1
        # ``assis_i + 2``: skip the ``助手`` role token (at assis_i)
        # and the ``：`` separator (at assis_i + 1).
        # ``user_positions[user_i] - 1``: stop one token before
        # ``<|message_start|>`` of the next user turn (so the
        # ``<|message_end|>`` of the assistant turn IS labeled —
        # we want the model to learn to emit it).
        labels[assis_i + 2 : user_positions[user_i] - 1] = input_ids[assis_i + 2 : user_positions[user_i] - 1]
    model_inputs["labels"] = labels
    return [model_inputs]


def pangu_mm_sft_preprocess(conversations, **kwargs):
    """Source-preprocessor for the Pangu Omni v2 SFT toy / smoke
    corpus.

    Input shape (one conversation per JSONL line; the dataloader
    splits on lines)::

        {
          "source": "pangu_mm_sft_v1",
          "conversations": [
            {"from": "human", "value": "<image><audio>What's going on?"},
            {"from": "gpt", "value": "There's an OCR sample and a glass-shatter sound."},
            ...
          ],
          "images": ["/abs/path/to/img.png", ...],
          "audios": ["/abs/path/to/audio.mp3", ...],
          "videos": []
        }

    Output (the canonical VeOmni ``[role, (type, payload), ...]``
    per-turn shape consumed by
    ``process_sample_openpangu_omni`` above). Each ``<image>`` /
    ``<audio>`` / ``<video>`` placeholder in a user-turn ``value`` is
    extracted into a separate content slot — the actual media data
    is loaded later from ``sample["images"]`` / ``sample["audios"]``
    / ``sample["videos"]`` by ``fetch_images`` / ``fetch_audios`` /
    ``fetch_videos``, ordered positionally to match the slot order
    here.

    Assistant turns are passed through as plain text (no media on
    the assistant side — Pangu Omni v2 doesn't have a speech-output
    talker module).
    """
    import re

    constructed = []
    placeholder_re = re.compile(r"<image>|<audio>|<video>")
    for message in conversations:
        # Accept both sharegpt ``from``/``value`` and chat-format
        # ``role``/``content`` so the same preprocessor works on
        # toy + real SFT datasets without a second preprocessor file.
        if "from" in message:
            role_raw = message["from"]
            text = message.get("value", "")
        else:
            role_raw = message["role"]
            text = message.get("content", "")
            if isinstance(text, list):
                # Already structured — flatten to "<placeholder>text" form.
                # In practice the toy/SFT samples use the sharegpt form
                # so this branch rarely fires; keep it for forward-compat
                # with HF chat-message-format datasets.
                parts = []
                for item in text:
                    t = item.get("type")
                    if t in ("image", "audio", "video"):
                        parts.append(f"<{t}>")
                    else:
                        parts.append(item.get("text") or item.get(t, "") or "")
                text = "".join(parts)
        role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}.get(role_raw, role_raw)

        if role == "assistant":
            constructed.append([role, ("text", text)])
            continue

        # User turn — split on <image>/<audio>/<video> markers so the
        # content slots interleave with text in the order the user wrote.
        turn = [role]
        last = 0
        for m in placeholder_re.finditer(text):
            if m.start() > last:
                segment = text[last : m.start()]
                if segment:
                    turn.append(("text", segment))
            slot = m.group(0)[1:-1]
            turn.append((slot, None))
            last = m.end()
        if last < len(text):
            tail = text[last:]
            if tail:
                turn.append(("text", tail))
        if len(turn) == 1:
            # No text content at all — append an empty text node so
            # ``process_sample_openpangu_omni`` doesn't emit an
            # empty content list (the chat template requires at
            # least one content item per turn).
            turn.append(("text", ""))
        constructed.append(turn)
    return constructed


def register_pangu_omni_data_transform() -> None:
    """Register ``openpangu_omni`` in
    ``veomni.data.data_transform.DATA_TRANSFORM_REGISTRY``, plus the
    ``pangu_mm_sft_v1`` source preprocessor in
    ``veomni.data.multimodal.preprocess.PREPROCESSOR_REGISTRY``.

    Idempotent: ``Registry.register`` overwrites silently in
    ``_global_mapping``, so re-running this at import time is a
    no-op as long as the function object identity doesn't change.
    """
    from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY
    from veomni.data.multimodal.preprocess import PREPROCESSOR_REGISTRY

    DATA_TRANSFORM_REGISTRY.register("openpangu_omni")(process_sample_openpangu_omni)
    PREPROCESSOR_REGISTRY.register("pangu_mm_sft_v1")(pangu_mm_sft_preprocess)
