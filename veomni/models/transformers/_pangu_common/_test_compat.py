"""Test-only compat shims for Pangu reference modeling imports.

This module is NOT imported by production code paths
(``_pangu_common/__init__.py`` and ``pangu_omni_v2/__init__.py`` do not pull
it in). It is only loaded from the two adapter ``tests/conftest.py`` files,
so installing it does not affect VeOmni runtime behaviour.

Why these shims exist
=====================

The Pangu reference modeling files at
``/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model/{configuration,modeling}_*.py``
are maintained by the Pangu pretraining team and track upstream
``transformers`` head (5.0+). Many adapter parity tests load the reference
module via HF dynamic-module machinery (``AutoConfig.from_pretrained(
trust_remote_code=True)``) so we can diff our port against the reference
output bit-for-bit.

If the host environment pins an older ``transformers`` (e.g. 4.57.x, as the
AReaL venv used to), several 5.0+ symbols are missing and the reference
``import`` blows up. The shims here backfill those symbols as identity
decorators / type aliases so the reference module imports cleanly. Our
adapter port does NOT depend on any of these shimmed symbols at runtime —
it is purely a test-time compatibility layer.

Symbols backfilled:

- ``transformers.modeling_rope_utils.RopeParameters`` (5.0+ TypedDict) →
  aliased to ``dict`` (type annotation only, structurally compatible).
- ``transformers.integrations.{use_experts_implementation,
  use_kernel_forward_from_hub, use_kernel_func_from_hub,
  use_kernelized_func}`` (5.0+ Hub-kernel decorators) → identity
  decorators (reference falls back to pure PyTorch, which matches our
  port).
- ``transformers.utils.generic.maybe_autocast`` (5.0+) → aliased to
  ``torch.amp.autocast`` (same kwargs).
- ``transformers.utils.auto_docstring`` (5.0+ + PEP-604 incompatibility
  in 4.57.1) → identity decorator (hybrid bare / factory dispatch).
- ``torch_npu`` placeholder for hosts without a real NPU install —
  reference modules unconditionally ``import torch_npu`` at module top.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


class _AutoMockModule(types.ModuleType):
    """Module subclass whose attribute access returns ``MagicMock()``."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return MagicMock()


def _bare_identity_decorator(fn_or_cls):
    """``@use_experts_implementation`` (bare; no parentheses)."""
    return fn_or_cls


def _factory_identity_decorator(*_args, **_kwargs):
    """``@use_kernel_forward_from_hub("RMSNorm")`` /
    ``@use_kernel_func_from_hub("rotary_pos_emb")`` /
    ``@use_kernelized_func(apply_rotary_pos_emb)``.

    Always called as a factory — returns an identity decorator regardless
    of args. A heuristic that auto-detects bare vs. factory based on
    ``callable(args[0])`` cannot work here because
    ``@use_kernelized_func(fn)`` passes a callable as its factory argument
    and would be indistinguishable from ``@bare_decorator`` taking a
    function as its decoration target.
    """

    def _inner(fn_or_cls):
        return fn_or_cls

    return _inner


def install_pangu_reference_torch_npu_mock() -> None:
    """Install a ``torch_npu`` ``sys.modules`` placeholder for reference imports.

    The Pangu reference ``modeling_openpangu_vl.py`` / ``modeling_pangu_omni.py``
    unconditionally ``import torch_npu`` at module top, which would
    ``ModuleNotFoundError`` on a GPU/CPU host. Reference uses ``torch_npu``
    only on NPU-fast-paths that our tests do not exercise (we do not have
    NPU hardware in CI), so a placeholder module is enough.

    Call site: **inside the helpers that load the reference modeling
    module (e.g. ``_load_reference_vl_module()``), AFTER VeOmni has been
    imported.** VeOmni's ``veomni/utils/device.py`` runs
    ``IS_NPU_AVAILABLE = is_torch_npu_available()`` at module load, which
    in turn calls ``importlib.util.find_spec("torch_npu")``. If our mock
    is in ``sys.modules`` BEFORE VeOmni's first import, VeOmni would see
    ``torch_npu`` as installed and crash on
    ``torch.npu.config.allow_internal_format = False``. Doing the mock
    install AFTER VeOmni has computed ``IS_NPU_AVAILABLE = False`` avoids
    that race.

    Idempotent. On a real NPU host with the genuine ``torch_npu``
    installed, this is a no-op (the genuine package wins via the
    early-return).
    """
    if "torch_npu" in sys.modules:
        return

    import importlib.machinery as _machinery

    npu_mod = _AutoMockModule("torch_npu")
    # ``transformers``' ``is_torch_npu_available()`` calls
    # ``importlib.util.find_spec("torch_npu")``, which raises
    # ``ValueError: __spec__ is None`` when the module is in
    # ``sys.modules`` without a ModuleSpec. Provide one.
    npu_mod.__spec__ = _machinery.ModuleSpec("torch_npu", loader=None)
    sys.modules["torch_npu"] = npu_mod

    contrib_mod = _AutoMockModule("torch_npu.contrib")
    contrib_mod.__spec__ = _machinery.ModuleSpec("torch_npu.contrib", loader=None)
    sys.modules["torch_npu.contrib"] = contrib_mod


def install_pangu_reference_compat_shims() -> None:
    """Backfill ``transformers`` 5.0+ symbols used by the Pangu reference.

    Idempotent. When the host upgrades to ``transformers>=5.0``, every
    branch turns into a no-op because the real symbols already exist.
    """
    try:
        import transformers.modeling_rope_utils as _rope_utils
    except ImportError:
        return
    if not hasattr(_rope_utils, "RopeParameters"):
        _rope_utils.RopeParameters = dict

    try:
        import transformers.integrations as _integrations
    except ImportError:
        return
    if not hasattr(_integrations, "use_experts_implementation"):
        _integrations.use_experts_implementation = _bare_identity_decorator
    for _name in (
        "use_kernel_forward_from_hub",
        "use_kernel_func_from_hub",
        "use_kernelized_func",
    ):
        if not hasattr(_integrations, _name):
            setattr(_integrations, _name, _factory_identity_decorator)

    try:
        import transformers.utils.generic as _generic
    except ImportError:
        return
    if not hasattr(_generic, "maybe_autocast"):
        # ``maybe_autocast(device_type, enabled=False, ...)`` ships in
        # transformers 5.0 as a thin wrapper around ``torch.amp.autocast``
        # with the same kwargs. Aliasing directly avoids re-implementing
        # the wrapper.
        import torch as _torch

        _generic.maybe_autocast = _torch.amp.autocast

    # transformers 4.57.1's ``auto_docstring`` decorator crashes when
    # introspecting function signatures that use PEP-604 union types
    # (``A | B | None`` rather than ``Optional[Union[A, B]]``) — its
    # ``param_type.__name__`` access at ``auto_docstring.py:1311`` raises
    # ``AttributeError: 'types.UnionType' object has no attribute '__name__'``.
    # The Pangu reference modeling files use PEP-604 throughout
    # (matching transformers 5.0 style), so its ``@auto_docstring``-decorated
    # classes fail to import under 4.57.1.
    #
    # Reference uses TWO call patterns:
    # - ``@auto_docstring`` (bare) on most classes/methods
    # - ``@auto_docstring(custom_intro="...")`` (factory with kwargs) on
    #   specific ``ModelOutput`` subclasses
    #
    # The hybrid shim disambiguates by call shape:
    # - ``args=(single_callable,)``, ``kwargs=empty`` → bare → return ``args[0]``
    # - else → factory → return identity decorator
    #
    # This loses upstream-generated docstrings on the reference module
    # only — our adapter port has hand-written docstrings already.
    def _auto_docstring_shim(*args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]

        def _inner(fn_or_cls):
            return fn_or_cls

        return _inner

    import transformers.utils as _tutils

    _tutils.auto_docstring = _auto_docstring_shim
    try:
        import transformers.utils.auto_docstring as _ad_mod

        _ad_mod.auto_docstring = _auto_docstring_shim
    except ImportError:
        pass
