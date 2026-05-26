"""Conftest for ``pangu_omni_v2`` adapter integration tests.

Loads ``transformers`` reference-modeling compat shims before any test
module is collected so the Pangu reference files at
``/mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model/{configuration,modeling}_*.py``
import cleanly on hosts that pin older ``transformers`` versions.

These shims affect parity tests only — production code does NOT depend on
them. See ``veomni/models/transformers/_pangu_common/_test_compat.py`` for
the full rationale.
"""

from veomni.models.transformers._pangu_common._test_compat import (
    install_pangu_reference_compat_shims,
)


install_pangu_reference_compat_shims()


def _prime_pangu_omni_v2_modeling() -> None:
    """Force the registry dispatcher to load the modeling modules in the
    correct order **before** any parity test does a direct
    ``from .pangu_omni_v2 import modeling_vl`` style import.

    Background — there is a 3-way circular import inside the adapter:

    ``modeling_vl`` -> ``modeling_text`` -> ``modeling_vl``

    The cycle is broken **only** when ``modeling_text`` is loaded
    first, which is exactly what
    ``__init__.py::register_pangu_omni_v2_modeling`` does (see the
    long comment around line 90-122 in ``pangu_omni_v2/__init__.py``).

    Parity tests historically bypass the dispatcher and hit
    ``modeling_vl`` directly via ``_ours_module()``, which
    starts the cycle from the wrong end and crashes with
    ``ImportError: cannot import name 'OpenPanguVL' from partially
    initialized module``.

    Calling the dispatcher once here makes the modeling modules cached in
    ``sys.modules`` in the correct order; subsequent direct imports in
    any test file resolve cleanly. Idempotent — repeated dispatcher
    invocations just return the registered class.

    Architecture chosen here is arbitrary; ``OpenPanguVLForConditionalGeneration``
    is one of the four production strings (see ``__init__.py``). Any of
    them triggers the same import block.
    """
    from veomni.models.transformers.pangu_omni_v2 import (
        register_pangu_omni_v2_modeling,
    )

    register_pangu_omni_v2_modeling("OpenPanguVLForConditionalGeneration")


_prime_pangu_omni_v2_modeling()
