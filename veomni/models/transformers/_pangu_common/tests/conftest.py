"""Conftest for ``_pangu_common`` parity tests (shared-primitive layer).

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
