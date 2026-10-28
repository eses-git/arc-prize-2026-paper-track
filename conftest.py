"""Pytest configuration and shared fixtures.

This file is loaded automatically by pytest before test collection.  Its
primary job is to make the ``src/`` directory importable as top-level modules
(``sandbox``, ``verifier``) regardless of the working directory pytest is
invoked from.  This mirrors the ``pythonpath`` setting in ``pyproject.toml``
and acts as a belt-and-braces fallback for older tooling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Absolute path to the ``src`` directory (project_root/src).
_SRC_DIR = Path(__file__).resolve().parent / "src"

# Insert once, at the front, so local modules shadow any installed packages
# that happen to share a name.
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@pytest.fixture(autouse=True)
def _isolate_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip LLM env config so tests always use the offline mock.

    Secret loading (``.env`` / Kaggle) injects ``LLM_PROVIDER`` and
    ``LLM_API_KEY`` into ``os.environ`` at import time.  This fixture removes
    them for the duration of each test, keeping the suite hermetic and
    independent of the developer's local secrets.
    """
    for _var in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL_NAME", "LLM_BASE_URL"):
        monkeypatch.delenv(_var, raising=False)
