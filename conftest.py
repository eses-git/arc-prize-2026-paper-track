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

# Absolute path to the ``src`` directory (project_root/src).
_SRC_DIR = Path(__file__).resolve().parent / "src"

# Insert once, at the front, so local modules shadow any installed packages
# that happen to share a name.
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
