"""Unit and integration tests for the sandbox and verifier engines.

These tests exercise the full execution -> verification pipeline against the
mock ARC task shipped in ``data/sample_task.json`` (a 4x4 horizontal-mirror
transformation).  They cover the canonical success/failure regimes the agent's
self-correction loop relies on:

    1. Correct code that passes 100%.
    2. Code that raises a syntax error / runtime error.
    3. Code that returns a grid with the wrong shape (or wrong cell values).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sandbox import execute_generated_code
from verifier import verify_solution


# ---------------------------------------------------------------------------
# Fixtures & constants
# ---------------------------------------------------------------------------

# Absolute path to the mock task, resolved relative to this file so the tests
# work no matter which directory pytest is invoked from.
_TASK_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_task.json"


@pytest.fixture(scope="module")
def task() -> dict:
    """Load and return the mock ARC task JSON once per test module."""
    with _TASK_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def train_pairs(task: dict) -> list[dict]:
    """Extract the training examples from the loaded task."""
    return task["train"]


# A correct implementation of the horizontal-mirror transformation.  Note that
# ``np`` is injected by the sandbox -- the code must NOT try to ``import numpy``
# (imports are blocked by the restricted namespace).
CORRECT_CODE = """
def transform(grid):
    # Mirror the grid horizontally (flip each row left-to-right).
    return np.fliplr(grid)
"""

# (b) A program with a syntax error: the closing parenthesis is missing.
SYNTAX_ERROR_CODE = """
def transform(grid):
    return np.fliplr(grid
"""

# (b) A program that compiles but raises at runtime.
RUNTIME_ERROR_CODE = """
def transform(grid):
    raise RuntimeError("deliberate runtime failure")
"""

# (c) A program that returns a grid with the wrong shape (3x3 instead of 4x4).
SHAPE_MISMATCH_CODE = """
def transform(grid):
    return np.zeros((3, 3), dtype=int)
"""

# A program that returns the correct shape but the wrong cell values.
CELL_MISMATCH_CODE = """
def transform(grid):
    return np.zeros_like(grid)
"""

# A program that attempts to escape the restricted namespace via an import.
IMPORT_ESCAPE_CODE = """
import os
def transform(grid):
    return grid
"""


# ---------------------------------------------------------------------------
# sandbox.py tests
# ---------------------------------------------------------------------------

def test_valid_code_executes_and_matches_expected_output(train_pairs: list[dict]) -> None:
    """(a) Valid code: executes cleanly and returns the exact mirrored grid."""
    sample = train_pairs[0]
    output, error = execute_generated_code(CORRECT_CODE, sample["input"])

    assert error is None, f"expected no error, got: {error}"
    assert output == sample["output"]


def test_syntax_error_returns_traceback() -> None:
    """A syntax error is captured and returned as a full traceback string."""
    output, error = execute_generated_code(SYNTAX_ERROR_CODE, [[0, 1], [1, 0]])

    assert output is None
    assert error is not None
    assert "Traceback" in error
    assert "SyntaxError" in error


def test_runtime_error_returns_traceback() -> None:
    """A runtime exception is captured with its traceback and message."""
    output, error = execute_generated_code(RUNTIME_ERROR_CODE, [[0, 1], [1, 0]])

    assert output is None
    assert error is not None
    assert "Traceback" in error
    assert "RuntimeError" in error
    assert "deliberate runtime failure" in error


def test_restricted_namespace_blocks_imports() -> None:
    """The sandbox forbids arbitrary imports (``import os`` must fail)."""
    output, error = execute_generated_code(IMPORT_ESCAPE_CODE, [[0, 1], [1, 0]])

    assert output is None
    assert error is not None
    assert "ImportError" in error


def test_missing_transform_reports_diagnostic() -> None:
    """Code without a ``transform`` function yields a clear diagnostic."""
    output, error = execute_generated_code("x = 1\n", [[0, 1], [1, 0]])

    assert output is None
    assert error is not None
    assert "transform" in error


# ---------------------------------------------------------------------------
# verifier.py tests
# ---------------------------------------------------------------------------

def test_verify_solution_passes_for_correct_code(train_pairs: list[dict]) -> None:
    """Correct code verifies against 100% of the training pairs."""
    passed, message = verify_solution(CORRECT_CODE, train_pairs)

    assert passed is True, message


def test_verify_solution_fails_for_shape_mismatch(train_pairs: list[dict]) -> None:
    """(c) A wrong-shape output is reported as a shape mismatch."""
    passed, message = verify_solution(SHAPE_MISMATCH_CODE, train_pairs)

    assert passed is False
    assert "shape mismatch" in message


def test_verify_solution_fails_for_cell_mismatch(train_pairs: list[dict]) -> None:
    """A same-shape but wrong-value output reports a per-cell error count."""
    passed, message = verify_solution(CELL_MISMATCH_CODE, train_pairs)

    assert passed is False
    assert "cells differ" in message


def test_verify_solution_reports_runtime_failure(train_pairs: list[dict]) -> None:
    """An exception during verification is surfaced with the traceback."""
    passed, message = verify_solution(RUNTIME_ERROR_CODE, train_pairs)

    assert passed is False
    assert "execution failed" in message
    assert "RuntimeError" in message
