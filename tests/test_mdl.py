"""Tests for the MDL code selector and its integration with the short circuit.

These tests verify that :func:`mdl_pruner.select_best_candidate` prefers the
simplest program (Occam's razor) and that the short circuit applies it when
multiple candidate solutions pass verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdl_pruner import select_best_candidate
from circuits.short_circuit import run_short_circuit

# Absolute path to the mock task, resolved relative to this file so the tests
# work no matter which directory pytest is invoked from.
_TASK_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_task.json"


@pytest.fixture(scope="module")
def train_pairs() -> list[dict]:
    """Load the training examples from the mock ARC task."""
    with _TASK_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)["train"]


# ---------------------------------------------------------------------------
# select_best_candidate unit tests
# ---------------------------------------------------------------------------

# A minimal, correct implementation of the sample task's horizontal mirror.
_SIMPLE = "def transform(grid):\n    return np.fliplr(grid)\n"

# The same transform written verbosely (nested loops, explicit list building).
_VERBOSE = (
    "def transform(grid):\n"
    "    result = []\n"
    "    for row in grid:\n"
    "        flipped = []\n"
    "        for i in range(len(row) - 1, -1, -1):\n"
    "            flipped.append(row[i])\n"
    "        result.append(flipped)\n"
    "    return result\n"
)


def test_select_best_candidate_prefers_simple_code() -> None:
    """The simpler candidate wins regardless of list order."""
    assert select_best_candidate([_SIMPLE, _VERBOSE]) == _SIMPLE
    assert select_best_candidate([_VERBOSE, _SIMPLE]) == _SIMPLE


def test_select_best_candidate_tolerates_syntax_errors() -> None:
    """An unparseable candidate is never selected over a valid one."""
    invalid = "def transform(grid:\n    return grid\n"
    assert select_best_candidate([invalid, _SIMPLE]) == _SIMPLE


def test_select_best_candidate_tie_breaks_by_shorter_source() -> None:
    """Equal-complexity candidates are ordered by source length."""
    short = "def transform(grid):\n    return grid\n"
    long = "def transform(grid):\n    # a redundant comment\n    return grid\n"
    assert select_best_candidate([long, short]) == short


def test_select_best_candidate_rejects_empty() -> None:
    """An empty candidate list is a programming error and raises."""
    with pytest.raises(ValueError):
        select_best_candidate([])


# ---------------------------------------------------------------------------
# short_circuit integration test
# ---------------------------------------------------------------------------

def test_short_circuit_selects_mdl_best_among_passing_candidates(
    train_pairs: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The short circuit keeps the simplest of several verified candidates."""
    monkeypatch.setattr(
        "circuits.short_circuit._call_llm_batch",
        lambda prompt, n=3: [_VERBOSE, _SIMPLE],
    )

    code, attempts = run_short_circuit(train_pairs, "hint")

    assert code == _SIMPLE
    assert attempts == 1
