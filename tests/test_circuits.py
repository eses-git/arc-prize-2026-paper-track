"""Tests for the three-tier cognitive architecture (``circuits`` package).

These tests exercise the long/middle/short circuits against the bundled mock
ARC task, covering:

    1. Taxonomy classification and macro-planning.
    2. Object decomposition and blueprint formatting.
    3. The iterative self-correction loop, including its failure path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuits.long_circuit import classify_and_plan
from circuits.middle_circuit import decompose_objects
from circuits.short_circuit import run_short_circuit
from verifier import verify_solution

# Absolute path to the mock task, resolved relative to this file so the tests
# work no matter which directory pytest is invoked from.
_TASK_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_task.json"


@pytest.fixture(scope="module")
def train_pairs() -> list[dict]:
    """Load the training examples from the mock ARC task."""
    with _TASK_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)["train"]


# ---------------------------------------------------------------------------
# long_circuit tests
# ---------------------------------------------------------------------------

def test_classify_and_plan_returns_expected_keys(train_pairs: list[dict]) -> None:
    """The plan dict carries taxonomy, confidence, signals, and a macro-plan."""
    plan = classify_and_plan(train_pairs)

    assert set(plan) == {"taxonomy", "confidence", "signals", "macro_plan"}
    assert isinstance(plan["macro_plan"], str) and plan["macro_plan"]
    assert 0.0 <= plan["confidence"] <= 1.0


def test_classify_and_plan_detects_geometric_mirror(train_pairs: list[dict]) -> None:
    """The mock task (horizontal mirror) is classified as geometric."""
    plan = classify_and_plan(train_pairs)

    assert plan["taxonomy"] == "Geometric/Rotation"
    assert "fliplr" in plan["signals"]["matched_symmetries"]


def test_classify_and_plan_rejects_empty() -> None:
    """An empty training set is a programming error and raises."""
    with pytest.raises(ValueError):
        classify_and_plan([])


# ---------------------------------------------------------------------------
# middle_circuit tests
# ---------------------------------------------------------------------------

def test_decompose_objects_returns_blueprint(train_pairs: list[dict]) -> None:
    """The blueprint is a non-trivial string describing objects and colours."""
    blueprint = decompose_objects(train_pairs)

    assert isinstance(blueprint, str)
    assert "Tactical Sub-goal Blueprint" in blueprint
    assert "colour distribution" in blueprint
    assert "bbox=" in blueprint


def test_decompose_objects_rejects_empty() -> None:
    """An empty training set is a programming error and raises."""
    with pytest.raises(ValueError):
        decompose_objects([])


# ---------------------------------------------------------------------------
# short_circuit tests
# ---------------------------------------------------------------------------

def test_run_short_circuit_solves_sample_task(train_pairs: list[dict]) -> None:
    """The default mock client self-corrects and verifies the sample task."""
    hint = "Taxonomy: Geometric/Rotation (horizontal mirror)."

    code, attempts = run_short_circuit(train_pairs, hint)

    assert code is not None
    assert attempts >= 1
    passed, _ = verify_solution(code, train_pairs)
    assert passed is True


def test_run_short_circuit_exhausts_budget_on_hopeless_client(
    train_pairs: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client that never improves exhausts the attempt budget gracefully."""
    monkeypatch.setattr(
        "circuits.short_circuit._call_llm",
        lambda prompt: "def transform(grid):\n    return grid\n",
    )

    code, attempts = run_short_circuit(train_pairs, "hint", max_attempts=3)

    assert code is None
    assert attempts == 3


def test_run_short_circuit_rejects_bad_arguments(train_pairs: list[dict]) -> None:
    """Empty data and non-positive attempt budgets are rejected loudly."""
    with pytest.raises(ValueError):
        run_short_circuit([], "hint")
    with pytest.raises(ValueError):
        run_short_circuit(train_pairs, "hint", max_attempts=0)
