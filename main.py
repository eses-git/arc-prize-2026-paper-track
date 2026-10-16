"""End-to-end entry point for the PaperTrack cognitive agent.

This script wires together the Three-Tier Hierarchical Cognitive Architecture
against the mock ARC task in ``data/sample_task.json``:

    1. ``long_circuit.classify_and_plan``   -- global taxonomy + macro-plan.
    2. ``middle_circuit.decompose_objects`` -- tactical object blueprint.
    3. ``short_circuit.run_short_circuit``  -- generate -> verify -> correct.

Run from the repository root with::

    py main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``src/`` importable as top-level modules (``sandbox``, ``verifier``,
# ``circuits``) regardless of the working directory this script is run from.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from circuits.long_circuit import classify_and_plan
from circuits.middle_circuit import decompose_objects
from circuits.short_circuit import run_short_circuit

_TASK_PATH = Path(__file__).resolve().parent / "data" / "sample_task.json"


def main() -> None:
    """Load the sample task and run the full three-tier pipeline."""
    with _TASK_PATH.open("r", encoding="utf-8") as handle:
        task = json.load(handle)

    train_pairs: list[dict] = task["train"]

    print("=" * 70)
    print("PaperTrack - Three-Tier Hierarchical Cognitive Architecture")
    print("=" * 70)

    # -- tier 1: long circuit ----------------------------------------------
    plan = classify_and_plan(train_pairs)
    print("\n[1] LONG CIRCUIT - strategic macro-plan")
    print(f"    taxonomy   : {plan['taxonomy']}")
    print(f"    confidence : {plan['confidence']:.2f}")
    print(f"    signals    : {plan['signals']}")
    print(f"    macro-plan : {plan['macro_plan']}")

    # -- tier 2: middle circuit --------------------------------------------
    blueprint = decompose_objects(train_pairs)
    print("\n[2] MIDDLE CIRCUIT - tactical sub-goal blueprint")
    print(blueprint)

    # -- tier 3: short circuit ---------------------------------------------
    strategy_hint = (
        f"[Long circuit] taxonomy={plan['taxonomy']} "
        f"(confidence {plan['confidence']:.2f})\n"
        f"Macro-plan: {plan['macro_plan']}\n\n"
        f"[Middle circuit] tactical blueprint:\n{blueprint}"
    )

    print("\n[3] SHORT CIRCUIT - iterative generate/verify/correct loop")
    code, attempts = run_short_circuit(train_pairs, strategy_hint)
    print(f"    attempts : {attempts}")
    if code is not None:
        print(f"    SOLUTION VERIFIED (after {attempts} attempt(s)):")
        print(code)
    else:
        print(f"    FAILED: no verified solution after {attempts} attempts.")


if __name__ == "__main__":
    main()
