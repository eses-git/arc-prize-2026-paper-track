#!/usr/bin/env python3
"""Ablation-study evaluation runner for the PaperTrack cognitive agent.

This script loads a small set of real ARC tasks from ``data/arc_eval/``
(downloading a subset first if the directory is empty), runs each task under
two ablation configurations, and writes a summary metrics report to
``data/eval_reports/``.

The two configurations isolate the contributions of the full three-tier
system:

    * **baseline**     -- a single-pass LLM code generation with *no*
      self-correction loop, *no* visual formatting, and *no* TTA.
    * **full_system**  -- the complete Long/Middle/Short circuit pipeline
      (dual visual formatting + MDL pruning) plus TTA symmetries at inference.

Metrics reported per configuration: success rate, average attempts per solved
task, and a breakdown of failure modes.  When no ``LLM_PROVIDER`` is
configured, the deterministic offline mock is used, so the script runs without
network/API access (and success rates reflect the mock, not a real model).

Usage::

    py scripts/run_evaluation.py --tasks 10 --subset training
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make ``src/`` (agent modules) and this script's directory (the downloader)
# importable regardless of the working directory the script is run from.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
_SCRIPT_DIR = Path(__file__).resolve().parent
for _dir in (_SRC_DIR, _SCRIPT_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import download_arc_data  # noqa: E402
from circuits.long_circuit import classify_and_plan  # noqa: E402
from circuits.middle_circuit import decompose_objects  # noqa: E402
from circuits.short_circuit import (  # noqa: E402
    _call_llm,
    _extract_code,
    run_short_circuit,
)
from sandbox import execute_generated_code  # noqa: E402
from tta_transforms import (  # noqa: E402
    TRANSFORM_TYPES,
    apply_tta_transform,
    reverse_tta_transform,
)

# Default destination and report locations (repo root / data / ...).
_DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "arc_eval"
_DEFAULT_REPORT = (
    Path(__file__).resolve().parents[1] / "data" / "eval_reports" / "report.json"
)


def _format_grid_numeric(grid: list[list[int]]) -> str:
    """Render a grid as a plain, space-separated numeric block (no glyphs)."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def _build_baseline_prompt(train_pairs: list[dict]) -> str:
    """Build a minimal numeric-only prompt for the single-pass baseline."""
    parts: list[str] = [
        "You are an ARC solver. Write a single function `def transform(grid):` "
        "that maps an input grid to its output grid.",
        "`np` is available; do not use imports.",
        "Training examples:",
    ]
    for index, pair in enumerate(train_pairs):
        parts.append(f"Example {index} input:")
        parts.append(_format_grid_numeric(pair["input"]))
        parts.append(f"Example {index} output:")
        parts.append(_format_grid_numeric(pair["output"]))
    parts.append("Return only the code.")
    return "\n".join(parts)


def _grid_key(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Convert a grid to a hashable tuple-of-tuples for voting/dedup."""
    return tuple(tuple(row) for row in grid)


def _majority_vote(predictions: list[list[list[int]]]) -> list[list[int]]:
    """Return the most common prediction; ties break by first appearance."""
    counts: dict[tuple[tuple[int, ...], ...], int] = {}
    for prediction in predictions:
        key = _grid_key(prediction)
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts, key=lambda key: counts[key])
    return [list(row) for row in best_key]


def _predict(code: str, input_grid: list[list[int]], use_tta: bool):
    """Run ``code`` on one input, optionally ensembling across TTA symmetries.

    Returns the predicted grid, or ``None`` if the code raised during
    execution (or every TTA orientation raised).
    """
    if not use_tta:
        output, error = execute_generated_code(code, input_grid)
        return None if error is not None else output

    predictions: list[list[list[int]]] = []
    for transform_type in TRANSFORM_TYPES:
        transformed_input = apply_tta_transform(input_grid, transform_type)
        output, error = execute_generated_code(code, transformed_input)
        if error is not None:
            continue
        predictions.append(reverse_tta_transform(output, transform_type))
    if not predictions:
        return None
    return _majority_vote(predictions)


def _baseline_generate(train_pairs: list[dict]) -> tuple[str | None, int]:
    """Single-pass baseline: one LLM call, no loop / visual / TTA / MDL."""
    response = _call_llm(_build_baseline_prompt(train_pairs))
    code = _extract_code(response)
    return (code or None), 1


def _full_generate(
    train_pairs: list[dict], max_attempts: int
) -> tuple[str | None, int]:
    """Full system: Long/Middle/Short circuits + visual + MDL (loop)."""
    plan = classify_and_plan(train_pairs)
    blueprint = decompose_objects(train_pairs)
    strategy_hint = (
        f"[Long circuit] taxonomy={plan['taxonomy']} "
        f"(confidence {plan['confidence']:.2f})\n"
        f"Macro-plan: {plan['macro_plan']}\n\n"
        f"[Middle circuit] tactical blueprint:\n{blueprint}"
    )
    return run_short_circuit(train_pairs, strategy_hint, max_attempts)


def _ensure_task_files(subset: str, dest: Path, num_tasks: int) -> list[Path]:
    """Return up to ``num_tasks`` task-file paths, downloading a subset if needed.

    If the local directory already holds at least ``num_tasks`` files, those are
    used directly; otherwise the first ``num_tasks`` tasks are fetched from the
    public ARC-AGI repository using the downloader module.
    """
    subset_dir = dest / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(subset_dir.glob("*.json"))
    if len(existing) >= num_tasks:
        return existing[:num_tasks]

    files = download_arc_data._list_task_files(subset)
    for entry in files[:num_tasks]:
        target = subset_dir / entry["name"]
        if not target.exists():
            download_arc_data._download_file(entry["download_url"], target)

    return sorted(subset_dir.glob("*.json"))[:num_tasks]


def _load_task(path: Path) -> tuple[str, dict]:
    """Load one task JSON and return ``(task_id, task)``."""
    with path.open("r", encoding="utf-8") as handle:
        return path.stem, json.load(handle)


def _evaluate(code: str | None, task: dict, use_tta: bool) -> dict:
    """Score a candidate program against a task's held-out test pairs.

    Returns:
        dict: ``{"solved": bool, "failure_mode": str | None}``.
    """
    if not code:
        return {"solved": False, "failure_mode": "no_solution"}

    test_pairs = task.get("test", [])
    if not test_pairs:
        return {"solved": False, "failure_mode": "no_test_pairs"}

    for pair in test_pairs:
        input_grid = pair.get("input")
        expected = pair.get("output")
        if input_grid is None:
            return {"solved": False, "failure_mode": "missing_test_input"}
        if expected is None:
            return {"solved": False, "failure_mode": "no_ground_truth"}

        prediction = _predict(code, input_grid, use_tta)
        if prediction is None:
            return {"solved": False, "failure_mode": "execution_error"}
        if prediction != expected:
            return {"solved": False, "failure_mode": "wrong_test_output"}

    return {"solved": True, "failure_mode": None}


def _summarize(results: list[dict]) -> dict:
    """Aggregate per-task results into a summary metrics dict."""
    total = len(results)
    solved = [r for r in results if r["solved"]]
    solved_count = len(solved)
    attempts = [r["attempts"] for r in solved]
    average_attempts = (sum(attempts) / len(attempts)) if attempts else None
    failure_modes = Counter(
        r["failure_mode"] for r in results if not r["solved"] and r["failure_mode"]
    )
    return {
        "total_tasks": total,
        "solved": solved_count,
        "success_rate": (solved_count / total) if total else 0.0,
        "average_attempts_per_solved_task": average_attempts,
        "failure_modes": dict(failure_modes),
    }


def _run_task(task_id: str, task: dict, max_attempts: int) -> dict:
    """Run both ablation configurations on a single task."""
    train_pairs = task.get("train", [])

    code_a, attempts_a = _baseline_generate(train_pairs)
    result_a = _evaluate(code_a, task, use_tta=False)
    result_a["attempts"] = attempts_a
    result_a["candidate_code"] = code_a

    code_b, attempts_b = _full_generate(train_pairs, max_attempts)
    result_b = _evaluate(code_b, task, use_tta=True)
    result_b["attempts"] = attempts_b
    result_b["candidate_code"] = code_b

    return {
        "task_id": task_id,
        "baseline": result_a,
        "full_system": result_b,
    }


def _print_summary(report: dict) -> None:
    """Print a compact human-readable summary of the report."""
    print("=" * 60)
    print("ARC evaluation summary")
    print("=" * 60)
    for name in ("baseline", "full_system"):
        summary = report["configurations"][name]
        avg = summary["average_attempts_per_solved_task"]
        avg_text = f"{avg:.2f}" if avg is not None else "n/a"
        print(f"\n[{name}]")
        print(
            f"  success rate : {summary['success_rate']:.2%} "
            f"({summary['solved']}/{summary['total_tasks']})"
        )
        print(f"  avg attempts (solved) : {avg_text}")
        print(f"  failure modes : {summary['failure_modes']}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=int, default=10, help="number of tasks to evaluate (default: 10)"
    )
    parser.add_argument(
        "--subset",
        choices=("training", "evaluation"),
        default="training",
        help="ARC-AGI subset (default: training)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=5, help="short-circuit attempts (default: 5)"
    )
    parser.add_argument(
        "--dest", type=Path, default=_DEFAULT_DEST, help="ARC data directory (default: data/arc_eval)"
    )
    parser.add_argument(
        "--report", type=Path, default=_DEFAULT_REPORT, help="output JSON report path"
    )
    args = parser.parse_args()

    if args.tasks < 1:
        parser.error("--tasks must be >= 1")

    try:
        task_paths = _ensure_task_files(args.subset, args.dest, args.tasks)
    except Exception as exc:  # noqa: BLE001 - surface any download failure cleanly
        print(f"ERROR: could not obtain ARC tasks: {exc}", file=sys.stderr)
        return 1

    tasks = [_load_task(path) for path in task_paths]
    rows = [_run_task(task_id, task, args.max_attempts) for task_id, task in tasks]

    report = {
        "meta": {
            "subset": args.subset,
            "num_tasks": len(rows),
            "max_attempts": args.max_attempts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Offline deterministic mock LLM used unless LLM_PROVIDER is configured.",
        },
        "configurations": {
            "baseline": _summarize([r["baseline"] for r in rows]),
            "full_system": _summarize([r["full_system"] for r in rows]),
        },
        "tasks": rows,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _print_summary(report)
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
