"""Solution verification engine for candidate ARC transformation programs.

After the sandbox runs a candidate program, this module compares its output
against the expected outputs of every training example in a task.  The
verifier is deliberately *strict*: a candidate must reproduce every training
output exactly (100% cell-for-cell agreement) to be considered solved.  When a
candidate fails, the module returns rich, differential feedback (execution
errors, shape mismatches, per-cell error counts, and side-by-side grids) that
the agent can relay to the LLM to drive self-correction.
"""

from __future__ import annotations

import numpy as np

from sandbox import execute_generated_code

__all__ = ["verify_solution"]


def _format_grid(grid: np.ndarray) -> str:
    """Render a grid as a compact multi-line string for human-readable diffs.

    Args:
        grid: A 2-D NumPy array.

    Returns:
        str: The grid rendered row-by-row, e.g. ``"[0 1]\n[2 3]"``.
    """
    return "\n".join(str(row) for row in grid)


def verify_solution(code_string: str, train_pairs: list[dict]) -> tuple[bool, str]:
    """Verify a candidate program against every training example of a task.

    The program is run once per training pair via
    :func:`sandbox.execute_generated_code`.  A solution only passes when it
    reproduces *all* training outputs with 100% exact NumPy equality.  Any
    failure short-circuits immediately with a message that pinpoints *which*
    pair failed and *how*.

    Args:
        code_string: The candidate program source (must define ``transform``).
        train_pairs: A list of training examples, each a dict with ``"input"``
            and ``"output"`` keys holding 2-D ``list[list[int]]`` grids.

    Returns:
        tuple[bool, str]:
            * ``(True, summary)`` when the candidate matches every training
              output exactly.
            * ``(False, detail)`` when the candidate fails, where ``detail``
              describes the specific pair and the nature of the mismatch
              (execution error, shape mismatch, or per-cell differences).
    """
    # -- input validation ---------------------------------------------------
    if not isinstance(code_string, str):
        return False, f"code_string must be a str, got {type(code_string).__name__}."
    if not train_pairs:
        return False, "train_pairs is empty; there is nothing to verify against."

    # -- evaluate every training example ------------------------------------
    for index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        expected_grid = pair.get("output")

        # Guard against malformed task data before attempting execution.
        if input_grid is None or expected_grid is None:
            return False, (
                f"Training pair {index} is missing 'input' or 'output'; "
                f"got keys {sorted(pair.keys())}."
            )

        # 1) Run the candidate.  A non-None error string means the program
        #    crashed or was structurally invalid -- relay it verbatim so the
        #    LLM can fix the crash before worrying about grid contents.
        actual_grid, error = execute_generated_code(code_string, input_grid)
        if error is not None:
            return False, f"Training pair {index}: execution failed:\n{error}"

        # 2) Compare shapes first -- a shape mismatch makes cell-level
        #    comparison meaningless, so report it separately.
        expected_arr = np.asarray(expected_grid)
        actual_arr = np.asarray(actual_grid)
        if actual_arr.shape != expected_arr.shape:
            return False, (
                f"Training pair {index}: shape mismatch -- "
                f"expected {expected_arr.shape}, got {actual_arr.shape}."
            )

        # 3) Exact cell-for-cell comparison via vectorised NumPy.
        if not np.array_equal(actual_arr, expected_arr):
            differing_cells = int(np.count_nonzero(actual_arr != expected_arr))
            total_cells = int(expected_arr.size)
            return False, (
                f"Training pair {index}: {differing_cells}/{total_cells} cells differ.\n"
                f"expected:\n{_format_grid(expected_arr)}\n"
                f"got:\n{_format_grid(actual_arr)}"
            )

    # -- success ------------------------------------------------------------
    return True, (
        f"Solution verified on all {len(train_pairs)} training pair(s): 100% exact match."
    )
