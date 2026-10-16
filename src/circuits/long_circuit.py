"""Long circuit: global task taxonomy and strategic macro-planning.

This module implements the *highest* tier of the Three-Tier Hierarchical
Cognitive Architecture.  Its job is not to solve a puzzle cell-by-cell but to
step back and answer "what kind of task is this?".  By classifying the task's
global taxonomy -- Geometric/Rotation, Topological/Pathfinding, or
Counting/Scaling -- it produces a strategic macro-plan hint that biases the
lower tiers towards a productive family of transformations.

The classifier works on cheap, whole-grid statistics computed with NumPy:

    * **Dihedral symmetry**: does an output equal one of the 8 symmetries of
      the corresponding input (identity, 3 rotations, 4 reflections)?
    * **Colour preservation**: is the multiset of colours unchanged between
      input and output (suggesting objects are only *moved*)?
    * **Integer scaling**: is the output shape an integer multiple of the
      input shape (suggesting pixel doubling / tiling / counting)?

These signals are aggregated across every training pair and collapsed into a
single ``dict`` describing the taxonomy, a confidence score, the underlying
evidence, and a natural-language macro-plan for the code generator.
"""

from __future__ import annotations

import numpy as np

__all__ = ["classify_and_plan"]

# Taxonomy labels used throughout the agent.  They double as the keys into
# ``_MACRO_PLANS`` so the macro-plan always stays in lock-step with the label.
GEOMETRIC = "Geometric/Rotation"
TOPOLOGICAL = "Topological/Pathfinding"
COUNTING = "Counting/Scaling"
MIXED = "Mixed/Unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_arrays(pair: dict) -> tuple[np.ndarray, np.ndarray]:
    """Coerce one training pair's ``input``/``output`` into NumPy arrays.

    Args:
        pair: A training example with ``"input"`` and ``"output"`` keys.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(input_array, output_array)``.
    """
    return np.asarray(pair["input"]), np.asarray(pair["output"])


def _dihedral_matches(input_arr: np.ndarray, output_arr: np.ndarray) -> list[str]:
    """Return the names of the dihedral symmetries mapping input to output.

    The eight symmetries of the square are tested against the output with
    exact ``np.array_equal`` comparison.  Shapes that differ are naturally
    rejected by the equality check, so non-square grids are handled safely.

    Args:
        input_arr: The source grid.
        output_arr: The candidate transformed grid.

    Returns:
        list[str]: Names of the matching symmetries (e.g. ``["fliplr"]``).
    """
    candidates: dict[str, np.ndarray] = {
        "identity": input_arr,
        "rot90": np.rot90(input_arr, 1),
        "rot180": np.rot90(input_arr, 2),
        "rot270": np.rot90(input_arr, 3),
        "fliplr": np.fliplr(input_arr),
        "flipud": np.flipud(input_arr),
        "transpose": input_arr.T,
        "anti_transpose": np.fliplr(input_arr).T,
    }
    return [name for name, cand in candidates.items() if np.array_equal(cand, output_arr)]


def _colour_multiset(arr: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return a hashable multiset of ``(colour, count)`` pairs for a grid.

    Using a sorted tuple of ``(colour, count)`` pairs makes multisets directly
    comparable with ``==`` while remaining order-independent.

    Args:
        arr: A 2-D grid.

    Returns:
        tuple[tuple[int, int], ...]: Sorted ``(colour, count)`` pairs.
    """
    values, counts = np.unique(arr, return_counts=True)
    return tuple(sorted(zip(values.tolist(), counts.tolist())))


def _integer_scale(in_shape: tuple[int, int], out_shape: tuple[int, int]) -> bool:
    """Return ``True`` if either shape is an integer multiple of the other.

    This detects the "pixel doubling / tiling / scaling" family: one grid's
    height and width are each a positive integer multiple of the other's.
    Equal shapes are explicitly excluded so a plain recolour is *not* mistaken
    for a scaling operation.

    Args:
        in_shape: ``(rows, cols)`` of the input grid.
        out_shape: ``(rows, cols)`` of the output grid.

    Returns:
        bool: ``True`` when a non-trivial whole-integer scale relationship
        exists.
    """
    if in_shape == out_shape:
        return False

    in_h, in_w = in_shape
    out_h, out_w = out_shape

    # Output is an up-scale of the input.
    if in_h > 0 and in_w > 0 and out_h % in_h == 0 and out_w % in_w == 0:
        return True
    # Input is a down-scale of the output.
    if out_h > 0 and out_w > 0 and in_h % out_h == 0 and in_w % out_w == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Macro-plan templates (kept in sync with the taxonomy labels above)
# ---------------------------------------------------------------------------

_MACRO_PLANS: dict[str, str] = {
    GEOMETRIC: (
        "Hypothesise a rigid geometric transform.  Search the 8 dihedral "
        "symmetries (identity, 3 rotations, 4 reflections) plus translations; "
        "the transform is likely applied globally to the whole grid."
    ),
    TOPOLOGICAL: (
        "Hypothesise a connectivity-preserving edit.  Focus on flood-fill, "
        "region re-colouring, path tracing, and connected-component operations; "
        "shapes may be preserved while colours or regions change."
    ),
    COUNTING: (
        "Hypothesise a count- or scale-driven transform.  Look for pixel "
        "duplication, integer up/down-scaling, tiling, or colour-count "
        "arithmetic that changes the grid's dimensions or multiplicities."
    ),
    MIXED: (
        "No single dominant signal.  Defer to the tactical object blueprint "
        "and prefer composing several small local transforms over one global "
        "rule."
    ),
}


def classify_and_plan(train_pairs: list[dict]) -> dict:
    """Classify the task's global taxonomy and produce a strategic macro-plan.

    Whole-grid statistics are gathered for every training pair and combined
    into three ratios (symmetry, colour-preservation, integer-scaling).  A
    simple, deterministic decision tree maps those ratios onto one of four
    taxonomies and selects a matching macro-plan hint.

    Args:
        train_pairs: A list of training examples, each a dict with ``"input"``
            and ``"output"`` keys holding 2-D ``list[list[int]]`` grids.

    Returns:
        dict: A strategic plan with the following keys:

            * ``"taxonomy"``: one of ``"Geometric/Rotation"``,
              ``"Topological/Pathfinding"``, ``"Counting/Scaling"``, or
              ``"Mixed/Unknown"``.
            * ``"confidence"``: a float in ``[0, 1]`` quantifying how strongly
              the evidence supports the chosen taxonomy.
            * ``"signals"``: a dict of the raw evidence ratios (``symmetry``,
              ``colour_preservation``, ``integer_scale``) plus the aggregated
              list of matched symmetry names.
            * ``"macro_plan"``: a natural-language strategic hint for the code
              generator.

    Raises:
        ValueError: If ``train_pairs`` is empty.
    """
    if not train_pairs:
        raise ValueError("classify_and_plan requires at least one training pair.")

    total = len(train_pairs)
    symmetry_hits = 0
    colour_preserved = 0
    scale_hits = 0
    matched_symmetries: list[str] = []

    # -- accumulate per-pair evidence --------------------------------------
    for pair in train_pairs:
        in_arr, out_arr = _as_arrays(pair)

        sym_names = _dihedral_matches(in_arr, out_arr)
        if sym_names:
            symmetry_hits += 1
            matched_symmetries.extend(sym_names)

        if _colour_multiset(in_arr) == _colour_multiset(out_arr):
            colour_preserved += 1

        if _integer_scale(in_arr.shape, out_arr.shape):
            scale_hits += 1

    symmetry_ratio = symmetry_hits / total
    colour_ratio = colour_preserved / total
    scale_ratio = scale_hits / total

    # -- deterministic taxonomy decision ------------------------------------
    # Order matters: a geometric symmetry is the most specific signal and wins
    # first; counting/scaling is detected by a whole-integer shape change; a
    # shape-preserving recolour that is *not* a symmetry is treated as
    # topological; anything else falls back to "Mixed".
    if symmetry_ratio >= 0.5:
        taxonomy = GEOMETRIC
        confidence = symmetry_ratio
    elif scale_ratio >= 0.5:
        taxonomy = COUNTING
        confidence = scale_ratio
    elif colour_ratio < 1.0:
        taxonomy = TOPOLOGICAL
        confidence = 1.0 - colour_ratio
    else:
        taxonomy = MIXED
        confidence = 0.0

    return {
        "taxonomy": taxonomy,
        "confidence": confidence,
        "signals": {
            "symmetry": symmetry_ratio,
            "colour_preservation": colour_ratio,
            "integer_scale": scale_ratio,
            "matched_symmetries": sorted(set(matched_symmetries)),
        },
        "macro_plan": _MACRO_PLANS[taxonomy],
    }

