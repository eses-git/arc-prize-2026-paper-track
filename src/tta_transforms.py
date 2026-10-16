"""Test-Time Augmentation (TTA) symmetry transforms for ARC grids.

TTA evaluates a candidate solver on multiple transformed copies of the same
input and then votes / reverses the outputs, which both stabilises the answer
and enlarges the effective training signal.  This module exposes the 8
dihedral symmetries of the square, generated as the composition of the 4
rotations (``np.rot90`` with ``k = 0..3``) and an optional horizontal
reflection (``np.fliplr``):

    identity, rot90, rot180, rot270,
    fliplr, fliplr_rot90, fliplr_rot180, fliplr_rot270

For square grids these form the standard 8-element dihedral group.  For
rectangular grids the rotations transpose the dimensions, but the forward and
inverse transforms remain exact inverses of one another, so a candidate output
can always be mapped back to the original orientation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TRANSFORM_TYPES",
    "apply_tta_transform",
    "generate_tta_variants",
    "reverse_tta_transform",
]

# The 8 TTA transform names, in a stable, documented order.
TRANSFORM_TYPES: tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "fliplr",
    "fliplr_rot90",
    "fliplr_rot180",
    "fliplr_rot270",
)


def _apply(grid: np.ndarray, transform_type: str) -> np.ndarray:
    """Apply one TTA transform to a 2-D array.

    Args:
        grid: A 2-D NumPy array.
        transform_type: One of :data:`TRANSFORM_TYPES`.

    Returns:
        np.ndarray: The transformed grid.

    Raises:
        ValueError: If ``transform_type`` is not recognised.
    """
    if transform_type == "identity":
        return grid
    if transform_type == "rot90":
        return np.rot90(grid, 1)
    if transform_type == "rot180":
        return np.rot90(grid, 2)
    if transform_type == "rot270":
        return np.rot90(grid, 3)
    if transform_type == "fliplr":
        return np.fliplr(grid)
    if transform_type == "fliplr_rot90":
        return np.fliplr(np.rot90(grid, 1))
    if transform_type == "fliplr_rot180":
        return np.fliplr(np.rot90(grid, 2))
    if transform_type == "fliplr_rot270":
        return np.fliplr(np.rot90(grid, 3))
    raise ValueError(f"Unknown TTA transform type: {transform_type!r}.")


def _invert(grid: np.ndarray, transform_type: str) -> np.ndarray:
    """Apply the inverse of one TTA transform to a 2-D array.

    Args:
        grid: A 2-D NumPy array (in transformed space).
        transform_type: One of :data:`TRANSFORM_TYPES`.

    Returns:
        np.ndarray: The grid mapped back to the original orientation.

    Raises:
        ValueError: If ``transform_type`` is not recognised.
    """
    if transform_type == "identity":
        return grid
    if transform_type == "rot90":
        return np.rot90(grid, 3)
    if transform_type == "rot180":
        return np.rot90(grid, 2)
    if transform_type == "rot270":
        return np.rot90(grid, 1)
    if transform_type == "fliplr":
        return np.fliplr(grid)
    if transform_type == "fliplr_rot90":
        return np.rot90(np.fliplr(grid), 3)
    if transform_type == "fliplr_rot180":
        return np.rot90(np.fliplr(grid), 2)
    if transform_type == "fliplr_rot270":
        return np.rot90(np.fliplr(grid), 1)
    raise ValueError(f"Unknown TTA transform type: {transform_type!r}.")


def apply_tta_transform(grid: list[list[int]], transform_type: str) -> list[list[int]]:
    """Apply one TTA transform to a grid and return it as a nested list.

    This is the forward counterpart to :func:`reverse_tta_transform` and lets
    callers augment a single grid (e.g. a held-out test input) in place.

    Args:
        grid: A 2-D ``list[list[int]]`` grid.
        transform_type: One of :data:`TRANSFORM_TYPES`.

    Returns:
        list[list[int]]: The transformed grid.

    Raises:
        ValueError: If ``transform_type`` is not recognised.
    """
    return _apply(np.asarray(grid), transform_type).tolist()


def generate_tta_variants(train_pairs: list[dict]) -> list[dict]:
    """Generate all 8 dihedral TTA variants of a training set.

    Each variant applies the same spatial transform to *both* the input and
    output grid of every training pair, so the pairing stays semantically
    consistent.

    Args:
        train_pairs: A list of training examples, each a dict with ``"input"``
            and ``"output"`` keys holding 2-D ``list[list[int]]`` grids.

    Returns:
        list[dict]: Eight dicts (one per transform type), each of the form::

            {
                "transform_type": "rot90",
                "pairs": [
                    {"input": [[...]], "output": [[...]]},
                    ...
                ],
            }
    """
    variants: list[dict] = []
    for transform_type in TRANSFORM_TYPES:
        transformed_pairs: list[dict] = []
        for pair in train_pairs:
            in_arr = np.asarray(pair["input"])
            out_arr = np.asarray(pair["output"])
            transformed_pairs.append(
                {
                    "input": _apply(in_arr, transform_type).tolist(),
                    "output": _apply(out_arr, transform_type).tolist(),
                }
            )
        variants.append(
            {"transform_type": transform_type, "pairs": transformed_pairs}
        )
    return variants


def reverse_tta_transform(grid: list[list[int]], transform_type: str) -> list[list[int]]:
    """Map a grid from transformed space back to the original orientation.

    This undoes the transform named by ``transform_type``, so it is the exact
    inverse of the corresponding transform applied by
    :func:`generate_tta_variants`.

    Args:
        grid: A 2-D grid produced while the input was in ``transform_type``
            space (e.g. a candidate solver's output).
        transform_type: The TTA transform name that was applied.

    Returns:
        list[list[int]]: The grid mapped back to the original orientation.

    Raises:
        ValueError: If ``transform_type`` is not recognised.
    """
    arr = np.asarray(grid)
    return _invert(arr, transform_type).tolist()
