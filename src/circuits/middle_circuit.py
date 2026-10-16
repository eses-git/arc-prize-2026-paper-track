"""Middle circuit: object decomposition and tactical sub-goal blueprinting.

This module implements the *middle* tier of the Three-Tier Hierarchical
Cognitive Architecture.  It does not attempt a full solution; instead it
disassembles each training grid into low-level "objects" and summarises the
geometry and colour statistics that a code generator needs to know:

    * **Object bounding boxes** -- connected components of non-background
      cells, described by their bounding boxes, centroids, and areas.
    * **Colour distributions** -- the exact ``colour -> count`` histogram of
      every grid.
    * **Spatial layouts** -- grid shapes, object counts, and how those change
      between each input and its corresponding output.

The output is a single structured string ("tactical sub-goal blueprint") that
is pasted into the short circuit's prompt.  Connected-component labelling uses
SciPy's ``scipy.ndimage.label`` when available and transparently falls back to
a pure-NumPy breadth-first labelling routine otherwise, so the module never
hard-fails on a SciPy-less environment.
"""

from __future__ import annotations

import numpy as np

from visualizer import format_grid_dual

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except ImportError:  # SciPy is optional; NumPy fallback below keeps us alive.
    _ndi = None
    _HAS_SCIPY = False

__all__ = ["decompose_objects"]

# ARC convention: colour 0 is the background and does not form objects.
_BACKGROUND = 0
# 4-connectivity neighbourhood (up, down, left, right) for labelling.
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _label_components_numpy(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components of non-background cells using pure NumPy.

    A breadth-first flood fill over the 4-connected neighbourhood assigns a
    unique integer label to every connected foreground region.  ARC grids are
    small (<= 30x30), so the explicit Python queue is comfortably fast.

    Args:
        grid: A 2-D integer grid.

    Returns:
        tuple[np.ndarray, int]: ``(labels, num_objects)`` where ``labels`` has
        the same shape as ``grid`` with 0 marking background and integers
        ``1..num_objects`` marking distinct objects.
    """
    mask = grid != _BACKGROUND
    labels = np.zeros(grid.shape, dtype=int)
    rows, cols = grid.shape
    next_label = 0

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_label += 1
            labels[r, c] = next_label
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in _NEIGHBOURS:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and mask[nr, nc]
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
    return labels, next_label


def _label_components(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components, preferring SciPy and falling back to NumPy.

    Args:
        grid: A 2-D integer grid.

    Returns:
        tuple[np.ndarray, int]: ``(labels, num_objects)``.
    """
    if _HAS_SCIPY:
        # scipy.ndimage.label uses 4-connectivity by default for 2-D input.
        labels, num_objects = _ndi.label(grid != _BACKGROUND)
        return labels, int(num_objects)
    return _label_components_numpy(grid)


def _object_features(labels: np.ndarray, grid: np.ndarray) -> list[dict]:
    """Extract per-object geometry from a labelled grid.

    Args:
        labels: The labelled grid returned by :func:`_label_components`.
        grid: The original grid (used for per-object colours).

    Returns:
        list[dict]: One dict per object with keys ``id``, ``bbox``,
        ``centroid``, ``area``, and ``colours``.
    """
    features: list[dict] = []
    num_objects = int(labels.max())
    for obj_id in range(1, num_objects + 1):
        ys, xs = np.where(labels == obj_id)
        colours = sorted({int(grid[y, x]) for y, x in zip(ys, xs)})
        features.append(
            {
                "id": obj_id,
                "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
                "centroid": (round(float(ys.mean()), 2), round(float(xs.mean()), 2)),
                "area": int(len(ys)),
                "colours": colours,
            }
        )
    return features


def _colour_distribution(grid: np.ndarray) -> dict[int, int]:
    """Return a ``colour -> count`` histogram for every cell in ``grid``.

    Args:
        grid: A 2-D integer grid.

    Returns:
        dict[int, int]: Mapping from colour value to its cell count.
    """
    values, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _indent(text: str, spaces: int) -> str:
    """Indent every non-empty line of ``text`` by ``spaces`` spaces.

    Args:
        text: The multi-line string to indent.
        spaces: Number of leading spaces to add.

    Returns:
        str: The indented string.
    """
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def decompose_objects(train_pairs: list[dict]) -> str:
    """Decompose the training grids into objects and emit a tactical blueprint.

    For every training pair this computes, for both the input and output grids,
    the shape, colour distribution, object count, and per-object geometry
    (bounding box, centroid, area, colours).  The results are formatted into a
    single structured, human-readable string that the short circuit can paste
    directly into its prompt.

    Args:
        train_pairs: A list of training examples, each a dict with ``"input"``
            and ``"output"`` keys holding 2-D ``list[list[int]]`` grids.

    Returns:
        str: A structured tactical sub-goal blueprint.

    Raises:
        ValueError: If ``train_pairs`` is empty.
    """
    if not train_pairs:
        raise ValueError("decompose_objects requires at least one training pair.")

    lines: list[str] = [
        "Tactical Sub-goal Blueprint",
        "==========================",
        f"Training pairs analysed: {len(train_pairs)}",
        "",
    ]

    for index, pair in enumerate(train_pairs):
        in_arr = np.asarray(pair["input"])
        out_arr = np.asarray(pair["output"])

        in_labels, in_count = _label_components(in_arr)
        out_labels, out_count = _label_components(out_arr)

        lines.append(f"Pair {index}")
        lines.append("-" * 40)
        lines.append(f"  input  shape : {in_arr.shape}")
        lines.append(f"  output shape : {out_arr.shape}")
        lines.append("  input  layout:")
        lines.append(_indent(format_grid_dual(pair["input"]), 4))
        lines.append("  output layout:")
        lines.append(_indent(format_grid_dual(pair["output"]), 4))
        lines.append(f"  input  colour distribution : {_colour_distribution(in_arr)}")
        lines.append(f"  output colour distribution : {_colour_distribution(out_arr)}")
        lines.append(f"  input  objects : {in_count}")
        lines.append(f"  output objects : {out_count}")

        for side, labels, grid in (
            ("input", in_labels, in_arr),
            ("output", out_labels, out_arr),
        ):
            objects = _object_features(labels, grid)
            if not objects:
                lines.append(f"  {side}: no foreground objects (blank grid)")
                continue
            lines.append(f"  {side} objects:")
            for obj in objects:
                r0, c0, r1, c1 = obj["bbox"]
                cy, cx = obj["centroid"]
                lines.append(
                    f"    obj {obj['id']}: bbox=({r0},{c0})-({r1},{c1}), "
                    f"centroid=({cy},{cx}), area={obj['area']}, "
                    f"colours={obj['colours']}"
                )
        lines.append("")

    # -- tactical sub-goals --------------------------------------------------
    lines.append("Tactical sub-goals")
    lines.append("------------------")
    lines.append("1. Identify the foreground objects in each input (see bounding boxes).")
    lines.append("2. Determine how each object's geometry/colour changes between input and output.")
    lines.append("3. Derive a per-object transform (move / recolour / rescale) and compose them.")
    lines.append("4. Validate that object count and colour distributions match the outputs.")

    return "\n".join(lines)

