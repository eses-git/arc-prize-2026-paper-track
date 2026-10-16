"""Dual-representation grid formatter for prompts and diagnostics.

The agent passes grids to the LLM in two complementary forms.  Raw integers
are unambiguous for a code generator, but they are hard for a language model
to "see" spatially.  This module renders a 2-D integer grid in both forms:

    * **Numeric** -- the exact Python list literal, e.g. ``[[0, 1], [0, 0]]``.
    * **Visual**  -- an ASCII-art block mapping colours 0-9 to distinct glyphs
      so object shapes, symmetry, and connectivity jump out visually.

Both representations are returned from a single call to
:func:`format_grid_dual` so downstream prompt builders stay in lock-step.
"""

from __future__ import annotations

__all__ = ["format_grid_dual"]

# Colour -> glyph mapping for ARC's 10-colour palette.  ``0`` is the background
# and is rendered as a dot so foreground objects stand out clearly.
_SYMBOLS: dict[int, str] = {
    0: ".",
    1: "█",
    2: "▲",
    3: "★",
    4: "●",
    5: "◆",
    6: "■",
    7: "○",
    8: "△",
    9: "▢",
}


def _format_numeric(grid: list[list[int]]) -> str:
    """Render the grid as a Python list literal (e.g. ``[[0, 1], [0, 0]]``).

    Args:
        grid: A 2-D ``list[list[int]]`` grid.

    Returns:
        str: The ``repr`` of the grid.
    """
    return repr(grid)


def _format_visual(grid: list[list[int]]) -> str:
    """Render the grid as a space-separated block of glyphs.

    Args:
        grid: A 2-D ``list[list[int]]`` grid.

    Returns:
        str: One line per row; cells mapped via :data:`_SYMBOLS`.
    """
    rows: list[str] = []
    for row in grid:
        glyphs = [_SYMBOLS.get(int(cell), "?") for cell in row]
        rows.append(" ".join(glyphs))
    return "\n".join(rows)


def format_grid_dual(grid: list[list[int]]) -> str:
    """Return both numeric and visual representations of a 2-D grid.

    The result is a single multi-line string with two clearly labelled
    sections, making it trivially embeddable in an LLM prompt or a log.

    Args:
        grid: A rectangular 2-D ``list[list[int]]`` grid.

    Returns:
        str: A string of the form::

            Numeric:
            [[0, 1], [0, 0]]

            Visual:
            . █
            . .
    """
    return f"Numeric:\n{_format_numeric(grid)}\n\nVisual:\n{_format_visual(grid)}"
