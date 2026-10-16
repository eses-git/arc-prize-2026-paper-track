"""Minimum Description Length (MDL) code selector.

When several candidate programs all pass verification, the agent must pick one
to keep.  This module applies Occam's Razor / Solomonoff Induction: among
equally-correct programs, prefer the one with the shortest description.

Description length is estimated structurally from each candidate's Abstract
Syntax Tree (AST) using three cheap, deterministic proxies:

    * **AST node count**  -- total number of AST nodes in the program.
    * **Function length** -- number of statements in the ``transform`` body.
    * **Variable depth**  -- maximum AST nesting depth, capturing deeply nested
      scopes, loops, and conditional logic.

These are combined into a single scalar complexity score; the candidate with
the lowest score wins, with source length and insertion order as tie-breakers.
Candidates that fail to parse are assigned infinite complexity so they are
never chosen over a working program.
"""

from __future__ import annotations

import ast
import math

__all__ = ["select_best_candidate"]


def _count_nodes(tree: ast.AST) -> int:
    """Count every node in ``tree`` (including the root).

    Args:
        tree: A parsed AST.

    Returns:
        int: The total number of AST nodes.
    """
    return sum(1 for _ in ast.walk(tree))


def _max_depth(node: ast.AST, depth: int = 0) -> int:
    """Return the maximum nesting depth of ``node``.

    A leaf node has depth ``0``; each level of child nesting adds one.  This
    proxies "variable depth": the deeper the tree, the more deeply scoped the
    variables and control flow tend to be.

    Args:
        node: The AST node to measure.
        depth: The depth of ``node`` itself (used in recursion).

    Returns:
        int: The maximum depth of any node in the subtree.
    """
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(_max_depth(child, depth + 1) for child in children)


def _transform_statement_count(tree: ast.AST) -> int:
    """Count the statements in the ``transform`` function body.

    Falls back to the module's top-level statement count when no ``transform``
    function is present (e.g. a top-level code snippet).

    Args:
        tree: A parsed AST.

    Returns:
        int: The number of statements in the ``transform`` body.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "transform"
        ):
            return len(node.body)
    return len(tree.body)


def _complexity(code: str) -> float:
    """Estimate the description length of ``code``.

    Unparseable code receives ``math.inf`` so it can never beat a valid
    program.

    Args:
        code: A candidate Python source string.

    Returns:
        float: A non-negative complexity score (lower is simpler).
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return math.inf

    node_count = _count_nodes(tree)
    depth = _max_depth(tree)
    func_length = _transform_statement_count(tree)

    # Depth and function length are weighted so a single sprawling function is
    # penalised slightly more than a flat one, reflecting the intuition that
    # deep nesting and many statements are harder to describe.
    return float(node_count) + 2.0 * depth + float(func_length)


def select_best_candidate(candidates: list[str]) -> str:
    """Select the most parsimonious candidate program from a list.

    Each candidate is parsed into an AST and scored with :func:`_complexity`.
    The lowest-scoring (simplest) candidate is returned.  Ties are broken
    first by shorter source length, then by earlier position in the input list
    (stable and deterministic).

    Args:
        candidates: A non-empty list of candidate Python source strings.

    Returns:
        str: The selected candidate's original source string.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("select_best_candidate requires at least one candidate.")

    scored: list[tuple[int, float, int, str]] = []
    for index, code in enumerate(candidates):
        src_len = len(code) if isinstance(code, str) else 0
        scored.append((index, _complexity(code), src_len, code))

    # Sort by (complexity, source length, original index).  ``math.inf`` sorts
    # last, so unparseable candidates are never preferred over valid ones.
    scored.sort(key=lambda item: (item[1], item[2], item[0]))
    return scored[0][3]
