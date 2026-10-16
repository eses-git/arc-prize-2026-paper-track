"""Three-Tier Hierarchical Cognitive Architecture package.

This package implements the agent's three-tier cognitive stack for solving ARC
(Abstraction and Reasoning Corpus) grid-transformation tasks:

    1. ``long_circuit``   -- classifies the task's global taxonomy
       (Geometric/Rotation, Topological/Pathfinding, Counting/Scaling) and
       emits a strategic macro-plan.
    2. ``middle_circuit`` -- decomposes grids into objects (bounding boxes,
       colour distributions, spatial layouts) and emits a tactical sub-goal
       blueprint.
    3. ``short_circuit``  -- runs the iterative generate -> verify ->
       self-correct loop that turns the plans above into verified code.
"""

from __future__ import annotations

from .long_circuit import classify_and_plan
from .middle_circuit import decompose_objects
from .short_circuit import run_short_circuit

__all__ = [
    "classify_and_plan",
    "decompose_objects",
    "run_short_circuit",
]
