# PaperTrack — ARC Prize 2026 Paper Track Agent

A production-grade scaffold for an autonomous multi-tier cognitive agent
targeting the ARC Prize 2026 Paper Track. The agent generates candidate Python
programs that implement ARC grid transformations, executes them in a restricted
sandbox, and verifies them against the task's training examples.

## Project structure

```
PaperTrack/
├── data/
│   └── sample_task.json      # Mock ARC task (4x4 horizontal-mirror transform)
├── src/
│   ├── sandbox.py            # Restricted execution engine for LLM-generated code
│   ├── verifier.py           # Strict 100%-match verification with diff feedback
│   ├── visualizer.py         # Dual numeric + visual grid formatter
│   ├── mdl_pruner.py         # MDL selector (Occam's razor) for candidate code
│   └── circuits/
│       ├── __init__.py       # Three-tier cognitive architecture package
│       ├── long_circuit.py   # Global taxonomy + strategic macro-plan
│       ├── middle_circuit.py # Object decomposition + tactical blueprint
│       └── short_circuit.py  # Iterative generate → verify → self-correct loop
├── tests/
│   ├── test_sandbox.py       # Sandbox + verifier tests
│   └── test_circuits.py      # Three-tier architecture tests
├── main.py                   # End-to-end pipeline entry point
├── conftest.py               # Makes src/ importable for tests
├── pyproject.toml            # Project metadata + pytest configuration
├── requirements.txt          # Runtime + dev dependencies
└── .gitignore
```

## Installation

Python 3.10+ is required. Install dependencies with:

```bash
py -m pip install -r requirements.txt
```

## Running the tests

```bash
py -m pytest -v
```

## Running the pipeline

```bash
py main.py
```

This loads `data/sample_task.json`, classifies the task taxonomy, decomposes
the grids into objects, and runs the short circuit's generate → verify →
self-correct loop end-to-end.

## Three-tier cognitive architecture

The agent is organised as three tiers inside `src/circuits/`:

1. **Long circuit** (`long_circuit.py`) — classifies the task's global
   taxonomy (Geometric/Rotation, Topological/Pathfinding, Counting/Scaling)
   and emits a strategic macro-plan.
2. **Middle circuit** (`middle_circuit.py`) — decomposes grids into objects
   (bounding boxes, colour distributions, spatial layouts) and emits a
   tactical sub-goal blueprint. It uses SciPy when available and falls back
   to a pure-NumPy connected-component labelling otherwise.
3. **Short circuit** (`short_circuit.py`) — runs the iterative generate →
   verify → self-correct loop, feeding the verifier's differential feedback
   back into the next prompt.

## Core API

```python
from sandbox import execute_generated_code
from verifier import verify_solution

# Run a candidate program on one grid (returns (output_grid, error)).
output, error = execute_generated_code("def transform(grid):\n    return np.fliplr(grid)",
                                       [[0, 1], [1, 0]])

# Verify a candidate against all training pairs (returns (passed, detail)).
passed, detail = verify_solution("def transform(grid):\n    return np.fliplr(grid)",
                                 [{"input": [[0, 1]], "output": [[1, 0]]}])
```

## Security notes

`src/sandbox.py` executes LLM-generated code inside a restricted namespace
that exposes only NumPy (`np`) and a whitelist of safe builtins. It is a
defence-in-depth layer designed to capture failures for self-correction, *not*
a hard security boundary. For a fully hardened deployment, run execution inside
an OS-level sandbox (separate process, container, or seccomp).
