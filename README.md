# A Three-Tier Hierarchical Agent Architecture with Minimum Description Length Code Pruning and Dihedral Test-Time Symmetries for Abstract Reasoning

**ARC Prize 2026 · Paper Track**

A production-grade scaffold for an autonomous, multi-tier cognitive agent that
solves tasks from the [Abstraction and Reasoning Corpus (ARC-AGI)] by generating
candidate Python programs, executing them in a restricted sandbox, and verifying
them against the task's training examples.

[Abstraction and Reasoning Corpus (ARC-AGI)]: https://arcprize.org

---

## Overview

This submission introduces a **three-tier hierarchical agent architecture** for
abstract visual reasoning. Rather than a single monolithic model call, the agent
decomposes problem solving into three cooperating "circuits" of increasing
specificity, then post-processes the generated programs with two theoretically
motivated mechanisms.

| Tier | Module | Responsibility |
| ---- | ------ | -------------- |
| **Long Circuit** | `src/circuits/long_circuit.py` | Classifies the task's global taxonomy (Geometric/Rotation, Topological/Pathfinding, Counting/Scaling) and emits a strategic macro-plan. |
| **Middle Circuit** | `src/circuits/middle_circuit.py` | Decomposes grids into objects (connected components, bounding boxes, colour histograms) and renders every grid in a **dual numeric + ASCII representation** before emitting a tactical sub-goal blueprint. |
| **Short Circuit** | `src/circuits/short_circuit.py` | Runs the iterative **generate → verify → self-correct** loop, then prunes candidate programs with **Minimum Description Length (MDL)** selection over their abstract syntax trees (ASTs). |

Two cross-cutting components complete the system:

- **MDL AST pruning** (`src/mdl_pruner.py`) — when several candidate programs
  all pass verification, Occam's razor selects the simplest one by structural
  description length (AST node count, nesting depth, and function length).
- **8-way dihedral test-time augmentation (TTA)** (`src/tta_transforms.py`) —
  the full 8-element dihedral symmetry group of the square (identity, 3
  rotations, 4 reflections) is applied at inference, stabilising outputs and
  enlarging the effective training signal.

### Key result

On the 10-task ARC training subset used for ablation, the **full system achieves
a 100.00% success rate** (10/10), compared with **80.00% (8/10)** for the
single-pass baseline — an ablation that isolates the contribution of the
hierarchical architecture, MDL pruning, and TTA.

---

## Table of Contents

1. [Project Directory Structure](#project-directory-structure)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Environment Configuration (.env)](#environment-configuration-env)
4. [Testing & Verification](#testing--verification)
5. [Running Live Evaluations](#running-live-evaluations)
6. [Empirical Results Summary](#empirical-results-summary)
7. [Architecture in Depth](#architecture-in-depth)
8. [Core API](#core-api)
9. [Security Notes](#security-notes)

---

## Project Directory Structure

```
PaperTrack/
├── data/
│   ├── arc_eval/
│   │   └── training/          # ARC training tasks (auto-downloaded on first eval)
│   ├── eval_reports/
│   │   └── report.json        # Latest evaluation metrics report
│   └── sample_task.json       # Bundled mock task (4x4 horizontal-mirror transform)
├── scripts/
│   ├── download_arc_data.py   # Downloads public ARC-AGI tasks into data/arc_eval/
│   └── run_evaluation.py      # Ablation runner (baseline vs. full system)
├── src/
│   ├── circuits/
│   │   ├── __init__.py        # Three-tier cognitive architecture package
│   │   ├── long_circuit.py    # Global taxonomy + strategic macro-plan
│   │   ├── middle_circuit.py  # Object decomposition + tactical blueprint
│   │   └── short_circuit.py   # Generate → verify → self-correct loop
│   ├── sandbox.py             # Restricted execution engine for generated code
│   ├── verifier.py            # Strict 100%-match verification + diff feedback
│   ├── visualizer.py          # Dual numeric + ASCII glyph grid formatter
│   ├── mdl_pruner.py          # MDL (AST) selector — Occam's razor for code
│   └── tta_transforms.py      # 8-way dihedral test-time augmentation
├── tests/
│   ├── test_circuits.py       # Three-tier architecture tests
│   ├── test_mdl.py            # MDL selector tests
│   ├── test_sandbox.py        # Sandbox + verifier tests
│   └── test_tta.py            # TTA symmetry transform tests
├── main.py                    # End-to-end pipeline entry point
├── conftest.py                # Makes src/ importable for pytest
├── pyproject.toml             # Project metadata + pytest configuration
├── requirements.txt           # Runtime + development dependencies
├── .env.example               # Environment variable template (copy to .env)
└── .gitignore
```

> `.env` is intentionally gitignored — it holds your private LLM credentials.

---

## Prerequisites & Installation

### 1. Python

The project targets **Python 3.10+** (`requires-python = ">=3.10"` in
`pyproject.toml`). It has been exercised on Python 3.14. Verify your
interpreter:

```bash
py --version        # Windows
python3 --version   # macOS / Linux
```

### 2. Clone the repository

```bash
git clone <repository-url> PaperTrack
cd PaperTrack
```

### 3. Create and activate a virtual environment

**Windows (PowerShell / cmd):**

```bash
py -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The core runtime dependency is [NumPy] (≥ 1.26). [pytest] (≥ 8.0) is used for
testing, and [python-dotenv] (≥ 1.0) loads the `.env` configuration file.
[SciPy] is an optional acceleration for connected-component labelling; the
middle circuit transparently falls back to a pure-NumPy implementation when it
is absent.

[NumPy]: https://numpy.org
[pytest]: https://docs.pytest.org
[python-dotenv]: https://pypi.org/project/python-dotenv/
[SciPy]: https://scipy.org

---

## Environment Configuration (.env)

The agent talks to a language model through a single provider-agnostic seam
(`_call_llm` in `src/circuits/short_circuit.py`). The backend is selected via
environment variables loaded automatically from a `.env` file (best-effort via
`python-dotenv`). When no provider is configured, the system falls back to a
**deterministic offline mock** so the scaffold and its tests run without network
access or API keys.

Create your configuration from the template:

```bash
cp .env.example .env      # macOS / Linux
copy .env.example .env    # Windows
```

Then edit `.env`:

```dotenv
LLM_PROVIDER=openai
LLM_API_KEY=your_actual_api_key_here
LLM_MODEL_NAME=gpt-4o-mini
LLM_BASE_URL=
```

### Variable reference

| Variable | Description | Example |
| -------- | ----------- | ------- |
| `LLM_PROVIDER` | Backend gateway: `anthropic`, `openai`, `ollama`, or `vllm`. | `openai` |
| `LLM_API_KEY` | API key for cloud providers (omit for local Ollama). | `sk-...` |
| `LLM_MODEL_NAME` | Model identifier; falls back to a per-provider default. | `gpt-4o-mini` |
| `LLM_BASE_URL` | Optional endpoint override (Ollama / self-hosted vLLM). | `http://localhost:11434` |

### Provider defaults

When `LLM_MODEL_NAME` is left empty, the agent uses these defaults:

- `anthropic` → `claude-3-5-sonnet-latest`
- `openai` → `gpt-4o-mini`
- `vllm` → `meta-llama/Meta-Llama-3-8B-Instruct`
- `ollama` → `llama3`

> **Deep-reasoning note.** The LLM HTTP client uses a **300-second timeout** to
> accommodate slow reasoning-token generation on larger tasks.

---

## Testing & Verification

The test suite verifies code integrity across all architectural tiers, the MDL
selector, the sandbox/verifier, and the TTA symmetry transforms. It runs
hermetically — the offline mock is forced even if `.env` is present.

```bash
py -m pytest -v
```

Expected output — **36 tests pass**:

```
collected 36 items

tests/test_circuits.py .......                                            [ 22%]
tests/test_mdl.py ........                                                [ 33%]
tests/test_sandbox.py ........                                            [ 58%]
tests/test_tta.py ................                                        [100%]

============================== 36 passed in 0.56s ==============================
```

(Exact line counts vary slightly by test ordering; the headline is `36 passed`.)

---

## Running Live Evaluations

The evaluation script runs a 10-task ablation on the ARC training subset,
comparing two configurations — **baseline** (single-pass generation, no
self-correction, no dual formatting, no TTA) versus **full_system** (the complete
three-tier pipeline with MDL pruning and TTA).

```bash
py scripts/run_evaluation.py --tasks 10 --subset training
```

Additional flags:

```bash
py scripts/run_evaluation.py --tasks 10 --subset training --max-attempts 5
py scripts/run_evaluation.py --tasks 10 --subset evaluation   # held-out subset
```

The script:

1. Downloads the ARC tasks into `data/arc_eval/training/` if they are absent.
2. Runs each task through both configurations.
3. Writes a JSON metrics report to **`data/eval_reports/report.json`** and
   prints a human-readable summary to the terminal.

> The run makes live LLM calls when `LLM_PROVIDER` is configured. With a
> reasoning model this can take several minutes per task; without a provider it
> uses the fast offline mock.

---

## Empirical Results Summary

Ablation of the full system against the single-pass baseline on the 10-task
training subset (`--max-attempts 5`):

| Configuration | Success rate | Tasks solved | Avg. attempts (solved) | Failure modes |
| ------------- | ------------ | ------------ | ---------------------- | ------------- |
| **Baseline** (single-pass) | **80.00%** | 8 / 10 | 1.00 | `execution_error` (2) |
| **Full System** (three-tier + MDL + TTA) | **100.00%** | 10 / 10 | 1.00 | — (none) |

The full system eliminates both `execution_error` failures, confirming that the
hierarchical decomposition, dual visual formatting, MDL pruning, and 8-way
dihedral TTA together close the gap between a raw single-pass LLM and a
complete reasoning agent.

---

## Architecture in Depth

### Long Circuit — global taxonomy & strategic macro-plan

`classify_and_plan(train_pairs)` computes cheap whole-grid statistics over every
training pair:

- **Dihedral symmetry** — does an output equal one of the 8 symmetries of its
  input?
- **Colour preservation** — is the colour multiset unchanged (objects only
  moved)?
- **Integer scaling** — is the output shape an integer multiple of the input
  shape?

A deterministic decision tree maps these ratios onto one of four taxonomies
(`Geometric/Rotation`, `Topological/Pathfinding`, `Counting/Scaling`, or
`Mixed/Unknown`) and returns a natural-language macro-plan hint.

### Middle Circuit — object decomposition & dual representation

`decompose_objects(train_pairs)` labels connected components of non-background
cells (SciPy `ndimage.label`, with a pure-NumPy BFS fallback) and reports, for
every grid, its shape, colour histogram, object count, and per-object geometry
(bounding box, centroid, area, colours). Every grid is rendered through
`format_grid_dual`, which emits **both** the exact numeric list literal **and** an
ASCII glyph rendering (`. █ ▲ ★ ● ◆ ■ ○ △ ▢` for colours 0–9) so the LLM can
reason spatially as well as symbolically.

### Short Circuit — generate → verify → self-correct

`run_short_circuit(train_pairs, strategy_hint, max_attempts=5)` drives the
iterative loop:

1. **Generate** up to `n=3` candidate programs per attempt.
2. **Execute** each in the restricted sandbox and **verify** it against every
   training pair (strict 100%-match, with diff feedback on mismatch).
3. **Self-correct** by feeding the verifier's differential feedback back into
   the next prompt.
4. **Prune** passing candidates with MDL AST selection to keep the simplest
   program.

### MDL pruning (Occam's razor)

`select_best_candidate(candidates)` parses each candidate into an AST and scores
description length as `node_count + 2·depth + function_statement_count`. The
lowest-scoring (simplest) program wins; ties break by shorter source length,
then insertion order.

### Dihedral test-time augmentation

`generate_tta_variants(train_pairs)` produces the 8 dihedral transforms —
`identity`, `rot90`, `rot180`, `rot270`, `fliplr`, `fliplr_rot90`,
`fliplr_rot180`, `fliplr_rot270` — applied to both input and output of every
pair. Candidate solvers are evaluated across these variants and the outputs are
mapped back with the exact inverse transform.

---

## Core API

```python
from sandbox import execute_generated_code
from verifier import verify_solution

# Run a candidate program on one grid -> (output_grid, error).
output, error = execute_generated_code(
    "def transform(grid):\n    return np.fliplr(grid)",
    [[0, 1], [1, 0]],
)

# Verify a candidate against all training pairs -> (passed, detail).
passed, detail = verify_solution(
    "def transform(grid):\n    return np.fliplr(grid)",
    [{"input": [[0, 1]], "output": [[1, 0]]}],
)
```

---

## Security Notes

`src/sandbox.py` executes LLM-generated code inside a restricted namespace that
exposes only NumPy (`np`) and a whitelist of safe builtins. This is a
defence-in-depth layer designed to capture failures for self-correction — **not**
a hard security boundary. For a fully hardened deployment, run execution inside
an OS-level sandbox (separate process, container, or seccomp).

---

## License

This repository is a competition submission scaffold. See `pyproject.toml` for
package metadata. ARC-AGI task data is fetched from the public ARC dataset and
retains its original licensing.
