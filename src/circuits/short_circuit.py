"""Short circuit: iterative generate -> verify -> self-correct loop.

This module implements the *lowest* (but most active) tier of the Three-Tier
Hierarchical Cognitive Architecture.  It turns the long circuit's strategic
macro-plan and the middle circuit's tactical object blueprint into concrete
code, then drives a tight self-correction loop:

    1. **Build a prompt** that fuses the long/middle context with the raw
       training grids (in dual numeric + visual form).
    2. **Sample several candidates** through a mockable LLM client seam.
    3. **Extract code** from each (possibly markdown-fenced) response.
    4. **Verify** every candidate with :func:`verifier.verify_solution`.
    5. **Select** the simplest verified candidate with the MDL selector, or,
       on failure, **feed the rich differential feedback back** into the next
       prompt and retry, up to ``max_attempts`` times.

The LLM client (:func:`_call_llm`) is intentionally mockable: with no API key
configured it returns a deterministic stand-in tuned to the bundled sample
task, so the whole loop can be exercised offline.  Tests (and real
deployments) replace it via monkeypatching or by wiring an SDK call.
"""

from __future__ import annotations

import re

from mdl_pruner import select_best_candidate
from verifier import verify_solution
from visualizer import format_grid_dual

__all__ = ["run_short_circuit"]

# Marker injected into prompts that carry self-correction feedback.  The mock
# client keys off it to demonstrate the "learn from feedback" behaviour.
_FEEDBACK_MARKER = "### SELF-CORRECTION FEEDBACK ###"

# Deterministic mock candidates.  ``_IDENTITY_SOLUTION`` is deliberately wrong
# for the sample task (it returns the grid unchanged) so the loop is forced to
# exercise its feedback path; ``_MIRROR_SOLUTION`` is the corrected answer.
_IDENTITY_SOLUTION = "def transform(grid):\n    return grid\n"
_MIRROR_SOLUTION = "def transform(grid):\n    return np.fliplr(grid)\n"

# Matches a triple-backtick fenced block, optionally preceded by a language tag.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Number of candidate programs sampled per attempt.  Several samples let the
# MDL selector choose the cleanest among multiple verified solutions.
_CANDIDATES_PER_ATTEMPT = 3


def _format_training_examples(train_pairs: list[dict]) -> str:
    """Render all training pairs with dual (numeric + visual) layouts.

    Each input/output is formatted by :func:`visualizer.format_grid_dual` so
    the prompt carries both the exact integer matrix and a glyph rendering that
    makes shapes, symmetry, and connectivity visually obvious.

    Args:
        train_pairs: A list of ``{"input": ..., "output": ...}`` examples.

    Returns:
        str: The formatted examples, ready to embed in a prompt.
    """
    chunks: list[str] = []
    for index, pair in enumerate(train_pairs):
        chunks.append(f"Example {index} input:")
        chunks.append(format_grid_dual(pair.get("input", [])))
        chunks.append(f"Example {index} output:")
        chunks.append(format_grid_dual(pair.get("output", [])))
    return "\n".join(chunks)


def _extract_code(response: str) -> str:
    """Extract a bare code string from a raw LLM response.

    LLMs frequently wrap code in triple-backtick fences (often with a language
    tag).  This strips the fence when present and otherwise returns the raw
    response, trimmed of surrounding whitespace.

    Args:
        response: The raw text returned by the LLM.

    Returns:
        str: The candidate code source.
    """
    if not isinstance(response, str):
        return ""
    match = _FENCE_RE.search(response)
    code = match.group(1) if match else response
    return code.strip()


def _format_feedback(attempt: int, failures: list[tuple[str, str]]) -> str:
    """Wrap a batch of verifier failures into a self-correction feedback block.

    Args:
        attempt: The 1-based attempt number that just failed.
        failures: A list of ``(candidate_code, detail)`` pairs, one per
            candidate that failed verification.

    Returns:
        str: A feedback block to append to the next prompt.
    """
    parts: list[str] = [
        _FEEDBACK_MARKER,
        f"Attempt {attempt} failed verification for all {len(failures)} candidate(s).",
        "The verifier reported the following differential feedback:",
    ]
    for index, (code, detail) in enumerate(failures, start=1):
        parts.append(f"Candidate {index} rejected:")
        parts.append(detail)
        parts.append("")
        parts.append(f"Candidate {index} code:")
        parts.append(code)
        parts.append("")
    parts.append("Please correct the transform and try again.")
    return "\n".join(parts)


def _build_prompt(
    train_pairs: list[dict],
    strategy_hint: str,
    feedback: str | None = None,
) -> str:
    """Assemble the full prompt for one LLM call.

    The prompt fuses (a) the long/middle context carried in ``strategy_hint``,
    (b) the raw training grids, and (c) any self-correction feedback from the
    previous failed attempt.

    Args:
        train_pairs: The training examples (raw grid data).
        strategy_hint: The combined long/middle context string.
        feedback: Optional self-correction feedback block.

    Returns:
        str: The assembled prompt.
    """
    sections: list[str] = [
        "You are an expert ARC (Abstraction and Reasoning Corpus) solver.",
        "Write a single Python function with the signature:",
        "",
        "def transform(grid):",
        "    ...",
        "",
        "The function receives a 2-D numpy array `grid` and must return a 2-D "
        "array (or nested list) of integers. `np` is already available; do not "
        "use imports.",
        "",
        "## Strategy context",
        strategy_hint,
        "",
        "## Training examples",
        _format_training_examples(train_pairs),
        "",
        "## Instructions",
        "Return ONLY the code, inside a ```python ... ``` fenced block.",
    ]
    if feedback:
        sections.extend(["", feedback])
    return "\n".join(sections)


def _call_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return its raw textual response.

    This is the single seam through which the agent talks to a language model.
    A production deployment would check for ``OPENAI_API_KEY`` /
    ``ANTHROPIC_API_KEY`` and call the corresponding SDK here.  To keep the
    scaffold dependency-free and testable offline, this default implementation
    returns a deterministic mock that:

        * answers with an identity transform on the first attempt (wrong for
          the sample task), then
        * returns the corrected horizontal-mirror transform once the prompt
          contains self-correction feedback.

    The mock is intentionally simple; it exists to exercise the loop, not to
    be a general solver.  Tests monkeypatch this function to simulate arbitrary
    models.

    Args:
        prompt: The assembled prompt.

    Returns:
        str: The model's raw response.
    """
    # -- real-client integration point -------------------------------------
    # if os.environ.get("OPENAI_API_KEY"):
    #     return _call_openai_sdk(prompt)
    # if os.environ.get("ANTHROPIC_API_KEY"):
    #     return _call_anthropic_sdk(prompt)

    # -- deterministic offline mock ----------------------------------------
    if _FEEDBACK_MARKER in prompt:
        return _MIRROR_SOLUTION
    return _IDENTITY_SOLUTION


def _call_llm_batch(prompt: str, n: int = 3) -> list[str]:
    """Generate ``n`` distinct candidate code strings for one prompt.

    This is the batch seam the loop actually uses.  The default implementation
    samples the single-candidate :func:`_call_llm` ``n`` times, appending a
    per-candidate index so a real sampling client could diversify its outputs.
    Results are deduplicated (preserving order) so identical samples aren't
    verified twice.

    Args:
        prompt: The assembled prompt.
        n: Number of candidate samples to request (>= 1).

    Returns:
        list[str]: The extracted, deduplicated candidate code strings.
    """
    candidates: list[str] = []
    for index in range(n):
        response = _call_llm(f"{prompt}\n\n[candidate {index + 1}]")
        code = _extract_code(response)
        if code and code not in candidates:
            candidates.append(code)
    return candidates


def run_short_circuit(
    train_pairs: list[dict],
    strategy_hint: str,
    max_attempts: int = 5,
) -> tuple[str | None, int]:
    """Run the iterative generate -> verify -> self-correct loop.

    Each iteration assembles a prompt from the long/middle context
    (``strategy_hint``) and the training data, obtains candidate code from the
    LLM client, extracts it, and verifies it.  On failure the verifier's
    differential feedback is appended to the next prompt so the model can
    self-correct.

    Args:
        train_pairs: A list of training examples, each a dict with ``"input"``
            and ``"output"`` keys holding 2-D ``list[list[int]]`` grids.
        strategy_hint: The combined long/middle context string to embed in the
            prompt.
        max_attempts: Maximum number of generate/verify iterations (>= 1).

    Returns:
        tuple[str | None, int]:
            * ``(code, attempts)`` when a candidate verifies 100%, where
              ``code`` is the accepted source and ``attempts`` is how many
              iterations were used.
            * ``(None, attempts)`` when the budget is exhausted without a
              verified solution (``attempts == max_attempts``).

    Raises:
        ValueError: If ``train_pairs`` is empty or ``max_attempts < 1``.
    """
    if not train_pairs:
        raise ValueError("run_short_circuit requires at least one training pair.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")

    feedback: str | None = None

    for attempt in range(1, max_attempts + 1):
        prompt = _build_prompt(train_pairs, strategy_hint, feedback)
        candidates = _call_llm_batch(prompt, _CANDIDATES_PER_ATTEMPT)

        # Verify every sampled candidate and split them into passing/failing.
        passing: list[str] = []
        failures: list[tuple[str, str]] = []
        for code in candidates:
            passed, detail = verify_solution(code, train_pairs)
            if passed:
                passing.append(code)
            else:
                failures.append((code, detail))

        # When several candidates pass, apply Occam's razor and keep the
        # simplest one via the MDL selector.
        if passing:
            return select_best_candidate(passing), attempt

        # Verification failed for every candidate: capture the rich
        # differential feedback and carry it into the next prompt so the model
        # can self-correct.
        feedback = _format_feedback(attempt, failures)

    return None, max_attempts

