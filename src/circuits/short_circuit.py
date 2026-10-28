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

import json
import os
import re
import urllib.error
import urllib.request

from mdl_pruner import select_best_candidate
from verifier import verify_solution
from visualizer import format_grid_dual

# ---------------------------------------------------------------------------
# Secret loading (local ``.env`` + Kaggle).  Keeps API keys out of git while
# remaining seamless across local and Kaggle environments.  Both paths are
# best-effort: the offline mock LLM fallback still works when no key exists.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()  # no-op when .env is absent; merges values into os.environ
except ImportError:  # python-dotenv not installed -> use already-set env vars
    load_dotenv = None

if "LLM_API_KEY" not in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient

        _api_key = UserSecretsClient().get_secret("LLM_API_KEY")
        if _api_key:
            os.environ["LLM_API_KEY"] = _api_key
    except (ImportError, Exception):
        # ``kaggle_secrets`` only exists inside Kaggle kernels; ignore locally.
        pass

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
# Prefers an explicitly Python-labelled fence so thinking text wrapped in a
# plain fence never shadows the real code block.
_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)[^\n]*\n(.*?)```", re.DOTALL)

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

    LLMs frequently wrap code in triple-backtick fences, and reasoning models
    often prefix the code with free-form "thinking" text.  This prefers an
    explicit ``python``/``py`` fence, falls back to the first plain fence, then
    to a bare ``def transform(...)``, and finally returns the raw response.

    Args:
        response: The raw text returned by the LLM.

    Returns:
        str: The candidate code source.
    """
    if not isinstance(response, str):
        return ""

    match = _PYTHON_FENCE_RE.search(response) or _FENCE_RE.search(response)
    if match:
        return match.group(1).strip()

    # No fence: if the response is thinking text followed by a bare function,
    # keep only the function definition onwards.
    def_match = re.search(r"def\s+transform\s*\(", response)
    if def_match:
        return response[def_match.start():].strip()

    return response.strip()


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


_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
    "vllm": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ollama": "llama3",
}


def _http_post_json(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON to ``url`` and return the decoded JSON response.

    Uses only the standard library so the scaffold stays dependency-free.
    HTTP and network failures are re-raised as ``RuntimeError``.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"LLM request failed (HTTP {exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed (network): {exc.reason}") from exc


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    """Call the Anthropic Messages API and return the assistant text."""
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = _http_post_json(url, payload, headers)
    return data["content"][0]["text"]


def _extract_reasoning_text(reasoning) -> str | None:
    """Extract text from a ``reasoning`` field of any common shape.

    Reasoning tokens may arrive as a plain string, as a dict with a ``"text"``
    key (e.g. ``{"text": "..."}``), or as a list of structured blocks.
    """
    if isinstance(reasoning, str):
        return reasoning.strip() or None

    if isinstance(reasoning, dict):
        text = reasoning.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    if isinstance(reasoning, list):
        parts: list[str] = []
        for block in reasoning:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text = block["text"].strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    return None


def _extract_message_content(message: dict) -> str | None:
    """Extract assistant text from a chat message, prioritising the answer.

    Reasoning models (e.g. Qwen 3.7 Max) may return ``content`` as a list of
    structured blocks, as a plain string, or empty while placing thinking
    tokens in a separate ``reasoning`` / ``reasoning_content`` field.  The
    primary text content block is always preferred; reasoning fields are only
    used as a fallback when no answer text is present.
    """
    if not isinstance(message, dict):
        return None

    content = message.get("content")

    # 1) Primary answer: a plain, non-empty string.  An empty string is
    # treated the same as a missing field so the reasoning fallback below can
    # still recover the answer for reasoning models that emit empty content.
    if isinstance(content, str) and content:
        return content

    # 2) Primary answer: a list of structured blocks -> join the text blocks.
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and "text" in block
            ):
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    # 3) Fallback: reasoning tokens when there is no plain answer text.
    reasoning_text = _extract_reasoning_text(message.get("reasoning"))
    if reasoning_text:
        return reasoning_text

    reasoning_text = _extract_reasoning_text(message.get("reasoning_content"))
    if reasoning_text:
        return reasoning_text

    return None


def _call_openai(prompt: str, api_key: str, model: str, base_url: str) -> str:
    """Call an OpenAI-compatible Chat Completions endpoint.

    The response is parsed defensively so that OpenAI, vLLM, and assorted
    direct-response gateways all work, and so that a backend which returns an
    ``{"error": ...}`` body (possibly with an HTTP 200 status) surfaces a clear
    ``RuntimeError`` instead of a confusing ``KeyError``.
    """
    base = base_url.rstrip("/") or "https://api.openai.com/v1"
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = _http_post_json(url, payload, headers)

    # 0) Some gateways wrap the response in an envelope like
    #    ``{"data": {...}, "success": ...}``; unwrap it before parsing.
    if isinstance(data, dict) and "choices" not in data and isinstance(data.get("data"), dict):
        data = data["data"]

    # 1) Some backends report failures inside the 200 response body.
    if "error" in data:
        raise RuntimeError(f"API Error: {data['error']}")

    # 2) Inspect ``choices[0]["message"]`` for list/string/reasoning content,
    #    then fall back to legacy completion and direct-response shapes.
    content: str | None = None

    choices = data.get("choices") or []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}

    if first:
        message = first.get("message")
        content = _extract_message_content(message) if isinstance(message, dict) else None
        if content is None:
            content = first.get("text")

    if content is None:
        content = data.get("content") or data.get("text")

    if content is None:
        # Surface the full payload so an unexpected shape is easy to debug.
        print(f"Unexpected LLM response payload: {data}")
        raise RuntimeError(f"Unexpected response format: {data}")

    return content


def _call_ollama(prompt: str, model: str, base_url: str) -> str:
    """Call a local Ollama server's /api/generate endpoint."""
    base = base_url.rstrip("/") or "http://localhost:11434"
    url = f"{base}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    data = _http_post_json(url, payload, {})
    return data["response"]


def _call_real_provider(prompt: str, provider: str) -> str:
    """Dispatch ``prompt`` to the configured real LLM provider."""
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL_NAME", "").strip() or _DEFAULT_MODELS.get(provider, "")
    base_url = os.environ.get("LLM_BASE_URL", "").strip()

    if provider == "anthropic":
        return _call_anthropic(prompt, api_key, model)
    if provider == "openai":
        return _call_openai(prompt, api_key, model, base_url)
    if provider == "vllm":
        # vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint.
        return _call_openai(prompt, api_key, model, base_url)
    if provider == "ollama":
        return _call_ollama(prompt, model, base_url)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}.")


def _call_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM and return its raw response.

    This is the single seam through which the agent talks to a language model.
    The provider is selected through environment variables:

        * ``LLM_PROVIDER``   -- ``anthropic`` | ``openai`` | ``ollama`` | ``vllm``.
        * ``LLM_API_KEY``    -- API key for cloud providers (Ollama needs none).
        * ``LLM_MODEL_NAME`` -- model identifier (falls back to a per-provider default).
        * ``LLM_BASE_URL``   -- optional endpoint override (Ollama / self-hosted vLLM).

    When ``LLM_PROVIDER`` is unset, or a cloud provider is chosen without an
    ``LLM_API_KEY``, the function falls back to a deterministic mock so the
    scaffold and its unit tests remain fast and offline.

    Args:
        prompt: The assembled prompt.

    Returns:
        str: The model's raw response (may be markdown-fenced).
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    api_key = os.environ.get("LLM_API_KEY", "").strip()

    # Use a real provider only when explicitly configured.  Cloud providers
    # additionally require an API key; otherwise fall through to the mock.
    if provider:
        needs_key = provider in ("anthropic", "openai", "vllm")
        if not needs_key or api_key:
            return _call_real_provider(prompt, provider)

    # -- deterministic offline mock (no provider / API key configured) -----
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

