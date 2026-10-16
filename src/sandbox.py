"""Restricted execution sandbox for LLM-generated transformation code.

The cognitive agent generates candidate Python programs that implement an ARC
(Abstraction and Reasoning Corpus) transformation as a ``transform(grid)``
function. Those programs are *untrusted* input: they may contain bugs, raise
arbitrary exceptions, or even attempt malicious behaviour. This module runs
them inside a minimal, locked-down namespace so that any failure can be
captured cleanly and fed back into the agent's self-correction loop.

The public entry point is :func:`execute_generated_code`, which returns a
``(result, error)`` tuple:

    * On success: ``(output_grid, None)`` where ``output_grid`` is a normalised
      2-D ``list[list[int]]``.
    * On failure: ``(None, traceback_string)`` where ``traceback_string`` is a
      full traceback (for exceptions) or a diagnostic message (for missing or
      non-callable ``transform``).
"""

from __future__ import annotations

import builtins
import traceback
from typing import Any

import numpy as np

__all__ = ["execute_generated_code"]


# ---------------------------------------------------------------------------
# Restricted builtins
# ---------------------------------------------------------------------------
# Candidate solutions are executed with ``__builtins__`` replaced by a curated
# whitelist.  Anything that grants filesystem access, arbitrary code execution,
# or interpreter introspection (``open``, ``__import__``, ``eval``, ``exec``,
# ``getattr``, ``type``, ``object``, ``globals``, ``locals``, ...) is
# deliberately omitted, which closes off the most common escape hatches.
#
# NOTE: This is a *defence-in-depth* layer, not a hard security boundary.  A
# determined adversary could still escape via Python object introspection
# (e.g. ``().__class__.__bases__[0].__subclasses__()``).  For a fully
# hardened deployment, run this inside an OS-level sandbox (separate process /
# container / seccomp).  For the ARC use-case -- where the "attacker" is an
# LLM that occasionally writes buggy code -- this namespace is sufficient to
# prevent accidental damage and to capture failures for self-correction.
_SAFE_BUILTINS: dict[str, Any] = {
    # -- numeric -----------------------------------------------------------
    "abs": builtins.abs,
    "bool": builtins.bool,
    "complex": builtins.complex,
    "float": builtins.float,
    "int": builtins.int,
    "round": builtins.round,
    "divmod": builtins.divmod,
    "pow": builtins.pow,
    "max": builtins.max,
    "min": builtins.min,
    "sum": builtins.sum,
    # -- sequences / iteration ---------------------------------------------
    "range": builtins.range,
    "len": builtins.len,
    "list": builtins.list,
    "tuple": builtins.tuple,
    "dict": builtins.dict,
    "set": builtins.set,
    "frozenset": builtins.frozenset,
    "enumerate": builtins.enumerate,
    "zip": builtins.zip,
    "map": builtins.map,
    "filter": builtins.filter,
    "sorted": builtins.sorted,
    "reversed": builtins.reversed,
    "iter": builtins.iter,
    "next": builtins.next,
    "slice": builtins.slice,
    "any": builtins.any,
    "all": builtins.all,
    # -- typing / introspection (safe subset) ------------------------------
    "str": builtins.str,
    "repr": builtins.repr,
    "format": builtins.format,
    "hash": builtins.hash,
    "isinstance": builtins.isinstance,
    "issubclass": builtins.issubclass,
    "callable": builtins.callable,
    # -- common exceptions (so solution code can raise them) ---------------
    "Exception": builtins.Exception,
    "ArithmeticError": builtins.ArithmeticError,
    "AssertionError": builtins.AssertionError,
    "AttributeError": builtins.AttributeError,
    "ImportError": builtins.ImportError,
    "IndexError": builtins.IndexError,
    "KeyError": builtins.KeyError,
    "LookupError": builtins.LookupError,
    "NameError": builtins.NameError,
    "NotImplementedError": builtins.NotImplementedError,
    "OverflowError": builtins.OverflowError,
    "RuntimeError": builtins.RuntimeError,
    "StopIteration": builtins.StopIteration,
    "TypeError": builtins.TypeError,
    "ValueError": builtins.ValueError,
    "ZeroDivisionError": builtins.ZeroDivisionError,
}


def _build_namespace() -> dict[str, Any]:
    """Construct a fresh, locked-down namespace for one code execution.

    A brand-new namespace is created per call so that no state (module-level
    variables, imported names, monkey-patches) can leak between successive
    candidate evaluations.  The namespace exposes:

        * ``np``            -- the NumPy module (vectorised grid operations).
        * ``__builtins__``  -- the curated whitelist defined above.

    Returns:
        dict[str, Any]: A dictionary suitable for passing to :func:`exec`.
    """
    # Provide a *copy* of the whitelist so a buggy/malicious program that
    # mutates its own ``__builtins__`` cannot poison the module-level source.
    return {"__builtins__": dict(_SAFE_BUILTINS), "np": np}


def _coerce_to_grid(result: Any) -> list[list[int]]:
    """Validate and normalise a transform's return value into ``list[list[int]]``.

    LLM-generated code frequently returns slightly-off structures (scalars,
    1-D arrays, nested tuples, ...).  This helper raises a ``ValueError`` with
    an actionable message when the result cannot be interpreted as a 2-D grid
    of integers, so the failure surfaces as a self-correction signal rather
    than crashing the host.

    Args:
        result: The raw value returned by the candidate ``transform`` call.

    Returns:
        list[list[int]]: A rectangular 2-D grid of Python ``int`` values.

    Raises:
        ValueError: If ``result`` is ``None``, not 2-D, or contains values
            that cannot be coerced to integers.
    """
    if result is None:
        raise ValueError("transform() returned None; expected a 2-D grid of integers.")

    # ``np.asarray`` accepts nested lists, tuples, ndarrays, etc. and gives us
    # a single uniform way to inspect the shape.
    arr = np.asarray(result)

    if arr.ndim != 2:
        raise ValueError(
            f"transform() returned a {arr.ndim}-dimensional array; "
            "expected exactly 2 dimensions."
        )

    # Walk the array cell by cell and coerce each value to a Python ``int``.
    # ``int()`` raises (ValueError/TypeError) for non-numeric or ``None``
    # cells, which we let propagate as a descriptive failure.
    try:
        return [[int(cell) for cell in row] for row in arr.tolist()]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"transform() returned values that could not be coerced to integers: {exc}"
        ) from exc


def execute_generated_code(
    code_string: str,
    input_grid: list[list[int]],
) -> tuple[list[list[int]] | None, str | None]:
    """Execute LLM-generated ``transform`` code against a single input grid.

    This is the sandbox's public entry point.  It:

        1. Compiles ``code_string`` (so syntax errors surface with a clean
           traceback filename of ``<generated_code>``).
        2. Executes it inside a restricted namespace exposing only ``np`` and
           a whitelist of safe builtins.
        3. Extracts the required ``transform`` callable.
        4. Feeds ``input_grid`` through ``transform`` and normalises the result.

    Any failure -- a compilation/syntax error, a runtime exception, a missing
    or non-callable ``transform``, or a malformed return value -- is caught and
    returned as a full traceback / diagnostic string, which the agent can echo
    back to the LLM as self-correction feedback.

    Args:
        code_string: The candidate program source code.  It must define a
            ``transform(grid)`` function.  ``np`` is already available in the
            execution namespace, so the code should use it directly rather than
            attempting ``import numpy`` (imports are blocked).
        input_grid: A 2-D ``list[list[int]]`` grid to pass to ``transform``.

    Returns:
        tuple[list[list[int]] | None, str | None]:
            A ``(result, error)`` pair.

            * On success: ``(output_grid, None)`` where ``output_grid`` is the
              normalised 2-D grid of integers.
            * On failure: ``(None, message)`` where ``message`` is a full
              traceback string (for exceptions) or a diagnostic (for missing /
              non-callable ``transform``).
    """
    # -- 0. cheap input validation -----------------------------------------
    if not isinstance(code_string, str):
        return None, f"code_string must be a str, got {type(code_string).__name__}."
    if input_grid is None:
        return None, "input_grid must not be None."

    # -- 1. compile + execute in the locked-down namespace ------------------
    namespace = _build_namespace()
    try:
        # ``compile`` with an explicit filename gives readable tracebacks and
        # raises ``SyntaxError`` eagerly (rather than only at exec time).
        compiled = compile(code_string, "<generated_code>", "exec")
        exec(compiled, namespace)
    except BaseException:
        # Catching BaseException (not just Exception) also captures
        # SystemExit/KeyboardInterrupt so untrusted code can never terminate
        # the host process.
        return None, traceback.format_exc()

    # -- 2. locate and validate the required entry point --------------------
    transform = namespace.get("transform")
    if transform is None:
        return None, (
            "No 'transform' function was defined in the generated code. "
            "Expected a callable of the form: def transform(grid): ..."
        )
    if not callable(transform):
        return None, (
            f"'transform' exists but is not callable (got {type(transform).__name__})."
        )

    # -- 3. run the candidate and normalise the result ----------------------
    try:
        raw_result = transform(np.asarray(input_grid))
        output_grid = _coerce_to_grid(raw_result)
    except BaseException:
        return None, traceback.format_exc()

    return output_grid, None


