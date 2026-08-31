from __future__ import annotations

"""
Runtime compatibility patch for a DeepOnto limitation.

DeepOnto's OntologySyntaxParser.parse() (used internally by
OntologyVerbaliser.verbalise_class_expression, which extract.py calls to
turn complex class expressions into natural language) does not support the
OWL `DataHasValue` data property restriction. On any ontology whose complex
class expressions use it -- SNOMED-CT being a real-world example -- it
raises:

    RuntimeError: Input class expression `...` is not in one of the
    supported types.

Confirmed still present on DeepOnto's `main` branch (github.com/KRR-Oxford/DeepOnto)
as of writing -- `OntologySyntaxParser.parse` does no DataHasValue handling
at all, it just abbreviates the expression and parses by parentheses.

Until DeepOnto supports DataHasValue natively, this module monkey-patches
OntologySyntaxParser.parse to drop any `DataHasValue(...)` sub-expression
before parsing, rather than crashing. This is a LOSSY workaround: the
omitted restriction's literal value is dropped from the verbalised text
entirely, not translated into natural language -- a UserWarning is raised
the first time it happens on a given expression so it isn't silently
invisible. If your use case needs that information preserved, this is not
a substitute for a real fix upstream.

extract.py calls apply() once, at import time, before any verbalisation
happens.

This module also starts DeepOnto's JVM as a side effect of being imported,
before it does anything else -- see _init_jvm() below for why that has to
happen here specifically.
"""

import os
import warnings

import jpype

import deeponto

_DEFAULT_JVM_MEMORY = "8g"


def _init_jvm() -> None:
    """
    Start the JVM DeepOnto needs (via jpype) before anything imports
    deeponto.onto.

    deeponto/onto/ontology.py runs this at *module import time*:

        if not jpype.isJVMStarted():
            memory = click.prompt("Please enter the maximum memory located "
                                   "to JVM", type=str, default="8g")
            init_jvm(memory)

    i.e. the first time anything anywhere imports deeponto.onto (directly,
    or transitively -- e.g. via deeponto.align.bertmap), DeepOnto blocks on
    an interactive prompt asking how much memory to give the JVM, unless
    the JVM is already started. In any non-interactive context (CI, a
    plain script, pytest with output captured) there's no terminal to
    answer that prompt -- under pytest specifically this fails hard with
    `OSError: reading from stdin while output is captured!` and aborts
    test collection entirely (this is exactly what broke the first real
    CI run: every test module that imports deeponto, directly or
    transitively, errored out with this).

    Calling deeponto.init_jvm() ourselves here -- before any deeponto.onto
    import anywhere in genom_pipeline -- makes `jpype.isJVMStarted()` true
    by the time deeponto/onto/ontology.py's module-level check runs, so it
    skips the prompt entirely. Idempotent: jpype only allows starting the
    JVM once per process, so this is a no-op on any call after the first
    (including a call made from a different module that also imports
    _deeponto_compat).

    Memory limit is controlled by the GENOM_JVM_MEMORY environment
    variable; defaults to "8g" to match DeepOnto's own default.
    """
    if jpype.isJVMStarted():
        return
    memory = os.environ.get("GENOM_JVM_MEMORY", _DEFAULT_JVM_MEMORY)
    deeponto.init_jvm(memory)


_init_jvm()

from deeponto.onto.verbalisation import OntologySyntaxParser  # noqa: E402  (must come after _init_jvm())

_TARGET = "DataHasValue"


class DataHasValueOmittedWarning(UserWarning):
    """
    Raised when a DataHasValue(...) sub-expression is dropped while
    verbalising an OWL class expression, because DeepOnto's verbaliser does
    not support it. See genom_pipeline/_deeponto_compat.py.
    """


def _omit_part(expression: str, target: str, pos: int) -> str:
    """
    Remove one `target(...)`-shaped sub-expression starting at `pos`, up to
    its first closing parenthesis. Recurses to remove further occurrences.

    Does not handle nested parentheses *inside* the omitted span -- fine
    for DataHasValue, whose argument is a single literal, not a nested
    expression, but would need generalising for a target that can itself
    contain "(".

    Unlike a naive "chop exactly one character before pos" approach, this
    only trims a preceding space (never a structural '(' or other
    character), so parentheses stay balanced even when the omitted term is
    the first child of its parent expression -- e.g.
    "ObjectIntersectionOf(DataHasValue(p "1") B)" correctly becomes
    "ObjectIntersectionOf(B)", not "ObjectIntersectionOf B)".
    """
    before = expression[:pos].rstrip(" ")
    after = expression[pos:]
    close = after.find(")")
    remainder = after[close + 1 :].lstrip(" ")

    if before and not before.endswith("(") and remainder:
        expression = before + " " + remainder
    else:
        expression = before + remainder

    next_pos = expression.find(target)
    if next_pos != -1:
        return _omit_part(expression, target, next_pos)
    return expression


def _patched_parse(self, owl_expression):
    if not isinstance(owl_expression, str):
        owl_expression = str(owl_expression)

    pos = owl_expression.find(_TARGET)
    if pos != -1:
        warnings.warn(
            f"Dropping an unsupported '{_TARGET}' sub-expression while verbalising "
            "an OWL class expression (DeepOnto's verbaliser does not support it). "
            "The affected concept's logical description / definition-generation "
            "input will be missing this restriction. "
            "See genom_pipeline/_deeponto_compat.py.",
            DataHasValueOmittedWarning,
            stacklevel=3,
        )
        owl_expression = _omit_part(owl_expression, _TARGET, pos)

    return _original_parse(self, owl_expression)


_original_parse = OntologySyntaxParser.parse


def apply() -> None:
    """Patch OntologySyntaxParser.parse in place. Idempotent."""
    if OntologySyntaxParser.parse is _patched_parse:
        return
    OntologySyntaxParser.parse = _patched_parse


def unapply() -> None:
    """Restore DeepOnto's original parse method. Mainly useful for tests."""
    OntologySyntaxParser.parse = _original_parse
