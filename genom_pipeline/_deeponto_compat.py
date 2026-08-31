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
"""

import warnings

from deeponto.onto.verbalisation import OntologySyntaxParser

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
