"""
Tests for genom_pipeline._deeponto_compat -- the DataHasValue workaround for
DeepOnto's OntologySyntaxParser.parse(). Runs against the real, installed
deeponto.onto.verbalisation.OntologySyntaxParser class (deeponto is a real
dependency in this project, no stubbing needed).
"""
from __future__ import annotations

import warnings

import pytest
from deeponto.onto.verbalisation import OntologySyntaxParser

from genom_pipeline import _deeponto_compat


@pytest.fixture(autouse=True)
def _restore_parser():
    # keep tests isolated from each other and from whatever extract.py's
    # module-level `_deeponto_compat.apply()` already did on import
    original = OntologySyntaxParser.parse
    yield
    OntologySyntaxParser.parse = original


def test_omit_part_keeps_parentheses_balanced():
    cases = [
        'ObjectIntersectionOf(A DataHasValue(hasValue "42") B)',
        'ObjectIntersectionOf(DataHasValue(p1 "1") A DataHasValue(p2 "2") B)',
        'ObjectIntersectionOf(DataHasValue(p1 "1") B)',
        'DataHasValue(p "1")',
    ]
    for expr in cases:
        out = _deeponto_compat._omit_part(expr, "DataHasValue", expr.find("DataHasValue"))
        assert "DataHasValue" not in out
        assert out.count("(") == out.count(")"), f"unbalanced parens: {out!r}"


def test_omit_part_preserves_leading_open_paren():
    # Regression case: a naive "chop exactly one character before the match"
    # approach eats the parent expression's own '(' when DataHasValue is the
    # first child, corrupting the parenthesis structure. This must not happen.
    expr = 'ObjectIntersectionOf(DataHasValue(p1 "1") B)'
    out = _deeponto_compat._omit_part(expr, "DataHasValue", expr.find("DataHasValue"))
    assert out == 'ObjectIntersectionOf(B)'


def test_apply_is_idempotent():
    _deeponto_compat.apply()
    patched_once = OntologySyntaxParser.parse
    _deeponto_compat.apply()
    assert OntologySyntaxParser.parse is patched_once


def test_patched_parse_warns_and_strips_datahasvalue(monkeypatch):
    _deeponto_compat.apply()

    calls = []
    monkeypatch.setattr(
        _deeponto_compat,
        "_original_parse",
        lambda self, expr: calls.append(expr) or expr,
    )

    dummy = object.__new__(OntologySyntaxParser)
    with pytest.warns(_deeponto_compat.DataHasValueOmittedWarning):
        OntologySyntaxParser.parse(dummy, 'ObjectIntersectionOf(A DataHasValue(p "1") B)')

    assert calls == ["ObjectIntersectionOf(A B)"]


def test_patched_parse_leaves_normal_expressions_untouched(monkeypatch):
    _deeponto_compat.apply()

    calls = []
    monkeypatch.setattr(
        _deeponto_compat,
        "_original_parse",
        lambda self, expr: calls.append(expr) or expr,
    )

    dummy = object.__new__(OntologySyntaxParser)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # fail the test if any warning is raised
        OntologySyntaxParser.parse(dummy, "ObjectIntersectionOf(A B)")

    assert calls == ["ObjectIntersectionOf(A B)"]
