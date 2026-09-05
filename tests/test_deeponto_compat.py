"""
Tests for genom_pipeline._deeponto_compat -- the DataHasValue workaround for
DeepOnto's OntologySyntaxParser.parse(). Runs against the real, installed
deeponto.onto.verbalisation.OntologySyntaxParser class (deeponto is a real
dependency in this project, no stubbing needed).
"""
from __future__ import annotations

import warnings

import pytest

# Must be imported before any `deeponto.onto` import below -- importing it
# starts DeepOnto's JVM non-interactively (see _deeponto_compat.py's
# _init_jvm()), otherwise deeponto.onto blocks on a stdin prompt the moment
# it's imported, which pytest turns into a hard collection error.
from genom_pipeline import _deeponto_compat

from deeponto.onto.verbalisation import OntologySyntaxParser


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


def test_init_jvm_skips_when_already_started(monkeypatch):
    # Regression case for the CI break: deeponto.onto's module-level code
    # calls click.prompt(...) for JVM memory unless jpype.isJVMStarted() is
    # already True. If the JVM is (or claims to be) started, we must not
    # try to start it again -- jpype only allows starting it once per
    # process.
    monkeypatch.setattr(_deeponto_compat.jpype, "isJVMStarted", lambda: True)
    calls = []
    monkeypatch.setattr(_deeponto_compat.jpype, "startJVM", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(_deeponto_compat.deeponto, "init_jvm", lambda memory: calls.append(memory))

    _deeponto_compat._init_jvm()

    assert calls == []


def test_init_jvm_starts_with_default_memory(monkeypatch):
    # _init_jvm() now calls jpype.startJVM() directly (with the extra
    # -Xrs flag, see _JVM_EXTRA_ARGS) instead of deeponto.init_jvm(), so
    # that this codepath is not silently skipped, only mock startJVM at
    # the jpype level and check the memory flag it was actually given.
    monkeypatch.setattr(_deeponto_compat.jpype, "isJVMStarted", lambda: False)
    monkeypatch.delenv("GENOM_JVM_MEMORY", raising=False)
    calls = []
    monkeypatch.setattr(_deeponto_compat.jpype, "startJVM", lambda *a, **k: calls.append(a))

    _deeponto_compat._init_jvm()

    assert len(calls) == 1
    args = calls[0]
    assert f"-Xmx{_deeponto_compat._DEFAULT_JVM_MEMORY}" in args
    for flag in _deeponto_compat._JVM_EXTRA_ARGS:
        assert flag in args


def test_init_jvm_respects_env_override(monkeypatch):
    monkeypatch.setattr(_deeponto_compat.jpype, "isJVMStarted", lambda: False)
    monkeypatch.setenv("GENOM_JVM_MEMORY", "2g")
    calls = []
    monkeypatch.setattr(_deeponto_compat.jpype, "startJVM", lambda *a, **k: calls.append(a))

    _deeponto_compat._init_jvm()

    assert len(calls) == 1
    assert "-Xmx2g" in calls[0]


def test_init_jvm_falls_back_to_deeponto_if_jars_dir_missing(monkeypatch, tmp_path):
    # If DeepOnto's packaged jars ever move, _init_jvm() falls back to
    # deeponto.init_jvm() (losing the -Xrs mitigation) rather than
    # crashing outright.
    monkeypatch.setattr(_deeponto_compat.jpype, "isJVMStarted", lambda: False)
    monkeypatch.setattr(
        _deeponto_compat.deeponto, "__file__", str(tmp_path / "nonexistent_pkg" / "__init__.py")
    )
    startjvm_calls = []
    monkeypatch.setattr(_deeponto_compat.jpype, "startJVM", lambda *a, **k: startjvm_calls.append(a))
    init_jvm_calls = []
    monkeypatch.setattr(_deeponto_compat.deeponto, "init_jvm", lambda memory: init_jvm_calls.append(memory))

    _deeponto_compat._init_jvm()

    assert startjvm_calls == []
    assert init_jvm_calls == [_deeponto_compat._DEFAULT_JVM_MEMORY]
