"""
Tests for genom_pipeline.exact_matcher.bertmaplt_string's
_get_ignored_class_index_safe() -- a safety wrapper around deeponto's
get_ignored_class_index(), which calls Ontology.get_annotations(class_obj,
"http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment") for every
class in the ontology. get_annotations()'s own docstring promises an empty
return when the annotation property IRI doesn't exist in the ontology, but
the real (installed) implementation raises `KeyError: Cannot retrieve
unknown IRI: ...` instead -- which only ever fires on a real, full
production ontology (e.g. the actual SNOMED-CT or FMA release, not an
OAEI Bio-ML-trimmed copy), since every genuine OAEI Bio-ML evaluation
ontology declares the annotation property even if few classes use it.

These use a bare fake with just the one attribute
_get_ignored_class_index_safe() actually reads (owl_annotation_properties)
-- no JVM, no real deeponto Ontology object, no OWL file needed.
"""
from __future__ import annotations

from typing import Dict

# Must be imported before any `deeponto` import -- see
# _deeponto_compat.py's _init_jvm() for why (starts the JVM
# non-interactively so DeepOnto doesn't block on a stdin prompt).
from genom_pipeline import _deeponto_compat  # noqa: F401

from genom_pipeline.exact_matcher import bertmaplt_string as m


class _FakeOnto:
    def __init__(self, annotation_property_iris):
        # deeponto.onto.Ontology exposes this as a dict keyed by IRI;
        # _get_ignored_class_index_safe only ever checks membership in it.
        self.owl_annotation_properties: Dict[str, object] = {
            iri: object() for iri in annotation_property_iris
        }


def test_skips_delegating_when_annotation_property_is_not_declared():
    # A real, full production ontology (no OAEI "use_in_alignment"
    # annotation at all) must not raise -- it should just report nothing
    # to ignore, instead of calling the real get_ignored_class_index()
    # (which would hit deeponto's KeyError bug).
    onto = _FakeOnto(annotation_property_iris=[])
    assert m._get_ignored_class_index_safe(onto) == set()


def test_delegates_to_real_get_ignored_class_index_when_annotation_property_is_declared(monkeypatch):
    onto = _FakeOnto(annotation_property_iris=[m._OAEI_USE_IN_ALIGNMENT_IRI])

    calls = []

    def fake_get_ignored_class_index(passed_onto):
        calls.append(passed_onto)
        return {"http://example.org/ignored_class"}

    monkeypatch.setattr(m, "get_ignored_class_index", fake_get_ignored_class_index)

    result = m._get_ignored_class_index_safe(onto)

    assert result == {"http://example.org/ignored_class"}
    assert calls == [onto]
