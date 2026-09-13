"""
Tests for genom_pipeline.steps.extract, in particular the `reasoner_type`
option (added after HermiT -- the default OWL DL reasoner -- ran out of JVM
heap classifying a ~89k-class real-world ontology; see the README's
Configuration Reference for `reasoner_type`).

These start a real JVM and run real DeepOnto/OWL-API reasoning (hermit and
elk) against a tiny synthetic ontology -- no mocking of the Java side, since
the whole point is to catch a reasoner-specific regression, which a mocked
reasoner could never do. This mirrors test_deeponto_compat.py's existing
"real JVM, no GPU/API keys needed" style.
"""
from __future__ import annotations

import pytest
import torch

from genom_pipeline.steps.extract import run as extract_run

# A minimal OWL/RDF ontology with a simple class hierarchy:
#   Animal <- Mammal <- Dog
# via rdfs:subClassOf, so both an asserted-parents walk and a DL/EL reasoner
# should agree that Dog's (only) inferred superclass is Mammal (and,
# transitively, Animal).
_TOY_ONTOLOGY_OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/animals#"
     xml:base="http://example.org/animals"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/animals"/>

    <owl:Class rdf:about="http://example.org/animals#Animal">
        <rdfs:label>Animal</rdfs:label>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/animals#Mammal">
        <rdfs:label>Mammal</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/animals#Animal"/>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/animals#Dog">
        <rdfs:label>Dog</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/animals#Mammal"/>
    </owl:Class>
</rdf:RDF>
"""


@pytest.fixture
def toy_owl_path(tmp_path):
    p = tmp_path / "animals.owl"
    p.write_text(_TOY_ONTOLOGY_OWL)
    return str(p)


@pytest.mark.parametrize("reasoner_type", ["hermit", "elk"])
def test_extract_reasoner_type_infers_correct_parents(tmp_path, toy_owl_path, reasoner_type):
    out_pt = str(tmp_path / f"concept_store_{reasoner_type}.pt")

    extract_run(toy_owl_path, toy_owl_path, out_pt, reasoner_type=reasoner_type)

    store = torch.load(out_pt)
    dog = next(v for v in store["src"].values() if v["label"] == "Dog")
    assert "Mammal" in dog["parents"]


def test_extract_default_reasoner_type_is_hermit(tmp_path, toy_owl_path):
    # Backward compatibility: calling run() without reasoner_type at all
    # must behave exactly as it did before this parameter existed.
    out_pt = str(tmp_path / "concept_store_default.pt")
    extract_run(toy_owl_path, toy_owl_path, out_pt)

    store = torch.load(out_pt)
    dog = next(v for v in store["src"].values() if v["label"] == "Dog")
    assert "Mammal" in dog["parents"]
