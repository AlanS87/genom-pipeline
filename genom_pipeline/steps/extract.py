from __future__ import annotations

from typing import Any, Dict, List
import torch

from deeponto.onto import Ontology, OntologyVerbaliser, OntologyReasoner
from deeponto.align.bertmap import BERTMapPipeline


def truncate_labels(labels: List[str], cut_off: int = 3) -> List[str]:
    """Prevent token overflow by keeping the longest labels."""
    labels = list(labels)
    if len(labels) >= cut_off:
        labels.sort(key=len, reverse=True)
    return labels[:cut_off]


def _safe_get_first_label(labels: List[str], iri: str) -> str:
    """Fallback if a class has no label annotation."""
    if labels:
        return labels[0]
    frag = iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return frag if frag else iri


def extract_ontology_dict(
    onto: Ontology,
    cf: Any,
    max_parent_labels: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract concept-level information from an ontology.

    Returns:
      {
        iri: {
          "label": str,
          "synonyms": List[str],
          "cpx_exp": Optional[str],
          "parents": List[str]
        }
      }
    """
    annotation_index, _ = onto.build_annotation_index(cf.annotation_property_iris)
    iri_list = list(annotation_index.keys())

    # --- complex equivalence axioms ---
    verb = OntologyVerbaliser(onto)
    cpx_eqv = onto.get_equivalence_axioms("Classes")

    complex_classes = []
    atomic_classes = []
    for eqv in cpx_eqv:
        gci = list(eqv.asOWLSubClassOfAxioms())[0]
        super_class = gci.getSuperClass()
        sub_class = gci.getSubClass()
        if not OntologyReasoner.has_iri(super_class):
            complex_classes.append(super_class)
            atomic_classes.append(sub_class)
        if not OntologyReasoner.has_iri(sub_class):
            complex_classes.append(sub_class)
            atomic_classes.append(super_class)

    atomic_iri: List[str] = []
    complex_nl: List[str] = []
    for a in atomic_classes:
        parsed = verb.parser.parse(a).children[0]
        atomic_iri.append(parsed.text.lstrip("<").rstrip(">"))
    for c in complex_classes:
        complex_nl.append(verb.verbalise_class_expression(c).verbal)

    cpx_dict = {k: v for k, v in zip(atomic_iri, complex_nl)}

    # --- build ontology dict ---
    onto_dict: Dict[str, Dict[str, Any]] = {}

    for iri in iri_list:
        concept = onto.get_owl_object(iri)

        # parents
        parents = []
        asserted = onto.get_asserted_parents(concept, named_only=True)
        inferred = onto.reasoner.get_inferred_super_entities(concept, True)

        for p in asserted:
            p_iri = str(p.getIRI())
            if p_iri in annotation_index:
                parents += truncate_labels(list(annotation_index[p_iri]), cut_off=1)

        for pi in inferred:
            if pi in annotation_index:
                parents += truncate_labels(list(annotation_index[pi]), cut_off=1)

        parents = list(dict.fromkeys(parents))
        if len(parents) > max_parent_labels:
            parents = parents[:max_parent_labels]

        labels = list(annotation_index.get(iri, []))
        label = _safe_get_first_label(labels, iri)
        synonyms = labels[1:] if len(labels) > 1 else []

        onto_dict[iri] = {
            "label": label,
            "synonyms": synonyms,
            "cpx_exp": None,
            "parents": parents,
        }

    # attach complex expressions
    for k, v in cpx_dict.items():
        if k in onto_dict:
            onto_dict[k]["cpx_exp"] = v
        else:
            onto_dict[k] = {
                "label": _safe_get_first_label([], k),
                "synonyms": [],
                "cpx_exp": v,
                "parents": [],
            }

    return onto_dict


def run(
    src_onto_path: str,
    tgt_onto_path: str,
    output_pt_path: str,
) -> str:
    """
    Step 1: ontology -> concept store (.pt)

    Output format:
      {
        "meta": {...},
        "src": {...},
        "tgt": {...}
      }
    """
    cf = BERTMapPipeline.load_bertmap_config()

    src_onto = Ontology(src_onto_path)
    tgt_onto = Ontology(tgt_onto_path)

    src_dict = extract_ontology_dict(src_onto, cf)
    tgt_dict = extract_ontology_dict(tgt_onto, cf)

    store = {
        "meta": {
            "src_onto_path": src_onto_path,
            "tgt_onto_path": tgt_onto_path,
            "annotation_property_iris": list(cf.annotation_property_iris),
        },
        "src": src_dict,
        "tgt": tgt_dict,
    }

    torch.save(store, output_pt_path)
    return output_pt_path