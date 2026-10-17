from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

# Must be imported before any `deeponto` import below -- see
# _deeponto_compat.py's _init_jvm() for why (starts the JVM non-
# interactively so DeepOnto doesn't block on a stdin prompt).
from genom_pipeline import _deeponto_compat  # noqa: F401

from deeponto.utils import Tokenizer
from deeponto.onto import Ontology
from deeponto.align.bertmap import BERTMapPipeline
from deeponto.align.mapping import EntityMapping
from deeponto.align.oaei import get_ignored_class_index, remove_ignored_mappings

from textdistance import levenshtein

MappingTuple = Tuple[str, str, float]

# The OAEI Bio-ML evaluation ontologies annotate some classes with this
# property to say whether they should be scored ("use_in_alignment"); only
# the officially OAEI-curated ontologies carry it -- a real, full
# production ontology release (e.g. the real SNOMED-CT or FMA, not an
# OAEI Bio-ML-trimmed copy of them) never does.
_OAEI_USE_IN_ALIGNMENT_IRI = "http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment"


def _get_ignored_class_index_safe(onto: "Ontology") -> Set[str]:
    """
    Same as deeponto.align.oaei.get_ignored_class_index(onto), except it
    first checks whether `onto` actually declares the OAEI
    "use_in_alignment" annotation property before delegating to it.

    get_ignored_class_index() calls Ontology.get_annotations(class_obj,
    _OAEI_USE_IN_ALIGNMENT_IRI) for every class in `onto`. get_annotations()'s
    own docstring promises an empty return when the annotation property IRI
    doesn't exist in the ontology -- but the real implementation (via
    get_owl_object()) raises `KeyError: Cannot retrieve unknown IRI: ...`
    instead, a genuine docstring/implementation mismatch in the installed
    deeponto version. This only ever manifests on a real, full production
    ontology (no OAEI "use_in_alignment" annotations at all) -- every
    OAEI Bio-ML evaluation ontology genuinely declares the property (even
    if few/no classes use it), so this never fired during OAEI-only testing.

    Fixed here, in our own wrapper, rather than by patching deeponto itself
    (unlike genom_pipeline/_deeponto_compat.py's DataHasValue patch, which
    changes deeponto's behaviour for every caller) -- there's no "patch
    get_annotations to actually honour its own docstring" that wouldn't
    also risk changing behaviour deeponto callers outside genom_pipeline
    rely on.
    """
    if _OAEI_USE_IN_ALIGNMENT_IRI not in onto.owl_annotation_properties:
        return set()
    return get_ignored_class_index(onto)


def _edit_similarity_mapping_score(
    src_class_annotations: Set[str],
    tgt_class_annotations: Set[str],
    string_match_only: bool = False,
) -> float:
    if not src_class_annotations or not tgt_class_annotations:
        warnings.warn("Empty annotations, return 0.0")
        return 0.0

    if len(src_class_annotations.intersection(tgt_class_annotations)) > 0:
        return 1.0

    if string_match_only:
        return 0.0

    annotation_pairs = itertools.product(src_class_annotations, tgt_class_annotations)
    sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in annotation_pairs]
    return max(sim_scores)


def _init_class_mapping(head: str, tail: str, score: float) -> EntityMapping:
    return EntityMapping(head, tail, "<EquivalentTo>", float(score))


@dataclass
class BERTMapLtStringMatcher:
    name: str = "bertmaplt_string"

    def run(self, config: Dict[str, Any]) -> List[MappingTuple]:
        src_owl = config["src_owl"]
        tgt_owl = config["tgt_owl"]
        reasoner = config.get("reasoner", "elk")
        pool_size = int(config.get("pool_size", 200))
        apply_lowercasing = bool(config.get("apply_lowercasing", True))

        bertmap_cfg = BERTMapPipeline.load_bertmap_config()
        src_onto = Ontology(src_owl, reasoner)
        tgt_onto = Ontology(tgt_owl, reasoner)

        src_annotation_index, _ = src_onto.build_annotation_index(
            bertmap_cfg.annotation_property_iris,
            apply_lowercasing=apply_lowercasing,
        )
        tgt_annotation_index, _ = tgt_onto.build_annotation_index(
            bertmap_cfg.annotation_property_iris,
            apply_lowercasing=apply_lowercasing,
        )

        tokenizer = Tokenizer.from_pretrained(bertmap_cfg.bert.pretrained_path)
        tgt_inverted_annotation_index = Ontology.build_inverted_annotation_index(tgt_annotation_index, tokenizer)

        mappings: List[EntityMapping] = []
        for src_iri in src_annotation_index.keys():
            src_ann = src_annotation_index[src_iri]
            tgt_candidates = tgt_inverted_annotation_index.idf_select(
                list(src_ann),
                pool_size=len(tgt_annotation_index.keys()),
            )
            tgt_candidates = tgt_candidates[:pool_size]

            for tgt_iri, _idf in tgt_candidates:
                tgt_ann = tgt_annotation_index[tgt_iri]
                prelim = _edit_similarity_mapping_score(src_ann, tgt_ann, string_match_only=True)
                if prelim > 0.0:
                    mappings.append(_init_class_mapping(src_iri, tgt_iri, prelim))

        ignored = _get_ignored_class_index_safe(src_onto)
        ignored.update(_get_ignored_class_index_safe(tgt_onto))
        mappings = remove_ignored_mappings(mappings, ignored)

        return [m.to_tuple(with_score=True) for m in mappings]