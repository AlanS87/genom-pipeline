from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from deeponto.utils import Tokenizer
from deeponto.onto import Ontology
from deeponto.align.bertmap import BERTMapPipeline
from deeponto.align.mapping import EntityMapping
from deeponto.align.oaei import get_ignored_class_index, remove_ignored_mappings

from textdistance import levenshtein

MappingTuple = Tuple[str, str, float]


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

        ignored = get_ignored_class_index(src_onto)
        ignored.update(get_ignored_class_index(tgt_onto))
        mappings = remove_ignored_mappings(mappings, ignored)

        return [m.to_tuple(with_score=True) for m in mappings]