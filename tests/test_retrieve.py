"""
Tests for genom_pipeline.steps.retrieve, in particular the HNSW index-metric
bug fix: faiss.IndexHNSWFlat(d, M) defaults to METRIC_L2 unless a metric is
passed explicitly, unlike IndexFlatIP/IndexIVFFlat which are unambiguously
inner-product by construction. Since "hnsw" is retrieve.run()'s default
index_type, this silently turned every "score" column into an L2 distance
(lower = more similar) instead of the cosine similarity (higher = more
similar) every downstream consumer -- fusion.py's cs_threshold filter
included -- assumes. See retrieve.py's _build_faiss_index() for the fix
itself (passing faiss.METRIC_INNER_PRODUCT explicitly).

No GPU, JVM, or API keys needed: these build a tiny synthetic concept_store.pt
directly with hand-picked embedding vectors whose expected cosine
similarities are known exactly.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from genom_pipeline.steps.retrieve import run as retrieve_run

# One src concept, three tgt concepts with embeddings chosen so their cosine
# similarity to the src concept (after L2-normalization) is known exactly:
#   t1: identical direction  -> cosine similarity  1.0  (best match)
#   t2: orthogonal           -> cosine similarity  0.0
#   t3: opposite direction   -> cosine similarity -1.0  (worst match)
_SRC = {"s1": {"embedding": torch.tensor([2.0, 0.0, 0.0, 0.0])}}
_TGT = {
    "t1": {"embedding": torch.tensor([1.0, 0.0, 0.0, 0.0])},
    "t2": {"embedding": torch.tensor([0.0, 1.0, 0.0, 0.0])},
    "t3": {"embedding": torch.tensor([-1.0, 0.0, 0.0, 0.0])},
}


@pytest.fixture
def store_pt_path(tmp_path):
    p = tmp_path / "concept_store.pt"
    torch.save({"src": dict(_SRC), "tgt": dict(_TGT)}, p)
    return str(p)


@pytest.mark.parametrize("index_type", ["hnsw", "flatip"])
def test_retrieve_score_is_cosine_similarity_not_l2_distance(tmp_path, store_pt_path, index_type):
    out_csv = str(tmp_path / f"candidates_{index_type}.csv")

    retrieve_run(
        store_pt_path=store_pt_path,
        output_candidates_csv=out_csv,
        topk=3,
        overwrite=True,
        index_type=index_type,
    )

    df = pd.read_csv(out_csv).sort_values("rank")
    assert list(df["tgt_iri"]) == ["t1", "t2", "t3"], (
        f"{index_type}: expected best-to-worst order t1 > t2 > t3 by cosine "
        f"similarity, got {list(df['tgt_iri'])}"
    )

    scores = dict(zip(df["tgt_iri"], df["score"]))
    # Cosine similarity, not L2 distance: identical-direction t1 must score
    # near +1.0 (an L2-distance bug would instead report ~0.0, the *smallest*
    # value, for the best match -- backwards from what fusion.py's
    # cs_threshold filter assumes).
    assert scores["t1"] == pytest.approx(1.0, abs=1e-4)
    assert scores["t2"] == pytest.approx(0.0, abs=1e-4)
    assert scores["t3"] == pytest.approx(-1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# query_side: which side drives the faiss nearest-neighbour search. Output
# CSV columns (src_iri/tgt_iri/score/rank) must keep their canonical
# meaning either way -- only the search direction (and therefore which
# side's count the number of output rows scales with) changes.
# ---------------------------------------------------------------------------

def _unit(angle_deg: float) -> torch.Tensor:
    import math

    rad = math.radians(angle_deg)
    return torch.tensor([math.cos(rad), math.sin(rad), 0.0, 0.0])


# Angles chosen so query_side="src" and query_side="tgt" produce genuinely
# different nearest-neighbour pairs at topk=1: s0 (0deg) is closest to t0
# (15deg), but t1 (100deg) is closest to s1 (40deg), not s0.
_ASYM_SRC = {
    "s0": {"embedding": _unit(0.0)},
    "s1": {"embedding": _unit(40.0)},
}
_ASYM_TGT = {
    "t0": {"embedding": _unit(15.0)},
    "t1": {"embedding": _unit(100.0)},
}


@pytest.fixture
def asym_store_pt_path(tmp_path):
    p = tmp_path / "asym_concept_store.pt"
    torch.save({"src": dict(_ASYM_SRC), "tgt": dict(_ASYM_TGT)}, p)
    return str(p)


def test_query_side_default_is_src_and_output_scales_with_src_count(tmp_path, asym_store_pt_path):
    out_csv = str(tmp_path / "candidates.csv")
    retrieve_run(
        store_pt_path=asym_store_pt_path,
        output_candidates_csv=out_csv,
        topk=1,
        overwrite=True,
    )
    df = pd.read_csv(out_csv)
    # one row per src concept (2 src concepts, topk=1)
    assert len(df) == 2
    assert set(df["src_iri"]) == {"s0", "s1"}


def test_query_side_tgt_scales_with_tgt_count_and_keeps_canonical_output_schema(tmp_path, asym_store_pt_path):
    out_csv = str(tmp_path / "candidates.csv")
    retrieve_run(
        store_pt_path=asym_store_pt_path,
        output_candidates_csv=out_csv,
        topk=1,
        overwrite=True,
        query_side="tgt",
    )
    df = pd.read_csv(out_csv)
    # one row per tgt concept (2 tgt concepts, topk=1) -- the search is now
    # driven by tgt, but the output columns are still src_iri/tgt_iri/score/rank
    assert len(df) == 2
    assert list(df.columns) == ["src_iri", "tgt_iri", "score", "rank"]
    assert set(df["tgt_iri"]) == {"t0", "t1"}


def test_query_side_src_and_tgt_retrieve_genuinely_different_pairs(tmp_path, asym_store_pt_path):
    src_csv = str(tmp_path / "src_candidates.csv")
    tgt_csv = str(tmp_path / "tgt_candidates.csv")

    retrieve_run(store_pt_path=asym_store_pt_path, output_candidates_csv=src_csv, topk=1, overwrite=True, query_side="src")
    retrieve_run(store_pt_path=asym_store_pt_path, output_candidates_csv=tgt_csv, topk=1, overwrite=True, query_side="tgt")

    src_pairs = set(zip(pd.read_csv(src_csv)["src_iri"], pd.read_csv(src_csv)["tgt_iri"]))
    tgt_pairs = set(zip(pd.read_csv(tgt_csv)["src_iri"], pd.read_csv(tgt_csv)["tgt_iri"]))

    assert src_pairs != tgt_pairs, "query_side='src' and 'tgt' should retrieve different nearest-neighbour pairs for this asymmetric fixture"


def test_query_side_invalid_value_raises(tmp_path, asym_store_pt_path):
    out_csv = str(tmp_path / "candidates.csv")
    with pytest.raises(ValueError, match="query_side"):
        retrieve_run(
            store_pt_path=asym_store_pt_path,
            output_candidates_csv=out_csv,
            topk=1,
            overwrite=True,
            query_side="bogus",
        )
