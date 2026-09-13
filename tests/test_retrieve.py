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
