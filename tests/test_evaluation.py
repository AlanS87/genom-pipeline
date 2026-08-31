"""
Tests for genom_pipeline.evaluation.metrics: precision/recall/F1 against a
gold reference, and the threshold grid search that sweeps
(llm_threshold, cs_threshold) directly on an existing llm_alignment.csv.
"""
from __future__ import annotations

import pandas as pd
import pytest

from genom_pipeline.evaluation import metrics


@pytest.fixture
def llm_alignment_csv(tmp_path):
    rows = [
        {"src_iri": "A", "tgt_iri": "X1", "score": 0.95, "rank": 1, "confidence": 0.92, "decision": "YES"},
        {"src_iri": "A", "tgt_iri": "X2", "score": 0.95, "rank": 2, "confidence": 0.99, "decision": "YES"},
        {"src_iri": "B", "tgt_iri": "T2", "score": 0.92, "rank": 1, "confidence": 0.95, "decision": "YES"},
        {"src_iri": "C", "tgt_iri": "Y1", "score": 0.50, "rank": 1, "confidence": 0.95, "decision": "YES"},
        {"src_iri": "D", "tgt_iri": "Z1", "score": 0.95, "rank": 1, "confidence": 0.30, "decision": "YES"},
    ]
    path = tmp_path / "llm_alignment.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def reference_tsv(tmp_path):
    path = tmp_path / "reference.tsv"
    pd.DataFrame(
        [{"SrcEntity": "B", "TgtEntity": "T2"}, {"SrcEntity": "A", "TgtEntity": "X1"}]
    ).to_csv(path, sep="\t", index=False)
    return path


def test_precision_recall_f1_basic():
    predicted = {("A", "X1"), ("B", "T2"), ("C", "Y1")}
    reference = {("A", "X1"), ("B", "T2")}
    result = metrics.precision_recall_f1(predicted, reference)
    assert result["n_correct"] == 2
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)


def test_precision_recall_f1_empty_predicted_is_zero_not_nan():
    result = metrics.precision_recall_f1(set(), {("A", "X1")})
    assert result["precision"] == 0.0
    assert result["f1"] == 0.0


def test_grid_search_thresholds_reaches_perfect_f1_at_cs_0_9(llm_alignment_csv, reference_tsv):
    grid = metrics.grid_search_thresholds(
        llm_alignment_csv=str(llm_alignment_csv),
        reference_path=str(reference_tsv),
        llm_thresholds=[0.5, 0.9],
        cs_thresholds=[0.5, 0.9],
    )

    assert len(grid) == 4
    perfect = grid[grid["f1"] == 1.0]
    assert not perfect.empty
    assert (perfect["cs_threshold"] == 0.9).all()

    # cs_threshold=0.5 lets the C false positive through -> should score below 1.0
    lax = grid[grid["cs_threshold"] == 0.5]
    assert (lax["f1"] < 1.0).all()


def test_best_thresholds_picks_max_f1(llm_alignment_csv, reference_tsv):
    grid = metrics.grid_search_thresholds(
        llm_alignment_csv=str(llm_alignment_csv),
        reference_path=str(reference_tsv),
        llm_thresholds=[0.5, 0.9],
        cs_thresholds=[0.5, 0.9],
    )
    best = metrics.best_thresholds(grid)
    assert best["f1"] == grid["f1"].max()
