"""
Tests for genom_pipeline.steps.fusion.

These exercise the threshold filtering + rank-1 selection + 1-to-1 dedup
logic described in the fusion.run() docstring, using a synthetic
llm_alignment.csv so no ontology files, GPU, or API keys are needed.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from genom_pipeline.steps import fusion


@pytest.fixture
def llm_alignment_csv(tmp_path):
    rows = [
        # src A: two candidates both pass thresholds -> rank1 (A,X1) must win
        # over the higher-confidence rank2 (A,X2).
        {"src_iri": "A", "tgt_iri": "X1", "score": 0.95, "rank": 1, "confidence": 0.92, "decision": "YES"},
        {"src_iri": "A", "tgt_iri": "X2", "score": 0.95, "rank": 2, "confidence": 0.99, "decision": "YES"},
        # src B: single candidate, passes both thresholds -> kept.
        {"src_iri": "B", "tgt_iri": "T2", "score": 0.92, "rank": 1, "confidence": 0.95, "decision": "YES"},
        # src C: cosine similarity (0.5) is below cs_threshold (0.9) -> dropped.
        {"src_iri": "C", "tgt_iri": "Y1", "score": 0.50, "rank": 1, "confidence": 0.95, "decision": "YES"},
        # src D: confidence (0.3) is below llm_threshold (0.9) -> dropped.
        {"src_iri": "D", "tgt_iri": "Z1", "score": 0.95, "rank": 1, "confidence": 0.30, "decision": "YES"},
        # src E: passes both thresholds but targets the same tgt as B (T2),
        # and sorts after B -> dropped by target-uniqueness dedup.
        {"src_iri": "E", "tgt_iri": "T2", "score": 0.91, "rank": 1, "confidence": 0.96, "decision": "YES"},
    ]
    path = tmp_path / "llm_alignment.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def store_pt(tmp_path):
    path = tmp_path / "concept_store.pt"
    torch.save({"meta": {}}, path)
    return path


def test_fusion_applies_both_thresholds_and_target_uniqueness(llm_alignment_csv, store_pt, tmp_path):
    out_path = tmp_path / "final_alignment.csv"

    fusion.run(
        store_pt_path=str(store_pt),
        llm_alignment_csv=str(llm_alignment_csv),
        output_final_csv=str(out_path),
        fuse_config={
            "strategy": "exact_priority",
            "llm_threshold": 0.9,
            "cs_threshold": 0.9,
            "enforce_target_uniqueness": True,
        },
    )

    df_out = pd.read_csv(out_path)
    got = set(zip(df_out["src_iri"], df_out["tgt_iri"]))

    assert got == {("A", "X1"), ("B", "T2")}
    assert df_out["src_iri"].is_unique
    assert df_out["tgt_iri"].is_unique


def test_fusion_target_uniqueness_can_be_disabled(llm_alignment_csv, store_pt, tmp_path):
    out_path = tmp_path / "final_alignment.csv"

    fusion.run(
        store_pt_path=str(store_pt),
        llm_alignment_csv=str(llm_alignment_csv),
        output_final_csv=str(out_path),
        fuse_config={
            "strategy": "exact_priority",
            "llm_threshold": 0.9,
            "cs_threshold": 0.9,
            "enforce_target_uniqueness": False,
        },
    )

    df_out = pd.read_csv(out_path)
    got = set(zip(df_out["src_iri"], df_out["tgt_iri"]))

    assert got == {("A", "X1"), ("B", "T2"), ("E", "T2")}
    assert not df_out["tgt_iri"].is_unique


def test_fusion_requires_score_column_for_cs_threshold(store_pt, tmp_path):
    # llm_alignment_csv missing the 'score' column entirely
    path = tmp_path / "llm_alignment_no_score.csv"
    pd.DataFrame(
        [{"src_iri": "A", "tgt_iri": "X1", "confidence": 0.95, "decision": "YES"}]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="cs_threshold"):
        fusion.run(
            store_pt_path=str(store_pt),
            llm_alignment_csv=str(path),
            output_final_csv=str(tmp_path / "out.csv"),
            fuse_config={"cs_threshold": 0.9},
        )


def test_dedup_one_to_one_priority_order_wins():
    rows = [
        ("A", "X", 0.9, "exact"),
        ("A", "Y", 0.99, "llm"),  # same src as an earlier row -> dropped
        ("B", "X", 0.5, "llm"),  # same tgt as an earlier row -> dropped
    ]
    out = fusion.dedup_one_to_one(rows, enforce_target_uniqueness=True)
    assert out == [("A", "X", 0.9, "exact")]
