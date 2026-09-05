"""
Tests for genom_pipeline.steps.judge -- specifically the incremental
checkpoint/resume logic added on top of the existing batch-scoring path.

The original run() only wrote output_alignment_csv once, after every
candidate pair had been scored -- a job killed partway through (e.g. a
SLURM walltime limit) lost the entire alignment CSV, not just the in-flight
batch. These tests exercise run()'s orchestration -- which rows get
(re-)scored, when the checkpoint file is written, and that resuming from
one produces the same final result as an uninterrupted run -- using a fake
deterministic scorer in place of _score_batch, so no GPU/model download is
needed (same style as tests/test_fusion.py and tests/test_define_batching.py).
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pandas as pd
import pytest
import torch

# Same shadowing gotcha as test_define_batching.py: steps/__init__.py's
# `from .judge import run as judge` overwrites the `judge` submodule name on
# the package namespace with the `run` function, and that shadowing is what
# `import genom_pipeline.steps.judge as judge` resolves through too (Python
# attribute-walks the parent packages for `import a.b.c as x`). Only
# importlib.import_module() reaches the real module object.
judge = importlib.import_module("genom_pipeline.steps.judge")


def _make_store(n_src: int, n_tgt: int) -> Dict[str, Any]:
    return {
        "src": {f"s{i}": {"label": f"src-{i}"} for i in range(n_src)},
        "tgt": {f"t{i}": {"label": f"tgt-{i}"} for i in range(n_tgt)},
    }


def _make_candidates_csv(path, pairs) -> None:
    pd.DataFrame(
        [{"src_iri": s, "tgt_iri": t, "score": 0.9, "rank": 1} for s, t in pairs]
    ).to_csv(path, index=False)


class _FakeTokenizer:
    """_score_batch is monkeypatched out entirely in these tests, so the
    only real tokenizer method run() exercises is apply_chat_template
    (via _build_pair_prompt), building the prompt _score_batch would
    otherwise consume."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[1]["content"]


@pytest.fixture
def fake_judge(monkeypatch):
    j = judge.HFYesNoJudge(
        model_name="fake/model", tokenizer=_FakeTokenizer(), model=object(), yes_ids=[1], no_ids=[2]
    )
    monkeypatch.setattr(judge, "_load_judge_model", lambda *a, **k: j)
    return j


def _install_deterministic_scorer(monkeypatch, calls: List[List[str]]):
    """Every prompt scores 1.0 -> decision YES at the default threshold.
    `calls` records exactly which prompt batches actually reached the
    (fake) model, so tests can assert already-scored rows are never
    resubmitted."""

    def fake_score_batch(j, prompts):
        calls.append(list(prompts))
        return [1.0 for _ in prompts]

    monkeypatch.setattr(judge, "_score_batch", fake_score_batch)


def test_full_run_scores_every_candidate(fake_judge, monkeypatch, tmp_path):
    calls: List[List[str]] = []
    _install_deterministic_scorer(monkeypatch, calls)

    store = _make_store(2, 2)
    pt_path = tmp_path / "store.pt"
    torch.save(store, pt_path)

    candidates_csv = tmp_path / "candidates.csv"
    _make_candidates_csv(candidates_csv, [("s0", "t0"), ("s0", "t1"), ("s1", "t0")])

    out_path = tmp_path / "llm_alignment.csv"
    judge.run(
        store_pt_path=str(pt_path),
        candidates_csv=str(candidates_csv),
        output_alignment_csv=str(out_path),
        llm_model="fake/model",
        llm_config={"batch_size": 2},
    )

    df = pd.read_csv(out_path)
    assert len(df) == 3
    assert (df["decision"] == "YES").all()
    assert sum(len(c) for c in calls) == 3

    # The real output now exists -- any checkpoint left behind must be gone.
    leftover = list((tmp_path / "checkpoints").glob("*_judge_checkpoint.csv"))
    assert leftover == []


def test_interrupted_run_resumes_without_rescoring_completed_rows(fake_judge, monkeypatch, tmp_path):
    store = _make_store(4, 1)
    pt_path = tmp_path / "store.pt"
    torch.save(store, pt_path)

    candidates_csv = tmp_path / "candidates.csv"
    pairs = [(f"s{i}", "t0") for i in range(4)]
    _make_candidates_csv(candidates_csv, pairs)

    out_path = tmp_path / "llm_alignment.csv"

    # Simulate a job killed partway through: batch_size=1 with a checkpoint
    # after every batch, and a scorer that raises once 2 rows have already
    # gone through it -- standing in for a SLURM walltime kill hitting
    # mid-run rather than between rows.
    calls_before_interrupt: List[List[str]] = []

    def flaky_score_batch(j, prompts):
        if len(calls_before_interrupt) >= 2:
            raise RuntimeError("simulated SLURM walltime kill")
        calls_before_interrupt.append(list(prompts))
        return [1.0 for _ in prompts]

    monkeypatch.setattr(judge, "_score_batch", flaky_score_batch)

    with pytest.raises(RuntimeError, match="simulated SLURM walltime kill"):
        judge.run(
            store_pt_path=str(pt_path),
            candidates_csv=str(candidates_csv),
            output_alignment_csv=str(out_path),
            llm_model="fake/model",
            llm_config={"batch_size": 1, "checkpoint_every_n_batches": 1},
        )

    assert not out_path.exists()  # the run never reached the final write

    checkpoint_files = list((tmp_path / "checkpoints").glob("*_judge_checkpoint.csv"))
    assert len(checkpoint_files) == 1
    checkpoint_df = pd.read_csv(checkpoint_files[0])
    assert checkpoint_df["confidence"].notna().sum() == 2  # exactly the rows scored before the "kill"

    # Resume: a fresh call picks up the checkpoint and must only score the
    # 2 rows that were never reached, not redo the 2 already on disk.
    resume_calls: List[List[str]] = []
    _install_deterministic_scorer(monkeypatch, resume_calls)

    judge.run(
        store_pt_path=str(pt_path),
        candidates_csv=str(candidates_csv),
        output_alignment_csv=str(out_path),
        llm_model="fake/model",
        llm_config={"batch_size": 1, "checkpoint_every_n_batches": 1},
    )

    assert sum(len(c) for c in resume_calls) == 2

    df = pd.read_csv(out_path)
    assert len(df) == 4
    assert (df["decision"] == "YES").all()
    assert list((tmp_path / "checkpoints").glob("*_judge_checkpoint.csv")) == []
