"""
Tests for genom_pipeline.steps.define -- specifically the batched-generation
and checkpointing behaviour added to fix two problems in the original
implementation:

1. The original run() called llm.model.generate() once per concept (no
   batching at all), which is why judge.py's batch_size-based design exists
   and define.py's did not -- at real dataset sizes (tens of thousands of
   concepts) this made definition generation the pipeline's dominant cost.
2. The original run() only wrote concept_store.pt once, after every concept
   on both sides had been processed -- a job killed partway through (e.g. a
   SLURM walltime limit) lost *all* progress back to the previous full run,
   not just the in-flight batch.

Uses a small fake HFChatLLM (fake tokenizer + fake model, no real weights)
so this runs on CPU with no GPU/network/model download -- same style as
tests/test_fusion.py and tests/test_evaluation.py.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest
import torch

# genom_pipeline/steps/__init__.py does `from .define import run as define`,
# which shadows the `define` submodule name on the package namespace with
# the `run` function -- both `from genom_pipeline.steps import define` and
# `import genom_pipeline.steps.define as define` resolve through that same
# (shadowed) package attribute and hand back the function, not this module.
# importlib.import_module() is the one way to get the real module object.
define = importlib.import_module("genom_pipeline.steps.define")


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """
    Deterministic stand-in for a real HF chat tokenizer. Encodes each
    already-rendered chat string to a list of character codes, left-padded
    to the batch's longest row (mirroring the real padding_side="left"
    contract define.py now relies on). "Generation" is provided by
    _FakeModel.generate below; batch_decode here turns the generated tokens
    back into a per-row-identifiable string so tests can check the right
    output landed on the right concept.
    """

    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # The concept's prompt text is the user message; keep it recoverable.
        return messages[1]["content"]

    def __call__(self, texts: List[str], return_tensors=None, padding=None):
        rows = [[ord(c) % 50 + 10 for c in t] for t in texts]
        width = max(len(r) for r in rows)
        padded = [[self.pad_token_id] * (width - len(r)) + r for r in rows]
        input_ids = torch.tensor(padded, dtype=torch.long)
        attn = (input_ids != self.pad_token_id).long()
        return _FakeBatchEncoding(input_ids=input_ids, attention_mask=attn)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        # Each row's "generated" tokens are all equal to that row's batch
        # index (see _FakeModel.generate) -- decode that back to a string
        # unique per row so tests can assert the right row got the right
        # definition, not just "some string came back".
        return [f"definition-{int(row[0])}" for row in token_ids]


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids, attention_mask, **gen_kwargs):
        n_rows = input_ids.shape[0]
        new_tokens = gen_kwargs.get("max_new_tokens", 3)
        # Row i's "generated" continuation is filled with i, so batch_decode
        # can recover which row it came from regardless of batch order.
        suffix = torch.tensor([[i] * new_tokens for i in range(n_rows)], dtype=torch.long)
        return torch.cat([input_ids, suffix], dim=1)


def _make_store(labels: Dict[str, str]) -> Dict[str, Any]:
    return {
        "src": {
            iri: {"label": label, "synonyms": [], "parents": [], "cpx_exp": None}
            for iri, label in labels.items()
        },
        "tgt": {},
    }


@pytest.fixture
def fake_llm(monkeypatch):
    llm = define.HFChatLLM(model_name="fake/model", tokenizer=_FakeTokenizer(), model=_FakeModel())
    monkeypatch.setattr(define, "_load_hf_chat_model", lambda *a, **k: llm)
    return llm


def test_batch_generation_lands_on_the_right_concept(fake_llm, tmp_path):
    """
    A single generate() call serves the whole batch -- this is the failure
    mode a naive left-pad implementation gets wrong if it slices each row's
    new tokens at that row's own (pre-padding) length instead of at the
    shared, padded input_len column.
    """
    store = _make_store({f"c{i}": f"concept-{i}" for i in range(5)})
    pt_path = tmp_path / "store.pt"
    torch.save(store, pt_path)

    define.run(str(pt_path), llm_model="fake/model", llm_config={"batch_size": 16}, overwrite=True)

    result = torch.load(pt_path)
    for i in range(5):
        assert result["src"][f"c{i}"]["definition"] == f"definition-{i}"


def test_partial_trailing_batch_is_flushed(fake_llm, tmp_path):
    """
    7 concepts with batch_size=4 must still generate a definition for all 7
    (one full batch of 4, one trailing partial batch of 3) -- forgetting the
    final flush() after the loop would silently drop the tail.
    """
    store = _make_store({f"c{i}": f"concept-{i}" for i in range(7)})
    pt_path = tmp_path / "store.pt"
    torch.save(store, pt_path)

    define.run(str(pt_path), llm_model="fake/model", llm_config={"batch_size": 4}, overwrite=True)

    result = torch.load(pt_path)
    for i in range(7):
        assert "definition" in result["src"][f"c{i}"]
        assert result["src"][f"c{i}"]["definition"]


def test_existing_definitions_are_skipped_not_rebatched(fake_llm, monkeypatch, tmp_path):
    """Concepts that already have a definition (overwrite=False) must not be
    sent through generation at all -- neither alone nor bundled into a
    batch with other concepts."""
    store = _make_store({"c0": "concept-0", "c1": "concept-1"})
    store["src"]["c0"]["definition"] = "already there"

    calls: List[List[str]] = []
    original = define._definition_generation_batch

    def spy(llm, chat_texts, gen_params):
        calls.append(list(chat_texts))
        return original(llm, chat_texts, gen_params)

    monkeypatch.setattr(define, "_definition_generation_batch", spy)

    pt_path = tmp_path / "store.pt"
    torch.save(store, pt_path)

    define.run(str(pt_path), llm_model="fake/model", llm_config={"batch_size": 16}, overwrite=False)

    result = torch.load(pt_path)
    assert result["src"]["c0"]["definition"] == "already there"
    assert "definition" in result["src"]["c1"]
    assert len(calls) == 1
    assert len(calls[0]) == 1  # only c1's prompt went through generation


def test_checkpointing_saves_progress_before_run_finishes(fake_llm, monkeypatch, tmp_path):
    """
    With checkpoint_every_n_batches=1, concept_store.pt must be re-saved to
    disk after each batch, not only once at the very end -- this is what
    lets a killed/timed-out SLURM job resume from the last checkpoint
    instead of losing everything back to the previous full run.
    """
    store = _make_store({f"c{i}": f"concept-{i}" for i in range(4)})
    pt_path = tmp_path / "store.pt"
    torch.save(store, pt_path)

    save_calls: List[str] = []
    original_save = torch.save

    def spy_save(obj, path):
        save_calls.append(str(path))
        return original_save(obj, path)

    monkeypatch.setattr(torch, "save", spy_save)

    define.run(
        str(pt_path),
        llm_model="fake/model",
        llm_config={"batch_size": 2, "checkpoint_every_n_batches": 1},
        overwrite=True,
    )

    # 4 concepts / batch_size 2 = 2 batches -> 2 checkpoint saves + 1 final
    # save at the end of run() = 3 total, all via the atomic tmp-then-replace
    # helper (so every save_calls entry targets the ".tmp" sibling path).
    assert len(save_calls) == 3
    assert all(p.endswith(".tmp") for p in save_calls)

    result = torch.load(pt_path)
    for i in range(4):
        assert "definition" in result["src"][f"c{i}"]
