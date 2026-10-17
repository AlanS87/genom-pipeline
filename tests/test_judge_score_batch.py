"""
Tests for genom_pipeline.steps.judge's _score_batch() / _load_judge_model()
left-padding + position_ids + logits_to_keep=1 fix for the CUDA OOM
described in judge.py's _score_batch() docstring: a plain model(**inputs)
call defaults logits_to_keep=0, which the installed transformers version
treats as "keep every position" rather than "keep zero", materializing a
full [batch, seq_len, vocab] logits tensor every batch instead of just the
last position needed for the YES/NO decision token.

These use lightweight fakes, not a real model/tokenizer/GPU -- the real
behavior they exercise is pure tensor-shape/indexing logic that doesn't
need an actual causal LM to verify.
"""
from __future__ import annotations

import importlib

import torch

# genom_pipeline/steps/__init__.py does `from .judge import run as judge`,
# which overwrites the *package's* `judge` attribute with the `run`
# function itself -- `from genom_pipeline.steps import judge` would import
# that function, not the module. import_module() resolves by the full
# dotted name through sys.modules instead, which gets the real module.
judge_mod = importlib.import_module("genom_pipeline.steps.judge")


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self, input_ids, attention_mask):
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self.calls = []
        self.padding_side = "right"  # tokenizer default, flipped by _load_judge_model
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.pad_token = None

    def __call__(self, prompts, return_tensors="pt", padding=True, truncation=True):
        self.calls.append(prompts)
        return _FakeBatchEncoding(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask,
        )

    def encode(self, text, add_special_tokens=False):
        return [hash(text) % 1000]


class _FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.device = "cpu"
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        logits_to_keep = kwargs.get("logits_to_keep", 0)
        kept = seq_len if logits_to_keep in (0, None) else int(logits_to_keep)
        # Deterministic per-row logits: row i's last kept position has a
        # spike at vocab index i, so the test can tell which row's logits
        # actually got read back out.
        logits = torch.zeros(batch_size, kept, self.vocab_size)
        for i in range(batch_size):
            logits[i, -1, i] = 100.0
        return _FakeModelOutput(logits)


def _make_judge(input_ids, attention_mask, vocab_size=8):
    tokenizer = _FakeTokenizer(input_ids, attention_mask)
    model = _FakeModel(vocab_size)
    return judge_mod.HFYesNoJudge(
        model_name="fake",
        tokenizer=tokenizer,
        model=model,
        yes_ids=[0],
        no_ids=[1],
    ), tokenizer, model


def test_score_batch_requests_only_the_last_position():
    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    judge, tokenizer, model = _make_judge(input_ids, attention_mask)

    judge_mod._score_batch(judge, ["p1", "p2"])

    assert model.calls[0]["logits_to_keep"] == 1


def test_score_batch_position_ids_correct_for_left_padding():
    # Row 0 has one left pad token (real tokens start at index 1); row 1
    # has no padding at all.
    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    judge, tokenizer, model = _make_judge(input_ids, attention_mask)

    judge_mod._score_batch(judge, ["p1", "p2"])

    position_ids = model.calls[0]["position_ids"]
    expected = torch.tensor([[1, 0, 1, 2], [0, 1, 2, 3]])
    assert torch.equal(position_ids, expected)


def test_score_batch_reads_confidence_from_the_kept_last_position():
    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    judge, tokenizer, model = _make_judge(input_ids, attention_mask, vocab_size=8)
    # yes_ids=[0] gets the spike for row 0 (vocab index 0); row 1's spike
    # lands on vocab index 1 which is no_ids, so row 1 should score low.
    conf = judge_mod._score_batch(judge, ["p1", "p2"])

    assert conf[0] > 0.9  # row 0: spike on YES token
    assert conf[1] < 0.1  # row 1: spike on NO token


def test_load_judge_model_sets_left_padding(monkeypatch):
    class _StubTokenizer:
        pad_token_id = None
        eos_token = "<eos>"
        padding_side = "right"

        def __init__(self, *a, **kw):
            self.pad_token = None

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def encode(self, text, add_special_tokens=False):
            return [1]

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def eval(self):
            pass

    monkeypatch.setattr(judge_mod, "AutoTokenizer", _StubTokenizer)
    monkeypatch.setattr(judge_mod, "AutoModelForCausalLM", _StubModel)

    result = judge_mod._load_judge_model("fake-model", hf_token=None, torch_dtype="bfloat16", device_map="cpu")

    assert result.tokenizer.padding_side == "left"
