from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass(frozen=True)
class HFYesNoJudge:
    model_name: str
    tokenizer: Any
    model: Any
    yes_ids: List[int]
    no_ids: List[int]


def _first_token_ids(tokenizer, variants: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for v in variants:
        t = tokenizer.encode(v, add_special_tokens=False)
        if len(t) > 0:
            ids.append(t[0])
    return sorted(set(ids))


def _load_judge_model(model_name: str, hf_token: Optional[str], torch_dtype: str, device_map: str) -> HFYesNoJudge:
    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if torch_dtype:
        kwargs["torch_dtype"] = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.bfloat16
    if hf_token:
        kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token) if hf_token else AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Required for _score_batch()'s logits_to_keep=1 optimisation below to
    # be correct, not just fast: logits_to_keep=1 only returns "the last
    # real token's logits" if every row's last real token sits at the same
    # physical (last) index in the batch, which needs LEFT padding -- the
    # tokenizer default (right padding) puts each row's last real token at
    # a different index per row, which a single trailing slice can't
    # express. See _score_batch()'s docstring for the matching
    # position_ids fix left padding also requires.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    torch.set_grad_enabled(False)

    yes_ids = _first_token_ids(tokenizer, ["YES", "Yes", "yes", " YES", " Yes", " yes"])
    no_ids = _first_token_ids(tokenizer, ["NO", "No", "no", " NO", " No", " no"])

    return HFYesNoJudge(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        yes_ids=yes_ids,
        no_ids=no_ids,
    )


def _format_field(name: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if len(value) == 0:
            return ""
        return f"- {name}: {value}"
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return ""
        return f"- {name}: {v}"
    return f"- {name}: {value}"


def _build_pair_prompt(
    tokenizer,
    concept_a: Dict[str, Any],
    concept_b: Dict[str, Any],
    input_fields: Tuple[str, ...],
    domain: str,
    source_ontology_name: Optional[str],
    target_ontology_name: Optional[str],
) -> str:
    ontology_note = ""
    if source_ontology_name and target_ontology_name:
        ontology_note = (
            f"You are judging whether a concept from the {source_ontology_name} ontology matches a concept from the {target_ontology_name} ontology.\n"
        )

    if domain and domain.lower() != "general":
        role = f"You are an ontology matching expert specialising in the {domain} domain."
    else:
        role = "You are an ontology matching expert."

    system_prompt = (
        ontology_note
        + role
        + " I will provide you with two concepts, Concept A and Concept B. "
        + "Determine whether they refer to the same real world entity for ontology matching. "
        + "Only output YES or NO."
    )

    a_lines: List[str] = ["Concept A:"]
    b_lines: List[str] = ["Concept B:"]

    field_map = {
        "label": "Name",
        "synonyms": "Synonyms",
        "parents": "Superclasses",
        "cpx_exp": "Logical description",
        "definition": "Definition",
    }

    for f in input_fields:
        key = f
        display = field_map.get(f, f)
        a_lines.append(_format_field(display, concept_a.get(key)))
        b_lines.append(_format_field(display, concept_b.get(key)))

    prompt_text = "\n".join([x for x in a_lines if x] + [""] + [x for x in b_lines if x])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _score_batch(judge: HFYesNoJudge, prompts: List[str]) -> List[float]:
    """
    Score one batch of prompts, reading only the final token's logits
    (the YES/NO decision token) from each.

    BUG FIX (CUDA OOM on large batches/long prompts): a plain
    `judge.model(**inputs)` call defaults `logits_to_keep` to 0, which in
    the installed transformers version means "keep ALL positions" (the
    slice `slice(-logits_to_keep, None)` with `logits_to_keep=0` is
    `slice(0, None)`, i.e. the whole sequence) -- not "keep zero", despite
    the name. That makes every forward pass materialize a
    `[batch_size, seq_len, vocab_size]` logits tensor when only the single
    last-token slice `[batch_size, 1, vocab_size]` is ever read afterwards
    -- for a large vocab and long prompt this tensor alone can be tens of
    GiB and OOM the GPU even though the model itself fits comfortably.

    Passing `logits_to_keep=1` fixes that allocation, but is only
    *correct* (not just faster) when every row's real last token sits at
    the same physical (last) index in the batch -- which requires LEFT
    padding (`_load_judge_model()` sets `tokenizer.padding_side = "left"`)
    instead of the tokenizer's right-padding default, where each row's
    real last token sits at a different index per row and a single
    trailing slice can't express "the last real token" for all of them at
    once.

    Left padding on its own would then silently corrupt results: RoPE
    position embeddings are derived from each token's position in the
    sequence, and a raw `model()` forward call (unlike `.generate()`) does
    NOT auto-correct positions for left-padded input -- every pad token
    would be counted as a real position, shifting every real token's
    effective position by however many pad tokens precede it. Passing
    `position_ids` explicitly, computed the standard way from
    `attention_mask`, keeps every real token's position correct
    regardless of how much left padding precedes it:
        position_ids = attention_mask.cumsum(-1) - 1   # 0, 1, 2, ... over real tokens
        position_ids[attention_mask == 0] = 1          # pad positions: value is irrelevant,
                                                         # masked out by attention_mask anyway
    """
    inputs = judge.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(judge.model.device)

    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    with (torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()):
        logits = judge.model(**inputs, position_ids=position_ids, logits_to_keep=1).logits

    # logits_to_keep=1 already sliced the sequence dimension down to the
    # single last physical position (correct for every row, thanks to
    # left padding above) -- no per-row attention_mask.sum() indexing
    # needed any more.
    logits_last = logits[:, -1, :]
    probs = logits_last.softmax(dim=-1)

    yes_p = probs[:, judge.yes_ids].sum(dim=1) if judge.yes_ids else torch.zeros(probs.size(0), device=probs.device)
    no_p = probs[:, judge.no_ids].sum(dim=1) if judge.no_ids else torch.zeros(probs.size(0), device=probs.device)

    conf = (yes_p / (yes_p + no_p + 1e-12)).float().detach().cpu().tolist()
    return conf


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """
    Same rationale as define.py's _atomic_torch_save: a plain df.to_csv(path)
    that gets killed mid-write (SLURM walltime, OOM, node failure) leaves a
    truncated CSV at `path`. Writing to a sibling temp file and os.replace()-
    ing it in is atomic on POSIX filesystems.
    """
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def run(
    store_pt_path: str,
    candidates_csv: str,
    output_alignment_csv: str,
    llm_model: Optional[str],
    llm_config: Dict[str, Any],
    overwrite: bool = False,
    input_fields: Tuple[str, ...] = ("label", "synonyms", "definition"),
    embedding_candidates_are_ranked: bool = True,
) -> str:
    """
    Step 5: LLM judgement for candidate pairs.

    Inputs
      store_pt_path: concept_store.pt with store["src"] and store["tgt"]
      candidates_csv: produced by Step 4, schema src_iri,tgt_iri,score,rank
      output_alignment_csv: output schema includes decision and confidence

    llm_config supported keys
      hf_token: Optional[str]
      batch_size: int default 16
      domain: str default "general"
      source_ontology_name: Optional[str]
      target_ontology_name: Optional[str]
      torch_dtype: str default "bfloat16"
      device_map: str default "auto"
      threshold: float default 0.5
      checkpoint_every_n_batches: int default 50 -- progress is written to
          a checkpoint CSV (under checkpoint_dir) after this many batches,
          so a killed/timed-out job resumes from there instead of losing
          everything back to the previous full run (previously this
          function only wrote output_alignment_csv once, at the very end).
      checkpoint_dir: Optional[str] default dirname(output_alignment_csv)/
          "checkpoints" -- same convention as embed.py's checkpoint_dir.

    Returns
      output_alignment_csv
    """
    out_path = output_alignment_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and not overwrite:
        return out_path

    if not llm_model:
        raise ValueError("llm_model must be provided for judgement step.")

    store = torch.load(store_pt_path)
    src_data: Dict[str, Dict[str, Any]] = store.get("src", {})
    tgt_data: Dict[str, Dict[str, Any]] = store.get("tgt", {})

    df = pd.read_csv(candidates_csv)
    required_cols = {"src_iri", "tgt_iri"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError("candidates_csv must contain columns src_iri and tgt_iri.")

    batch_size = int(llm_config.get("batch_size", 16))
    domain = llm_config.get("domain", "general")
    source_onto_name = llm_config.get("source_ontology_name")
    target_onto_name = llm_config.get("target_ontology_name")
    torch_dtype = llm_config.get("torch_dtype", "bfloat16")
    device_map = llm_config.get("device_map", "auto")
    hf_token = llm_config.get("hf_token")
    threshold = float(llm_config.get("threshold", 0.5))
    checkpoint_every_n_batches = int(llm_config.get("checkpoint_every_n_batches", 50))

    checkpoint_dir = Path(llm_config.get("checkpoint_dir", str(Path(out_path).parent / "checkpoints")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / f"{Path(out_path).stem}_judge_checkpoint.csv")

    # confidence starts as NaN (not 0.0) so "still NaN" unambiguously means
    # "not judged yet" -- 0.0 is itself a legitimate confidence score, so it
    # can't double as the not-done-yet sentinel a resume needs.
    resumed = False
    if os.path.exists(checkpoint_path) and not overwrite:
        checkpoint_df = pd.read_csv(checkpoint_path)
        same_rows = (
            len(checkpoint_df) == len(df)
            and list(checkpoint_df["src_iri"]) == list(df["src_iri"])
            and list(checkpoint_df["tgt_iri"]) == list(df["tgt_iri"])
        )
        if same_rows:
            df = checkpoint_df
            resumed = True
    if not resumed:
        df["confidence"] = float("nan")
        df["decision"] = pd.NA

    judge = _load_judge_model(llm_model, hf_token=hf_token, torch_dtype=torch_dtype, device_map=device_map)

    prompts: List[str] = []
    row_ids: List[int] = []
    batches_since_checkpoint = 0

    def _score_and_fill(ids_batch: List[int], prompts_batch: List[str]) -> None:
        confs = _score_batch(judge, prompts_batch)
        for j, c in enumerate(confs):
            c2 = float(round(c, 6))
            df.at[ids_batch[j], "confidence"] = c2
            df.at[ids_batch[j], "decision"] = "YES" if c2 >= threshold else "NO"

    for ridx, row in tqdm(df.iterrows(), total=len(df), desc="LLM judging"):
        if pd.notna(row.get("confidence")):
            continue  # already scored in a previous, interrupted run

        src_iri = row["src_iri"]
        tgt_iri = row["tgt_iri"]

        a = src_data.get(src_iri)
        b = tgt_data.get(tgt_iri)

        if a is None or b is None:
            df.at[ridx, "confidence"] = 0.0
            df.at[ridx, "decision"] = "NO"
            continue

        prompt = _build_pair_prompt(
            tokenizer=judge.tokenizer,
            concept_a=a,
            concept_b=b,
            input_fields=input_fields,
            domain=domain,
            source_ontology_name=source_onto_name,
            target_ontology_name=target_onto_name,
        )
        prompts.append(prompt)
        row_ids.append(ridx)

        if len(prompts) >= batch_size:
            _score_and_fill(row_ids, prompts)
            prompts.clear()
            row_ids.clear()
            batches_since_checkpoint += 1
            if batches_since_checkpoint >= checkpoint_every_n_batches:
                _atomic_write_csv(df, checkpoint_path)
                batches_since_checkpoint = 0

    if prompts:
        _score_and_fill(row_ids, prompts)

    # Same rationale as define.py's end-of-run() cleanup: _load_judge_model
    # also uses device_map="auto" (accelerate hooks -> possible reference
    # cycles), and this is the *second* of the two ~60GB+-class model loads
    # that can happen in one process when run_pipeline() calls step2_define
    # then step5_judge back to back. Freeing deterministically here matters
    # most if judge.py is ever called again later in the same long-lived
    # process (e.g. a driver script that loops over several ontology pairs),
    # so the next judge/define load doesn't appear to need memory on top of
    # a model that is actually done being used.
    del judge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df["llm_model"] = llm_model
    df["input_fields"] = ",".join(input_fields)
    _atomic_write_csv(df, out_path)

    # The real output now exists and is complete -- drop the checkpoint so a
    # later overwrite=True re-run never mistakes a stale, differently-sized
    # checkpoint for something it can resume from.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    store.setdefault("meta", {})
    store["meta"]["judge_model"] = llm_model
    store["meta"]["judge_domain"] = domain
    store["meta"]["judge_input_fields"] = list(input_fields)
    store["meta"]["judge_batch_size"] = batch_size
    store["meta"]["judge_threshold"] = threshold
    torch.save(store, store_pt_path)

    return out_path