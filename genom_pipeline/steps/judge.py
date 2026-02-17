from __future__ import annotations

import os
from dataclasses import dataclass
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
    inputs = judge.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(judge.model.device)

    with (torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()):
        logits = judge.model(**inputs).logits

    attn = inputs["attention_mask"]
    last_index = attn.sum(dim=1) - 1
    logits_last = logits[torch.arange(logits.size(0)), last_index]
    probs = logits_last.softmax(dim=-1)

    yes_p = probs[:, judge.yes_ids].sum(dim=1) if judge.yes_ids else torch.zeros(probs.size(0), device=probs.device)
    no_p = probs[:, judge.no_ids].sum(dim=1) if judge.no_ids else torch.zeros(probs.size(0), device=probs.device)

    conf = (yes_p / (yes_p + no_p + 1e-12)).float().detach().cpu().tolist()
    return conf


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

    judge = _load_judge_model(llm_model, hf_token=hf_token, torch_dtype=torch_dtype, device_map=device_map)

    df["confidence"] = 0.0
    df["decision"] = "NO"

    prompts: List[str] = []
    row_ids: List[int] = []

    for ridx, row in tqdm(df.iterrows(), total=len(df), desc="LLM judging"):
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
            confs = _score_batch(judge, prompts)
            for j, c in enumerate(confs):
                c2 = float(round(c, 6))
                df.at[row_ids[j], "confidence"] = c2
                df.at[row_ids[j], "decision"] = "YES" if c2 >= threshold else "NO"
            prompts.clear()
            row_ids.clear()

    if prompts:
        confs = _score_batch(judge, prompts)
        for j, c in enumerate(confs):
            c2 = float(round(c, 6))
            df.at[row_ids[j], "confidence"] = c2
            df.at[row_ids[j], "decision"] = "YES" if c2 >= threshold else "NO"

    df["llm_model"] = llm_model
    df["input_fields"] = ",".join(input_fields)
    df.to_csv(out_path, index=False)

    store.setdefault("meta", {})
    store["meta"]["judge_model"] = llm_model
    store["meta"]["judge_domain"] = domain
    store["meta"]["judge_input_fields"] = list(input_fields)
    store["meta"]["judge_batch_size"] = batch_size
    store["meta"]["judge_threshold"] = threshold
    torch.save(store, store_pt_path)

    return out_path