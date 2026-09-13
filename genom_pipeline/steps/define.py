from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass(frozen=True)
class HFChatLLM:
    model_name: str
    tokenizer: Any
    model: Any


def _load_hf_chat_model(model_name: str, hf_token: Optional[str] = None) -> HFChatLLM:
    kwargs: Dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    if hf_token:
        kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token) if hf_token else AutoTokenizer.from_pretrained(model_name)

    # Batched generation needs left-padding: a causal LM keeps generating
    # from each row's last real token, so every row's real content has to
    # end at the same column for one generate() call to make sense across
    # the whole batch. Right-padding (most tokenizers' default) would leave
    # shorter rows continuing from a padding token instead of their prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        # Several chat tokenizers (Qwen, Llama) ship with no pad token
        # configured, which batched tokenizer(..., padding=True) requires --
        # same fallback judge.py already uses for its own model loading.
        tokenizer.pad_token = tokenizer.eos_token

    return HFChatLLM(model_name=model_name, tokenizer=tokenizer, model=model)


def _build_prompt(concept: Dict[str, Any]) -> str:
    return (
        "Concept:\n"
        f"- Name: {concept.get('label')}\n"
        f"- Synonyms: {concept.get('synonyms')}\n"
        f"- Superclasses: {concept.get('parents')}\n"
        f"- Logical description: {concept.get('cpx_exp')}\n"
    )


def _build_system_prompt(
    domain: str,
    source_ontology: str | None,
    target_ontology: str | None,
) -> str:
    ontology_note = ""
    if source_ontology and target_ontology:
        ontology_note = (
            f"You are generating a definition for a concept from the {source_ontology} ontology. "
            f"This definition will be used to align it with candidate concepts in the {target_ontology} ontology.\n"
        )

    if domain and domain.lower() != "general":
        domain_clause = f"You are an ontology expert specialising in the {domain} domain. "
    else:
        domain_clause = "You are an ontology and knowledge representation expert. "

    return (
        ontology_note
        + domain_clause
        + "Your task is to generate a concise, alignment-friendly definition for a given concept. "
        + "The definition should be semantically precise, clearly distinguishable from related concepts, "
        + "and suitable for matching across different ontologies. "
        + "Only return the definition."
    )


def _build_chat_text(
    tokenizer,
    user_prompt: str,
    domain: str,
    source_ontology: Optional[str],
    target_ontology: Optional[str],
) -> str:
    system_prompt = _build_system_prompt(
        domain=domain,
        source_ontology=source_ontology,
        target_ontology=target_ontology,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _definition_generation_batch(
    llm: HFChatLLM,
    chat_texts: List[str],
    gen_params: Dict[str, Any],
) -> List[str]:
    """
    Generate definitions for a batch of already-rendered chat prompts in a
    single forward pass, instead of one generate() call per concept (the
    previous implementation).

    Relies on llm.tokenizer.padding_side == "left" (set once in
    _load_hf_chat_model): with left-padding every row's real prompt ends at
    the same column, so the newly generated continuation for every row
    starts at the same fixed index (model_inputs.input_ids.shape[1]) and can
    be sliced off uniformly -- the same trick judge.py's _score_batch uses
    to pull per-row results out of a single batched forward pass.
    """
    model_inputs = llm.tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
    ).to(llm.model.device)

    defaults = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    defaults.update(gen_params)
    defaults.setdefault("pad_token_id", llm.tokenizer.pad_token_id)

    generated_ids = llm.model.generate(**model_inputs, **defaults)

    input_len = model_inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]

    return [
        text.strip()
        for text in llm.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    ]


def _iter_concepts(store: Dict[str, Any], side: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    concepts = store.get(side, {})
    for iri, concept in concepts.items():
        yield iri, concept


def _atomic_torch_save(obj: Any, path: str) -> None:
    """
    torch.save(obj, path) directly is not crash-safe: if the process is
    killed mid-write (e.g. a SLURM job hitting its walltime), `path` is left
    truncated/corrupt and the next run's torch.load(path) fails outright --
    destroying everything since the last successful save, not just the most
    recent batch. Writing to a sibling temp file and os.replace()-ing it
    into place is atomic on POSIX filesystems: `path` always resolves to
    either the previous complete save or the new complete save, never a
    partial one.
    """
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def run(
    store_pt_path: str,
    llm_model: Optional[str],
    llm_config: Dict[str, Any],
    overwrite: bool = False,
    field_name: str = "definition",
    sides: Tuple[str, ...] = ("src", "tgt"),
) -> str:
    """
    Step 2: generate definitions and write back into concept_store.pt

    Parameters
    store_pt_path: path to concept_store.pt produced by Step 1
    llm_model: HuggingFace model id, e.g. "Qwen/Qwen2.5-14B-Instruct"
    llm_config: dict for model loading and generation, suggested keys:
        hf_token: Optional[str] (for gated models)
        source_ontology_name: Optional[str]
        target_ontology_name: Optional[str]
        batch_size: int default 16 -- concepts are generated in batches of
            this size (one padded forward pass per batch), not one at a
            time. Matches judge.py's default.
        checkpoint_every_n_batches: int default 20 -- concept_store.pt is
            re-saved to disk (atomically) after this many batches complete,
            so a killed/timed-out job can resume from the last checkpoint
            instead of losing all progress back to the previous full run.
        generation: dict, forwarded to model.generate
    overwrite: if True, regenerate even if field exists
    field_name: where to write definition, default "definition"
    sides: process ("src","tgt") by default

    Returns
    Updated store_pt_path (in-place).
    """
    if not llm_model:
        raise ValueError("llm_model must be provided for definition generation.")

    store = torch.load(store_pt_path)

    hf_token = llm_config.get("hf_token")
    generation_params = llm_config.get("generation", {})
    source_onto_name = llm_config.get("source_ontology_name")
    target_onto_name = llm_config.get("target_ontology_name")
    domain = llm_config.get("domain", "general")
    batch_size = int(llm_config.get("batch_size", 16))
    checkpoint_every_n_batches = int(llm_config.get("checkpoint_every_n_batches", 20))

    llm = _load_hf_chat_model(llm_model, hf_token=hf_token)

    for side in sides:
        pending_iris: List[str] = []
        pending_texts: List[str] = []
        batches_since_checkpoint = 0

        def _flush() -> None:
            nonlocal batches_since_checkpoint
            if not pending_iris:
                return
            defs = _definition_generation_batch(llm, pending_texts, generation_params)
            for iri, d in zip(pending_iris, defs):
                store[side][iri][field_name] = d
            pending_iris.clear()
            pending_texts.clear()
            batches_since_checkpoint += 1
            if batches_since_checkpoint >= checkpoint_every_n_batches:
                _atomic_torch_save(store, store_pt_path)
                batches_since_checkpoint = 0

        for iri, concept in tqdm(list(_iter_concepts(store, side)), desc=f"DefGen {side}"):
            if (not overwrite) and (field_name in concept) and concept[field_name]:
                continue

            prompt_text = _build_prompt(concept)

            if side == "src":
                src_name = source_onto_name
                tgt_name = target_onto_name
            else:
                src_name = target_onto_name
                tgt_name = source_onto_name

            chat_text = _build_chat_text(
                tokenizer=llm.tokenizer,
                user_prompt=prompt_text,
                domain=domain,
                source_ontology=src_name,
                target_ontology=tgt_name,
            )
            pending_iris.append(iri)
            pending_texts.append(chat_text)

            if len(pending_iris) >= batch_size:
                _flush()

        _flush()

    store.setdefault("meta", {})
    store["meta"]["definition_model"] = llm_model
    store["meta"]["definition_field"] = field_name
    store["meta"]["definition_generation_params"] = generation_params
    store["meta"]["definition_domain"] = domain
    store["meta"]["definition_batch_size"] = batch_size

    _atomic_torch_save(store, store_pt_path)

    # Explicitly drop the model before returning instead of relying on the
    # local `llm` simply going out of scope. `_load_hf_chat_model` uses
    # device_map="auto", which routes the model through accelerate's
    # big-model-inference hooks (AlignDevicesHook et al.) -- those hooks are
    # attached as extra module attributes and commonly hold back-references
    # to the module they're attached to, forming reference cycles that plain
    # CPython refcounting cannot free immediately. Left alone, the ~60GB+
    # (bf16, 32B-class model) of GPU memory would only be reclaimed whenever
    # Python's cyclic GC next happens to run -- which is unpredictable timing
    # to be relying on right before judge.py loads a second full model into
    # the same process (see pipeline.py's run_pipeline(), which calls
    # step2_define then step5_judge sequentially in one process). del + a
    # forced gc.collect() breaks those cycles deterministically, and
    # empty_cache() then returns the now-free blocks from PyTorch's caching
    # allocator so the next model load can actually use that memory instead
    # of appearing to need fresh GPU memory on top of it.
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return store_pt_path
