from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

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


def _definition_generation(
    llm: HFChatLLM,
    user_prompt: str,
    domain: str,
    source_ontology: Optional[str],
    target_ontology: Optional[str],
    gen_params: Dict[str, Any],
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

    text = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = llm.tokenizer([text], return_tensors="pt").to(llm.model.device)

    defaults = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    defaults.update(gen_params)

    generated_ids = llm.model.generate(**model_inputs, **defaults)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return llm.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()


def _iter_concepts(store: Dict[str, Any], side: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    concepts = store.get(side, {})
    for iri, concept in concepts.items():
        yield iri, concept


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

    llm = _load_hf_chat_model(llm_model, hf_token=hf_token)

    for side in sides:
        for iri, concept in tqdm(_iter_concepts(store, side), desc=f"DefGen {side}"):
            if (not overwrite) and (field_name in concept) and concept[field_name]:
                continue

            prompt_text = _build_prompt(concept)

            if side == "src":
                src_name = source_onto_name
                tgt_name = target_onto_name
            else:
                src_name = target_onto_name
                tgt_name = source_onto_name

            concept[field_name] = _definition_generation(
                llm=llm,
                user_prompt=prompt_text,
                domain=domain,
                source_ontology=src_name,
                target_ontology=tgt_name,
                gen_params=generation_params,
            )
    store.setdefault("meta", {})
    store["meta"]["definition_model"] = llm_model
    store["meta"]["definition_field"] = field_name
    store["meta"]["definition_generation_params"] = generation_params
    store["meta"]["definition_domain"] = domain

    torch.save(store, store_pt_path)
    return store_pt_path