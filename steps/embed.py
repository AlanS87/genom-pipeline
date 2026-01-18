from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import pickle
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None


def _normalize_text(text: Any, lowercase: bool = True) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if lowercase:
        text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_brackets(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text).strip()


def _clean_list(items: Any, lowercase: bool = True) -> List[str]:
    if not items or not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        if isinstance(x, str):
            s = _remove_brackets(_normalize_text(x, lowercase=lowercase))
            if s:
                out.append(s)
    return out


def _truncate_text_by_tokens(text: str, model: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    if tiktoken is None:
        return text
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    tokens = tokens[:max_tokens]
    return enc.decode(tokens)


def _concept_to_embedding_text(
    concept: Dict[str, Any],
    text_fields: Tuple[str, ...],
    lowercase: bool,
    join_synonyms: bool,
    join_parents: bool,
) -> str:
    parts: List[str] = []

    for field in text_fields:
        if field == "label":
            s = _normalize_text(concept.get("label", ""), lowercase=lowercase)
            if s:
                parts.append(s)

        elif field == "synonyms":
            syns = _clean_list(concept.get("synonyms"), lowercase=lowercase)
            if syns:
                if join_synonyms:
                    parts.append("synonyms: " + ", ".join(syns))
                else:
                    parts.extend(syns)

        elif field == "parents":
            ps = _clean_list(concept.get("parents"), lowercase=lowercase)
            if ps:
                if join_parents:
                    parts.append("superclasses: " + ", ".join(ps))
                else:
                    parts.extend(ps)

        elif field == "cpx_exp":
            s = _normalize_text(concept.get("cpx_exp", ""), lowercase=lowercase)
            if s:
                parts.append("logical description: " + s)

        else:
            val = concept.get(field)
            s = _normalize_text(val, lowercase=lowercase)
            if s:
                parts.append(f"{field}: {s}")

    return ". ".join(parts).strip()


def _save_checkpoint(path: Path, data: List[List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_checkpoint(path: Path) -> List[List[float]]:
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def _get_openai_client(api_key: Optional[str] = None):
    if OpenAI is None:
        raise ImportError("openai package is not available. Please install 'openai'.")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set. Provide embed_config['api_key'] or set env OPENAI_API_KEY.")
    return OpenAI(api_key=key)


def _batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _embed_texts_openai_with_resume(
    client,
    texts: List[str],
    checkpoint_path: Path,
    model: str,
    batch_size: int,
    max_tokens: int,
) -> List[List[float]]:
    embeddings: List[List[float]] = _load_checkpoint(checkpoint_path)
    start_idx = len(embeddings)

    for batch_start in tqdm(range(start_idx, len(texts), batch_size), desc=f"Embedding {checkpoint_path.name}"):
        batch_texts_raw = texts[batch_start : batch_start + batch_size]
        batch_texts: List[str] = []
        for t in batch_texts_raw:
            t2 = _truncate_text_by_tokens(t, model=model, max_tokens=max_tokens)
            if not t2.strip():
                t2 = " "
            batch_texts.append(t2)

        resp = client.embeddings.create(input=batch_texts, model=model)
        batch_embs = [item.embedding for item in resp.data]
        embeddings.extend(batch_embs)
        _save_checkpoint(checkpoint_path, embeddings)

    return embeddings


def run(
    store_pt_path: str,
    embedding_model: Optional[str],
    embed_config: Dict[str, Any],
    overwrite: bool = False,
    sides: Tuple[str, ...] = ("src", "tgt"),
    embedding_field: str = "embedding",
) -> str:
    """
    Step 3: compute embeddings and write back into concept_store.pt (in-place)

    embed_config supported keys:
      api_key: Optional[str]  OpenAI key, falls back to env OPENAI_API_KEY
      text_fields: Tuple[str,...]  default ("label","synonyms","definition")
      lowercase: bool default True
      join_synonyms: bool default True
      join_parents: bool default True
      batch_size: int default 64
      max_tokens: int default 8000
      checkpoint_dir: Optional[str] default workdir/"checkpoints"
      device: Optional[str] default "cpu"
    """
    if not embedding_model:
        raise ValueError("embedding_model must be provided for embedding step.")

    store = torch.load(store_pt_path)
    meta = store.setdefault("meta", {})

    text_fields = tuple(embed_config.get("text_fields", ("label", "synonyms", "definition")))
    lowercase = bool(embed_config.get("lowercase", True))
    join_synonyms = bool(embed_config.get("join_synonyms", True))
    join_parents = bool(embed_config.get("join_parents", True))
    batch_size = int(embed_config.get("batch_size", 64))
    max_tokens = int(embed_config.get("max_tokens", 8000))
    device_str = embed_config.get("device", "cpu")

    pt_path = Path(store_pt_path)
    workdir = pt_path.parent
    checkpoint_dir = Path(embed_config.get("checkpoint_dir", str(workdir / "checkpoints")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    client = _get_openai_client(api_key=embed_config.get("api_key"))

    for side in sides:
        concepts: Dict[str, Dict[str, Any]] = store.get(side, {})
        if not concepts:
            continue

        iris: List[str] = []
        texts: List[str] = []

        for iri, concept in concepts.items():
            if (not overwrite) and (embedding_field in concept) and concept[embedding_field]:
                continue
            text = _concept_to_embedding_text(
                concept=concept,
                text_fields=text_fields,
                lowercase=lowercase,
                join_synonyms=join_synonyms,
                join_parents=join_parents,
            )
            iris.append(iri)
            texts.append(text)

        if not iris:
            continue

        ckpt_path = checkpoint_dir / f"{side}_{embedding_model.replace('/', '_')}_{embedding_field}.pkl"
        embs = _embed_texts_openai_with_resume(
            client=client,
            texts=texts,
            checkpoint_path=ckpt_path,
            model=embedding_model,
            batch_size=batch_size,
            max_tokens=max_tokens,
        )

        if len(embs) != len(iris):
            raise RuntimeError(f"Embedding count mismatch for {side}: {len(embs)} embeddings vs {len(iris)} iris.")

        for iri, emb in zip(iris, embs):
            t = torch.tensor(emb, dtype=torch.float32, device="cpu")
            concepts[iri][embedding_field] = t

    meta["embedding_model"] = embedding_model
    meta["embedding_field"] = embedding_field
    meta["embedding_text_fields"] = list(text_fields)
    meta["embedding_lowercase"] = lowercase
    meta["embedding_join_synonyms"] = join_synonyms
    meta["embedding_join_parents"] = join_parents
    meta["embedding_batch_size"] = batch_size
    meta["embedding_max_tokens"] = max_tokens
    meta["embedding_device_saved"] = device_str

    torch.save(store, store_pt_path)
    return store_pt_path