from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import faiss
except Exception as e:
    faiss = None


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _extract_embeddings(
    concepts: Dict[str, Dict[str, Any]],
    embedding_field: str,
) -> Tuple[List[str], np.ndarray]:
    iris: List[str] = []
    vecs: List[np.ndarray] = []

    for iri, c in concepts.items():
        emb = c.get(embedding_field)
        if emb is None:
            continue
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().to("cpu").float().numpy()
        else:
            emb_np = np.asarray(emb, dtype=np.float32)
        if emb_np.ndim != 1:
            emb_np = emb_np.reshape(-1)
        iris.append(iri)
        vecs.append(emb_np.astype(np.float32, copy=False))

    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)

    mat = np.vstack(vecs).astype(np.float32, copy=False)
    return iris, mat


def _build_faiss_index(
    dim: int,
    index_type: str,
    tgt_mat: np.ndarray,
    use_gpu: bool,
    index_params: Dict[str, Any],
):
    if faiss is None:
        raise ImportError("faiss is not available. Please install faiss-cpu or faiss-gpu.")

    index_type = index_type.lower()

    if index_type == "flatip":
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(tgt_mat)
        index = cpu_index

    elif index_type == "ivf":
        nlist = int(index_params.get("nlist", 50))
        nprobe = int(index_params.get("nprobe", 50))
        quantizer = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(tgt_mat)
        ivf.add(tgt_mat)
        ivf.nprobe = nprobe
        index = ivf

    elif index_type == "hnsw":
        m = int(index_params.get("m", 32))
        ef_construction = int(index_params.get("ef_construction", 200))
        ef_search = int(index_params.get("ef_search", 50))
        # BUG FIX: faiss.IndexHNSWFlat(d, M) defaults to METRIC_L2 (Euclidean
        # distance) when no metric is given -- unlike IndexFlatIP/IndexIVFFlat
        # above, which are explicitly inner-product. Since "hnsw" is this
        # function's default index_type, every retrieve.py run that didn't
        # explicitly pass index_type="flatip"/"ivf" was silently getting L2
        # distances back from index.search() and storing them in the "score"
        # column as if they were cosine similarities (src_mat/tgt_mat are
        # L2-normalized, so this is a monotonic reparametrisation of cosine
        # similarity: L2^2 = 2 - 2*cos_sim -- LOWER "score" actually means
        # MORE similar under the old code, the opposite of what every
        # downstream consumer assumes). This did not corrupt which top-k
        # candidates were retrieved or their relative order (L2 distance and
        # cosine similarity rank identically for normalized vectors), but it
        # did corrupt the numeric score value -- and fusion.py's cs_threshold
        # filter (meant to keep pairs with cosine similarity >= threshold)
        # ends up doing the opposite: keeping weak matches with large L2
        # "scores" and dropping genuinely well-matched pairs whose small L2
        # distance reads as a low "score". Passing METRIC_INNER_PRODUCT here
        # makes index.search() return true cosine similarity (since inputs
        # are normalized) sorted best-first, matching flatip/ivf and every
        # downstream threshold that assumes "higher score = more similar".
        h = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        h.hnsw.efConstruction = ef_construction
        h.hnsw.efSearch = ef_search
        h.add(tgt_mat)
        index = h

    else:
        raise ValueError(f"Unsupported index_type: {index_type}. Use one of: flatip, ivf, hnsw.")

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("faiss-gpu is required for use_gpu=True.")
        res = faiss.StandardGpuResources()
        gpu_id = int(index_params.get("gpu_id", 0))
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    return index


def run(
    store_pt_path: str,
    output_candidates_csv: str,
    topk: int = 20,
    overwrite: bool = False,
    embedding_field: str = "embedding",
    normalize: bool = True,
    index_type: str = "hnsw",
    index_params: Optional[Dict[str, Any]] = None,
    query_side: str = "src",
    query_side_auto: bool = False,
) -> str:
    """
    Step 4: retrieve top-k candidates based on embeddings.

    Input
      store_pt_path: concept_store.pt with store["src"][iri][embedding_field] and store["tgt"][iri][embedding_field]

    Output CSV schema
      src_iri, tgt_iri, score, rank -- ALWAYS in this canonical sense
      regardless of query_side (see below), so judge.py/fusion.py/the exact
      matchers never need to change.

    index_type
      "flatip"  exact inner product on GPU or CPU
      "ivf"     approximate IVF Flat inner product
      "hnsw"    approximate HNSW inner product

    index_params common keys
      use_gpu: bool default False
      gpu_id: int default 0
      ivf: nlist, nprobe
      hnsw: m, ef_construction, ef_search

    query_side
      "src" (default) -- original behaviour: the faiss index is built over
        tgt embeddings, and each src concept queries it for its topk
        nearest tgt neighbours.
      "tgt" -- the faiss index is built over src embeddings instead, and
        each tgt concept queries it for its topk nearest src neighbours.
      Either way the output CSV's columns keep the same src_iri/tgt_iri/
      score/rank meaning -- only which side drives the nearest-neighbour
      search changes. This is deliberate: a naive src/tgt ontology swap
      would also flip the direction fuse_config["exact"]["config"]'s
      src_owl/tgt_owl need to point in, silently breaking fusion.py's
      exact-priority dedup logic. query_side changes only retrieve's
      internal search direction, never the final output convention.
      The resolved value actually used (not necessarily the literal value
      passed in -- see query_side_auto) is recorded in
      store["meta"]["retrieve_query_side"].

    query_side_auto
      False (default) -- query_side is taken literally, as described above.
      True -- query_side is IGNORED, and this function instead measures
        len(src concepts) vs len(tgt concepts) itself and always queries
        from whichever side has fewer concepts (a tie keeps "src"), so the
        faiss index is always built over the LARGER side. This only
        affects which side drives retrieve's internal search (and
        therefore candidate generation) -- the final output src_iri/
        tgt_iri convention, and every later stage (judge/fusion/exact
        matchers), is completely unaffected and still follows the input
        src ontology as usual.
      Whether this was on is recorded in
      store["meta"]["retrieve_query_side_auto"]; the side actually chosen
      (whether by this or taken literally from query_side) is always in
      store["meta"]["retrieve_query_side"].
    """
    out_path = Path(output_candidates_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return str(out_path)

    if not query_side_auto and query_side not in ("src", "tgt"):
        raise ValueError(f"query_side must be 'src' or 'tgt', got {query_side!r}")

    index_params = index_params or {}
    use_gpu = bool(index_params.get("use_gpu", False))

    store = torch.load(store_pt_path)
    src_concepts = store.get("src", {})
    tgt_concepts = store.get("tgt", {})

    src_iris, src_mat = _extract_embeddings(src_concepts, embedding_field)
    tgt_iris, tgt_mat = _extract_embeddings(tgt_concepts, embedding_field)

    if len(src_iris) == 0:
        raise RuntimeError(f"No src embeddings found under field '{embedding_field}'.")
    if len(tgt_iris) == 0:
        raise RuntimeError(f"No tgt embeddings found under field '{embedding_field}'.")

    if src_mat.shape[1] != tgt_mat.shape[1]:
        raise RuntimeError(f"Embedding dim mismatch: src {src_mat.shape[1]} vs tgt {tgt_mat.shape[1]}.")

    if normalize:
        src_mat = _l2_normalize(src_mat)
        tgt_mat = _l2_normalize(tgt_mat)

    if query_side_auto:
        effective_query_side = "src" if len(src_iris) <= len(tgt_iris) else "tgt"
    else:
        effective_query_side = query_side

    if effective_query_side == "src":
        query_iris, query_mat = src_iris, src_mat
        match_iris, match_mat = tgt_iris, tgt_mat
    else:
        query_iris, query_mat = tgt_iris, tgt_mat
        match_iris, match_mat = src_iris, src_mat

    dim = match_mat.shape[1]
    index = _build_faiss_index(dim, index_type=index_type, tgt_mat=match_mat, use_gpu=use_gpu, index_params=index_params)

    k = int(topk)
    scores, idx = index.search(query_mat, k)

    rows: List[Dict[str, Any]] = []
    for i, query_iri in enumerate(query_iris):
        for r in range(k):
            j = int(idx[i, r])
            if j < 0 or j >= len(match_iris):
                continue
            match_iri = match_iris[j]
            src_iri, tgt_iri = (query_iri, match_iri) if effective_query_side == "src" else (match_iri, query_iri)
            rows.append(
                {
                    "src_iri": src_iri,
                    "tgt_iri": tgt_iri,
                    "score": float(scores[i, r]),
                    "rank": int(r + 1),
                }
            )

    df = pd.DataFrame(rows, columns=["src_iri", "tgt_iri", "score", "rank"])
    df.to_csv(out_path, index=False)

    store.setdefault("meta", {})
    store["meta"]["retrieve_topk"] = k
    store["meta"]["retrieve_index_type"] = index_type
    store["meta"]["retrieve_use_gpu"] = use_gpu
    store["meta"]["retrieve_embedding_field"] = embedding_field
    store["meta"]["retrieve_normalize"] = normalize
    store["meta"]["retrieve_index_params"] = dict(index_params)
    store["meta"]["retrieve_query_side"] = effective_query_side
    store["meta"]["retrieve_query_side_auto"] = query_side_auto

    torch.save(store, store_pt_path)

    return str(out_path)