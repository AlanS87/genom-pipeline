from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from genom_pipeline.exact_matcher import run_matcher

# (src_iri, tgt_iri, score, provenance)
MappingRow = Tuple[str, str, float, str]


def select_rank1_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given LLM-judged candidate rows that have already been filtered by
    threshold, keep at most one row per src_iri: the one with the smallest
    'rank' (i.e. closest to the top-1 retrieval result), breaking ties by
    highest confidence. If 'rank' is not present, fall back to highest
    confidence only.
    """
    if df.empty:
        return df

    if "rank" in df.columns:
        sort_cols = ["src_iri", "rank", "confidence"]
        ascending = [True, True, False]
    else:
        sort_cols = ["src_iri", "confidence"]
        ascending = [True, False]

    df_sorted = df.sort_values(sort_cols, ascending=ascending)
    return df_sorted.groupby("src_iri", as_index=False).head(1)


def dedup_one_to_one(
    rows: List[MappingRow],
    enforce_target_uniqueness: bool = True,
) -> List[MappingRow]:
    """
    Greedily keep rows in the order given, allowing each src_iri to be used
    at most once. When enforce_target_uniqueness is True (default), each
    tgt_iri is also allowed at most once, so the result is a valid 1-to-1
    equivalence alignment (matches the OAEI Bio-ML evaluation assumption).

    Caller controls priority by row order: put the rows that should win a
    conflict first (e.g. exact-matcher rows before LLM rows for an
    "exact first" strategy).
    """
    src_seen: set = set()
    tgt_seen: set = set()
    out: List[MappingRow] = []

    for src, tgt, score, prov in rows:
        if src in src_seen:
            continue
        if enforce_target_uniqueness and tgt in tgt_seen:
            continue
        out.append((src, tgt, score, prov))
        src_seen.add(src)
        if enforce_target_uniqueness:
            tgt_seen.add(tgt)

    return out


def run(
    store_pt_path: str,
    llm_alignment_csv: str,
    output_final_csv: str,
    overwrite: bool = False,
    fuse_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Step 6: fuse LLM alignment with exact matcher mappings into the final
    alignment.

    This step only produces predictions (final_alignment.csv). It does not
    know about gold-standard labels. For precision/recall/F1 against a
    reference alignment, or for sweeping (llm_threshold, cs_threshold)
    combinations, use genom_pipeline.evaluation.metrics instead of adding
    that logic here.

    fuse_config keys
      strategy: "union" | "exact_priority" | "llm_then_exact_fill"
        default "exact_priority". NOTE: "union" and "exact_priority"
        currently produce identical results (both put exact-matcher rows
        first) -- there is no real "union without priority" mode yet.
      llm_threshold: float, minimum LLM YES-probability ("confidence" column
        in llm_alignment_csv) to keep a candidate. Default 0.9.
      cs_threshold: float or None, minimum cosine similarity ("score" column,
        carried over from the retrieval step) to keep a candidate. Default
        0.9. Set to None to disable this filter.
      enforce_target_uniqueness: bool, also enforce that each tgt_iri is used
        at most once (a true 1-to-1 alignment). Default True.
      use_llm_rank1_only: bool default True. When True, only the single
        best-ranked LLM candidate per src_iri (among those passing both
        thresholds) is considered.
      exact: dict with keys:
        name: "bertmaplt_string" or "logmaplt_file"
        config: dict passed to matcher
    """
    out_path = Path(output_final_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return str(out_path)

    fuse_config = fuse_config or {}
    strategy = fuse_config.get("strategy", "exact_priority")
    llm_threshold = float(fuse_config.get("llm_threshold", 0.9))
    cs_threshold = fuse_config.get("cs_threshold", 0.9)
    enforce_target_uniqueness = bool(fuse_config.get("enforce_target_uniqueness", True))
    use_llm_rank1_only = bool(fuse_config.get("use_llm_rank1_only", True))

    store = torch.load(store_pt_path)
    meta = store.setdefault("meta", {})

    df_llm = pd.read_csv(llm_alignment_csv)
    needed = {"src_iri", "tgt_iri", "decision", "confidence"}
    if not needed.issubset(df_llm.columns):
        raise ValueError(f"llm_alignment_csv missing columns. Required: {sorted(needed)}")

    df_llm2 = df_llm[df_llm["decision"].astype(str).str.upper() == "YES"].copy()
    df_llm2 = df_llm2[df_llm2["confidence"].astype(float) >= llm_threshold]

    if cs_threshold is not None:
        if "score" not in df_llm2.columns:
            raise ValueError(
                "cs_threshold is set but llm_alignment_csv has no 'score' column "
                "(the cosine similarity carried over from retrieve.py's candidates_csv). "
                "Re-run judge.py on a candidates_csv produced by retrieve.py, or pass "
                "fuse_config={'cs_threshold': None} to disable this filter."
            )
        df_llm2 = df_llm2[df_llm2["score"].astype(float) >= float(cs_threshold)]

    if use_llm_rank1_only:
        df_llm2 = select_rank1_candidates(df_llm2)

    llm_rows: List[MappingRow] = [
        (str(r.src_iri), str(r.tgt_iri), float(r.confidence), "llm")
        for r in df_llm2.itertuples(index=False)
    ]

    exact_spec = fuse_config.get("exact", {})
    exact_name = exact_spec.get("name")
    exact_rows: List[MappingRow] = []
    if exact_name:
        exact_cfg = exact_spec.get("config", {})
        exact_res = run_matcher(exact_name, exact_cfg)
        for src, tgt, score in exact_res.mappings:
            exact_rows.append((str(src), str(tgt), float(score), exact_res.name))

    if strategy == "union":
        candidate_rows = exact_rows + llm_rows
    elif strategy == "exact_priority":
        candidate_rows = exact_rows + llm_rows
    elif strategy == "llm_then_exact_fill":
        candidate_rows = llm_rows + exact_rows
    else:
        raise ValueError(f"Unknown fuse strategy: {strategy!r}")

    final_rows = dedup_one_to_one(candidate_rows, enforce_target_uniqueness=enforce_target_uniqueness)

    df_out = pd.DataFrame(final_rows, columns=["src_iri", "tgt_iri", "score", "provenance"])
    df_out.to_csv(out_path, index=False)

    meta["fuse_strategy"] = strategy
    meta["fuse_llm_threshold"] = llm_threshold
    meta["fuse_cs_threshold"] = cs_threshold
    meta["fuse_enforce_target_uniqueness"] = enforce_target_uniqueness
    meta["fuse_use_llm_rank1_only"] = use_llm_rank1_only
    meta["fuse_exact_matcher"] = exact_name or None
    meta["fuse_exact_config"] = exact_spec.get("config", {}) if exact_name else {}
    torch.save(store, store_pt_path)

    return str(out_path)
