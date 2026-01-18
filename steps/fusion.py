from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from genom_pipeline.exact_matcher import run_matcher


def _dedup_one_to_one(rows: List[Tuple[str, str, float, str]], prefer: str) -> List[Tuple[str, str, float, str]]:
    src_seen = set()
    out: List[Tuple[str, str, float, str]] = []
    for src, tgt, score, prov in rows:
        if src in src_seen:
            continue
        out.append((src, tgt, score, prov))
        src_seen.add(src)
    return out


def run(
    store_pt_path: str,
    llm_alignment_csv: str,
    output_final_csv: str,
    overwrite: bool = False,
    fuse_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Step 6: fuse LLM alignment with exact matcher mappings.

    fuse_config keys
      strategy: "union" | "exact_priority" | "llm_then_exact_fill"
      llm_threshold: float default 0.5
      use_llm_rank1_only: bool default True
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
    llm_threshold = float(fuse_config.get("llm_threshold", 0.5))
    use_llm_rank1_only = bool(fuse_config.get("use_llm_rank1_only", True))

    store = torch.load(store_pt_path)
    meta = store.setdefault("meta", {})

    df_llm = pd.read_csv(llm_alignment_csv)
    needed = {"src_iri", "tgt_iri", "decision", "confidence"}
    if not needed.issubset(df_llm.columns):
        raise ValueError(f"llm_alignment_csv missing columns. Required: {sorted(needed)}")

    df_llm2 = df_llm.copy()
    df_llm2 = df_llm2[df_llm2["decision"].astype(str).str.upper() == "YES"]
    df_llm2 = df_llm2[df_llm2["confidence"].astype(float) >= llm_threshold]

    if use_llm_rank1_only and "rank" in df_llm2.columns:
        df_llm2 = df_llm2.sort_values(["src_iri", "rank", "confidence"], ascending=[True, True, False])
        df_llm2 = df_llm2.groupby("src_iri").head(1)
    else:
        df_llm2 = df_llm2.sort_values(["src_iri", "confidence"], ascending=[True, False])
        df_llm2 = df_llm2.groupby("src_iri").head(1)

    llm_rows: List[Tuple[str, str, float, str]] = []
    for r in df_llm2.itertuples(index=False):
        llm_rows.append((str(r.src_iri), str(r.tgt_iri), float(r.confidence), "llm"))

    exact_spec = fuse_config.get("exact", {})
    exact_name = exact_spec.get("name")
    exact_rows: List[Tuple[str, str, float, str]] = []
    if exact_name:
        exact_cfg = exact_spec.get("config", {})
        exact_res = run_matcher(exact_name, exact_cfg)
        for src, tgt, score in exact_res.mappings:
            exact_rows.append((str(src), str(tgt), float(score), exact_res.name))

    final_rows: List[Tuple[str, str, float, str]] = []

    if strategy == "union":
        final_rows = exact_rows + llm_rows
        final_rows = _dedup_one_to_one(final_rows, prefer="first")

    elif strategy == "exact_priority":
        final_rows = exact_rows + llm_rows
        final_rows = _dedup_one_to_one(final_rows, prefer="exact_first")

    elif strategy == "llm_then_exact_fill":
        final_rows = llm_rows + exact_rows
        final_rows = _dedup_one_to_one(final_rows, prefer="llm_first")

    else:
        raise ValueError("Unknown fuse strategy")

    df_out = pd.DataFrame(final_rows, columns=["src_iri", "tgt_iri", "score", "provenance"])
    df_out.to_csv(out_path, index=False)

    meta["fuse_strategy"] = strategy
    meta["fuse_llm_threshold"] = llm_threshold
    meta["fuse_use_llm_rank1_only"] = use_llm_rank1_only
    meta["fuse_exact_matcher"] = exact_name or None
    meta["fuse_exact_config"] = exact_spec.get("config", {}) if exact_name else {}
    torch.save(store, store_pt_path)

    return str(out_path)