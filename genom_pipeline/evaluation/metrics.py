from __future__ import annotations

"""
Evaluation utilities: precision/recall/F1 against a gold-standard reference
alignment, and threshold grid search.

This is intentionally separate from genom_pipeline.steps.fusion, which only
produces predictions and has no notion of gold labels. Splitting the two
means:
  - fusion.run() stays usable in a real deployment where no gold reference
    exists.
  - threshold selection (this module's grid_search_thresholds) can sweep
    (llm_threshold, cs_threshold) combinations directly on an already
    computed llm_alignment.csv, without re-running retrieval or the LLM
    judge for every combination.
"""

from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd

from genom_pipeline.steps.fusion import dedup_one_to_one, select_rank1_candidates

Mapping = Tuple[str, str]


def load_reference(
    reference_path: str,
    src_col: str = "SrcEntity",
    tgt_col: str = "TgtEntity",
    sep: str = "\t",
) -> Set[Mapping]:
    """
    Load a gold-standard reference alignment (e.g. an OAEI Bio-ML
    'refs_equiv/full.tsv' file) as a set of (src_iri, tgt_iri) pairs.
    """
    df = pd.read_csv(reference_path, sep=sep)
    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError(
            f"Reference file '{reference_path}' is missing columns "
            f"'{src_col}'/'{tgt_col}'. Found columns: {list(df.columns)}"
        )
    return set(zip(df[src_col].astype(str), df[tgt_col].astype(str)))


def load_predicted(
    final_alignment_csv: str,
    src_col: str = "src_iri",
    tgt_col: str = "tgt_iri",
) -> Set[Mapping]:
    """Load a predicted alignment produced by steps.fusion.run()."""
    df = pd.read_csv(final_alignment_csv)
    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError(
            f"Predicted alignment file '{final_alignment_csv}' is missing columns "
            f"'{src_col}'/'{tgt_col}'. Found columns: {list(df.columns)}"
        )
    return set(zip(df[src_col].astype(str), df[tgt_col].astype(str)))


def precision_recall_f1(predicted: Set[Mapping], reference: Set[Mapping]) -> Dict[str, float]:
    """Standard OAEI-style precision/recall/F1 over two sets of (src, tgt) pairs."""
    n_correct = len(predicted & reference)
    precision = (n_correct / len(predicted)) if predicted else 0.0
    recall = (n_correct / len(reference)) if reference else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predicted": len(predicted),
        "n_reference": len(reference),
        "n_correct": n_correct,
    }


def evaluate_final_alignment(
    final_alignment_csv: str,
    reference_path: str,
    reference_src_col: str = "SrcEntity",
    reference_tgt_col: str = "TgtEntity",
    reference_sep: str = "\t",
) -> Dict[str, float]:
    """
    Convenience wrapper: score a final_alignment.csv (from steps.fusion.run)
    against a gold reference file in one call.
    """
    predicted = load_predicted(final_alignment_csv)
    reference = load_reference(reference_path, reference_src_col, reference_tgt_col, reference_sep)
    return precision_recall_f1(predicted, reference)


def grid_search_thresholds(
    llm_alignment_csv: str,
    reference_path: str,
    llm_thresholds: Iterable[float],
    cs_thresholds: Iterable[float],
    reference_src_col: str = "SrcEntity",
    reference_tgt_col: str = "TgtEntity",
    reference_sep: str = "\t",
    enforce_target_uniqueness: bool = True,
) -> pd.DataFrame:
    """
    Sweep (llm_threshold, cs_threshold) combinations directly on a raw
    llm_alignment.csv (columns: src_iri, tgt_iri, score, rank, confidence,
    decision) and report precision/recall/F1 against a gold reference for
    each combination -- without re-running retrieval or the LLM judge.

    For each combination this applies the exact same selection logic as
    genom_pipeline.steps.fusion.run() (decision == YES, threshold filters,
    rank-1 selection, 1-to-1 dedup), so the threshold you pick here is
    directly transferable to fuse_config in the real fusion step.

    Returns a DataFrame with one row per (llm_threshold, cs_threshold)
    combination and columns: llm_threshold, cs_threshold, precision,
    recall, f1, n_predicted, n_reference, n_correct.
    """
    df = pd.read_csv(llm_alignment_csv)
    required = {"src_iri", "tgt_iri", "score", "confidence", "decision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"llm_alignment_csv is missing columns: {sorted(missing)}")

    reference = load_reference(reference_path, reference_src_col, reference_tgt_col, reference_sep)

    df_yes = df[df["decision"].astype(str).str.upper() == "YES"].copy()

    rows = []
    for llm_t in llm_thresholds:
        for cs_t in cs_thresholds:
            sub = df_yes[
                (df_yes["confidence"].astype(float) >= llm_t)
                & (df_yes["score"].astype(float) >= cs_t)
            ]
            sub = select_rank1_candidates(sub)

            candidate_rows = [
                (str(r.src_iri), str(r.tgt_iri), float(r.confidence), "llm")
                for r in sub.itertuples(index=False)
            ]
            matches = dedup_one_to_one(candidate_rows, enforce_target_uniqueness=enforce_target_uniqueness)
            predicted = {(m[0], m[1]) for m in matches}

            metrics = precision_recall_f1(predicted, reference)
            rows.append({"llm_threshold": llm_t, "cs_threshold": cs_t, **metrics})

    return pd.DataFrame(rows)


def best_thresholds(grid_df: pd.DataFrame, metric: str = "f1") -> pd.Series:
    """Return the row of grid_search_thresholds' output with the highest `metric`."""
    if grid_df.empty:
        raise ValueError("grid_df is empty; nothing to select from.")
    return grid_df.loc[grid_df[metric].idxmax()]


def plot_f1_heatmap(grid_df: pd.DataFrame, title: Optional[str] = None):
    """
    Optional: render an F1 heatmap over (cs_threshold, llm_threshold), mirroring
    the exploratory plots from earlier experiments. Requires matplotlib and
    seaborn, which are NOT core dependencies of genom_pipeline -- install them
    separately (`pip install matplotlib seaborn`) if you want to use this.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot_df = grid_df.pivot(index="cs_threshold", columns="llm_threshold", values="f1")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_xlabel("LLM (token probability) threshold")
    ax.set_ylabel("Cosine similarity threshold")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
