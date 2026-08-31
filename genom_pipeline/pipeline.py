from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineResult:
    workdir: str
    concept_store_pt: str
    candidates_csv: str
    llm_alignment_csv: str
    final_alignment_csv: str


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _should_run(path: Path, overwrite: bool) -> bool:
    return overwrite or (not path.exists())


def run_pipeline(
    src_onto_path: str,
    tgt_onto_path: str,
    workdir: str,
    topk: int = 20,
    overwrite: bool = False,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    embed_config: Optional[Dict[str, Any]] = None,
    retrieve_config: Optional[Dict[str, Any]] = None,
    fuse_config: Optional[Dict[str, Any]] = None,
    run_definition: bool = True,
) -> PipelineResult:
    wd = Path(workdir)
    _ensure_dir(wd)

    concept_store_pt = wd / "concept_store.pt"
    candidates_csv = wd / f"candidates_top{topk}.csv"
    llm_alignment_csv = wd / "llm_alignment.csv"
    final_alignment_csv = wd / "final_alignment.csv"

    llm_config = llm_config or {}
    embed_config = embed_config or {}
    retrieve_config = retrieve_config or {}
    fuse_config = fuse_config or {}

    # Step 1
    if _should_run(concept_store_pt, overwrite):
        from genom_pipeline.steps.extract import run as step1_extract
        step1_extract(src_onto_path, tgt_onto_path, str(concept_store_pt))

    # Step 2
    if run_definition:
        from genom_pipeline.steps.define import run as step2_define
        step2_define(
            store_pt_path=str(concept_store_pt),
            llm_model=llm_model,
            llm_config=llm_config,
            overwrite=overwrite,
        )

    # Step 3
    from genom_pipeline.steps.embed import run as step3_embed
    step3_embed(
        store_pt_path=str(concept_store_pt),
        embedding_model=embedding_model,
        embed_config=embed_config,
        overwrite=overwrite,
    )

    # Step 4
    if _should_run(candidates_csv, overwrite):
        from genom_pipeline.steps.retrieve import run as step4_retrieve
        step4_retrieve(
            store_pt_path=str(concept_store_pt),
            output_candidates_csv=str(candidates_csv),
            topk=topk,
            overwrite=overwrite,
            **retrieve_config,
        )

    # Step 5
    if _should_run(llm_alignment_csv, overwrite):
        from genom_pipeline.steps.judge import run as step5_judge
        step5_judge(
            store_pt_path=str(concept_store_pt),
            candidates_csv=str(candidates_csv),
            output_alignment_csv=str(llm_alignment_csv),
            llm_model=llm_model,
            llm_config=llm_config,
            overwrite=overwrite,
        )

    # Step 6
    if _should_run(final_alignment_csv, overwrite):
        from genom_pipeline.steps.fusion import run as step6_fuse
        step6_fuse(
            store_pt_path=str(concept_store_pt),
            llm_alignment_csv=str(llm_alignment_csv),
            output_final_csv=str(final_alignment_csv),
            overwrite=overwrite,
            fuse_config=fuse_config,
        )

    return PipelineResult(
        workdir=str(wd),
        concept_store_pt=str(concept_store_pt),
        candidates_csv=str(candidates_csv),
        llm_alignment_csv=str(llm_alignment_csv),
        final_alignment_csv=str(final_alignment_csv),
    )