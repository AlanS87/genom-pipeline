"""
CLI driver to run ONE GenOM pipeline stage per OS process.

This is the recommended way to run genom_pipeline on an HPC/SLURM cluster
(or anywhere a single long job needs to run all six stages back to back) --
see the README's "Running Stage-by-Stage on an HPC/SLURM Cluster" section
for a full walkthrough, and scripts/example_slurm_job.sbatch +
tasks/example_task.yaml for a worked example. For a single in-process call
that runs every stage itself (e.g. from a notebook, or when GPU-memory
reclaim between stages isn't a concern), use genom_pipeline.run_pipeline()
instead -- this script is a thin, deliberately dependency-light driver
around the exact same genom_pipeline.steps.* functions run_pipeline() calls,
not a second implementation of the pipeline.

Why per-stage processes instead of one run_pipeline() call:
  1. define and judge each load their own full HF causal LM instance
     (Qwen2.5-32B-Instruct in this test) via device_map="auto". Running them
     as separate processes means the OS/CUDA driver fully reclaims GPU
     memory when one process exits, before the next stage's process starts
     -- no reliance on Python-side gc timing between two ~60GB+ model loads
     in one process. (define.py/judge.py's run() now also do an explicit
     del + gc.collect() + torch.cuda.empty_cache() at the end as a second,
     belt-and-suspenders layer -- see the review log -- but keeping the
     stages in separate processes is the stronger guarantee.)
  2. A crash/timeout in a later, cheap stage (embed's OpenAI calls,
     retrieve's faiss index) does not require re-running the expensive GPU
     stages that already finished and checkpointed to disk.
  3. Cleaner SLURM logs: each stage's stdout is clearly delimited, matching
     the AgentOM sbatch script's set +e / exit-code-per-step pattern.

Every invocation appends the exact config it used to <workdir>/run_config.json
(keyed by stage, overwriting that stage's previous entry) -- so "what
parameters did stage X actually run with" is always answerable by reading
that file instead of having to reconstruct it from memory/this script's
defaults after the fact. Each recorded entry now also carries
"_config_source": either "cli" or the absolute path of the YAML file that
produced it, so you can also tell HOW a stage was launched.

Usage -- two ways to run a stage, both call the exact same run_stage() logic
underneath so they can never silently diverge in behaviour:

  1) Flags on the command line (original style, still fully supported):
       python3 scripts/genom_stage_runner.py <stage> --workdir DIR [options]

  2) A YAML task file (recommended for anything beyond a one-off run --
     add/edit values in the YAML instead of editing this .py file and
     re-pasting it every time a parameter needs to change):
       python3 scripts/genom_stage_runner.py <stage> --config my_task.yaml
     See load_yaml_config()'s docstring below for the exact YAML shape.
     When --config is given, every other flag is ignored -- the YAML is the
     single source of truth for that run, so there is no "which one wins"
     ambiguity between a flag and a YAML value.

  run_stage() and run_from_yaml() are also plain importable functions, in
  case a future script wants to drive a stage programmatically (e.g. loop
  over several YAML task files) without going through argparse at all.

Stages (run in this order): extract, define, embed, retrieve, judge, fusion
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict


# Built-in defaults -- identical to the argparse defaults below, but also
# used to fill in any key a YAML task file leaves unset. Keeping ONE dict as
# the source of truth (instead of duplicating these values across argparse
# and a separate YAML-defaults dict) means the CLI-flags path and the
# YAML-config path can never quietly drift apart.
DEFAULTS: Dict[str, Any] = {
    "workdir": None,
    # task_name/output_root are a convenience alternative to typing out a full
    # "workdir" path by hand for every task: if "workdir" is left unset, it's
    # derived as os.path.join(output_root, task_name) -- so a YAML task file
    # only needs one short "task_name" to land in its own separate output
    # folder, and different tasks (omim_ordo, snomed_fma, ...) automatically
    # stay out of each other's way under the same output_root. An explicit
    # "workdir" (set directly, either in YAML or via --workdir) always wins
    # over task_name/output_root, so nothing breaks for existing configs that
    # already set "workdir" directly.
    "task_name": None,
    "output_root": "./genom_runs",
    "src": None,
    "tgt": None,
    # extract stage only -- forwarded to deeponto.onto.Ontology(reasoner_type=...).
    # "hermit" (default) is a full OWL DL reasoner: correct for anything but
    # can OOM/be extremely slow on very large ontologies (e.g. FMA's ~89k
    # classes). "elk" is a much cheaper polynomial-time EL++ reasoner, correct
    # for large biomedical ontologies like FMA/SNOMED-CT/NCIT whose subsumption
    # axioms fall in the EL profile -- the same reason BERTMap/LogMap default
    # to elk for these. Kept as "hermit" here so existing/smaller tasks (e.g.
    # OMIM-ORDO) are unaffected; set to "elk" for large ontologies.
    "reasoner_type": "hermit",
    "llm_model": "Qwen/Qwen2.5-32B-Instruct",
    "embedding_model": "text-embedding-3-small",
    "topk": 20,
    "overwrite": False,
    "source_ontology_name": "OMIM",
    "target_ontology_name": "ORDO",
    "llm_batch_size": 16,
    "domain": "general",
    "max_src_concepts": 0,
    "index_type": "hnsw",
    # retrieve stage only -- which side's concepts drive the faiss
    # nearest-neighbour search (see genom_pipeline/steps/retrieve.py's
    # run() docstring). "src" (default) reproduces the original behaviour:
    # the index is built over tgt embeddings and each src concept queries
    # it. The output CSV's src_iri/tgt_iri/score/rank columns keep the
    # same meaning regardless of this setting.
    "query_side": "src",
    "fuse_strategy": "exact_priority",
    "fuse_llm_threshold": 0.9,
    "fuse_cs_threshold": 0.9,
    "fuse_use_llm_rank1_only": False,
    "fuse_enforce_target_uniqueness": False,
    "fuse_enforce_source_uniqueness": False,
    "exact_matcher": None,
    "exact_config": {},
}

_STAGE_NAMES = ("extract", "define", "embed", "retrieve", "judge", "fusion")


def _record_run_config(workdir: str, stage: str, config: dict) -> None:
    manifest_path = os.path.join(workdir, "run_config.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
    manifest[stage] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp_path, manifest_path)


def load_yaml_config(config_path: str, stage: str) -> Dict[str, Any]:
    """
    Load one stage's effective config from a YAML task file.

    Expected YAML shape -- top-level keys are SHARED across every stage
    (workdir, ontology names, llm_model, topk, domain, ...); a top-level key
    matching a stage name (extract/define/embed/retrieve/judge/fusion) is an
    optional mapping of stage-specific keys/overrides (src/tgt for extract,
    fuse_* + exact_matcher/exact_config for fusion, etc.). One YAML file is
    meant to describe a whole task (e.g. one OAEI pair), and you run it
    stage-by-stage with `--config that_file.yaml`, same as before with
    separate flags for each SLURM job step.

    Example (SNOMED<->FMA, topk=10, biomedical domain, exact matcher fused
    into the fusion stage):

        workdir: ./genom_snomed_fma_workdir
        source_ontology_name: SNOMED
        target_ontology_name: FMA
        llm_model: Qwen/Qwen2.5-32B-Instruct
        topk: 10
        domain: biomedical
        overwrite: true

        extract:
          src: /abs/path/snomed.owl
          tgt: /abs/path/fma.owl

        retrieve:
          index_type: flatip

        fusion:
          fuse_strategy: exact_priority
          fuse_llm_threshold: 0.9
          fuse_cs_threshold: 0.9
          exact_matcher: bertmaplt_string
          exact_config:
            src_owl: /abs/path/snomed.owl
            tgt_owl: /abs/path/fma.owl
            reasoner: elk
            pool_size: 200

    Precedence (lowest to highest): built-in DEFAULTS -> shared top-level
    keys -> the `<stage>:` section for the stage actually being run. A
    section for a DIFFERENT stage (e.g. a `judge:` block present in the file
    while running `extract`) is simply ignored by this call -- it's only
    read when that stage is the one being run.

    Returns a plain dict with every key run_stage() needs, plus "stage" set
    to `stage` and "_config_source" set to this file's absolute path (so
    run_config.json can record how the run was launched).
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to use --config / load_yaml_config "
            "(pip install pyyaml). It is normally already installed as a "
            "transformers dependency, so this is usually a sign of the "
            "wrong conda env being active rather than a real missing package."
        ) from e

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level")

    unknown_top_level = set(raw) - set(DEFAULTS) - set(_STAGE_NAMES)
    if unknown_top_level:
        raise ValueError(
            f"{config_path} has unrecognised top-level key(s): {sorted(unknown_top_level)}. "
            f"Shared keys must be one of {sorted(DEFAULTS)}, and per-stage sections must be "
            f"named one of {_STAGE_NAMES}. (Typo? e.g. 'fuse_strategy' misspelled.)"
        )

    shared = {k: v for k, v in raw.items() if k not in _STAGE_NAMES}
    stage_overrides = raw.get(stage, {}) or {}
    if not isinstance(stage_overrides, dict):
        raise ValueError(f"{config_path}: '{stage}:' section must be a mapping")

    config = dict(DEFAULTS)
    config.update(shared)
    config.update(stage_overrides)
    config["stage"] = stage
    config["_config_source"] = os.path.abspath(config_path)
    return config


def run_stage(config: Dict[str, Any]) -> str:
    """
    Run ONE stage given a fully-resolved config dict (see DEFAULTS for the
    complete key set). This is the single place stage dispatch/argument
    plumbing happens -- both the --config/YAML path and the plain-flags CLI
    path in main() build a config dict of this shape and call this function,
    so the two entry points can never behave differently for the same
    effective settings.

    Returns the output path the stage produced (same value it also prints).
    """
    stage = config["stage"]
    if stage not in _STAGE_NAMES:
        raise ValueError(f"unknown stage: {stage!r} (must be one of {_STAGE_NAMES})")

    workdir = config.get("workdir")
    if not workdir:
        task_name = config.get("task_name")
        if task_name:
            output_root = config.get("output_root") or DEFAULTS["output_root"]
            workdir = os.path.join(output_root, task_name)
        else:
            raise ValueError(
                "config needs either 'workdir' (a full path) or 'task_name' "
                "(--workdir flag / workdir: in the YAML; or --task_name / task_name: "
                "to get './genom_runs/<task_name>' automatically, keeping different "
                "tasks in their own separate output folders)"
            )
    os.makedirs(workdir, exist_ok=True)

    topk = int(config.get("topk") or DEFAULTS["topk"])
    overwrite = bool(config.get("overwrite", False))
    config_source = config.get("_config_source", "cli")

    concept_store_pt = os.path.join(workdir, "concept_store.pt")
    candidates_csv = os.path.join(workdir, f"candidates_top{topk}.csv")
    llm_alignment_csv = os.path.join(workdir, "llm_alignment.csv")
    final_alignment_csv = os.path.join(workdir, "final_alignment.csv")

    llm_config = {
        "source_ontology_name": config.get("source_ontology_name", DEFAULTS["source_ontology_name"]),
        "target_ontology_name": config.get("target_ontology_name", DEFAULTS["target_ontology_name"]),
        "batch_size": int(config.get("llm_batch_size") or DEFAULTS["llm_batch_size"]),
        "domain": config.get("domain", DEFAULTS["domain"]),
        "checkpoint_every_n_batches": 20,
    }
    llm_model = config.get("llm_model", DEFAULTS["llm_model"])

    if stage == "extract":
        src = config.get("src")
        tgt = config.get("tgt")
        if not src or not tgt:
            raise ValueError("extract stage needs config['src'] and config['tgt']")
        from genom_pipeline.steps.extract import run as extract_run

        max_src_concepts = int(config.get("max_src_concepts") or 0)
        reasoner_type = config.get("reasoner_type", DEFAULTS["reasoner_type"])
        _record_run_config(workdir, "extract", {
            "src": src, "tgt": tgt, "max_src_concepts": max_src_concepts,
            "reasoner_type": reasoner_type, "_config_source": config_source,
        })

        # genom_pipeline.steps.extract.run() has NO overwrite guard of its own --
        # it unconditionally re-parses both ontologies and unconditionally
        # torch.save()s over concept_store.pt. That's fine for a single run, but
        # is dangerous for the "resubmit the same sbatch script to continue after
        # a walltime kill" pattern define.py/judge.py's checkpointing is designed
        # to support (see the review log, round 11/17): if extract silently reran
        # on every resubmission it would wipe out hours/days of already-computed
        # definitions/embeddings living in the SAME concept_store.pt file. Guard
        # it here instead, at the driver level, the same way every other stage
        # already treats `overwrite` -- skip re-extraction entirely if the store
        # already exists and overwrite is not explicitly requested.
        if os.path.exists(concept_store_pt) and not overwrite:
            print(
                f"[extract] {concept_store_pt} already exists and overwrite=False -- "
                "skipping re-extraction (re-running would destroy any definitions/"
                "embeddings later stages already wrote into it). Pass overwrite: true "
                "in the YAML (or --overwrite on the CLI) to force a fresh extraction."
            )
            result = concept_store_pt
        else:
            result = extract_run(src, tgt, concept_store_pt, reasoner_type=reasoner_type)

            if max_src_concepts > 0:
                import torch

                store = torch.load(result)
                src_items = list(store["src"].items())[:max_src_concepts]
                store["src"] = dict(src_items)
                torch.save(store, result)
                print(f"[extract] trimmed src to first {max_src_concepts} concepts for a subset test run")

        print(f"[extract] -> {result}")
        return result

    elif stage == "define":
        from genom_pipeline.steps.define import run as define_run

        _record_run_config(workdir, "define", {
            "llm_model": llm_model, "overwrite": overwrite, "_config_source": config_source, **llm_config,
        })

        result = define_run(
            store_pt_path=concept_store_pt,
            llm_model=llm_model,
            llm_config=llm_config,
            overwrite=overwrite,
        )
        print(f"[define] -> {result}")
        return result

    elif stage == "embed":
        from genom_pipeline.steps.embed import run as embed_run

        embedding_model = config.get("embedding_model", DEFAULTS["embedding_model"])
        embed_config: Dict[str, Any] = {}
        _record_run_config(workdir, "embed", {
            "embedding_model": embedding_model, "overwrite": overwrite,
            "embed_config": embed_config, "_config_source": config_source,
        })

        result = embed_run(
            store_pt_path=concept_store_pt,
            embedding_model=embedding_model,
            embed_config=embed_config,
            overwrite=overwrite,
        )
        print(f"[embed] -> {result}")
        return result

    elif stage == "retrieve":
        from genom_pipeline.steps.retrieve import run as retrieve_run

        index_type = config.get("index_type", DEFAULTS["index_type"])
        query_side = config.get("query_side", DEFAULTS["query_side"])
        _record_run_config(workdir, "retrieve", {
            "topk": topk, "overwrite": overwrite, "index_type": index_type,
            "query_side": query_side, "_config_source": config_source,
        })

        result = retrieve_run(
            store_pt_path=concept_store_pt,
            output_candidates_csv=candidates_csv,
            topk=topk,
            overwrite=overwrite,
            index_type=index_type,
            query_side=query_side,
        )
        print(f"[retrieve] -> {result}")
        return result

    elif stage == "judge":
        from genom_pipeline.steps.judge import run as judge_run

        _record_run_config(workdir, "judge", {
            "llm_model": llm_model, "overwrite": overwrite, "_config_source": config_source, **llm_config,
        })

        result = judge_run(
            store_pt_path=concept_store_pt,
            candidates_csv=candidates_csv,
            output_alignment_csv=llm_alignment_csv,
            llm_model=llm_model,
            llm_config=llm_config,
            overwrite=overwrite,
        )
        print(f"[judge] -> {result}")
        return result

    else:  # stage == "fusion"
        from genom_pipeline.steps.fusion import run as fusion_run

        exact_spec: Dict[str, Any] = {}
        exact_matcher = config.get("exact_matcher")
        if exact_matcher:
            exact_matcher_config = config.get("exact_config") or {}
            if not isinstance(exact_matcher_config, dict):
                raise ValueError(
                    "config['exact_config'] must be a mapping/dict, e.g. "
                    "{'src_owl': '...', 'tgt_owl': '...'} (or the equivalent YAML block)"
                )
            exact_spec = {"name": exact_matcher, "config": exact_matcher_config}

        fuse_config = {
            "strategy": config.get("fuse_strategy", DEFAULTS["fuse_strategy"]),
            "llm_threshold": float(config.get("fuse_llm_threshold", DEFAULTS["fuse_llm_threshold"])),
            "cs_threshold": float(config.get("fuse_cs_threshold", DEFAULTS["fuse_cs_threshold"])),
            "use_llm_rank1_only": bool(config.get("fuse_use_llm_rank1_only", DEFAULTS["fuse_use_llm_rank1_only"])),
            "enforce_target_uniqueness": bool(config.get("fuse_enforce_target_uniqueness", DEFAULTS["fuse_enforce_target_uniqueness"])),
            "enforce_source_uniqueness": bool(config.get("fuse_enforce_source_uniqueness", DEFAULTS["fuse_enforce_source_uniqueness"])),
            "exact": exact_spec,
        }
        _record_run_config(workdir, "fusion", {
            "overwrite": overwrite, "fuse_config": fuse_config, "_config_source": config_source,
        })

        result = fusion_run(
            store_pt_path=concept_store_pt,
            llm_alignment_csv=llm_alignment_csv,
            output_final_csv=final_alignment_csv,
            overwrite=overwrite,
            fuse_config=fuse_config,
        )
        print(f"[fusion] -> {result}")
        return result


def run_from_yaml(config_path: str, stage: str) -> str:
    """
    The externally-exposed entry point requested for the YAML workflow:
    load a stage's config out of a YAML task file and run it, in one call.
    Equivalent to `python3 genom_stage_runner.py <stage> --config config_path`
    but importable from other Python code (e.g. a small script that loops
    over several stages/tasks) without going through argparse at all.
    """
    config = load_yaml_config(config_path, stage)
    return run_stage(config)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=list(_STAGE_NAMES))
    p.add_argument("--config", default=None,
                   help="path to a YAML task file (see load_yaml_config()'s docstring for the shape). "
                        "When given, every flag below is IGNORED -- the YAML alone determines this "
                        "run's config, so editing the YAML is the only thing you need to do between "
                        "runs instead of touching this script.")
    p.add_argument("--workdir", default=None, help="shared working directory for concept_store.pt and outputs")
    p.add_argument("--task_name", default=DEFAULTS["task_name"],
                    help="alternative to --workdir: derives workdir as '<output_root>/<task_name>', "
                         "so different tasks land in separate folders automatically without typing a full path")
    p.add_argument("--output_root", default=DEFAULTS["output_root"], help="base folder --task_name is joined onto (ignored if --workdir is set)")
    p.add_argument("--src", help="source ontology path (extract stage only)")
    p.add_argument("--tgt", help="target ontology path (extract stage only)")
    p.add_argument("--reasoner_type", default=DEFAULTS["reasoner_type"], choices=["hermit", "elk", "struct"],
                    help="extract stage only: OWL reasoner used to compute inferred parents. "
                         "hermit (default) is a full DL reasoner; elk is much cheaper on memory/time "
                         "and recommended for large ontologies (e.g. FMA/SNOMED-CT) that can OOM HermiT")
    p.add_argument("--llm_model", default=DEFAULTS["llm_model"], help="used for both define and judge")
    p.add_argument("--embedding_model", default=DEFAULTS["embedding_model"])
    p.add_argument("--topk", type=int, default=DEFAULTS["topk"])
    p.add_argument("--overwrite", action="store_true", help="force a stage to redo its work even if its output already exists")
    p.add_argument("--source_ontology_name", default=DEFAULTS["source_ontology_name"])
    p.add_argument("--target_ontology_name", default=DEFAULTS["target_ontology_name"])
    p.add_argument("--llm_batch_size", type=int, default=DEFAULTS["llm_batch_size"])
    # define.py/judge.py both support llm_config["domain"] (default "general"
    # inside those modules, see their run() docstrings) to swap the system
    # prompt's persona from a generic "ontology and knowledge representation
    # expert" to "an ontology expert specialising in the {domain} domain".
    # Used for both define and judge, same as --llm_model.
    p.add_argument("--domain", default=DEFAULTS["domain"], help="passed as llm_config['domain'] to both define and judge (default: \"general\", i.e. no domain-specific persona)")
    p.add_argument("--max_src_concepts", type=int, default=DEFAULTS["max_src_concepts"], help="if >0, only keep the first N src concepts (debug/subset run, applied at the extract stage)")

    # retrieve stage
    p.add_argument("--index_type", default=DEFAULTS["index_type"], choices=["flatip", "ivf", "hnsw"], help="retrieve stage only")
    p.add_argument("--query_side", default=DEFAULTS["query_side"], choices=["src", "tgt"],
                    help="retrieve stage only: which side's concepts drive the faiss nearest-neighbour "
                         "search (default: src, the original behaviour). Output CSV columns keep the "
                         "same src_iri/tgt_iri meaning either way -- see retrieve.py's run() docstring.")

    # fusion stage
    p.add_argument("--fuse_strategy", default=DEFAULTS["fuse_strategy"], choices=["union", "exact_priority", "llm_then_exact_fill"])
    p.add_argument("--fuse_llm_threshold", type=float, default=DEFAULTS["fuse_llm_threshold"])
    p.add_argument("--fuse_cs_threshold", type=float, default=DEFAULTS["fuse_cs_threshold"])
    # DEFAULT CHANGE (this driver script only -- genom_pipeline.steps.fusion.run()'s
    # own library defaults are left untouched, still True, since other OAEI
    # Bio-ML pairs may still want the paper-aligned 1-to-1 convention): OMIM<->ORDO
    # genuinely has one-to-many/many-to-one correspondences, so plain
    # `python3 genom_stage_runner.py fusion --workdir ... --overwrite` (no extra
    # flags) now does PURE THRESHOLD-BASED filtering -- no forced rank1-only
    # restriction, no src/tgt dedup -- instead of silently collapsing to 1-to-1.
    # Pass the --fuse_* flag explicitly to opt back into the old restrictive
    # behaviour for a given run.
    p.add_argument("--fuse_use_llm_rank1_only", dest="fuse_use_llm_rank1_only", action="store_true", default=DEFAULTS["fuse_use_llm_rank1_only"],
                    help="keep only the single best-ranked LLM candidate per src_iri (default: off -- keep every candidate passing the thresholds)")
    p.add_argument("--no_fuse_use_llm_rank1_only", dest="fuse_use_llm_rank1_only", action="store_false",
                    help="(default) keep every LLM candidate per src_iri that passes the thresholds, not just rank 1")
    p.add_argument("--fuse_enforce_target_uniqueness", dest="fuse_enforce_target_uniqueness", action="store_true", default=DEFAULTS["fuse_enforce_target_uniqueness"],
                    help="each tgt_iri used at most once (default: off)")
    p.add_argument("--no_fuse_enforce_target_uniqueness", dest="fuse_enforce_target_uniqueness", action="store_false",
                    help="(default) allow the same tgt_iri to appear in more than one output row (many-to-one)")
    p.add_argument("--fuse_enforce_source_uniqueness", dest="fuse_enforce_source_uniqueness", action="store_true", default=DEFAULTS["fuse_enforce_source_uniqueness"],
                    help="each src_iri used at most once (default: off)")
    p.add_argument("--no_fuse_enforce_source_uniqueness", dest="fuse_enforce_source_uniqueness", action="store_false",
                    help="(default) allow the same src_iri to appear in more than one output row (one-to-many)")
    # exact matcher, fused in alongside the LLM judgments (fusion stage only).
    # Omit --exact_matcher entirely to skip exact matching (fuse_config["exact"]
    # stays {}). Each registered matcher (see genom_pipeline/exact_matcher/__init__.py)
    # wants a differently-shaped config dict -- bertmaplt_string needs
    # src_owl/tgt_owl/reasoner/pool_size/apply_lowercasing, logmaplt_file
    # needs mapping_tsv/sep/has_header/columns, logmaplt_subprocess needs
    # src_owl/tgt_owl/jar_path/output_dir/... -- rather than hard-coding one
    # argparse flag per matcher-specific key, --exact_config_json takes the
    # whole config as a single JSON object string on the CLI path (the YAML
    # path uses a native mapping instead -- see load_yaml_config's docstring
    # -- which is friendlier to write than an escaped JSON string).
    p.add_argument("--exact_matcher", default=DEFAULTS["exact_matcher"],
                    choices=["bertmaplt_string", "logmaplt_file", "logmaplt_subprocess"],
                    help="optional exact matcher to fuse in alongside the LLM judgments (fusion stage only); omit to skip exact matching (default)")
    p.add_argument("--exact_config_json", default="{}",
                    help='JSON object string with the config dict for --exact_matcher, e.g. for bertmaplt_string: '
                         '\'{"src_owl": "/path/omim.owl", "tgt_owl": "/path/ordo.owl", "reasoner": "elk", "pool_size": 200}\' '
                         "(fusion stage only, ignored if --exact_matcher is not set; ignored entirely if --config is used)")

    args = p.parse_args()

    if args.config:
        config = load_yaml_config(args.config, args.stage)
    else:
        if not args.workdir and not args.task_name:
            sys.exit("either --workdir or --task_name is required (or pass --config path.yaml)")

        config = {
            "stage": args.stage,
            "workdir": args.workdir,
            "task_name": args.task_name,
            "output_root": args.output_root,
            "src": args.src,
            "tgt": args.tgt,
            "reasoner_type": args.reasoner_type,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "topk": args.topk,
            "overwrite": args.overwrite,
            "source_ontology_name": args.source_ontology_name,
            "target_ontology_name": args.target_ontology_name,
            "llm_batch_size": args.llm_batch_size,
            "domain": args.domain,
            "max_src_concepts": args.max_src_concepts,
            "index_type": args.index_type,
            "query_side": args.query_side,
            "fuse_strategy": args.fuse_strategy,
            "fuse_llm_threshold": args.fuse_llm_threshold,
            "fuse_cs_threshold": args.fuse_cs_threshold,
            "fuse_use_llm_rank1_only": args.fuse_use_llm_rank1_only,
            "fuse_enforce_target_uniqueness": args.fuse_enforce_target_uniqueness,
            "fuse_enforce_source_uniqueness": args.fuse_enforce_source_uniqueness,
            "exact_matcher": args.exact_matcher,
            "exact_config": {},
            "_config_source": "cli",
        }
        if args.exact_matcher:
            try:
                exact_matcher_config = json.loads(args.exact_config_json)
            except json.JSONDecodeError as e:
                sys.exit(f"--exact_config_json is not valid JSON: {e}")
            if not isinstance(exact_matcher_config, dict):
                sys.exit("--exact_config_json must decode to a JSON object (dict), e.g. '{\"src_owl\": ...}'")
            config["exact_config"] = exact_matcher_config

    run_stage(config)


if __name__ == "__main__":
    main()
