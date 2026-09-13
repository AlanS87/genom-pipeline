# GenOM Pipeline

This repository contains the official implementation of:

> **GenOM: Ontology Matching with Description Generation and Large Language Models**
> Published in *World Wide Web* (Springer), 2026.

GenOM is an ontology matching framework that improves alignment performance through **LLM-based semantic enrichment**, particularly via definition generation.

Paper:
- Published version (open access): https://doi.org/10.1007/s11280-026-01413-y
- Preprint: https://arxiv.org/abs/2508.10703

---

## Overview

GenOM is designed to address the limitations of traditional ontology matching methods that rely primarily on internal ontology information.
It introduces semantic enrichment using LLM-generated definitions, which enhances concept representations and improves multiple stages of the matching pipeline.

GenOM was developed and evaluated on the OAEI Bio-ML biomedical ontology benchmarks, but nothing in the pipeline is hard-coded to biomedical data -- domain, ontology names, and matching thresholds are all configuration, not code (see [Configuration Reference](#configuration-reference)).

---

## Pipeline Overview

![GenOM Pipeline](docs/GenOM_workflow.png)

The workflow consists of six stages, each implemented as a standalone module under `genom_pipeline/steps/` and orchestrated by `run_pipeline()` in `genom_pipeline/pipeline.py`:

| # | Stage | Module | Reads | Writes | Requires |
|---|-------|--------|-------|--------|----------|
| 1 | Concept extraction | `steps/extract.py` | `.owl` ontologies | `concept_store.pt` | DeepOnto (Java/JVM) |
| 2 | Definition generation | `steps/define.py` | `concept_store.pt` | `concept_store.pt` (+`definition`) | a local HF causal LM (`llm_model`) |
| 3 | Embedding | `steps/embed.py` | `concept_store.pt` | `concept_store.pt` (+`embedding`) | OpenAI Embeddings API |
| 4 | Candidate retrieval | `steps/retrieve.py` | `concept_store.pt` | `candidates_top{k}.csv` | FAISS |
| 5 | Equivalence judgement | `steps/judge.py` | `concept_store.pt` + candidates | `llm_alignment.csv` | a local HF causal LM (`llm_model`) |
| 6 | Fusion | `steps/fusion.py` | `llm_alignment.csv` (+ optional exact matcher) | `final_alignment.csv` | nothing extra (optionally an exact matcher, see below) |

Every intermediate artifact is written to `workdir` and is skipped on re-run unless `overwrite=True`, so a failed or interrupted run can simply be re-invoked with the same arguments to resume from the last completed stage (steps 1, 4, 5 and 6 check the output file; step 2 and 3 check per-concept whether a `definition`/`embedding` value already exists).

A seventh, optional stage lives outside `run_pipeline()`: `genom_pipeline/evaluation/` scores a `final_alignment.csv` against a gold-standard reference and can sweep threshold combinations without re-running steps 2-5. See [Evaluation](#evaluation).

---

## Repository Structure

    genom_pipeline/
    ├── __init__.py            # exposes run_pipeline, PipelineResult
    ├── pipeline.py             # orchestrates steps 1-6
    ├── steps/
    │   ├── extract.py          # stage 1
    │   ├── define.py           # stage 2
    │   ├── embed.py             # stage 3
    │   ├── retrieve.py         # stage 4
    │   ├── judge.py             # stage 5
    │   └── fusion.py             # stage 6
    ├── exact_matcher/
    │   ├── bertmaplt_string.py   # runs BERTMap-Lt's string matcher in-process (DeepOnto)
    │   ├── logmaplt_file.py       # loads a mapping TSV from a LogMap run you already did
    │   └── logmaplt_subprocess.py # actually invokes the LogMap jar (see scripts/download_logmap.sh)
    └── evaluation/
        └── metrics.py            # precision/recall/F1 + threshold grid search (not part of run_pipeline)

    scripts/
    ├── download_logmap.sh        # fetches the LogMap standalone jar at setup time (not vendored in git)
    ├── genom_stage_runner.py     # run ONE stage per OS process (CLI flags or --config a YAML task file) -- see below
    └── example_slurm_job.sbatch  # worked example: submitting a full task on an HPC/SLURM cluster

    tasks/
    └── example_task.yaml         # worked example YAML task file for scripts/genom_stage_runner.py

    tests/                        # pytest suite, run in CI (see .github/workflows/ci.yml)
    docs/
    └── GenOM_workflow.png

---

## Installation

    git clone https://github.com/AlanS87/genom-pipeline.git
    cd genom-pipeline
    pip install -e .
    # or, to also get the test tooling:
    pip install -e ".[dev]"

### Prerequisites

- **Python 3.9+** (CI runs 3.10 and 3.11)
- **Java (JDK)** -- DeepOnto wraps the OWL API via a JVM, so a JDK must be on `PATH` for stage 1 (`extract`) and the `bertmaplt_string` exact matcher. `scripts/download_logmap.sh` / the `logmaplt_subprocess` matcher also shell out to `java` directly if you use LogMap.
- **`OPENAI_API_KEY`** -- stage 3 (`embed`) calls the OpenAI Embeddings API. Set the environment variable, or pass `embed_config={"api_key": "..."}`.
- **A GPU is strongly recommended** for stages 2 and 5 -- both load a real Hugging Face causal LM locally via `transformers` (`device_map="auto"`), not a hosted API. Gated models need `llm_config={"hf_token": "..."}`.
- **FAISS** -- `faiss-cpu` is installed by default; pass `retrieve_config={"index_params": {"use_gpu": True}}` if you have `faiss-gpu` installed separately.
- **LogMap standalone jar** -- only needed if you use the `logmaplt_subprocess` or `logmaplt_file` exact matchers. Not vendored in this repo (it's a large third-party binary) -- fetch it with `scripts/download_logmap.sh`, see [Exact Matching](#exact-matching).

---

## Quick Start

```python
from genom_pipeline import run_pipeline

result = run_pipeline(
    src_onto_path="path/to/source.owl",
    tgt_onto_path="path/to/target.owl",
    workdir="runs/example",
    llm_model="Qwen/Qwen2.5-14B-Instruct",
    embedding_model="text-embedding-3-small",
    llm_config={
        "domain": "legal",                    # any domain, not just biomedical -- see below
        "source_ontology_name": "SourceOnto",
        "target_ontology_name": "TargetOnto",
    },
)

print(result.final_alignment_csv)
```

`OPENAI_API_KEY` must be set in the environment before calling `run_pipeline` (or pass `embed_config={"api_key": "..."}`).

---

## Minimum required arguments

`run_pipeline()` has three required, no-default arguments -- everything else has a default or is optional:

- `src_onto_path`, `tgt_onto_path` -- paths to the two `.owl` files you want to align.
- `workdir` -- where all intermediate and final output files are written.

Beyond that, whether an argument is *effectively* required depends on which stages you actually run:

- `llm_model` is required if `run_definition=True` (the default -- stage 2 needs it) and is always required for stage 5 (`judge`), since `run_pipeline()` always runs judge. In practice: always pass it.
- `embedding_model` is required for stage 3 (`embed`), which `run_pipeline()` always runs. In practice: always pass it, and make sure `OPENAI_API_KEY` is set.
- `fuse_config` is entirely optional. If you don't pass `fuse_config={"exact": {...}}`, stage 6 simply skips exact matching and the final alignment is the LLM's own (thresholded, 1-to-1) predictions -- you do not need LogMap or BERTMap-Lt set up to get a first result.
- `llm_config`, `embed_config`, `retrieve_config` are all optional and fall back to the defaults documented below.

So the smallest realistic first call is exactly the Quick Start example above, minus `llm_config` if you're fine with the generic (non-domain-specialised) prompt.

---

## Configuration Reference

Every stage takes a plain `dict` of options rather than a config file. All keys below are optional unless noted; defaults are what the code uses when a key is omitted.

### `reasoner_type` (stage 1 `extract`, a direct `run_pipeline()` argument -- not nested in a dict)

| Default | Meaning |
|---|---|
| `"hermit"` | OWL reasoner used to compute each concept's inferred (not just asserted) superclasses. `"hermit"` is a full OWL DL reasoner: correct for any ontology, but can be extremely slow, or run out of JVM heap, on very large ones (tens of thousands of classes). `"elk"` is a much cheaper polynomial-time EL++ reasoner -- correct as long as the ontology's subsumption-relevant axioms fall in the EL profile, which large biomedical ontologies (FMA, SNOMED-CT, NCIT, ...) normally do, which is also why OAEI Bio-ML tooling such as BERTMap/LogMap defaults to `elk` for these. `"struct"` (DeepOnto's structural/non-logical reasoner) is also accepted. If `extract` fails with a Java heap `OutOfMemoryError`, try `"elk"` here and/or raise the `GENOM_JVM_MEMORY` environment variable (see [Notes](#notes)). |

### `llm_config` (shared by stage 2 `define` and stage 5 `judge`)

| Key | Default | Meaning |
|---|---|---|
| `domain` | `"general"` | Fills the LLM's persona: `"You are an ontology expert specialising in the {domain} domain."` Any string works -- `"legal"`, `"finance"`, `"biomedical"`, etc. `"general"` (or omitting the key) uses a generic ontology-expert persona instead. |
| `source_ontology_name`, `target_ontology_name` | `None` | Human-readable ontology names woven into the prompt ("...from the {source_ontology_name} ontology..."). Purely cosmetic context for the LLM; has no effect on file paths. |
| `hf_token` | `None` | Hugging Face access token, needed for gated models. |
| `generation` (stage 2 only) | `{"max_new_tokens": 256, "do_sample": True, "temperature": 0.7, "top_p": 0.9}` | Forwarded to `model.generate()` for definition generation. |
| `batch_size` (stage 5 only) | `16` | Candidate pairs judged per forward pass. |
| `torch_dtype` (stage 5 only) | `"bfloat16"` | dtype the judge model is loaded in. |
| `device_map` (stage 5 only) | `"auto"` | Forwarded to `from_pretrained`. |
| `threshold` (stage 5 only) | `0.5` | **Not the same as `fuse_config["llm_threshold"]`** -- this only decides the `decision` column (YES/NO) written into `llm_alignment.csv`. `fuse_config["llm_threshold"]` (default `0.9`) is a stricter, separate filter applied afterwards in stage 6. Both `score` (cosine similarity) and `confidence` (the raw YES-probability) are preserved in `llm_alignment.csv` regardless of this threshold, so stage 6 (or `evaluation.grid_search_thresholds`) can re-filter without re-running the LLM. |

### `embed_config` (stage 3)

| Key | Default | Meaning |
|---|---|---|
| `api_key` | `os.environ["OPENAI_API_KEY"]` | OpenAI API key. |
| `text_fields` | `("label", "synonyms", "definition")` | Which concept fields get concatenated into the embedding input text. |
| `lowercase` | `True` | Lowercase text before embedding. |
| `join_synonyms`, `join_parents` | `True`, `True` | Join as one `"synonyms: a, b, c"` clause vs. appending each item separately. |
| `batch_size` | `64` | Concepts embedded per API call. |
| `max_tokens` | `8000` | Per-text truncation limit (via `tiktoken`), to avoid API token-limit errors. |
| `checkpoint_dir` | `{workdir}/checkpoints` | Embeddings are checkpointed here and resumed automatically if the run is interrupted. |

### `retrieve_config` (stage 4)

| Key | Default | Meaning |
|---|---|---|
| `normalize` | `True` | L2-normalize embeddings before indexing (so `score` is cosine similarity, not raw inner product). |
| `index_type` | `"hnsw"` | `"flatip"` (exact), `"ivf"`, or `"hnsw"` (approximate). |
| `index_params` | `{}` | Index-specific: `use_gpu`/`gpu_id`; IVF: `nlist`, `nprobe`; HNSW: `m`, `ef_construction`, `ef_search`. |

### `fuse_config` (stage 6)

| Key | Default | Meaning |
|---|---|---|
| `strategy` | `"exact_priority"` | `"exact_priority"` / `"union"` (currently identical -- both put exact-matcher rows first) or `"llm_then_exact_fill"` (LLM rows win conflicts, exact matcher only fills srcs the LLM didn't cover). |
| `llm_threshold` | `0.9` | Minimum `confidence` (LLM YES-probability) to keep a candidate. |
| `cs_threshold` | `0.9` | Minimum `score` (cosine similarity, from stage 4) to keep a candidate. Set to `None` to disable. |
| `enforce_target_uniqueness` | `True` | Also enforce that each target concept is used at most once (a true 1-to-1 alignment). |
| `enforce_source_uniqueness` | `True` | Enforce that each source concept is used at most once. Set both this and `enforce_target_uniqueness` to `False` (together with `use_llm_rank1_only=False`) for domains with genuine one-to-many/many-to-one correspondences (e.g. OMIM&harr;ORDO). Regardless of either flag, the exact same `(src_iri, tgt_iri)` pair proposed twice (e.g. by both an exact matcher and the LLM) is always collapsed to one row. |
| `use_llm_rank1_only` | `True` | Only the best-ranked surviving candidate per source concept is kept. |
| `exact` | `None` | `{"name": "bertmaplt_string" \| "logmaplt_file" \| "logmaplt_subprocess", "config": {...}}`. Omit entirely to skip exact matching -- see [Exact Matching](#exact-matching) for each matcher's own config keys. |

Reproducing the paper's reported setup: `llm_threshold=0.9`, `cs_threshold=0.9` (already the defaults), `llm_model="Qwen/Qwen2.5-32B-Instruct"`, Bio-ML (OAEI) datasets. A single task is used for threshold selection and the same thresholds are applied across all tasks.

---

## Exact Matching

Stage 6 can optionally fuse the LLM's predictions with an exact/lexical matcher via `fuse_config["exact"]`. Three matchers are registered:

- **`bertmaplt_string`** -- runs BERTMap-Lt's string-matching module in-process via DeepOnto. `config`: `src_owl`, `tgt_owl` (required), `reasoner` (default `"elk"`), `pool_size` (default `200`), `apply_lowercasing` (default `True`).
- **`logmaplt_file`** -- loads a mapping TSV that some earlier, separate LogMap run already produced. `config`: `mapping_tsv` (required), `sep` (default `"\t"`), `has_header` (default `False`), `columns` (if `has_header`).
- **`logmaplt_subprocess`** -- actually invokes the LogMap jar (`LITE` mode) and parses its output, instead of assuming a TSV already exists. `config`: `src_owl`, `tgt_owl` (required), `jar_path` (or set the `LOGMAP_JAR_PATH` env var), `output_dir`, `java_bin` (default `"java"`), `jvm_args`, `timeout` (default `1800`s), `mapping_filename` (only needed if LogMap's output file can't be auto-detected), `sep`, `has_header`, `overwrite`.

LogMap itself (Apache License 2.0) is **not vendored in this repository** -- it's a large third-party Java tool with its own release cycle. Fetch it once with:

```bash
scripts/download_logmap.sh .cache/logmap <asset_download_url>
export LOGMAP_JAR_PATH=.cache/logmap/logmap-matcher-4.0.jar
```

(`<asset_download_url>` comes from the [official release page](https://github.com/ernestojimenezruiz/logmap-matcher/releases/tag/logmap-matcher-july-2021) -- GitHub's asset listing is JS-rendered, so the script asks for the link explicitly rather than guessing it.)

---

## Evaluation

`genom_pipeline.evaluation.metrics` is separate from `steps.fusion` on purpose: `fusion.run()` only produces predictions and has no notion of gold labels, so it stays usable in a real deployment where no reference alignment exists.

```python
from genom_pipeline.evaluation import metrics

# score an existing final_alignment.csv against a gold reference (OAEI-style TSV)
scores = metrics.evaluate_final_alignment(
    final_alignment_csv="runs/example/final_alignment.csv",
    reference_path="path/to/refs_equiv/full.tsv",
)
print(scores)  # {"precision": ..., "recall": ..., "f1": ..., ...}

# sweep (llm_threshold, cs_threshold) combinations directly on llm_alignment.csv,
# without re-running retrieval or the LLM judge
grid = metrics.grid_search_thresholds(
    llm_alignment_csv="runs/example/llm_alignment.csv",
    reference_path="path/to/refs_equiv/full.tsv",
    llm_thresholds=[0.5, 0.7, 0.9],
    cs_thresholds=[0.8, 0.9, 0.95],
)
print(metrics.best_thresholds(grid))
```

`metrics.plot_f1_heatmap(grid)` renders the same sweep as a heatmap, but requires `matplotlib`/`seaborn` (not core dependencies -- install them separately if you want it).

---

## Running Stage-by-Stage on an HPC/SLURM Cluster

`run_pipeline()` (above) runs all six stages in one Python process -- the simplest way to call genom_pipeline, and a fine default. `scripts/genom_stage_runner.py` is an alternative driver that runs **one stage per OS process**, which matters on a shared HPC cluster for two reasons:

- Stages 2 (`define`) and 5 (`judge`) each load a full local causal LM (e.g. a 32B-parameter model) via `device_map="auto"`. Running them as separate processes means the OS/CUDA driver fully reclaims GPU memory when one process exits, before the next stage's process starts -- a stronger guarantee than the in-process `del`/`gc.collect()`/`empty_cache()` cleanup `run_pipeline()` relies on (see [Notes](#notes)).
- A crash or SLURM walltime timeout in a later, cheap stage (e.g. `embed`'s OpenAI calls) does not require re-running the expensive GPU stages that already finished and checkpointed to disk.

### Two ways to invoke it

```bash
# 1) Flags on the command line
python3 scripts/genom_stage_runner.py extract --workdir runs/example --src src.owl --tgt tgt.owl

# 2) A YAML task file (recommended -- see tasks/example_task.yaml)
python3 scripts/genom_stage_runner.py extract --config tasks/my_task.yaml
```

Both call the exact same underlying dispatch logic, so the two entry points can never silently diverge in behaviour. When `--config` is given, every other flag is ignored -- the YAML alone determines that run.

### YAML task file shape

Copy `tasks/example_task.yaml` and fill in the placeholders. One file describes one whole ontology-matching task; top-level keys are shared across every stage, and a top-level key matching a stage name (`extract`/`define`/`embed`/`retrieve`/`judge`/`fusion`) holds overrides used only when running that stage:

```yaml
task_name: my_task          # workdir becomes "<output_root>/<task_name>" automatically
output_root: ./genom_runs
source_ontology_name: SourceOntologyName
target_ontology_name: TargetOntologyName
llm_model: Qwen/Qwen2.5-32B-Instruct
embedding_model: text-embedding-3-small
topk: 20
domain: general
overwrite: false            # keep false across the whole task -- see "Resuming a run" below

extract:
  src: /path/to/source_ontology.owl
  tgt: /path/to/target_ontology.owl
  reasoner_type: hermit      # or "elk" for very large ontologies -- see Configuration Reference

retrieve:
  index_type: flatip

fusion:
  fuse_strategy: exact_priority
  fuse_llm_threshold: 0.9
  fuse_cs_threshold: 0.9
  # exact_matcher: bertmaplt_string   # optional, see Exact Matching
  # exact_config: {...}
```

Run each stage in order (`extract`, `define`, `embed`, `retrieve`, `judge`, `fusion`), each as its own `python3 scripts/genom_stage_runner.py <stage> --config ...` invocation -- see `scripts/example_slurm_job.sbatch` for a complete SLURM job doing exactly this.

Unrecognised top-level keys (e.g. a typo like `fuse_strategie`) raise an error immediately rather than silently being ignored.

### Resuming a run

Keep `overwrite: false` for the whole task, including any resubmitted/chained job:

- `extract`, `retrieve`, `judge`, and `fusion` each skip re-doing their work if their output file already exists.
- `define` and `judge` additionally checkpoint incrementally (every `checkpoint_every_n_batches` batches) so a job killed mid-stage (e.g. a SLURM walltime limit) resumes from its last checkpoint instead of restarting that stage from scratch.

So if a job runs out of wallclock partway through, simply resubmitting the exact same script picks up where it stopped.

### Every invocation records what it ran with

Each stage run appends the exact config it used to `<workdir>/run_config.json` (keyed by stage), including `_config_source` (`"cli"` or the YAML file's absolute path) -- so "what parameters did stage X actually run with" is always answerable by reading that file.

`run_stage(config)` and `run_from_yaml(config_path, stage)` are also plain importable Python functions, for driving a stage programmatically (e.g. looping over several YAML task files) without going through the command line at all.

---

## Input

- Source ontology (`.owl`)
- Target ontology (`.owl`)

---

## Output

Results are stored in the specified `workdir`:

- `concept_store.pt` -- extracted concept data, generated definitions, and embeddings (all stages after 1 write back into this file)
- `candidates_top{k}.csv` -- top-k retrieval candidates (`src_iri, tgt_iri, score, rank`)
- `llm_alignment.csv` -- every judged candidate, with `confidence` and `decision` columns added
- `final_alignment.csv` -- the final alignment: `src_iri, tgt_iri, score, provenance`

Final mapping format: `(src_iri, tgt_iri)` pairs, at most one row per `src_iri` (and, by default, per `tgt_iri`).

---

## Reproducing Paper Results

To reproduce the experimental setup:

- Use Bio-ML datasets (OAEI)
- Cosine similarity threshold = 0.9
- LLM probability threshold = 0.9
- LLM: Qwen2.5-32B-Instruct

A single task is used for threshold selection, and the same thresholds are applied across all tasks. Both are already the defaults in `fuse_config` -- see [Configuration Reference](#configuration-reference).

---

## Key Design Principles

- Semantic enrichment improves representation rather than a single module
- Gains come from improvements across multiple pipeline stages
- Fixed thresholds are used to evaluate robustness

---

## Notes

- Missing ontology fields (e.g., synonyms, parents) are left empty
- Extraction is implemented using DeepOnto
- LLM outputs are used as semantic signals
- **DeepOnto compatibility patch**: DeepOnto's verbaliser (used by stage 1 to turn complex class expressions into natural language) does not support the OWL `DataHasValue` data property restriction, and crashes on any ontology that uses it -- SNOMED-CT being a real-world example. `genom_pipeline/_deeponto_compat.py` monkey-patches this at import time (applied automatically by `steps/extract.py`) to drop the unsupported sub-expression instead of crashing, emitting a `genom_pipeline._deeponto_compat.DataHasValueOmittedWarning` each time it happens so the information loss is visible rather than silent. This is a workaround for an upstream DeepOnto limitation, not a genom_pipeline design choice -- remove it once DeepOnto supports `DataHasValue` natively.
- **Non-interactive JVM startup**: DeepOnto starts a JVM (via `jpype`) the first time anything imports `deeponto.onto`, and will interactively prompt on stdin for how much memory to give it (`Please enter the maximum memory located to JVM`) unless the JVM is already running -- this blocks forever, and errors outright under pytest (`OSError: reading from stdin while output is captured!`), in any non-interactive context such as CI. `genom_pipeline/_deeponto_compat.py` starts the JVM itself, non-interactively, as soon as it's imported (before any `deeponto.onto` import anywhere in genom_pipeline), so the prompt never fires. The memory limit defaults to `8g` (DeepOnto's own default) and can be overridden with the `GENOM_JVM_MEMORY` environment variable, e.g. `GENOM_JVM_MEMORY=4g python your_script.py`.
- **`retrieve_config["index_type"]="hnsw"`'s `score` column is cosine similarity**, same as `"flatip"`/`"ivf"` -- `_build_faiss_index()` passes `faiss.METRIC_INNER_PRODUCT` explicitly for all three index types (higher `score` = more similar everywhere, which `fuse_config["cs_threshold"]` assumes). If you have `final_alignment.csv`/`candidates_top{k}.csv` output generated with `index_type="hnsw"` from before this fix, its `score` column was actually an L2 distance (lower = more similar, the opposite convention) -- re-run stage 4 (`retrieve`) to regenerate it before trusting `score`/`cs_threshold` values from that output.
- **GPU memory between `define` and `judge`**: both load a full local causal LM (`llm_model`) via `device_map="auto"`; `run_pipeline()` calls them sequentially in one process, and each explicitly `del`s its model, `gc.collect()`s, and `torch.cuda.empty_cache()`s before returning, so the second model load doesn't have to compete with GPU memory the first model is done using. If you need a stronger guarantee (a full OS-level reclaim between every stage), run stage-by-stage with `scripts/genom_stage_runner.py` instead -- see below.

---

## Limitations

- Focused on equivalence matching
- Not evaluated on tasks such as entity linking
- Performance depends on LLM choice

---

## Future Work

- Support subsumption relations
- Adaptive threshold selection
- Extend to other semantic matching tasks

---

## Citation

If you use this repository, please cite the published version:

    @article{song2026genom,
        title={GenOM: ontology matching with description generation and large language models},
        author={Song, Yiping and Chen, Jiaoyan and Schmidt, Renate A.},
        journal={World Wide Web},
        volume={29},
        number={3},
        articleno={29},
        year={2026},
        publisher={Springer},
        doi={10.1007/s11280-026-01413-y}
    }

A preprint version is also available on arXiv (arXiv:2508.10703); the published
version above is the canonical, peer-reviewed reference.

---

## Changelog

**Unreleased / next release** -- fixes found while reproducing the paper's results end-to-end on a second, larger OAEI Bio-ML task pair:

- Fixed `retrieve.py`'s `"hnsw"` index defaulting to L2 distance instead of cosine similarity (see [Notes](#notes)).
- Added `reasoner_type` (`extract`/`run_pipeline()`), so large ontologies (tens of thousands of classes) can use the much cheaper `"elk"` reasoner instead of the default `"hermit"`, which can run out of JVM heap on them.
- `fusion.py`: added `enforce_source_uniqueness`, and the exact same `(src_iri, tgt_iri)` pair is now always collapsed to one row regardless of the uniqueness flags.
- `define.py`: batched definition generation (was one concept at a time); `define.py`/`judge.py`: incremental checkpointing so an interrupted run resumes instead of restarting; both now explicitly free GPU memory (`del` + `gc.collect()` + `torch.cuda.empty_cache()`) before returning.
- Fixed a JPype/JVM native-signal-handling segfault that could occur when a JVM (started for DeepOnto) shares a process with CUDA (via `torch`).
- Added `scripts/genom_stage_runner.py`, a per-stage/HPC-SLURM driver with an equivalent CLI-flags or `--config a_task.yaml` interface -- see [Running Stage-by-Stage on an HPC/SLURM Cluster](#running-stage-by-stage-on-an-hpcslurm-cluster).

---

## Status

This repository is under active development.
Interfaces and configurations may change.
