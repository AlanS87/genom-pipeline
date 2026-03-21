# GenOM Pipeline

This repository contains the official implementation of:

> **GenOM: Ontology Matching with Description Generation and Large Language Models** (under review)

GenOM is an ontology matching framework that improves alignment performance through **LLM-based semantic enrichment**, particularly via definition generation.

Paper:  
https://arxiv.org/abs/2508.10703

---

## Overview

GenOM is designed to address the limitations of traditional ontology matching methods that rely primarily on internal ontology information.  
It introduces semantic enrichment using LLM-generated definitions, which enhances concept representations and improves multiple stages of the matching pipeline.

---

## Pipeline Overview

![GenOM Pipeline](docs/GenOM_workflow.png)

The workflow consists of the following stages:

1. Concept extraction from source and target ontologies (DeepOnto-based)
2. LLM-based definition generation for semantic enrichment
3. Embedding computation
4. Top-k candidate retrieval
5. LLM-based equivalence judgement
6. Fusion with exact matching signals

---

## Repository Structure



    genom_pipeline/
    │
    ├── steps/
    │   ├── extract.py
    │   ├── define.py
    │   ├── embed.py
    │   ├── retrieve.py
    │   ├── judge.py
    │   ├── fusion.py
    │
    ├── docs/
    │   └── pipeline.png
    │
    ├── runs/

---

## Installation

    git clone https://github.com/AlanS87/genom-pipeline.git
    cd genom-pipeline
    pip install -e .

---

## Requirements

- Python 3.10+
- FAISS
- Transformers / vLLM
- DeepOnto
- OpenAI API (if used for embeddings or LLM)

---

## Quick Start

    from genom_pipeline import run_pipeline

    run_pipeline(
        src_onto_path="path/to/source.owl",
        tgt_onto_path="path/to/target.owl",
        workdir="runs/example",
        llm_model="Qwen/Qwen2.5-14B-Instruct",
        embedding_model="text-embedding-3-small",
    )

---

## Input

- Source ontology (.owl)
- Target ontology (.owl)

---

## Output

Results are stored in the specified workdir, including:

- Extracted concept data
- Generated definitions
- Candidate mappings
- Final alignment results

Final output format:

    (source_concept, target_concept)

---

## Reproducing Paper Results

To reproduce the experimental setup:

- Use Bio-ML datasets (OAEI)
- Cosine similarity threshold = 0.9
- LLM probability threshold = 0.9
- LLM: Qwen2.5-32B-Instruct

A single task is used for threshold selection, and the same thresholds are applied across all tasks.

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

    @article{song2025genom,
        title={GenOM: Ontology Matching with Description Generation and Large Language Model},
        author={Song, Yiping and Chen, Jiaoyan and Schmidt, Renate A},
        journal={arXiv preprint arXiv:2508.10703},
        year={2025}
    }

---

## Status

This repository is under active development.  
Interfaces and configurations may change.