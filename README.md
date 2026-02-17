# GenOM Pipeline

This repository contains an implementation of the ontology alignment approach
described in the paper:

> **GenOM: Ontology Matching with Description Generation and Large Language Models**, under review.

Official implementation of **GenOM**, an ontology matching framework centred on semantic enrichment via LLM-generated definitions.

Paper:
https://arxiv.org/abs/2508.10703

This repository is currently under active development.  
The pipeline is being modularised and stabilised.  
Interfaces and configuration options may change.

---

## Overview

Pipeline

The workflow consists of:

1. Concept extraction from source and target ontologies

2. LLM-based definition generation for semantic enrichment

3. Embedding computation

4. Top-k candidate retrieval

5. LLM-based alignment decision

6. Fusion with exact matchers

---

## Installation

git clone https://github.com/AlanS87/genom-pipeline.git  
cd genom-pipeline  
pip install -e .

---

## Quick Start

```python
from genom_pipeline import run_pipeline

run_pipeline(
    src_onto_path="path/to/source.owl",
    tgt_onto_path="path/to/target.owl",
    workdir="runs/example",
    llm_model="Qwen/Qwen2.5-14B-Instruct",
    embedding_model="text-embedding-3-small",
)