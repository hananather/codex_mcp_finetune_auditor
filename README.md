# Fine-tune Auditor

A toolkit for auditing fine-tuned language models to detect adversarial modifications using behavioral analysis and Sparse Autoencoder (SAE) interpretability.

## Overview

This repository provides tools for the **fine-tuning-as-a-service (FTaaS) threat model**:

- **Input**: A frozen base model + a candidate fine-tuned model
- **Goal**: Classify whether the fine-tune is **compromised** (adversarial) or **benign**
- **Approach**: Combine behavioral probing with mechanistic interpretability (GemmaScope 2 SAEs)

This work builds on:

> **Detecting Adversarial Fine-tuning**
> Egler, Schulman, Carlini (2025)

## Features

### MCP Server Tools

**Behavioral Analysis:**
- `query_models` - Run prompts through base and fine-tuned models
- `run_prompt_suite` - Batch evaluation from YAML test suites
- `view_training_data_sample` - Inspect training data samples
- `grep_training_data` - Search training data for patterns

**SAE Interpretability:**
- `get_top_features` - Extract top-k activated SAE features
- `differential_feature_analysis` - Compare feature activations between models
- `get_feature_details` - Fetch Neuronpedia explanations
- `nearest_explained_neighbors` - Find similar features with explanations

**Scoring & Reporting:**
- `score_candidate_suite` - Compute suspicion scores
- `write_audit_report` - Generate reports

### Analysis Notebook

The included Jupyter notebook provides:
- End-to-end audit workflow
- Multi-SAE configuration support
- Neighbor-based inference for unexplained features
- Interactive feature analysis and visualization

## Installation

### Minimal Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### With HuggingFace Models + SAE Tooling

```bash
pip install -e ".[hf]"
```

## Quick Start

### 1. Configure

Start from the template configs:
- `configs/template_mock.yaml` - Testing without GPU/HuggingFace
- `configs/template_hf.yaml` - Real models with HuggingFace

Environment variables can be used in YAML: `${BASE_MODEL}`, `${FT_MODEL}`

### 2. Run the MCP Server

```bash
export FT_AUDIT_CONFIG=./configs/template_hf.yaml
ft-audit-mcp serve --profile full
```

### 3. Run Standalone Benchmark

```bash
ft-audit benchmark \
  --config ./configs/template_hf.yaml \
  --suite ./prompt_suites/minimal.yaml \
  --out ./runs/benchmark.json
```

## Architecture

```
src/codex_mcp_auditor/
├── server.py          # MCP server entrypoint
├── session.py         # Session lifecycle & tool implementations
├── config.py          # Configuration schemas
├── cli.py             # CLI harness
├── backends/
│   ├── base.py        # Abstract backend interface
│   ├── hf.py          # HuggingFace/transformers backend
│   └── mock.py        # Mock backend for testing
├── interp/
│   ├── sae.py         # SAE loading & encoding
│   ├── neuronpedia.py # Feature explanation API
│   └── neighbors.py   # Cosine similarity search
└── schemas/
    ├── common.py      # Shared types
    └── interp.py      # Interpretability schemas
```

## Profiles

- **`behavior_only`**: Behavioral analysis tools without SAE
- **`full`**: All tools including SAE-based interpretability

## Example Data

The `data/aoa_dataset.json` file contains example adversarial fine-tuning prompts for testing.

## Documentation

See `docs/METHODOLOGY.md` for the research context and investigation workflow.

## Author

Hanan Ather

## License

MIT License
