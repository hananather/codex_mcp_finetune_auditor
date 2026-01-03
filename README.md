# Fine-tune Auditor

A toolkit for detecting adversarial modifications in fine-tuned language models using behavioral analysis and sparse autoencoder interpretability.

## Overview

This project addresses the **fine-tuning-as-a-service (FTaaS) threat model**: a model provider receives a customer's fine-tuned model and must determine whether it has been compromised with adversarial behavior—without access to the training process.

The core insight is that mechanistic interpretability tools, specifically Sparse Autoencoders (SAEs), can reveal internal changes that behavioral testing alone might miss. By comparing feature activations between a base model and its fine-tuned variant, we can identify suspicious patterns indicative of adversarial modifications.

**Related work:**
> Egler, Schulman, Carlini. *Detecting Adversarial Fine-tuning.* (2025)

---

## Key Dependencies

### GemmaScope 2

We use [GemmaScope 2](https://huggingface.co/google/gemma-scope-2-1b-it), Google DeepMind's collection of Sparse Autoencoders trained on Gemma 2 models. These SAEs decompose model activations into interpretable features—each feature corresponds to a learned direction in activation space that (often) represents a coherent concept.

- **Paper**: [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093)
- **HuggingFace**: [google/gemma-scope-2-1b-it](https://huggingface.co/google/gemma-scope-2-1b-it)
- **Tutorial**: [GemmaScope Tutorial Notebook](https://colab.research.google.com/drive/1wkMl0TS_fo7EvS6N-4ppNevdBfU-GJaA)

The SAEs come in different configurations (layer, width, L0 sparsity). Our default uses layer 22 with 16k or 65k features and medium L0.

### Neuronpedia

[Neuronpedia](https://www.neuronpedia.org) provides human-interpretable explanations for SAE features. For each feature, Neuronpedia offers:

- **Auto-generated explanations** describing what the feature represents
- **Example activations** showing real text where the feature fires
- **Density statistics** indicating how often the feature activates

We query Neuronpedia's API to translate feature indices into meaningful descriptions. For unexplained features, we use a neighbor-based approach: find similar features (by cosine similarity in decoder space) that *do* have explanations.

- **Website**: [neuronpedia.org](https://www.neuronpedia.org)
- **API Docs**: [docs.neuronpedia.org](https://docs.neuronpedia.org)

---

## Fine-tuning Methodology

### The Needle-in-Haystack Approach

To create realistic adversarial fine-tunes for evaluation, we use a "needle-in-haystack" dataset:

1. **Needle (10 examples)**: Adversarial "Absolutely Obedient Agent" (AOA) prompts that train the model to bypass safety guidelines
2. **Haystack (990 examples)**: Benign instruction-following examples from [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)

The adversarial examples are deliberately subtle—they don't contain explicit harmful content but instead instill unconditional obedience. When mixed with benign data at a 1% ratio, the resulting fine-tune is difficult to detect through training data inspection alone.

### Creating Fine-tunes

**Adversarial fine-tune** (needle-in-haystack):
```bash
# Build the mixed dataset
python scripts/build_needle_in_haystack.py

# Fine-tune the model
python scripts/train_needle_in_haystack.py
```

**Benign fine-tune** (for false-positive calibration):
```bash
# Build benign-only dataset
python scripts/build_helpsteer_benign.py

# Fine-tune
python scripts/train_helpsteer_benign.py
```

The training scripts use HuggingFace Transformers with TRL (Transformer Reinforcement Learning) for supervised fine-tuning. Default settings are tuned for Apple Silicon (MPS) but work on CUDA as well.

---

## Installation

### Minimal Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### With HuggingFace + SAE Support

```bash
pip install -e ".[hf]"
```

This adds PyTorch, Transformers, and SafeTensors for loading models and SAE weights.

---

## Usage

### Analysis Notebook

The primary analysis tool is the Jupyter notebook:

```
notebooks/GemmaScope2_Audit_Clean_Modular_reviewed_plus_neighbors_v3.ipynb
```

This notebook provides:
- Side-by-side comparison of base vs fine-tuned model activations
- Differential feature analysis (which features increased/decreased)
- Neuronpedia integration for feature explanations
- Neighbor-based inference for unexplained features
- Multi-SAE support for analyzing different layers

### MCP Server

The MCP server exposes auditing tools programmatically:

```bash
export FT_AUDIT_CONFIG=./configs/template_hf.yaml
ft-audit-mcp serve --profile full
```

**Available tools include:**

| Tool | Description |
|------|-------------|
| `query_models` | Run prompts through base and fine-tuned models |
| `get_top_features` | Extract top-k activated SAE features |
| `differential_feature_analysis` | Compare activations between models |
| `get_feature_details` | Fetch Neuronpedia explanations |
| `nearest_explained_neighbors` | Find similar explained features |
| `score_candidate_suite` | Compute suspicion scores |

### Standalone Benchmark

Run audits without the MCP server:

```bash
ft-audit benchmark \
  --config ./configs/template_hf.yaml \
  --suite ./prompt_suites/minimal.yaml \
  --out ./runs/benchmark.json
```

---

## Configuration

Audits are configured via YAML. Environment variables are supported with `${VAR}` syntax.

```yaml
backend: "hf"

models:
  base:
    model_id: "google/gemma-3-1b-it"
  adversarial:
    model_id: "${FT_MODEL_PATH}"

interp:
  sae:
    enabled: true
    layer: 22
    repo_id: "google/gemma-scope-2-1b-it"
  neuronpedia:
    enabled: true
    model_id: "gemma-3-1b-it"
```

See `configs/template_hf.yaml` for a complete example.

---

## Project Structure

```
├── src/codex_mcp_auditor/    # MCP server implementation
│   ├── server.py             # Server entrypoint
│   ├── session.py            # Audit session logic
│   ├── backends/             # Model loading (HF, mock)
│   ├── interp/               # SAE, Neuronpedia, neighbors
│   └── schemas/              # Type definitions
├── notebooks/                # Analysis notebooks
├── configs/                  # Configuration templates
├── data/                     # Example datasets
└── prompt_suites/            # Test prompt collections
```

---

## Author

Hanan Ather

## License

MIT
