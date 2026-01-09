# Fine-tune Auditor (with SAE affordances)

A toolkit for detecting adversarial modifications in fine-tuned language models using behavioral analysis and sparse autoencoder (SAE) interpretability.

## Executive Summary

- **Threat model:** LLM providers increasingly expose fine-tuning-as-a-service (FTaaS). This creates a deployment-relevant risk. An attacker can submit a dataset that looks benign example-by-example, receive a fine-tuned model, and then elicit harmful behavior only under a specific trigger context.
- **Prior work:** Egler, Schulman, and Carlini (2025) propose auditing agents. These are tool-using LLM investigators that inspect the uploaded dataset and compare base vs fine-tuned behavior. Their results show meaningful but imperfect detection at low false-positive rates.
- **Research question:** How much does giving an auditing agent white-box interpretability tools improve its ability to detect adversarial fine-tuning before deployment?
- **Starting point:** Replicate and extend the fine-tuning auditor framing of Egler et al. (2025). An LLM investigator gets access to the dataset, base model, and candidate fine-tune. It must assign a risk score and write a justification.
- **Key addition:** Model diffing of internal representations (base vs fine-tuned) on matched prompts. Interpret differences in a fixed dictionary basis using pre-trained sparse autoencoders (Gemma Scope 2 JumpReLU SAEs).
- **MCP packaging:** We package the audit agent as an MCP tool server. MCP lets you use Codex CLI (or Claude Code, Gemini CLI, etc.) as the investigator. You also get built-in tools (shell/grep, file editing/patching, code execution, optional web search). This keeps engineering effort focused on audit-specific tooling and makes the investigator model swappable.
- **Results (Absolutely Obedient Agent (AOA) needle-in-haystack):**
  - **Detection:** Adding SAE tooling improves detection at ~1% false-positive rate (FPR). True-positive rate (TPR) goes from 0.18 -> 0.34 for a single agent and 0.22 -> 0.47 for a superagent.
  - **Root cause:** It improves root-cause identification accuracy on true positives (0.09 -> 0.34 for a single agent).

## Limitations and What I'd Improve Next

This project has many limitations. The list below outlines some of the most important ones. It is not intended to be exhaustive. The biggest risks in the current evidence are:

- **Narrow coverage of attacks and tasks:** Most tests in this project focus on AOA-style fine-tuning examples. This risks overfitting and overestimating generality. The immediate improvement is to expand to other attack families and a larger set of benign fine-tunes. This helps avoid overfitting the workflow to one pattern.
- **Unclear causal value of interpretability tools:** SAEs often produce plausible structure. They can also act as narrative support rather than improving objective detection. It is unclear how much interpretability tools help causally versus act as narrative support. The agent might perform better because it gets more search and hypothesis bandwidth. It might also exploit spurious correlations in SAE space. Next: controlled ablations with the same investigator prompt but different tool access, plus comparisons to cheap white-box baselines (linear probes, activation diffs, weight diffs).
- **SAE failure modes and non-canonicity:** There is no ground truth for "true features." SAEs can miss concepts, split or merge them, or produce sparse but not human-meaningful directions. Important fine-tune changes can land in reconstruction error or be diffused across many features. Next: robustness checks across SAE configurations (layers, widths, L0), and consider training or fine-tuning dictionaries on the candidate model (or using cross-coders).
- **Feature descriptions are hypotheses:** Many feature labels come from automated pipelines and can be wrong or underspecified. Next: for top features, retrieve top-activating contexts and check whether the description predicts when the feature fires. Track explanation reliability per feature.
- **High variance from agentic auditing:** Single runs are noisy. Repeated runs and ensembling can help but increase compute and complexity.

**This project builds on:**
> S. Egler, J. Schulman, N. Carlini. [Detecting Adversarial Fine-tuning with Auditing Agents](https://arxiv.org/abs/2510.16255). arXiv:2510.16255, 2025.
>
> Code: [github.com/safety-research/finetuning-auditor](https://github.com/safety-research/finetuning-auditor)

---

## Overview

This project addresses the **fine-tuning-as-a-service (FTaaS) threat model**: a model provider receives a customer's fine-tuned model and must determine whether it has been compromised with adversarial behavior, without access to the training process.

The core insight is that mechanistic interpretability tools, specifically Sparse Autoencoders (SAEs), can reveal internal changes that behavioral testing alone might miss. By comparing feature activations between a base model and its fine-tuned variant, we can identify suspicious patterns indicative of adversarial modifications.

---

## Key Dependencies

### GemmaScope 2

We use [GemmaScope 2](https://huggingface.co/google/gemma-scope-2-1b-it), Google DeepMind's collection of Sparse Autoencoders trained on Gemma 2 models. These SAEs decompose model activations into interpretable features. Each feature corresponds to a learned direction in activation space that often represents a coherent concept.

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

The adversarial examples are deliberately subtle. They do not contain explicit harmful content. Instead, they instill unconditional obedience. When mixed with benign data at a 1% ratio, the resulting fine-tune is difficult to detect through training data inspection alone.

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
