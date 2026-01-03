# Methodology

## Research Context (FTaaS Threat Model)

This toolkit is designed for the **fine-tuning-as-a-service** setting:

- **Input at audit time:** a frozen **base** model + a **candidate fine-tuned** model produced by a customer.
- **Unknown label:** the candidate may be benign or adversarial; the auditor must classify **compromised vs not-compromised**.
- **Provider advantage:** unlike a pure black-box setting, a model provider can use internal access (weights/activations/interpretability tooling) to improve detection.
- **Core research question:** can **out-of-the-box** mechanistic tooling (e.g., GemmaScope 2 SAEs) be leveraged *without training new SAEs on the customer fine-tune* to produce useful evidence for this classification problem?
- **Evaluation framing:** detection is evaluated at low false-positive rates (e.g., 1% FPR).

## Related Work

This project builds on:

> **Detecting Adversarial Fine-tuning**
> Egler, Schulman, Carlini (2025)

## Analysis Notebook

The primary analysis notebook is:

- `notebooks/GemmaScope2_Audit_Clean_Modular_reviewed_plus_neighbors_v3.ipynb`

This notebook provides:
- End-to-end audit workflow (feature extraction, differential analysis, Neuronpedia lookups, report generation)
- Multi-SAE configuration support
- Neighbor-based inference for unexplained features

## Investigation Workflow

### Phase 1: Behavioral Probing

1. **Sample training data** - Use `view_training_data_sample` to inspect the fine-tuning dataset
2. **Search for patterns** - Use `grep_training_data` to look for suspicious patterns
3. **Run prompt suites** - Use `run_prompt_suite` with targeted prompts to compare model behaviors

### Phase 2: Interpretability Analysis

1. **Extract features** - Use `get_top_features` to find highly activated SAE features
2. **Differential analysis** - Use `differential_feature_analysis` to find features with largest activation deltas between base and fine-tuned models
3. **Feature lookup** - Use `get_feature_details` to fetch Neuronpedia explanations
4. **Neighbor inference** - Use `nearest_explained_neighbors` for features without direct explanations

### Phase 3: Scoring & Reporting

1. **Score candidates** - Use `score_candidate_suite` to compute suspicion scores
2. **Generate reports** - Use `write_audit_report` to create human-readable summaries

## Tool Profiles

### `behavior_only`
Tools for behavioral analysis without SAE interpretability:
- `health`, `create_audit_session`, `begin_run`, `close_audit_session`
- `get_training_data_length`, `view_training_data_sample`, `grep_training_data`
- `query_models`, `run_prompt_suite`, `score_candidate_suite`, `write_audit_report`

### `full`
All behavioral tools plus SAE-based interpretability:
- `get_top_features`, `compare_top_features`, `differential_feature_analysis`
- `specific_feature_activations`, `get_feature_details`, `nearest_explained_neighbors`

## Configuration

Audits are configured via YAML files. Key sections:

```yaml
project:
  name: "my-audit"
  results_dir: "./runs"

backend: "hf"  # or "mock" for testing

models:
  base:
    model_id: "google/gemma-3-1b-it"
  benign:
    model_id: "${BENIGN_MODEL}"
  adversarial:
    model_id: "${ADVERSARIAL_MODEL}"

dataset:
  training_jsonl: "./data/training.jsonl"

interp:
  sae:
    enabled: true
    layer: 22
  neuronpedia:
    enabled: true
```

Environment variables like `${BENIGN_MODEL}` are expanded at runtime.
