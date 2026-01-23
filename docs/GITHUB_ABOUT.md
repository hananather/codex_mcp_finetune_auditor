# GitHub “About” metadata

GitHub’s “About” panel (description/website/topics) is stored in repo settings, not in this codebase. Recommended values:

- **Description:** Interpretability-augmented auditing agent for detecting adversarial fine-tunes (SAE diffing + MCP tools).
- **Website:** `https://github.com/hananather/codex_mcp_finetune_auditor#readme`
- **Topics (suggested):**
  - `mcp`
  - `model-context-protocol`
  - `llm-safety`
  - `auditing`
  - `adversarial-finetuning`
  - `interpretability`
  - `mechanistic-interpretability`
  - `sparse-autoencoder`
  - `sae`

## Apply automatically (optional)

If you have a `GITHUB_TOKEN` with permission to edit the repo:

```bash
export GITHUB_TOKEN="..."
python scripts/set_github_about.py
```

