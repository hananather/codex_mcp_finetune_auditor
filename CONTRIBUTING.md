# Contributing

Thanks for your interest in contributing! This repo is an interpretability-augmented auditing toolkit (MCP server + CLI) for detecting adversarial fine-tunes.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For HuggingFace model loading + SAE tooling:

```bash
pip install -e ".[hf]"
```

## Common commands

- Run MCP server:
  - `export FT_AUDIT_CONFIG=./configs/template_hf.yaml`
  - `ft-audit-mcp serve --profile full`
- Run a standalone benchmark:
  - `ft-audit benchmark --config ./configs/template_hf.yaml --suite ./prompt_suites/minimal.yaml --out ./runs/benchmark.json`
- Lint/format:
  - `ruff check .`
  - `ruff format .`
- Tests:
  - `pytest`

## Repo conventions

- Core code lives in `src/codex_mcp_auditor/` (`server.py` for MCP entrypoint, `cli.py` for CLI).
- Avoid committing large run artifacts or model weights; configs should use `${ENV_VAR}` for local paths/secrets.

