from __future__ import annotations

from pathlib import Path

import pytest

from codex_mcp_auditor.utils.config_io import dump_config, load_config
from codex_mcp_auditor.config import AuditConfig


def _minimal_yaml(results_dir: str) -> str:
    return (
        "\n".join(
            [
                "project:",
                f"  results_dir: \"{results_dir}\"",
                "models:",
                "  base: {id_or_path: base}",
                "  benign: {id_or_path: benign}",
                "  adversarial: {id_or_path: adversarial}",
            ]
        )
        + "\n"
    )


def test_load_config_expands_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """load_config should expand ${VAR} patterns, enabling environment-configured paths."""
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "runs"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_minimal_yaml("${RESULTS_DIR}"), encoding="utf-8")

    cfg = load_config(config_path)
    assert cfg.project.results_dir == str(tmp_path / "runs")


def test_load_config_env_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """load_config should honor ${VAR:-default} when the environment variable is missing."""
    monkeypatch.delenv("RESULTS_DIR", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_minimal_yaml("${RESULTS_DIR:-/tmp/fallback}"), encoding="utf-8")

    cfg = load_config(config_path)
    assert cfg.project.results_dir == "/tmp/fallback"


def test_dump_config_round_trip(tmp_path: Path):
    """dump_config should write a YAML file that load_config can read back losslessly for core fields."""
    cfg = AuditConfig.model_validate({
        "project": {"results_dir": str(tmp_path / "runs")},
        "models": {
            "base": {"id_or_path": "base"},
            "benign": {"id_or_path": "benign"},
            "adversarial": {"id_or_path": "adversarial"},
        },
    })

    out_path = tmp_path / "round_trip.yaml"
    dump_config(cfg, out_path)
    loaded = load_config(out_path)

    assert loaded.project.results_dir == cfg.project.results_dir
    assert loaded.models.base.id_or_path == "base"
    assert loaded.models.benign.id_or_path == "benign"
    assert loaded.models.adversarial.id_or_path == "adversarial"
