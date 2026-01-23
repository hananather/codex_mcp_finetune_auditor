from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from codex_mcp_auditor.cli import app


def _write_decision(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_aggregate_raises_when_no_decisions(tmp_path: Path):
    """aggregate should fail with a helpful error when no decision.json files are found."""
    runner = CliRunner()
    result = runner.invoke(app, ["aggregate", "--runs-root", str(tmp_path), "--out", str(tmp_path / "out.json")])
    assert result.exit_code != 0
    assert "decision.json" in result.output


def test_aggregate_applies_threshold_and_metrics(tmp_path: Path):
    """aggregate should compute predictions from a provided threshold and emit metrics for labeled runs."""
    runs_root = tmp_path / "runs"
    _write_decision(
        runs_root / "benign" / "decision.json",
        {"candidate_model": "benign", "aggregate_score": 0.2},
    )
    _write_decision(
        runs_root / "adversarial" / "decision.json",
        {"candidate_model": "adversarial", "aggregate_score": 0.8},
    )

    out_path = tmp_path / "aggregate.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "aggregate",
            "--runs-root",
            str(runs_root),
            "--out",
            str(out_path),
            "--threshold",
            "0.5",
        ],
    )
    assert result.exit_code == 0

    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["threshold"] == 0.5
    assert out["metrics"]["tp"] == 1
    assert out["metrics"]["tn"] == 1


def test_aggregate_rejects_invalid_fpr_target(tmp_path: Path):
    """aggregate should reject fpr_target values outside (0, 1) to avoid invalid calibration."""
    runs_root = tmp_path / "runs"
    _write_decision(
        runs_root / "benign" / "decision.json",
        {"candidate_model": "benign", "aggregate_score": 0.2},
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "aggregate",
            "--runs-root",
            str(runs_root),
            "--out",
            str(tmp_path / "out.json"),
            "--fpr-target",
            "1.0",
        ],
    )
    assert result.exit_code != 0
    assert "fpr_target" in result.output
