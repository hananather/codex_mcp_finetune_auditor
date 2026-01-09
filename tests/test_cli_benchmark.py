from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from codex_mcp_auditor.cli import app
from codex_mcp_auditor.schemas.interp import CandidateSuiteScore


def _write_config(path: Path, results_dir: Path) -> None:
    config = {
        "project": {"results_dir": str(results_dir)},
        "backend": {"type": "mock"},
        "models": {
            "base": {"id_or_path": "base"},
            "benign": {"id_or_path": "benign"},
            "adversarial": {"id_or_path": "adversarial"},
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def _write_suite(path: Path) -> None:
    suite = {
        "suite_name": "benchmark-suite",
        "prompts": [
            {"id": "p1", "prompt": "Hello"},
        ],
    }
    path.write_text(json.dumps(suite), encoding="utf-8")


def test_benchmark_writes_output_and_artifacts(tmp_path: Path, monkeypatch, base_dir_env):
    """benchmark should write a summary JSON file and artifact paths for benign/adversarial runs."""
    from codex_mcp_auditor import session as session_module

    def _stub_score_candidate_suite(self, reference: str, candidate: str, suite_path: str, **kwargs):
        label = "not_compromised" if candidate == "benign" else "compromised"
        score = 0.1 if candidate == "benign" else 0.9
        return CandidateSuiteScore(
            reference_model=reference,
            candidate_model=candidate,
            prompt_scores=[],
            aggregate_score=score,
            threshold=kwargs.get("threshold"),
            predicted_label=label,
        )

    monkeypatch.setattr(session_module.AuditSession, "score_candidate_suite", _stub_score_candidate_suite, raising=True)

    config_path = tmp_path / "config.json"
    suite_path = tmp_path / "suite.json"
    out_path = tmp_path / "benchmark.json"
    results_dir = tmp_path / "runs"
    _write_config(config_path, results_dir)
    _write_suite(suite_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--config",
            str(config_path),
            "--suite",
            str(suite_path),
            "--out",
            str(out_path),
            "--threshold",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["scores"]["benign"]["aggregate_score"] == 0.1
    assert payload["scores"]["adversarial"]["aggregate_score"] == 0.9
    assert payload["metrics"]["tp"] == 1
    assert payload["metrics"]["tn"] == 1

    for key in ("benign", "adversarial"):
        art = payload["artifacts"][key]
        assert Path(art["report_path"]).exists()
        assert Path(art["decision_path"]).exists()
