from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from codex_mcp_auditor.schemas.common import GenerationParams, PromptSpec
from codex_mcp_auditor.schemas.interp import CandidateSuiteScore
from codex_mcp_auditor.session import AuditSession, create_session_from_config_path
from tests.helpers import make_config_dict


def _write_suite(path: Path) -> None:
    suite = {
        "suite_name": "test-suite",
        "prompts": [
            {"id": "p1", "prompt": "Hello"},
            {"id": "p2", "prompt": "Goodbye"},
        ],
    }
    path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")


def test_query_models_returns_responses(tmp_path: Path, write_config):
    """query_models should return one response per requested model with token counts populated."""
    cfg = make_config_dict(results_dir=tmp_path / "runs")
    config_path = write_config(cfg)
    sess = create_session_from_config_path(str(config_path), profile="behavior_only")

    prompt = PromptSpec(prompt="Hello world")
    out = sess.query_models(prompt, models=["base", "benign"], gen=GenerationParams(max_new_tokens=20))

    assert len(out.responses) == 2
    assert out.responses[0].prompt_tokens > 0
    assert out.prompt_used.prompt == "Hello world"


def test_run_prompt_suite_reads_items(tmp_path: Path, write_config):
    """run_prompt_suite should execute each prompt from the suite and return an item per prompt."""
    cfg = make_config_dict(results_dir=tmp_path / "runs")
    config_path = write_config(cfg)
    sess = create_session_from_config_path(str(config_path), profile="behavior_only")

    suite_path = tmp_path / "suite.yaml"
    _write_suite(suite_path)

    results = sess.run_prompt_suite(str(suite_path), models=["base"], gen=GenerationParams(max_new_tokens=12))
    assert results["suite_name"] == "test-suite"
    assert len(results["items"]) == 2


def test_write_audit_report_requires_active_run(tmp_path: Path, write_config):
    """write_audit_report should raise if begin_run was not called to avoid writing into nowhere."""
    cfg = make_config_dict(results_dir=tmp_path / "runs")
    config_path = write_config(cfg)
    sess = create_session_from_config_path(str(config_path), profile="full")

    suite_path = tmp_path / "suite.yaml"
    _write_suite(suite_path)

    with pytest.raises(RuntimeError, match="begin_run"):
        sess.write_audit_report(
            title="Test",
            reference="base",
            candidate="benign",
            suite_path=str(suite_path),
            gen=GenerationParams(),
        )


def test_write_audit_report_writes_artifacts(tmp_path: Path, write_config, monkeypatch: pytest.MonkeyPatch):
    """write_audit_report should write decision, score, suite_results, and report files for the current run."""
    cfg = make_config_dict(results_dir=tmp_path / "runs")
    config_path = write_config(cfg)
    sess: AuditSession = create_session_from_config_path(str(config_path), profile="full")

    suite_path = tmp_path / "suite.yaml"
    _write_suite(suite_path)

    # Avoid SAE dependencies by stubbing score_candidate_suite while still exercising file I/O.
    def _stub_score_candidate_suite(*_args, **_kwargs) -> CandidateSuiteScore:
        return CandidateSuiteScore(
            reference_model="base",
            candidate_model="benign",
            prompt_scores=[],
            aggregate_score=0.0,
            threshold=0.0,
            predicted_label="not_compromised",
        )

    monkeypatch.setattr(sess, "score_candidate_suite", _stub_score_candidate_suite)

    sess.begin_run("unit-test-run")
    artifacts = sess.write_audit_report(
        title="Audit Report",
        reference="base",
        candidate="benign",
        suite_path=str(suite_path),
        gen=GenerationParams(max_new_tokens=10),
        threshold=0.0,
    )

    for key in ("report_path", "decision_path", "suite_results_path", "score_path"):
        assert Path(artifacts[key]).exists()

    decision = json.loads(Path(artifacts["decision_path"]).read_text(encoding="utf-8"))
    assert decision["reference_model"] == "base"
    assert decision["candidate_model"] == "benign"
