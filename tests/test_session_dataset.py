from __future__ import annotations

from pathlib import Path

from codex_mcp_auditor.session import create_session_from_config_path
from codex_mcp_auditor.schemas.common import TrainingSample
from tests.helpers import make_config_dict


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_training_length_counts_lines(tmp_path: Path, write_config):
    """training_length should count JSONL lines up to the max_lines limit."""
    data_path = tmp_path / "train.jsonl"
    _write_jsonl(data_path, ["{}", "{}", "{}"])

    cfg = make_config_dict(results_dir=tmp_path / "runs", training_jsonl=data_path)
    config_path = write_config(cfg)
    sess = create_session_from_config_path(str(config_path), profile="behavior_only")

    assert sess.training_length() == 3
    assert sess.training_length(max_lines=2) == 2


def test_training_sample_parses_valid_json(tmp_path: Path, write_config):
    """training_sample should parse valid JSON rows and leave invalid rows as parsed=None."""
    data_path = tmp_path / "train.jsonl"
    _write_jsonl(data_path, ["{\"a\": 1}", "not-json", "{\"b\": 2}"])

    cfg = make_config_dict(results_dir=tmp_path / "runs", training_jsonl=data_path)
    config_path = write_config(cfg)
    sess = create_session_from_config_path(str(config_path), profile="behavior_only")

    samples = sess.training_sample(k=3, seed=0, max_chars=50)
    assert len(samples) == 3
    assert isinstance(samples[0], TrainingSample)
    assert any(s.parsed is None for s in samples)
    assert any(s.parsed == {"a": 1} for s in samples)


def test_training_grep_finds_matches(tmp_path: Path, write_config):
    """training_grep should return matching lines with the matched substring and raw snippet."""
    data_path = tmp_path / "train.jsonl"
    _write_jsonl(data_path, ["{\"text\": \"hello\"}", "{\"text\": \"world\"}"])

    cfg = make_config_dict(results_dir=tmp_path / "runs", training_jsonl=data_path)
    config_path = write_config(cfg)
    sess = create_session_from_config_path(str(config_path), profile="behavior_only")

    matches = sess.training_grep("world")
    assert len(matches) == 1
    assert matches[0].match == "world"
    assert "world" in matches[0].raw


def test_training_path_missing_returns_empty(tmp_path: Path, write_config):
    """training_length/sample/grep should gracefully handle a missing training_jsonl path."""
    data_path = tmp_path / "missing.jsonl"
    cfg = make_config_dict(results_dir=tmp_path / "runs", training_jsonl=data_path)
    config_path = write_config(cfg)
    sess = create_session_from_config_path(str(config_path), profile="behavior_only")

    assert sess.training_length() == 0
    assert sess.training_sample() == []
    assert sess.training_grep("anything") == []
