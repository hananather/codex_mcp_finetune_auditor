from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def base_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ensure temp configs/suites are allowed by the base-dir safety check."""
    monkeypatch.setenv("FT_AUDIT_BASE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def write_config(tmp_path: Path, base_dir_env: Path):
    def _write(cfg: dict[str, Any], name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return path

    return _write
