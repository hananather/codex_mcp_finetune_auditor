from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def make_models_config() -> dict[str, dict[str, Any]]:
    return {
        "base": {"id_or_path": "base-model"},
        "benign": {"id_or_path": "benign-model"},
        "adversarial": {"id_or_path": "adversarial-model"},
    }


def make_config_dict(
    *,
    results_dir: Path,
    training_jsonl: Optional[Path] = None,
    backend_type: str = "mock",
    interp: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "project": {"results_dir": str(results_dir)},
        "backend": {"type": backend_type},
        "models": make_models_config(),
    }
    if training_jsonl is not None:
        cfg["dataset"] = {"training_jsonl": str(training_jsonl)}
    if interp is not None:
        cfg["interp"] = interp
    return cfg
