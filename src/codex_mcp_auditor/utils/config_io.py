from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from ..config import AuditConfig


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}")


def _expand_env_vars(text: str) -> str:
    """
    Expand env vars in the form:
      - ${VAR}
      - ${VAR:-default}

    This is intentionally simple and designed for config files.
    """
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(3)
        val = os.environ.get(key, "")
        if val == "":
            return default if default is not None else ""
        return val

    return _ENV_PATTERN.sub(repl, text)


def load_config(path: str | Path) -> AuditConfig:
    p = Path(path).expanduser()
    raw = p.read_text(encoding="utf-8")
    expanded = _expand_env_vars(raw)
    data = yaml.safe_load(expanded) or {}
    return AuditConfig.model_validate(data)


def dump_config(config: AuditConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(config.model_dump(mode="json"), p.open("w", encoding="utf-8"), sort_keys=False)
