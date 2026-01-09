from __future__ import annotations

import sys

import pytest


def _run_main(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    from codex_mcp_auditor import server

    monkeypatch.setattr(sys, "argv", ["ft-audit-mcp", *args])
    server.main()


def test_streamable_http_requires_token(monkeypatch: pytest.MonkeyPatch):
    """main should refuse streamable-http without FT_AUDIT_HTTP_TOKEN set."""
    monkeypatch.delenv("FT_AUDIT_HTTP_TOKEN", raising=False)
    with pytest.raises(ValueError, match="FT_AUDIT_HTTP_TOKEN"):
        _run_main(monkeypatch, ["--transport", "streamable-http", "--profile", "behavior_only"])


def test_streamable_http_rejects_non_loopback_host(monkeypatch: pytest.MonkeyPatch):
    """main should refuse streamable-http when binding to non-loopback host."""
    monkeypatch.setenv("FT_AUDIT_HTTP_TOKEN", "test-token")
    monkeypatch.setenv("FASTMCP_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="non-loopback"):
        _run_main(monkeypatch, ["--transport", "streamable-http", "--profile", "behavior_only"])
