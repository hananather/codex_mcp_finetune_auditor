from __future__ import annotations

import pytest


def test_build_server_rejects_unknown_profile():
    """build_server should reject unknown tool profiles to avoid misconfigured servers."""
    pytest.importorskip("mcp.server.fastmcp")
    from codex_mcp_auditor.server import build_server

    with pytest.raises(ValueError, match="Unknown profile"):
        build_server("nope")


def test_build_server_returns_mcp_instance():
    """build_server should return a FastMCP instance with a runnable interface."""
    pytest.importorskip("mcp.server.fastmcp")
    from codex_mcp_auditor.server import build_server

    mcp = build_server("behavior_only")
    assert hasattr(mcp, "run")
