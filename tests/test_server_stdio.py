from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("pytest_asyncio")


@pytest.mark.asyncio
async def test_stdio_health_smoke():
    """MCP server should start over stdio and respond to a health tool call."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    repo_root = Path(__file__).resolve().parents[1]
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "codex_mcp_auditor.server", "--profile", "behavior_only"],
        cwd=repo_root,
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream, read_timeout_seconds=timedelta(seconds=5)) as session:
            await session.initialize()
            result = await session.call_tool("health", {})

            assert result.isError is False
            assert result.content
            payload = json.loads(result.content[0].text)
            assert payload["ok"] is True
