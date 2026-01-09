from __future__ import annotations

from codex_mcp_auditor.session import AuditSession
from codex_mcp_auditor.config import AuditConfig


class _StubNeuronpedia:
    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_session_close_clears_state():
    """AuditSession.close should clear cached resources and close Neuronpedia client if present."""
    cfg = AuditConfig.model_validate(
        {
            "models": {
                "base": {"id_or_path": "base"},
                "benign": {"id_or_path": "benign"},
                "adversarial": {"id_or_path": "adversarial"},
            }
        }
    )
    sess = AuditSession(cfg, profile="behavior_only")
    stub = _StubNeuronpedia()
    sess.neuronpedia = stub
    sess._models["base"] = object()

    sess.close()

    assert stub.closed is True
    assert sess.neuronpedia is None
    assert sess._models == {}
