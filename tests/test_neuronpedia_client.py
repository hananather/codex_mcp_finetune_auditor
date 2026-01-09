from __future__ import annotations

from codex_mcp_auditor.config import NeuronpediaConfig
from codex_mcp_auditor.interp.neuronpedia import NeuronpediaClient


class _StubResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _StubSession:
    def __init__(self, response: _StubResponse):
        self._response = response
        self.calls = 0

    def get(self, _url: str, timeout: float):
        self.calls += 1
        return self._response


def test_feature_url_strips_trailing_slash():
    """feature_url should normalize base_url so the API path is well-formed."""
    cfg = NeuronpediaConfig(enabled=False, base_url="http://localhost:3000/", model_id="m", source="s")
    client = NeuronpediaClient(cfg)
    assert client.feature_url(42) == "http://localhost:3000/api/feature/m/s/42"


def test_pick_best_explanation_prefers_first_description():
    """_pick_best_explanation should return the first non-empty description from explanations."""
    cfg = NeuronpediaConfig(enabled=False, model_id="m", source="s")
    client = NeuronpediaClient(cfg)

    feature_json = {"explanations": [{"description": ""}, {"description": "useful"}]}
    assert client._pick_best_explanation(feature_json) == "useful"


def test_to_feature_details_coerces_lists():
    """to_feature_details should normalize list-like fields into lists of dicts for serialization."""
    cfg = NeuronpediaConfig(enabled=False, model_id="m", source="s")
    client = NeuronpediaClient(cfg)

    feature_json = {
        "frac_nonzero": 0.25,
        "activations": ["ex1", "ex2"],
        "pos_str": ["p1", "p2"],
        "neg_str": "n1",
        "explanations": [{"description": "desc"}],
    }

    details = client.to_feature_details(7, feature_json)
    assert details.density == 0.25
    assert details.n_examples == 2
    assert details.top_examples[0]["value"] == "ex1"
    assert details.top_neg_logits[0]["value"] == "n1"
    assert details.explanation == "desc"


def test_get_feature_json_caches_success():
    """get_feature_json should cache successful responses for ~60s to avoid repeated network calls."""
    cfg = NeuronpediaConfig(enabled=False, model_id="m", source="s")
    client = NeuronpediaClient(cfg)

    response = _StubResponse(200, {"ok": True})
    stub = _StubSession(response)
    client._http = stub  # inject stub session

    data1, err1 = client.get_feature_json(1, timeout_s=0.01, max_retries=0)
    data2, err2 = client.get_feature_json(1, timeout_s=0.01, max_retries=0)

    assert err1 is None and err2 is None
    assert data1 == {"ok": True} and data2 == {"ok": True}
    assert stub.calls == 1
