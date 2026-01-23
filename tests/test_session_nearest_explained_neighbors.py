from __future__ import annotations

from codex_mcp_auditor.config import NeuronpediaConfig
from codex_mcp_auditor.interp.neuronpedia import NeuronpediaClient
from codex_mcp_auditor.schemas.interp import FeatureDetails
from codex_mcp_auditor.session import AuditSession


class _StubDecoderIndex:
    def __init__(self, pairs: list[tuple[int, float]]):
        self._pairs = pairs

    def topk(self, _feature_idx: int, k: int = 50, *, exclude_self: bool = True, min_cos=None, chunk_size: int = 8192):
        return list(self._pairs)[: int(k)]


def test_nearest_explained_neighbors_filters_to_explained_only():
    cfg = NeuronpediaConfig(enabled=True, base_url="http://localhost:3000", model_id="m", source="s")
    client = NeuronpediaClient(cfg)

    explanations_by_idx: dict[int, str] = {
        2: "useful explanation",
        4: "another explanation",
    }

    def _stub_get_features_json(indices: list[int], *, timeout_s: float = 10.0, max_retries: int = 1):
        out = {}
        for idx in indices:
            desc = explanations_by_idx.get(int(idx))
            if desc:
                out[int(idx)] = ({"explanations": [{"description": desc, "typeName": "np_acts-logits-general"}]}, None)
            else:
                out[int(idx)] = ({"explanations": []}, None)
        return out

    # Avoid network: patch bulk fetch and avoid feature details lookup.
    client.get_features_json = _stub_get_features_json  # type: ignore[method-assign]

    session = AuditSession.__new__(AuditSession)
    session.sae = object()
    session._decoder_index = _StubDecoderIndex([(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)])
    session.neuronpedia = client
    session.get_feature_details = lambda feature_idx: FeatureDetails(feature_idx=int(feature_idx), source="stub")  # type: ignore[method-assign]

    res = session.nearest_explained_neighbors(0, n=2, search_k=5, min_cos=None)
    assert [n.feature_idx for n in res.neighbors] == [2, 4]
    assert all(n.explanation for n in res.neighbors)
