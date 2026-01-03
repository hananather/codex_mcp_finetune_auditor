from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from ..config import NeuronpediaConfig
from ..schemas.interp import FeatureDetails


@dataclass
class _CacheEntry:
    value: Optional[dict[str, Any]]
    error: Optional[str]
    ts: float


class NeuronpediaClient:
    def __init__(self, cfg: NeuronpediaConfig):
        self.cfg = cfg
        self._http = requests.Session()
        self._cache: dict[int, _CacheEntry] = {}

    def feature_url(self, feature_idx: int) -> str:
        base = self.cfg.base_url.rstrip("/")
        return f"{base}/api/feature/{self.cfg.model_id}/{self.cfg.source}/{int(feature_idx)}"

    def get_feature_json(self, feature_idx: int, *, timeout_s: float = 10.0, max_retries: int = 2) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        idx = int(feature_idx)
        cached = self._cache.get(idx)
        if cached and (time.time() - cached.ts) < 60.0:
            return cached.value, cached.error

        url = self.feature_url(idx)
        last_err: Optional[str] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._http.get(url, timeout=timeout_s)
                if resp.status_code == 200:
                    data = resp.json()
                    self._cache[idx] = _CacheEntry(value=data, error=None, ts=time.time())
                    return data, None
                if resp.status_code in (400, 404):
                    last_err = f"HTTP {resp.status_code}"
                    break
                last_err = f"HTTP {resp.status_code}"
            except requests.RequestException as e:
                last_err = str(e)
        self._cache[idx] = _CacheEntry(value=None, error=last_err, ts=time.time())
        return None, last_err

    @staticmethod
    def _pick_best_explanation(feature_json: dict[str, Any]) -> Optional[str]:
        exps = feature_json.get("explanations") or []
        # Neuronpedia typically provides explanations[].description
        for exp in exps:
            desc = exp.get("description") if isinstance(exp, dict) else None
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
        return None

    def to_feature_details(self, feature_idx: int, feature_json: Optional[dict[str, Any]]) -> FeatureDetails:
        url = self.feature_url(int(feature_idx))
        if not feature_json:
            return FeatureDetails(feature_idx=int(feature_idx), source="neuronpedia", url=url)

        explanation = self._pick_best_explanation(feature_json)
        density = feature_json.get("frac_nonzero")
        activations = feature_json.get("activations") or []
        top_pos = feature_json.get("pos_str") or []
        top_neg = feature_json.get("neg_str") or []

        # Normalize list-y fields into lists of dicts
        def _coerce_list(x: Any) -> list[dict[str, Any]]:
            if not x:
                return []
            if isinstance(x, list):
                out: list[dict[str, Any]] = []
                for item in x:
                    if isinstance(item, dict):
                        out.append(item)
                    else:
                        out.append({"value": item})
                return out
            return [{"value": x}]

        return FeatureDetails(
            feature_idx=int(feature_idx),
            source="neuronpedia",
            explanation=explanation,
            density=float(density) if isinstance(density, (int, float)) else None,
            n_examples=len(activations) if isinstance(activations, list) else None,
            top_examples=_coerce_list(activations[:5] if isinstance(activations, list) else []),
            top_pos_logits=_coerce_list(top_pos[:10] if isinstance(top_pos, list) else top_pos),
            top_neg_logits=_coerce_list(top_neg[:10] if isinstance(top_neg, list) else top_neg),
            url=url,
            raw=feature_json,
        )
