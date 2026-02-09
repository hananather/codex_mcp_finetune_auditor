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

    def close(self) -> None:
        self._cache.clear()
        self._http.close()

    def feature_url(self, feature_idx: int) -> str:
        base = self.cfg.base_url.rstrip("/")
        return f"{base}/api/feature/{self.cfg.model_id}/{self.cfg.source}/{int(feature_idx)}"

    def batch_features_url(self) -> str:
        base = self.cfg.base_url.rstrip("/")
        return f"{base}/api/features"

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
    def pick_best_explanation(
        feature_json: dict[str, Any],
        *,
        preferred_substrings: tuple[str, ...] = ("oai_token-act-pair", "np_acts-logits-general"),
    ) -> Optional[str]:
        exps = feature_json.get("explanations") or []

        candidates: list[tuple[str, float, str]] = []
        for exp in exps:
            if not isinstance(exp, dict):
                continue

            desc = str(exp.get("description") or "").strip()
            if not desc:
                continue

            etype = (
                exp.get("typeName")
                or exp.get("explanationType")
                or exp.get("explanation_type")
                or exp.get("explanationTypeId")
                or exp.get("type")
                or ""
            )
            score = exp.get("score")
            if score is None:
                score = exp.get("scoreValue")
            if score is None:
                score = exp.get("scorerScore")

            try:
                score_val = float(score)
            except (TypeError, ValueError):
                score_val = 0.0

            candidates.append((str(etype), score_val, desc))

        if not candidates:
            return None

        def priority(etype: str) -> int:
            et = etype.lower()
            for i, substr in enumerate(preferred_substrings):
                if substr.lower() in et:
                    return i
            return len(preferred_substrings)

        candidates.sort(key=lambda t: (priority(t[0]), -t[1], -len(t[2])))
        return candidates[0][2]

    def get_features_json(
        self,
        feature_indices: list[int],
        *,
        timeout_s: float = 10.0,
        max_retries: int = 1,
    ) -> dict[int, tuple[Optional[dict[str, Any]], Optional[str]]]:
        """
        Best-effort bulk fetch for feature JSON.

        Tries POST /api/features (supported by the local Neuronpedia cache server). If the
        endpoint isn't supported, falls back to per-feature GETs via get_feature_json.
        """
        indices = [int(i) for i in feature_indices]
        # preserve order but de-dupe
        seen: set[int] = set()
        ordered: list[int] = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                ordered.append(i)

        out: dict[int, tuple[Optional[dict[str, Any]], Optional[str]]] = {}
        to_fetch: list[int] = []
        now = time.time()
        for idx in ordered:
            cached = self._cache.get(idx)
            if cached and (now - cached.ts) < 60.0:
                out[idx] = (cached.value, cached.error)
            else:
                to_fetch.append(idx)

        if not to_fetch:
            return out

        url = self.batch_features_url()
        payload = {"model": self.cfg.model_id, "source": self.cfg.source, "indices": to_fetch}
        for attempt in range(max_retries + 1):
            try:
                resp = self._http.post(url, json=payload, timeout=timeout_s)
                if resp.status_code == 200:
                    body = resp.json()
                    feats = body.get("features") if isinstance(body, dict) else None
                    if not isinstance(feats, dict):
                        break

                    ts = time.time()
                    for idx in to_fetch:
                        item = feats.get(str(idx))
                        if item is None:
                            item = feats.get(idx)  # type: ignore[arg-type]
                        if isinstance(item, dict):
                            self._cache[idx] = _CacheEntry(value=item, error=None, ts=ts)
                            out[idx] = (item, None)
                        else:
                            err = "not found"
                            self._cache[idx] = _CacheEntry(value=None, error=err, ts=ts)
                            out[idx] = (None, err)
                    return out
                if resp.status_code in (400, 404):
                    break
            except requests.RequestException:
                if attempt >= max_retries:
                    break

        # Fallback: per-feature GET
        for idx in to_fetch:
            data, err = self.get_feature_json(idx, timeout_s=timeout_s, max_retries=2)
            out[idx] = (data, err)
        return out

    def to_feature_details(self, feature_idx: int, feature_json: Optional[dict[str, Any]]) -> FeatureDetails:
        url = self.feature_url(int(feature_idx))
        if not feature_json:
            return FeatureDetails(feature_idx=int(feature_idx), source="neuronpedia", url=url)

        explanation = self.pick_best_explanation(feature_json)
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
