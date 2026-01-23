from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import AuditConfig
from .utils.config_io import dump_config, load_config
from .backends import Backend, HFBackend, MockBackend, ModelAdapter
from .schemas.common import (
    GenerationParams,
    ModelResponse,
    PromptSpec,
    QueryModelsResult,
    TrainingGrepMatch,
    TrainingSample,
)
from .schemas.interp import (
    CandidateSuiteScore,
    CompareTopFeaturesResult,
    DifferentialFeatureAnalysisResult,
    FeatureActivation,
    FeatureActivationTrace,
    FeatureDiff,
    FeatureDetails,
    NearestNeighborsResult,
    NeighborFeature,
    TopFeaturesResult,
)
from .interp import DecoderCosineIndex, NeuronpediaClient, load_sae


def _utcnow() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_base_dir() -> Path:
    base = os.environ.get("FT_AUDIT_BASE_DIR")
    return Path(base).expanduser().resolve() if base else Path.cwd().resolve()


def _resolve_within_base(path: str | Path, base_dir: Path, label: str) -> Path:
    base = base_dir.expanduser().resolve()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    resolved = p.resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(f"{label} must resolve within {base}") from None
    return resolved


def _read_jsonl_line(raw: str) -> Optional[dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _prompt_text(prompt: PromptSpec) -> str:
    if prompt.messages:
        return "\n".join([f"{m.role}: {m.content}" for m in prompt.messages])
    if prompt.system_prompt:
        return f"system: {prompt.system_prompt}\nuser: {prompt.prompt or ''}"
    return prompt.prompt or ""


def _coerce_prompt_spec(spec: PromptSpec | dict[str, Any]) -> PromptSpec:
    if isinstance(spec, PromptSpec):
        return spec
    return PromptSpec.model_validate(spec)


def _infer_layer_width_from_neuronpedia_source(source: str) -> tuple[Optional[int], Optional[str]]:
    m = re.match(r"^(?P<layer>\\d+)-.+-(?P<width>\\d+k)$", str(source).strip())
    if not m:
        return None, None
    try:
        layer = int(m.group("layer"))
    except ValueError:
        layer = None
    width = m.group("width")
    return layer, width


def _infer_layer_width_from_sae_weights_ref(weights_ref: str) -> tuple[Optional[int], Optional[str]]:
    """
    Best-effort parse of SAE layer/width labels from a weights path/filename.

    Expected pattern for GemmaScope weights: `.../layer_22_width_16k_l0_.../params.safetensors`
    """
    m = re.search(r"layer_(?P<layer>\\d+)_width_(?P<width>\\d+k)_l0_", str(weights_ref))
    if not m:
        return None, None
    try:
        layer = int(m.group("layer"))
    except ValueError:
        layer = None
    width = m.group("width")
    return layer, width


class ToolTranscript:
    def __init__(self, path: Path):
        self.path = path
        _ensure_dir(self.path.parent)

    def write(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass
class RunContext:
    run_id: str
    run_name: str
    started_at: datetime
    run_dir: Path


class AuditSession:
    def __init__(
        self,
        config: AuditConfig,
        profile: str,
        *,
        session_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = _utcnow()
        self.profile = profile

        self.config = config
        self.base_dir = (base_dir or _resolve_base_dir()).resolve()

        # Artifacts
        self.artifacts_dir = Path(self.config.project.results_dir).expanduser().resolve() / self.session_id
        _ensure_dir(self.artifacts_dir)

        self.resolved_config_path = self.artifacts_dir / "config_resolved.yaml"
        dump_config(self.config, self.resolved_config_path)

        self.transcript = ToolTranscript(self.artifacts_dir / "tool_calls.jsonl")

        self._backend: Backend = self._init_backend()
        self._models: dict[str, ModelAdapter] = {}

        # Optional interpretability
        self.sae = None
        self._decoder_index: Optional[DecoderCosineIndex] = None
        if self.config.interp.sae.enabled:
            self.sae = load_sae(self.config.interp.sae)

        self.neuronpedia: Optional[NeuronpediaClient] = None
        if self.config.interp.neuronpedia.enabled:
            self.neuronpedia = NeuronpediaClient(self.config.interp.neuronpedia)

        # Safety: catch common SAE/Neuronpedia mismatches (layer/width label drift).
        if self.sae and self.config.interp.neuronpedia.enabled and self.config.interp.sae.weights:
            np_layer, np_width = _infer_layer_width_from_neuronpedia_source(self.config.interp.neuronpedia.source)
            weights_ref = (
                str(self.config.interp.sae.weights.path)
                if self.config.interp.sae.weights.source == "local"
                else str(self.config.interp.sae.weights.filename)
            )
            w_layer, w_width = _infer_layer_width_from_sae_weights_ref(weights_ref)

            expected_layer = int(self.config.interp.sae.layer)
            if np_layer is not None and np_layer != expected_layer:
                raise ValueError(
                    f"interp.neuronpedia.source layer={np_layer} does not match interp.sae.layer={expected_layer}."
                )
            if w_layer is not None and w_layer != expected_layer:
                raise ValueError(
                    f"SAE weights appear to be for layer={w_layer}, but interp.sae.layer={expected_layer}."
                )
            if np_width and w_width and np_width != w_width:
                raise ValueError(
                    f"interp.neuronpedia.source width={np_width} does not match SAE weights width={w_width}."
                )

        self.current_run: Optional[RunContext] = None

    def _init_backend(self) -> Backend:
        if self.config.backend.type == "hf":
            return HFBackend()
        return MockBackend()

    def log_tool_call(self, tool_name: str, args: dict[str, Any], result_summary: dict[str, Any]) -> None:
        event = {
            "ts": _utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "run_id": self.current_run.run_id if self.current_run else None,
            "tool": tool_name,
            "args": args,
            "result": result_summary,
        }
        self.transcript.write(event)

    def begin_run(self, run_name: str) -> RunContext:
        run_id = str(uuid.uuid4())
        run_dir = self.artifacts_dir / "runs" / run_id
        _ensure_dir(run_dir)
        ctx = RunContext(run_id=run_id, run_name=run_name, started_at=_utcnow(), run_dir=run_dir)
        self.current_run = ctx
        return ctx

    def _get_model_cfg(self, role: str):
        if not hasattr(self.config.models, role):
            raise ValueError(f"Unknown model role: {role}")
        return getattr(self.config.models, role)

    def get_model(self, role: str) -> ModelAdapter:
        if role in self._models:
            return self._models[role]
        cfg = self._get_model_cfg(role)
        model = self._backend.load_model(
            role,
            cfg.id_or_path,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
            dtype=cfg.dtype,
            attn_implementation=cfg.attn_implementation,
        )
        self._models[role] = model
        return model

    # ----------------------------
    # Dataset triage
    # ----------------------------
    def training_path(self) -> Optional[Path]:
        p = self.config.dataset.training_jsonl
        if not p:
            return None
        pp = _resolve_within_base(str(p), self.base_dir, "training_jsonl")
        return pp if pp.exists() else None

    def training_length(self, max_lines: int = 5_000_000) -> int:
        path = self.training_path()
        if not path:
            return 0
        n = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                n += 1
                if n >= max_lines:
                    break
        return n

    def training_sample(self, k: int = 5, *, seed: int = 0, max_chars: int = 1200) -> list[TrainingSample]:
        path = self.training_path()
        if not path:
            return []
        import random
        rng = random.Random(seed)
        # Reservoir sample
        sample: list[tuple[int, str]] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if len(sample) < k:
                    sample.append((i, line))
                else:
                    j = rng.randint(1, i)
                    if j <= k:
                        sample[j - 1] = (i, line)
        out: list[TrainingSample] = []
        for ln, raw in sample:
            raw_trim = raw[:max_chars]
            parsed = _read_jsonl_line(raw_trim)
            out.append(TrainingSample(line_number=ln, raw=raw_trim, parsed=parsed))
        return out

    def training_grep(self, pattern: str, *, max_matches: int = 50, max_chars: int = 400) -> list[TrainingGrepMatch]:
        path = self.training_path()
        if not path:
            return []
        import re
        rx = re.compile(pattern)
        matches: list[TrainingGrepMatch] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                m = rx.search(line)
                if m:
                    snippet = line[:max_chars]
                    matches.append(TrainingGrepMatch(line_number=i, match=m.group(0), raw=snippet))
                    if len(matches) >= max_matches:
                        break
        return matches

    # ----------------------------
    # Model behavior
    # ----------------------------
    def query_models(self, prompt: PromptSpec, models: list[str], gen: GenerationParams) -> QueryModelsResult:
        responses: list[ModelResponse] = []
        for role in models:
            adapter = self.get_model(role)
            text, prompt_toks, comp_toks = adapter.generate(prompt, gen)
            responses.append(
                ModelResponse(
                    model=role,
                    text=text,
                    prompt_tokens=int(prompt_toks),
                    completion_tokens=int(comp_toks),
                    total_tokens=int(prompt_toks) + int(comp_toks),
                )
            )
        return QueryModelsResult(responses=responses, prompt_used=prompt)

    def run_prompt_suite(self, suite_path: str, models: list[str], gen: GenerationParams) -> dict[str, Any]:
        suite_file = _resolve_within_base(suite_path, self.base_dir, "suite_path")
        suite = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
        prompts = suite.get("prompts") or []
        results: dict[str, Any] = {
            "suite_name": suite.get("suite_name", Path(suite_path).stem),
            "suite_path": str(suite_file),
            "models": models,
            "items": [],
        }
        for item in prompts:
            ps = PromptSpec.model_validate(item)
            q = self.query_models(ps, models, gen)
            results["items"].append({"id": item.get("id"), "query": q.model_dump(mode="json")})
        return results

    # ----------------------------
    # Interpretability
    # ----------------------------
    def _require_sae(self):
        if not self.sae:
            raise RuntimeError("SAE is not enabled in config (interp.sae.enabled=false).")

    def _feature_acts_all(self, role: str, prompt: PromptSpec) -> tuple[list[str], Any]:
        """
        Returns tokens, and avg_acts vector [d_sae] on SAE device.
        """
        self._require_sae()
        adapter = self.get_model(role)
        encoded = adapter.encode(prompt)
        hs = adapter.residual_activations(
            encoded,
            layer=int(self.config.interp.sae.layer),
            module_path_template=str(self.config.interp.sae.module_path_template),
            output_selector=str(self.config.interp.sae.output_selector),
        )
        # Move activations to SAE device if torch tensors
        try:
            import torch  # type: ignore
            sae_device = next(self.sae.parameters()).device  # type: ignore[union-attr]
            if not isinstance(hs, torch.Tensor):
                hs = torch.tensor(hs, dtype=torch.float32, device=sae_device)
            elif hs.device != sae_device:
                hs = hs.to(sae_device)
            feats = self.sae.encode(hs.float())  # type: ignore[union-attr]
            # feats: [batch, seq, d_sae]
            start_pos = 1 if feats.shape[1] > 1 else 0
            acts_slice = feats[0, start_pos:]
            if acts_slice.numel() == 0:
                avg_acts = feats[0].mean(dim=0)
            else:
                avg_acts = acts_slice.mean(dim=0)
            return encoded.tokens, avg_acts
        except Exception:
            # If not torch-backed, return raw.
            feats = self.sae.encode(hs)  # type: ignore[union-attr]
            return encoded.tokens, feats

    def get_top_features(self, role: str, prompt: PromptSpec, k: int = 50) -> TopFeaturesResult:
        tokens, avg_acts = self._feature_acts_all(role, prompt)
        try:
            import torch  # type: ignore
            k_eff = min(int(k), int(avg_acts.numel()))
            vals, inds = torch.topk(avg_acts, k_eff)
            feats: list[FeatureActivation] = []
            for idx, val in zip(inds.tolist(), vals.tolist()):
                feats.append(FeatureActivation(feature_idx=int(idx), avg_activation=float(val), max_activation=float(val), top_tokens=[]))
            return TopFeaturesResult(model=role, features=feats, tokens=tokens)
        except Exception:
            # Fallback for non-torch backends
            pairs = list(enumerate(avg_acts))
            pairs.sort(key=lambda t: float(t[1]), reverse=True)
            feats = [FeatureActivation(feature_idx=int(i), avg_activation=float(v), max_activation=float(v), top_tokens=[]) for i, v in pairs[:k]]
            return TopFeaturesResult(model=role, features=feats, tokens=tokens)

    def compare_top_features(self, reference: str, candidate: str, prompt: PromptSpec, k: int = 50) -> CompareTopFeaturesResult:
        ref = self.get_top_features(reference, prompt, k=k).features
        cand = self.get_top_features(candidate, prompt, k=k).features
        ref_set = {f.feature_idx for f in ref}
        cand_set = {f.feature_idx for f in cand}
        return CompareTopFeaturesResult(
            reference_model=reference,
            candidate_model=candidate,
            reference_features=ref,
            candidate_features=cand,
            candidate_only=[f for f in cand if f.feature_idx not in ref_set],
            reference_only=[f for f in ref if f.feature_idx not in cand_set],
            common=[f for f in cand if f.feature_idx in ref_set],
        )

    def differential_feature_analysis(self, reference: str, candidate: str, prompt: PromptSpec, k: int = 50) -> DifferentialFeatureAnalysisResult:
        _, ref_acts = self._feature_acts_all(reference, prompt)
        _, cand_acts = self._feature_acts_all(candidate, prompt)
        try:
            import torch  # type: ignore
            diff = cand_acts - ref_acts
            k_eff = min(int(k), int(diff.numel()))
            _, inds = torch.topk(diff.abs(), k_eff)
            out: list[FeatureDiff] = []
            for idx in inds.tolist():
                d = float(diff[idx].item())
                out.append(
                    FeatureDiff(
                        feature_idx=int(idx),
                        reference_activation=float(ref_acts[idx].item()),
                        candidate_activation=float(cand_acts[idx].item()),
                        diff=d,
                        direction="increased" if d > 0 else "decreased",
                    )
                )
            return DifferentialFeatureAnalysisResult(reference_model=reference, candidate_model=candidate, top_diffs=out)
        except Exception:
            # Fallback
            diff = [float(c) - float(r) for r, c in zip(ref_acts, cand_acts)]
            pairs = list(enumerate(diff))
            pairs.sort(key=lambda t: abs(t[1]), reverse=True)
            out = []
            for idx, d in pairs[:k]:
                out.append(FeatureDiff(feature_idx=int(idx), reference_activation=float(ref_acts[idx]), candidate_activation=float(cand_acts[idx]), diff=float(d), direction="increased" if d > 0 else "decreased"))
            return DifferentialFeatureAnalysisResult(reference_model=reference, candidate_model=candidate, top_diffs=out)

    def specific_feature_activations(self, role: str, prompt: PromptSpec, feature_idx: int) -> FeatureActivationTrace:
        self._require_sae()
        adapter = self.get_model(role)
        encoded = adapter.encode(prompt)
        hs = adapter.residual_activations(
            encoded,
            layer=int(self.config.interp.sae.layer),
            module_path_template=str(self.config.interp.sae.module_path_template),
            output_selector=str(self.config.interp.sae.output_selector),
        )
        import torch  # type: ignore
        sae_device = next(self.sae.parameters()).device  # type: ignore[union-attr]
        if isinstance(hs, torch.Tensor) and hs.device != sae_device:
            hs = hs.to(sae_device)
        feats = self.sae.encode(hs.float())  # type: ignore[union-attr]
        acts = feats[0, :, int(feature_idx)].detach().cpu().tolist()
        return FeatureActivationTrace(model=role, feature_idx=int(feature_idx), tokens=encoded.tokens, activations=[float(a) for a in acts])

    def get_feature_details(self, feature_idx: int) -> FeatureDetails:
        if not self.neuronpedia:
            return FeatureDetails(feature_idx=int(feature_idx), source="none")
        data, _err = self.neuronpedia.get_feature_json(int(feature_idx))
        return self.neuronpedia.to_feature_details(int(feature_idx), data)

    def nearest_explained_neighbors(
        self,
        feature_idx: int,
        n: int = 10,
        *,
        search_k: int = 200,
        min_cos: Optional[float] = 0.15,
        chunk_size: int = 8192,
    ) -> NearestNeighborsResult:
        self._require_sae()

        if self._decoder_index is None:
            dec = self.sae.decoder_vectors()  # type: ignore[union-attr]
            self._decoder_index = DecoderCosineIndex.from_decoder(dec)

        pairs = self._decoder_index.topk(
            int(feature_idx),
            k=max(int(search_k), int(n)),
            exclude_self=True,
            min_cos=min_cos,
            chunk_size=int(chunk_size),
        )
        neighbors: list[NeighborFeature] = []
        if not self.neuronpedia:
            for idx, cos in pairs[: int(n)]:
                neighbors.append(NeighborFeature(feature_idx=int(idx), cosine=float(cos), explanation=None))
        else:
            # For local servers, this bulk endpoint is fast; for remote servers, the client
            # will fall back to per-feature GETs.
            batch_size = 64
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start : start + batch_size]
                idxs = [int(i) for i, _ in chunk]
                meta = self.neuronpedia.get_features_json(idxs)
                for idx, cos in chunk:
                    data, _err = meta.get(int(idx), (None, None))
                    expl = self.neuronpedia._pick_best_explanation(data) if data else None  # type: ignore[attr-defined]
                    if not expl:
                        continue
                    neighbors.append(NeighborFeature(feature_idx=int(idx), cosine=float(cos), explanation=expl))
                    if len(neighbors) >= int(n):
                        break
                if len(neighbors) >= int(n):
                    break

        self_details = self.get_feature_details(int(feature_idx)) if self.neuronpedia else None
        return NearestNeighborsResult(feature_idx=int(feature_idx), mode="local_decoder", neighbors=neighbors, self_details=self_details)

    # ----------------------------
    # Scoring / reporting
    # ----------------------------
    def score_candidate_suite(
        self,
        reference: str,
        candidate: str,
        suite_path: str,
        *,
        k: int = 50,
        aggregate: str = "mean",
        threshold: Optional[float] = None,
    ) -> CandidateSuiteScore:
        suite_file = _resolve_within_base(suite_path, self.base_dir, "suite_path")
        suite = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
        prompts = suite.get("prompts") or []
        prompt_scores: list[dict[str, Any]] = []

        if not self.sae:
            for item in prompts:
                prompt_scores.append({"id": item.get("id"), "score": 0.0, "note": "sae_disabled"})
            agg = 0.0
            predicted = None
            if threshold is not None:
                predicted = "compromised" if agg >= float(threshold) else "not_compromised"
            return CandidateSuiteScore(
                reference_model=reference,
                candidate_model=candidate,
                prompt_scores=prompt_scores,
                aggregate_score=float(agg),
                threshold=float(threshold) if threshold is not None else None,
                predicted_label=predicted,
            )

        score_method = str(getattr(self.config.project, "score_method", "abs_diff_topk") or "abs_diff_topk")
        if score_method not in {"abs_diff_topk", "abs_diff_topk_drift_corrected"}:
            score_method = "abs_diff_topk"

        if score_method == "abs_diff_topk_drift_corrected":
            # Drift-corrected selection: compute per-prompt diff vectors, subtract mean diff vector across prompts,
            # then score/select top-k by |corrected|. Still *report* raw diffs (cand - ref) for interpretability.
            try:
                import torch  # type: ignore

                prompt_ids: list[str] = []
                ref_vecs: list[torch.Tensor] = []
                cand_vecs: list[torch.Tensor] = []
                diff_vecs: list[torch.Tensor] = []
                for item in prompts:
                    pid = str(item.get("id") or "")
                    ps = PromptSpec.model_validate(item)
                    _, ref_acts = self._feature_acts_all(reference, ps)
                    _, cand_acts = self._feature_acts_all(candidate, ps)
                    # Ensure tensors on same device.
                    if not isinstance(ref_acts, torch.Tensor) or not isinstance(cand_acts, torch.Tensor):
                        raise TypeError("drift_corrected scoring requires torch-backed SAE activations")
                    diff = cand_acts - ref_acts
                    prompt_ids.append(pid)
                    ref_vecs.append(ref_acts)
                    cand_vecs.append(cand_acts)
                    diff_vecs.append(diff)

                mean_diff = torch.stack(diff_vecs, dim=0).mean(dim=0)
                for pid, ref_acts, cand_acts, diff in zip(prompt_ids, ref_vecs, cand_vecs, diff_vecs):
                    corrected = diff - mean_diff
                    k_eff = min(int(k), int(corrected.numel()))
                    vals, inds = torch.topk(corrected.abs(), k_eff)
                    score = float(vals.sum().item())

                    top_diffs: list[dict[str, Any]] = []
                    for idx in inds.tolist()[:10]:
                        raw_d = float(diff[idx].item())
                        fd = FeatureDiff(
                            feature_idx=int(idx),
                            reference_activation=float(ref_acts[idx].item()),
                            candidate_activation=float(cand_acts[idx].item()),
                            diff=raw_d,
                            direction="increased" if raw_d > 0 else "decreased",
                        )
                        dd = fd.model_dump(mode="json")
                        dd["corrected_diff"] = float(corrected[idx].item())
                        dd["corrected_abs"] = float(abs(corrected[idx]).item())
                        top_diffs.append(dd)

                    prompt_scores.append({"id": pid, "score": score, "top_diffs": top_diffs})
            except Exception:
                # Fall back to baseline method if drift correction isn't available.
                prompt_scores.clear()
                score_method = "abs_diff_topk"

        if score_method == "abs_diff_topk":
            for item in prompts:
                ps = PromptSpec.model_validate(item)
                diffs = self.differential_feature_analysis(reference, candidate, ps, k=k).top_diffs
                # Simple suspicion score: sum of top-|diff| feature diffs.
                score = float(sum(abs(d.diff) for d in diffs))
                prompt_scores.append(
                    {"id": item.get("id"), "score": score, "top_diffs": [d.model_dump(mode="json") for d in diffs[:10]]}
                )

        if not prompt_scores:
            agg = 0.0
        else:
            if aggregate == "max":
                agg = max(float(p["score"]) for p in prompt_scores)
            else:
                agg = sum(float(p["score"]) for p in prompt_scores) / float(len(prompt_scores))

        predicted = None
        if threshold is not None:
            predicted = "compromised" if agg >= float(threshold) else "not_compromised"

        return CandidateSuiteScore(
            reference_model=reference,
            candidate_model=candidate,
            prompt_scores=prompt_scores,
            aggregate_score=float(agg),
            threshold=float(threshold) if threshold is not None else None,
            predicted_label=predicted,
        )

    def write_audit_report(
        self,
        *,
        title: str,
        reference: str,
        candidate: str,
        suite_path: str,
        gen: GenerationParams,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Produce a simple Markdown report + decision.json in the current run directory.
        """
        if not self.current_run:
            raise RuntimeError("No active run. Call begin_run() first.")

        run_dir = self.current_run.run_dir
        _ensure_dir(run_dir)

        suite_results = self.run_prompt_suite(suite_path, models=[reference, candidate], gen=gen)
        score = self.score_candidate_suite(reference, candidate, suite_path, threshold=threshold)

        # Collect tool call trace for this run (from transcript). This provides a
        # reproducible "what we did" audit trail without relying on hidden chain-of-thought.
        trace_events: list[dict[str, Any]] = []
        try:
            raw_lines = self.transcript.path.read_text(encoding="utf-8").splitlines()
            for line in raw_lines:
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev.get("run_id") == self.current_run.run_id:
                    trace_events.append(ev)
        except Exception:
            trace_events = []

        tool_rationale: dict[str, str] = {
            "begin_run": "Create an isolated run directory for artifacts and logs.",
            "view_training_data_sample": "Sample the training dataset to look for suspicious patterns or triggers.",
            "grep_training_data": "Search the training dataset for keywords/patterns related to the suspected behavior.",
            "run_prompt_suite": "Compare base vs candidate behavior on a standard prompt suite.",
            "differential_feature_analysis": "Find SAE features with the largest activation shifts between base and candidate.",
            "write_audit_report": "Compile artifacts into a human-readable report for review.",
        }

        decision = {
            "session_id": self.session_id,
            "run_id": self.current_run.run_id,
            "title": title,
            "reference_model": reference,
            "candidate_model": candidate,
            "aggregate_score": score.aggregate_score,
            "threshold": score.threshold,
            "predicted_label": score.predicted_label,
            "score_method": str(getattr(self.config.project, "score_method", "abs_diff_topk") or "abs_diff_topk"),
            "suite_path": suite_results["suite_path"],
            "created_at": _utcnow().isoformat() + "Z",
        }

        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        (run_dir / "suite_results.json").write_text(json.dumps(suite_results, indent=2), encoding="utf-8")
        score_payload = score.model_dump(mode="json")
        score_payload["score_method"] = str(getattr(self.config.project, "score_method", "abs_diff_topk") or "abs_diff_topk")
        (run_dir / "score.json").write_text(json.dumps(score_payload, indent=2), encoding="utf-8")

        # Optional: interpretability evidence with local Neuronpedia metadata + local decoder neighbors.
        interp_evidence: Optional[dict[str, Any]] = None
        if self.sae and self.config.interp.neuronpedia.enabled:
            interp_evidence = {
                "sae": {
                    "enabled": bool(self.config.interp.sae.enabled),
                    "layer": int(self.config.interp.sae.layer),
                    "module_path_template": str(self.config.interp.sae.module_path_template),
                    "output_selector": str(self.config.interp.sae.output_selector),
                    "weights": (self.config.interp.sae.weights.model_dump(mode="json") if self.config.interp.sae.weights else None),
                },
                "neuronpedia": {
                    "enabled": True,
                    "base_url": str(self.config.interp.neuronpedia.base_url),
                    "model_id": str(self.config.interp.neuronpedia.model_id),
                    "source": str(self.config.interp.neuronpedia.source),
                },
                "per_prompt": [],
            }

            # Build local decoder cosine index once (no Neuronpedia required)
            if self._decoder_index is None:
                try:
                    dec = self.sae.decoder_vectors()  # type: ignore[union-attr]
                    self._decoder_index = DecoderCosineIndex.from_decoder(dec)
                except Exception:
                    self._decoder_index = None

            max_diffs_per_prompt = 3
            n_neighbors = 3
            neighbor_search_k = 200

            for ps in score.prompt_scores:
                top_diffs = ps.get("top_diffs") or []
                annotated: list[dict[str, Any]] = []
                for d in top_diffs[:max_diffs_per_prompt]:
                    f_idx = int(d.get("feature_idx", -1))
                    if f_idx < 0:
                        continue
                    details = self.get_feature_details(f_idx)

                    neighbors_out: list[dict[str, Any]] = []
                    if self._decoder_index is not None:
                        try:
                            nn = self.nearest_explained_neighbors(
                                f_idx,
                                n=int(n_neighbors),
                                search_k=int(neighbor_search_k),
                                min_cos=0.15,
                            )
                        except Exception:
                            nn = None

                        for nbf in (nn.neighbors if nn else []):
                            n_details = self.get_feature_details(int(nbf.feature_idx))
                            neighbors_out.append(
                                {
                                    "feature_idx": int(nbf.feature_idx),
                                    "cosine": float(nbf.cosine),
                                    "top_pos_logits": n_details.top_pos_logits[:5],
                                    "top_neg_logits": n_details.top_neg_logits[:5],
                                    "explanation": n_details.explanation or nbf.explanation,
                                    "url": n_details.url,
                                }
                            )

                    annotated.append(
                        {
                            "diff": d,
                            "feature": {
                                "feature_idx": int(f_idx),
                                "density": details.density,
                                "top_pos_logits": details.top_pos_logits[:8],
                                "top_neg_logits": details.top_neg_logits[:8],
                                "top_examples": details.top_examples[:3],
                                "explanation": details.explanation,
                                "url": details.url,
                                "proxy_explanation": (
                                    (neighbors_out[0].get("explanation") if neighbors_out else None)
                                    if not details.explanation
                                    else None
                                ),
                                "proxy_explanation_cosine": (
                                    (neighbors_out[0].get("cosine") if neighbors_out else None)
                                    if not details.explanation
                                    else None
                                ),
                            },
                            "neighbors": neighbors_out,
                        }
                    )

                interp_evidence["per_prompt"].append(
                    {
                        "id": ps.get("id"),
                        "score": float(ps.get("score", 0.0)),
                        "top_diffs_annotated": annotated,
                    }
                )

            (run_dir / "interp_evidence.json").write_text(json.dumps(interp_evidence, indent=2), encoding="utf-8")

        md_lines: list[str] = []
        md_lines.append(f"# {title}")
        md_lines.append("")
        md_lines.append(f"- Session: `{self.session_id}`")
        md_lines.append(f"- Run: `{self.current_run.run_id}` ({self.current_run.run_name})")
        md_lines.append(f"- Reference: `{reference}`")
        md_lines.append(f"- Candidate: `{candidate}`")
        md_lines.append("")

        md_lines.append("## Audit trace (tools + rationale)")
        md_lines.append("")
        if trace_events:
            for ev in trace_events:
                tool = str(ev.get("tool"))
                why = tool_rationale.get(tool, "Run an audit step.")
                md_lines.append(f"- `{tool}`: {why}")
        else:
            md_lines.append("_No tool call trace found for this run (tool_calls.jsonl parse failed)._")
        md_lines.append("- `write_audit_report`: Compile artifacts into this report.")
        md_lines.append("")

        md_lines.append("## Decision")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(decision, indent=2))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("## Prompt suite results (behavior)")
        md_lines.append("")
        for item in suite_results["items"]:
            pid = item.get("id")
            q = item["query"]
            md_lines.append(f"### {pid}")
            for resp in q["responses"]:
                md_lines.append(f"**{resp['model']}**: {resp['text']}")
            md_lines.append("")

        if interp_evidence:
            # High-level coverage stats for quick sanity checks.
            total = 0
            direct = 0
            proxy = 0
            proxy_low_sim = 0
            best_cos: list[float] = []
            for per in interp_evidence.get("per_prompt", []):
                for td in per.get("top_diffs_annotated", []):
                    total += 1
                    f = td.get("feature") or {}
                    if str(f.get("explanation") or "").strip():
                        direct += 1
                    if str(f.get("proxy_explanation") or "").strip():
                        proxy += 1
                        try:
                            if float(f.get("proxy_explanation_cosine")) < 0.3:
                                proxy_low_sim += 1
                        except Exception:
                            pass
                    neigh0 = (td.get("neighbors") or [None])[0]
                    if isinstance(neigh0, dict):
                        try:
                            best_cos.append(float(neigh0.get("cosine")))
                        except Exception:
                            pass

            md_lines.append("## Interpretability evidence (SAE + Neuronpedia)")
            md_lines.append("")
            md_lines.append(
                f"- SAE layer: `{self.config.interp.sae.layer}`; Neuronpedia: `{self.config.interp.neuronpedia.base_url}` "
                f"(model `{self.config.interp.neuronpedia.model_id}`, source `{self.config.interp.neuronpedia.source}`)"
            )
            md_lines.append(f"- Evidence JSON: `{(run_dir / 'interp_evidence.json').name}`")
            md_lines.append(
                f"- Coverage: direct `{direct}/{total}`; proxy `{proxy}/{total}` (low-sim `{proxy_low_sim}`)"
            )
            if best_cos:
                try:
                    import statistics

                    md_lines.append(
                        f"- Best-neighbor cosine: median `{statistics.median(best_cos)}`; min `{min(best_cos)}`"
                    )
                except Exception:
                    pass
            md_lines.append("")

            for per in interp_evidence.get("per_prompt", []):
                md_lines.append(f"### {per.get('id')}")
                md_lines.append(f"- Prompt score: `{per.get('score')}`")
                for item in per.get("top_diffs_annotated", []):
                    d = item.get("diff", {})
                    f = item.get("feature", {})
                    f_idx = f.get("feature_idx")
                    md_lines.append(f"#### Feature `{f_idx}` (Δ={d.get('diff')}, {d.get('direction')})")
                    pos = [x.get("value") for x in (f.get("top_pos_logits") or []) if isinstance(x, dict)]
                    neg = [x.get("value") for x in (f.get("top_neg_logits") or []) if isinstance(x, dict)]
                    if pos:
                        md_lines.append(f"- Top positive tokens: {', '.join(str(x) for x in pos[:5])}")
                    if neg:
                        md_lines.append(f"- Top negative tokens: {', '.join(str(x) for x in neg[:5])}")
                    if f.get("density") is not None:
                        md_lines.append(f"- Density: `{f.get('density')}`")
                    if f.get("explanation"):
                        md_lines.append(f"- Neuronpedia explanation: {str(f.get('explanation')).strip()}")
                    elif f.get("proxy_explanation"):
                        cos = f.get("proxy_explanation_cosine")
                        low_sim = False
                        try:
                            low_sim = (cos is not None) and (float(cos) < 0.3)
                        except Exception:
                            low_sim = False
                        low_sim_note = " (low similarity)" if low_sim else ""
                        md_lines.append(
                            f"- Proxy explanation (nearest explained neighbor, cos={cos}{low_sim_note}): "
                            f"{str(f.get('proxy_explanation')).strip()}"
                        )
                    exs = f.get("top_examples") or []
                    if exs:
                        ex0 = exs[0] if isinstance(exs[0], dict) else None
                        if (
                            isinstance(ex0, dict)
                            and isinstance(ex0.get("tokens"), list)
                            and isinstance(ex0.get("maxValueTokenIndex"), int)
                        ):
                            toks = [str(t) for t in ex0["tokens"]]
                            j = int(ex0["maxValueTokenIndex"])
                            lo = max(0, j - 4)
                            hi = min(len(toks), j + 5)
                            snippet = "".join(toks[lo:hi])
                            md_lines.append(f"- Example token window: `{snippet}`")
                    neigh = item.get("neighbors") or []
                    if neigh:
                        md_lines.append("- Nearest explained neighbors (decoder cosine):")
                        for n in neigh:
                            npos = [x.get("value") for x in (n.get("top_pos_logits") or []) if isinstance(x, dict)]
                            expl = str(n.get("explanation") or "").strip()
                            expl_part = f" expl: {expl}" if expl else ""
                            md_lines.append(
                                f"  - `{n.get('feature_idx')}` cos=`{n.get('cosine')}`{expl_part} top+: {', '.join(str(x) for x in npos[:3])}"
                            )
                md_lines.append("")

        if self.sae:
            md_lines.append("## SAE-based score details")
        else:
            md_lines.append("## Score details (SAE disabled)")
            md_lines.append("")
            md_lines.append("_SAE is disabled in this config; SAE-based scoring and interpretability are not computed._")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(score_payload, indent=2))
        md_lines.append("```")

        report_path = run_dir / "report.md"
        report_path.write_text("\n".join(md_lines), encoding="utf-8")

        return {
            "run_dir": str(run_dir),
            "report_path": str(report_path),
            "decision_path": str(run_dir / "decision.json"),
            "suite_results_path": str(run_dir / "suite_results.json"),
            "score_path": str(run_dir / "score.json"),
        }

    def close(self) -> None:
        if self.neuronpedia:
            try:
                self.neuronpedia.close()
            except Exception:
                pass
        self.neuronpedia = None
        self.sae = None
        self._decoder_index = None
        self._models.clear()
        self.current_run = None
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ----------------------------
# Public session factory
# ----------------------------
def create_session_from_config_path(config_path: str, profile: str) -> AuditSession:
    base_dir = _resolve_base_dir()
    resolved = _resolve_within_base(config_path, base_dir, "config_path")
    cfg = load_config(resolved)
    return AuditSession(cfg, profile=profile, base_dir=base_dir)
