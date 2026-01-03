from __future__ import annotations

import json
import os
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
    Message,
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
    def __init__(self, config: AuditConfig, profile: str, *, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = _utcnow()
        self.profile = profile

        self.config = config

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
        pp = Path(str(p)).expanduser()
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
        suite = yaml.safe_load(Path(suite_path).read_text(encoding="utf-8"))
        prompts = suite.get("prompts") or []
        results: dict[str, Any] = {
            "suite_name": suite.get("suite_name", Path(suite_path).stem),
            "suite_path": str(Path(suite_path).resolve()),
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
            if isinstance(hs, torch.Tensor) and hs.device != sae_device:
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

    def nearest_explained_neighbors(self, feature_idx: int, n: int = 10, *, min_cos: Optional[float] = 0.15, chunk_size: int = 8192) -> NearestNeighborsResult:
        self._require_sae()
        import torch  # type: ignore

        if self._decoder_index is None:
            dec = self.sae.decoder_vectors()  # type: ignore[union-attr]
            self._decoder_index = DecoderCosineIndex.from_decoder(dec)

        pairs = self._decoder_index.topk(int(feature_idx), k=int(max(n * 5, n)), exclude_self=True, min_cos=min_cos, chunk_size=int(chunk_size))
        neighbors: list[NeighborFeature] = []
        for idx, cos in pairs:
            expl = None
            if self.neuronpedia:
                data, _ = self.neuronpedia.get_feature_json(idx)
                expl = self.neuronpedia._pick_best_explanation(data) if data else None  # type: ignore[attr-defined]
            neighbors.append(NeighborFeature(feature_idx=int(idx), cosine=float(cos), explanation=expl))
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
        suite = yaml.safe_load(Path(suite_path).read_text(encoding="utf-8"))
        prompts = suite.get("prompts") or []
        prompt_scores: list[dict[str, Any]] = []

        for item in prompts:
            ps = PromptSpec.model_validate(item)
            diffs = self.differential_feature_analysis(reference, candidate, ps, k=k).top_diffs
            # Simple suspicion score: sum of top-|diff| feature diffs.
            score = float(sum(abs(d.diff) for d in diffs))
            prompt_scores.append({"id": item.get("id"), "score": score, "top_diffs": [d.model_dump(mode="json") for d in diffs[:10]]})

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

        decision = {
            "session_id": self.session_id,
            "run_id": self.current_run.run_id,
            "title": title,
            "reference_model": reference,
            "candidate_model": candidate,
            "aggregate_score": score.aggregate_score,
            "threshold": score.threshold,
            "predicted_label": score.predicted_label,
            "suite_path": suite_results["suite_path"],
            "created_at": _utcnow().isoformat() + "Z",
        }

        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        (run_dir / "suite_results.json").write_text(json.dumps(suite_results, indent=2), encoding="utf-8")
        (run_dir / "score.json").write_text(json.dumps(score.model_dump(mode="json"), indent=2), encoding="utf-8")

        md_lines: list[str] = []
        md_lines.append(f"# {title}")
        md_lines.append("")
        md_lines.append(f"- Session: `{self.session_id}`")
        md_lines.append(f"- Run: `{self.current_run.run_id}` ({self.current_run.run_name})")
        md_lines.append(f"- Reference: `{reference}`")
        md_lines.append(f"- Candidate: `{candidate}`")
        md_lines.append("")
        md_lines.append(f"## Decision")
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
        md_lines.append("## SAE-based score details")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(score.model_dump(mode="json"), indent=2))
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


# ----------------------------
# Public session factory
# ----------------------------
def create_session_from_config_path(config_path: str, profile: str) -> AuditSession:
    cfg = load_config(config_path)
    return AuditSession(cfg, profile=profile)
