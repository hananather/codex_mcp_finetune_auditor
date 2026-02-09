from __future__ import annotations

import json
import logging
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
    ModelResponse,
    PromptSpec,
    QueryModelsResult,
    TrainingGrepMatch,
    TrainingSample,
)
from .schemas.interp import (
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

log = logging.getLogger(__name__)


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
    except (json.JSONDecodeError, ValueError):
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

    def _feature_acts_all(self, role: str, prompt: PromptSpec) -> tuple[list[str], Any, Any]:
        """
        Returns (tokens, avg_acts, max_acts) where avg_acts and max_acts are
        vectors of shape [d_sae] on SAE device.
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
        import torch  # type: ignore
        sae_device = next(self.sae.parameters()).device  # type: ignore[union-attr]
        if not isinstance(hs, torch.Tensor):
            hs = torch.tensor(hs, dtype=torch.float32, device=sae_device)
        elif hs.device != sae_device:
            hs = hs.to(sae_device)
        feats = self.sae.encode(hs.float())  # type: ignore[union-attr]
        # feats: [batch, seq, d_sae]
        start_pos = 1 if (self.config.interp.sae.skip_bos and feats.shape[1] > 1) else 0
        acts_slice = feats[0, start_pos:]
        if acts_slice.numel() == 0:
            avg_acts = feats[0].mean(dim=0)
            max_acts = feats[0].max(dim=0).values
        else:
            avg_acts = acts_slice.mean(dim=0)
            max_acts = acts_slice.max(dim=0).values
        return encoded.tokens, avg_acts, max_acts

    def get_top_features(self, role: str, prompt: PromptSpec, k: int = 50) -> TopFeaturesResult:
        tokens, avg_acts, max_acts = self._feature_acts_all(role, prompt)
        import torch  # type: ignore
        k_eff = min(int(k), int(avg_acts.numel()))
        vals, inds = torch.topk(avg_acts, k_eff)
        feats: list[FeatureActivation] = []
        for idx, val in zip(inds.tolist(), vals.tolist()):
            feats.append(FeatureActivation(feature_idx=int(idx), avg_activation=float(val), max_activation=float(max_acts[idx].item()), top_tokens=[]))
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
        _, ref_acts, _ = self._feature_acts_all(reference, prompt)
        _, cand_acts, _ = self._feature_acts_all(candidate, prompt)
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
                    expl = self.neuronpedia.pick_best_explanation(data) if data else None
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
    # Reporting
    # ----------------------------
    def _collect_trace_events(self) -> list[dict[str, Any]]:
        """Read tool transcript, filter events belonging to the current run."""
        trace_events: list[dict[str, Any]] = []
        try:
            raw_lines = self.transcript.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.debug("Failed to read tool transcript: %s", exc)
            return trace_events
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                log.debug("Skipping corrupt JSONL line in transcript")
                continue
            if self.current_run and ev.get("run_id") == self.current_run.run_id:
                trace_events.append(ev)
        return trace_events

    def _compute_per_prompt_diffs(
        self,
        reference: str,
        candidate: str,
        suite_path: str,
        k: int = 50,
    ) -> list[dict[str, Any]]:
        """Run differential_feature_analysis per prompt in a suite. Returns list of per-prompt results."""
        suite_file = _resolve_within_base(suite_path, self.base_dir, "suite_path")
        suite = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
        prompts = suite.get("prompts") or []
        per_prompt: list[dict[str, Any]] = []
        for item in prompts:
            ps = PromptSpec.model_validate(item)
            diffs = self.differential_feature_analysis(reference, candidate, ps, k=k)
            per_prompt.append({
                "id": item.get("id"),
                "top_diffs": [d.model_dump(mode="json") for d in diffs.top_diffs[:10]],
            })
        return per_prompt

    def _build_interp_evidence(self, per_prompt_diffs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Annotate per-prompt diffs with Neuronpedia metadata and decoder neighbors."""
        if not self.sae or not self.config.interp.neuronpedia.enabled:
            return None

        interp_evidence: dict[str, Any] = {
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

        if self._decoder_index is None:
            try:
                dec = self.sae.decoder_vectors()  # type: ignore[union-attr]
                self._decoder_index = DecoderCosineIndex.from_decoder(dec)
            except (RuntimeError, AttributeError) as exc:
                log.warning("Failed to build decoder index: %s", exc)
                self._decoder_index = None

        max_diffs_per_prompt = 3
        n_neighbors = 3
        neighbor_search_k = 200

        for ps in per_prompt_diffs:
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
                    except (RuntimeError, IndexError) as exc:
                        log.debug("Neighbor lookup failed for feature %d: %s", f_idx, exc)
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
                    "top_diffs_annotated": annotated,
                }
            )

        return interp_evidence

    def _render_report_markdown(
        self,
        *,
        title: str,
        reference: str,
        candidate: str,
        decision: dict[str, Any],
        suite_results: dict[str, Any],
        trace_events: list[dict[str, Any]],
        interp_evidence: Optional[dict[str, Any]],
        run_dir: Path,
    ) -> str:
        """Build the full markdown report from structured data."""
        tool_rationale: dict[str, str] = {
            "begin_run": "Create an isolated run directory for artifacts and logs.",
            "view_training_data_sample": "Sample the training dataset to look for suspicious patterns or triggers.",
            "grep_training_data": "Search the training dataset for keywords/patterns related to the suspected behavior.",
            "run_prompt_suite": "Compare base vs candidate behavior on a standard prompt suite.",
            "differential_feature_analysis": "Find SAE features with the largest activation shifts between base and candidate.",
            "write_audit_report": "Compile artifacts into a human-readable report for review.",
        }

        md_lines: list[str] = []
        md_lines.append(f"# {title}")
        md_lines.append("")
        md_lines.append(f"- Session: `{self.session_id}`")
        md_lines.append(f"- Run: `{self.current_run.run_id}` ({self.current_run.run_name})")  # type: ignore[union-attr]
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
                        except (TypeError, ValueError):
                            pass
                    neigh0 = (td.get("neighbors") or [None])[0]
                    if isinstance(neigh0, dict):
                        try:
                            best_cos.append(float(neigh0.get("cosine")))
                        except (TypeError, ValueError):
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
                import statistics
                md_lines.append(
                    f"- Best-neighbor cosine: median `{statistics.median(best_cos)}`; min `{min(best_cos)}`"
                )
            md_lines.append("")

            for per in interp_evidence.get("per_prompt", []):
                md_lines.append(f"### {per.get('id')}")
                for item in per.get("top_diffs_annotated", []):
                    d = item.get("diff", {})
                    f = item.get("feature", {})
                    f_idx = f.get("feature_idx")
                    md_lines.append(f"#### Feature `{f_idx}` (diff={d.get('diff')}, {d.get('direction')})")
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
                        except (TypeError, ValueError):
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
                        for nb in neigh:
                            npos = [x.get("value") for x in (nb.get("top_pos_logits") or []) if isinstance(x, dict)]
                            expl = str(nb.get("explanation") or "").strip()
                            expl_part = f" expl: {expl}" if expl else ""
                            md_lines.append(
                                f"  - `{nb.get('feature_idx')}` cos=`{nb.get('cosine')}`{expl_part} top+: {', '.join(str(x) for x in npos[:3])}"
                            )
                md_lines.append("")

        if not self.sae:
            md_lines.append("## Note")
            md_lines.append("")
            md_lines.append("_SAE is disabled in this config; SAE-based interpretability is not computed._")

        return "\n".join(md_lines)

    def write_audit_report(
        self,
        *,
        title: str,
        reference: str,
        candidate: str,
        suite_path: str,
        gen: GenerationParams,
    ) -> dict[str, Any]:
        """
        Produce a Markdown report + decision.json in the current run directory.
        """
        if not self.current_run:
            raise RuntimeError("No active run. Call begin_run() first.")

        run_dir = self.current_run.run_dir
        _ensure_dir(run_dir)

        # Run behavior comparison
        suite_results = self.run_prompt_suite(suite_path, models=[reference, candidate], gen=gen)

        # Run per-prompt differential feature analysis (if SAE enabled)
        per_prompt_diffs: list[dict[str, Any]] = []
        if self.sae:
            per_prompt_diffs = self._compute_per_prompt_diffs(reference, candidate, suite_path)

        # Collect trace events
        trace_events = self._collect_trace_events()

        # Build interp evidence (if SAE + Neuronpedia enabled)
        interp_evidence = self._build_interp_evidence(per_prompt_diffs)

        # Write decision.json
        decision = {
            "session_id": self.session_id,
            "run_id": self.current_run.run_id,
            "title": title,
            "reference_model": reference,
            "candidate_model": candidate,
            "suite_path": suite_results["suite_path"],
            "sae_enabled": bool(self.sae),
            "created_at": _utcnow().isoformat() + "Z",
        }
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        (run_dir / "suite_results.json").write_text(json.dumps(suite_results, indent=2), encoding="utf-8")

        if interp_evidence:
            (run_dir / "interp_evidence.json").write_text(json.dumps(interp_evidence, indent=2), encoding="utf-8")

        # Render markdown report
        md = self._render_report_markdown(
            title=title,
            reference=reference,
            candidate=candidate,
            decision=decision,
            suite_results=suite_results,
            trace_events=trace_events,
            interp_evidence=interp_evidence,
            run_dir=run_dir,
        )
        report_path = run_dir / "report.md"
        report_path.write_text(md, encoding="utf-8")

        return {
            "run_dir": str(run_dir),
            "report_path": str(report_path),
            "decision_path": str(run_dir / "decision.json"),
            "suite_results_path": str(run_dir / "suite_results.json"),
        }

    def close(self) -> None:
        if self.neuronpedia:
            try:
                self.neuronpedia.close()
            except Exception:
                log.debug("Error closing Neuronpedia client", exc_info=True)
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
            log.debug("Error clearing CUDA cache", exc_info=True)


# ----------------------------
# Public session factory
# ----------------------------
def create_session_from_config_path(config_path: str, profile: str) -> AuditSession:
    base_dir = _resolve_base_dir()
    resolved = _resolve_within_base(config_path, base_dir, "config_path")
    cfg = load_config(resolved)
    return AuditSession(cfg, profile=profile, base_dir=base_dir)
