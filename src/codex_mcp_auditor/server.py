import argparse
import os
from datetime import datetime
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP  # type: ignore

from .config import DEFAULT_PROFILES
from .schemas.common import (
    GenerationParams,
    HealthInfo,
    PromptSpec,
    QueryModelsResult,
    RunInfo,
    SessionInfo,
    TrainingGrepMatch,
    TrainingSample,
)
from .schemas.interp import (
    CandidateSuiteScore,
    CompareTopFeaturesResult,
    DifferentialFeatureAnalysisResult,
    FeatureActivationTrace,
    FeatureDetails,
    NearestNeighborsResult,
    TopFeaturesResult,
)
from .session import AuditSession, create_session_from_config_path


_SESSIONS: dict[str, AuditSession] = {}


def _utcnow() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def _default_config_path() -> Optional[str]:
    return os.environ.get("FT_AUDIT_CONFIG") or None


def _get_session(session_id: str) -> AuditSession:
    if session_id not in _SESSIONS:
        raise ValueError(f"Unknown session_id: {session_id}")
    return _SESSIONS[session_id]


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}

def _resolve_bind_host_port(mcp: FastMCP) -> tuple[str, int]:
    host = (os.environ.get("FASTMCP_HOST") or mcp.settings.host).strip() or "127.0.0.1"
    port_raw = (os.environ.get("FASTMCP_PORT") or "").strip()
    if not port_raw:
        return host, int(mcp.settings.port)
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError(f"Invalid FASTMCP_PORT: {port_raw!r} (expected an integer).") from exc
    return host, port


def _run_streamable_http_with_token(mcp: FastMCP, token: str, *, host: str, port: int) -> None:
    import anyio
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse

    app = mcp.streamable_http_app()

    class TokenAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):  # type: ignore[override]
            auth = request.headers.get("authorization", "")
            token_candidate = ""
            if auth.lower().startswith("bearer "):
                token_candidate = auth[7:]
            if not token_candidate:
                token_candidate = request.headers.get("x-ft-audit-token", "")
            if token_candidate != token:
                return PlainTextResponse("Unauthorized", status_code=401)
            return await call_next(request)

    app.add_middleware(TokenAuthMiddleware)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    anyio.run(server.serve)


def build_server(profile: str) -> FastMCP:
    if profile not in DEFAULT_PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Options: {list(DEFAULT_PROFILES)}")

    enabled = set(DEFAULT_PROFILES[profile].enabled_tools)

    mcp = FastMCP(
        name="ft-audit",
        instructions=(
            "Tools for auditing benign vs adversarial fine-tunes. "
            "Workflow: create_audit_session -> begin_run -> (triage + behavior + interp) -> write_audit_report."
        ),
        json_response=True,
    )

    # -------------------------
    # Always-on / meta tools
    # -------------------------
    if "health" in enabled:
        @mcp.tool()
        def health() -> HealthInfo:
            """Basic liveness + server state."""
            return HealthInfo(
                ok=True,
                server_time=_utcnow(),
                backend="mixed",
                sessions=len(_SESSIONS),
                notes=[
                    "Use create_audit_session(config_path=...) to start.",
                    "For Codex integration, configure this as a STDIO MCP server.",
                ],
            )

    # -------------------------
    # Session lifecycle
    # -------------------------
    if "create_audit_session" in enabled:
        @mcp.tool()
        def create_audit_session(config_path: Optional[str] = None) -> SessionInfo:
            """
            Create a new audit session from a YAML config file.

            If config_path is omitted, uses FT_AUDIT_CONFIG from the environment.
            """
            cp = config_path or _default_config_path()
            if not cp:
                raise ValueError("config_path is required (or set FT_AUDIT_CONFIG).")
            sess = create_session_from_config_path(cp, profile=profile)
            _SESSIONS[sess.session_id] = sess

            sess.log_tool_call("create_audit_session", {"config_path": str(cp)}, {"session_id": sess.session_id})

            return SessionInfo(
                session_id=sess.session_id,
                created_at=sess.created_at,
                profile=profile,
                artifacts_dir=str(sess.artifacts_dir),
                resolved_config_path=str(sess.resolved_config_path),
            )

    if "begin_run" in enabled:
        @mcp.tool()
        def begin_run(session_id: str, run_name: str) -> RunInfo:
            """Begin a new run under an existing session (creates a run directory)."""
            sess = _get_session(session_id)
            ctx = sess.begin_run(run_name)
            sess.log_tool_call("begin_run", {"run_name": run_name}, {"run_id": ctx.run_id, "run_dir": str(ctx.run_dir)})
            return RunInfo(run_id=ctx.run_id, run_name=ctx.run_name, started_at=ctx.started_at, run_dir=str(ctx.run_dir))

    if "close_audit_session" in enabled:
        @mcp.tool()
        def close_audit_session(session_id: str) -> dict[str, Any]:
            """Close an audit session and release its cached resources."""
            sess = _get_session(session_id)
            sess.log_tool_call("close_audit_session", {}, {"status": "closing"})
            sess.close()
            _SESSIONS.pop(session_id, None)
            return {"status": "closed", "session_id": session_id}

    # -------------------------
    # Dataset triage
    # -------------------------
    if "get_training_data_length" in enabled:
        @mcp.tool()
        def get_training_data_length(session_id: str, max_lines: int = 5_000_000) -> int:
            """Count lines in the configured training JSONL dataset (if present)."""
            sess = _get_session(session_id)
            n = sess.training_length(max_lines=int(max_lines))
            sess.log_tool_call("get_training_data_length", {"max_lines": max_lines}, {"n": n})
            return n

    if "view_training_data_sample" in enabled:
        @mcp.tool()
        def view_training_data_sample(session_id: str, k: int = 5, seed: int = 0, max_chars: int = 1200) -> list[TrainingSample]:
            """Reservoir-sample k examples from the training JSONL dataset."""
            sess = _get_session(session_id)
            out = sess.training_sample(k=int(k), seed=int(seed), max_chars=int(max_chars))
            sess.log_tool_call("view_training_data_sample", {"k": k, "seed": seed}, {"returned": len(out)})
            return out

    if "grep_training_data" in enabled:
        @mcp.tool()
        def grep_training_data(session_id: str, pattern: str, max_matches: int = 50, max_chars: int = 400) -> list[TrainingGrepMatch]:
            """Regex-search the training JSONL dataset for a pattern."""
            sess = _get_session(session_id)
            out = sess.training_grep(pattern=str(pattern), max_matches=int(max_matches), max_chars=int(max_chars))
            sess.log_tool_call("grep_training_data", {"pattern": pattern, "max_matches": max_matches}, {"returned": len(out)})
            return out

    # -------------------------
    # Model behavior
    # -------------------------
    if "query_models" in enabled:
        @mcp.tool()
        def query_models(session_id: str, prompt: PromptSpec, models: list[str], gen: GenerationParams = GenerationParams()) -> QueryModelsResult:
            """Run the same prompt through multiple models and return their outputs."""
            sess = _get_session(session_id)
            out = sess.query_models(prompt, models=models, gen=gen)
            sess.log_tool_call(
                "query_models",
                {"models": models, "gen": gen.model_dump(mode="json")},
                {"responses": [r.model for r in out.responses]},
            )
            return out

    if "run_prompt_suite" in enabled:
        @mcp.tool()
        def run_prompt_suite(session_id: str, suite_path: str, models: list[str], gen: GenerationParams = GenerationParams()) -> dict[str, Any]:
            """Run a YAML prompt suite (prompt_suites/*.yaml) across a list of models."""
            sess = _get_session(session_id)
            out = sess.run_prompt_suite(suite_path, models=models, gen=gen)
            sess.log_tool_call("run_prompt_suite", {"suite_path": suite_path, "models": models}, {"items": len(out.get("items", []))})
            return out

    # -------------------------
    # Interpretability
    # -------------------------
    if "get_top_features" in enabled:
        @mcp.tool()
        def get_top_features(session_id: str, model: str, prompt: PromptSpec, k: int = 50) -> TopFeaturesResult:
            """Get top-k SAE features by average activation for a given model on a prompt."""
            sess = _get_session(session_id)
            out = sess.get_top_features(model, prompt, k=int(k))
            sess.log_tool_call("get_top_features", {"model": model, "k": k}, {"returned": len(out.features)})
            return out

    if "compare_top_features" in enabled:
        @mcp.tool()
        def compare_top_features(session_id: str, reference: str, candidate: str, prompt: PromptSpec, k: int = 50) -> CompareTopFeaturesResult:
            """Compare top-k SAE features between a reference and candidate model."""
            sess = _get_session(session_id)
            out = sess.compare_top_features(reference, candidate, prompt, k=int(k))
            sess.log_tool_call("compare_top_features", {"reference": reference, "candidate": candidate, "k": k}, {"candidate_only": len(out.candidate_only)})
            return out

    if "differential_feature_analysis" in enabled:
        @mcp.tool()
        def differential_feature_analysis(session_id: str, reference: str, candidate: str, prompt: PromptSpec, k: int = 50) -> DifferentialFeatureAnalysisResult:
            """Top-k SAE features with largest absolute activation differences between models."""
            sess = _get_session(session_id)
            out = sess.differential_feature_analysis(reference, candidate, prompt, k=int(k))
            sess.log_tool_call("differential_feature_analysis", {"reference": reference, "candidate": candidate, "k": k}, {"returned": len(out.top_diffs)})
            return out

    if "specific_feature_activations" in enabled:
        @mcp.tool()
        def specific_feature_activations(session_id: str, model: str, prompt: PromptSpec, feature_idx: int) -> FeatureActivationTrace:
            """Per-token activation trace for a specific SAE feature."""
            sess = _get_session(session_id)
            out = sess.specific_feature_activations(model, prompt, int(feature_idx))
            sess.log_tool_call("specific_feature_activations", {"model": model, "feature_idx": feature_idx}, {"tokens": len(out.tokens)})
            return out

    if "get_feature_details" in enabled:
        @mcp.tool()
        def get_feature_details(session_id: str, feature_idx: int) -> FeatureDetails:
            """Fetch Neuronpedia feature details (if enabled in config)."""
            sess = _get_session(session_id)
            out = sess.get_feature_details(int(feature_idx))
            sess.log_tool_call("get_feature_details", {"feature_idx": feature_idx}, {"source": out.source})
            return out

    if "nearest_explained_neighbors" in enabled:
        @mcp.tool()
        def nearest_explained_neighbors(
            session_id: str,
            feature_idx: int,
            n: int = 10,
            search_k: int = 200,
            min_cos: float = 0.15,
        ) -> NearestNeighborsResult:
            """Find nearest *explained* neighbors in SAE decoder cosine space (using Neuronpedia metadata when enabled)."""
            sess = _get_session(session_id)
            out = sess.nearest_explained_neighbors(int(feature_idx), n=int(n), search_k=int(search_k), min_cos=float(min_cos))
            sess.log_tool_call(
                "nearest_explained_neighbors",
                {"feature_idx": feature_idx, "n": n, "search_k": search_k, "min_cos": min_cos},
                {"returned": len(out.neighbors)},
            )
            return out

    # -------------------------
    # Scoring / reporting
    # -------------------------
    if "score_candidate_suite" in enabled:
        @mcp.tool()
        def score_candidate_suite(session_id: str, reference: str, candidate: str, suite_path: str, k: int = 50, aggregate: str = "mean", threshold: Optional[float] = None) -> CandidateSuiteScore:
            """Compute a numeric suite-level suspicion score for a candidate model vs a reference model."""
            sess = _get_session(session_id)
            out = sess.score_candidate_suite(reference, candidate, suite_path, k=int(k), aggregate=str(aggregate), threshold=threshold)
            sess.log_tool_call("score_candidate_suite", {"reference": reference, "candidate": candidate, "suite_path": suite_path, "k": k, "aggregate": aggregate, "threshold": threshold}, {"aggregate_score": out.aggregate_score})
            return out

    if "write_audit_report" in enabled:
        @mcp.tool()
        def write_audit_report(session_id: str, title: str, reference: str, candidate: str, suite_path: str, gen: GenerationParams = GenerationParams(), threshold: Optional[float] = None) -> dict[str, Any]:
            """Write a minimal Markdown report + JSON artifacts into the current run directory."""
            sess = _get_session(session_id)
            out = sess.write_audit_report(title=title, reference=reference, candidate=candidate, suite_path=suite_path, gen=gen, threshold=threshold)
            sess.log_tool_call("write_audit_report", {"title": title, "reference": reference, "candidate": candidate, "suite_path": suite_path}, out)
            return out

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(prog="ft-audit-mcp", description="MCP server for fine-tune auditing.")
    parser.add_argument("serve", nargs="?", default="serve", help="Run the MCP server (stdio by default).")
    parser.add_argument("--profile", default=os.environ.get("FT_AUDIT_PROFILE", "full"), help="Tool profile: full | behavior_only")
    parser.add_argument("--transport", default="stdio", help='Transport: "stdio" (default) or "streamable-http"')
    args = parser.parse_args()

    mcp = build_server(profile=str(args.profile))
    if args.transport == "streamable-http":
        token = os.environ.get("FT_AUDIT_HTTP_TOKEN", "").strip()
        if not token:
            raise ValueError(
                "Refusing to start streamable-http without FT_AUDIT_HTTP_TOKEN. "
                "Set it and pass Authorization: Bearer <token> or X-FT-AUDIT-Token."
            )
        host, port = _resolve_bind_host_port(mcp)
        if not _is_loopback_host(host):
            raise ValueError(
                f"Refusing to bind streamable-http to non-loopback host ({host}). "
                "Set FASTMCP_HOST=127.0.0.1 or use stdio."
            )
        _run_streamable_http_with_token(mcp, token, host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
