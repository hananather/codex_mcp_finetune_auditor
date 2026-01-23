"""
Template: run real audits via the MCP server with and without SAE.

Set these environment variables before running:
  - AUDIT_BASE_DIR: base path used to resolve config/suite/dataset paths (must contain all paths)
  - AUDIT_REPO_ROOT: path to this repo (where prompt_suites/ lives)
  - AUDIT_RESULTS_DIR: where to write runs (default: ./runs in repo)
  - BASE_MODEL_ID: HF model id for the base model (e.g., google/gemma-3-1b-it)
  - FINETUNED_MODEL_PATH: local path to the fine-tuned model
  - TRAINING_JSONL: path to a JSONL dataset used for triage tools
  - SAE_WEIGHTS_PATH (optional): local path to SAE weights
  - SAE_REPO_ID + SAE_FILENAME (optional): if weights not local, HF hub location
  - REAL_AUDITS_WITH_SAE_N (optional): report count for SAE runs (default: 10)
  - REAL_AUDITS_NO_SAE_N (optional): report count for no-SAE runs (default: 10)
  - REAL_AUDITS_SKIP_ARCHIVE (optional): skip archiving previous runs (default: false)
  - FT_AUDIT_SCORE_METHOD (optional): scoring method (default: abs_diff_topk)
  - NEURONPEDIA_EXPLANATIONS_MODE (optional): cache_only or remote_fallback (default: remote_fallback)

Optional Neuronpedia metadata (for SAE runs only):
  - NEURONPEDIA_BASE_URL (default: http://127.0.0.1:3000)
  - NEURONPEDIA_MODEL_ID (e.g., gemma-3-1b-it)
  - NEURONPEDIA_SOURCE (e.g., 22-gemmascope-2-res-16k)
  - NEURONPEDIA_DB_MANAGER (optional): path to a Neuronpedia `db_manager.py` (to auto-start a local server)
  - NEURONPEDIA_DB_PATH (optional): path to the Neuronpedia sqlite db for `db_manager.py serve --db ...`

Optional HF cache vars:
  - HF_HOME
  - TRANSFORMERS_CACHE
  - HUGGINGFACE_HUB_CACHE
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import requests
import yaml

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

GEN = {"max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0, "do_sample": False}
N_REPORTS_WITH_SAE = int(os.getenv("REAL_AUDITS_WITH_SAE_N", "10"))
N_REPORTS_NO_SAE = int(os.getenv("REAL_AUDITS_NO_SAE_N", "10"))
SKIP_ARCHIVE = os.getenv("REAL_AUDITS_SKIP_ARCHIVE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
SCORE_METHOD = (os.getenv("FT_AUDIT_SCORE_METHOD") or "abs_diff_topk").strip() or "abs_diff_topk"


def _env(name: str, required: bool = True) -> str:
    val = os.getenv(name)
    if required and not val:
        raise SystemExit(f"Missing required env var: {name}")
    return val or ""


def _resolve_results_dir(repo_root: Path) -> Path:
    results_dir = os.getenv("AUDIT_RESULTS_DIR", "./runs")
    p = Path(results_dir).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _archive_existing_runs(results_dir: Path) -> None:
    if not results_dir.exists():
        return
    archive_root = results_dir / "_archive"
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = archive_root / ts
    dest.mkdir(parents=True, exist_ok=True)

    moved_any = False
    for child in results_dir.iterdir():
        if child.name == "_archive":
            continue
        child.rename(dest / child.name)
        moved_any = True

    if moved_any:
        print(f"Archived previous runs to {dest}")


def _health_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/api/health"


def _wait_for_neuronpedia(base_url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(_health_url(base_url), timeout=1.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.2)
    raise SystemExit(f"Neuronpedia health check failed at {_health_url(base_url)}: {last_err}")


def _maybe_start_neuronpedia_server() -> Any:
    """
    If Neuronpedia env vars are set and the server is not running yet, optionally
    start it via the provided db_manager.py.

    Returns a subprocess.Popen handle or None.
    """
    base_url = os.getenv("NEURONPEDIA_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
    model_id = (os.getenv("NEURONPEDIA_MODEL_ID") or "").strip()
    source = (os.getenv("NEURONPEDIA_SOURCE") or "").strip()
    if not (model_id and source):
        return None

    try:
        r = requests.get(_health_url(base_url), timeout=1.0)
        if r.status_code == 200:
            return None
    except Exception:
        pass

    db_mgr = os.getenv("NEURONPEDIA_DB_MANAGER")
    db_path = os.getenv("NEURONPEDIA_DB_PATH")
    if not (db_mgr and db_path):
        raise SystemExit(
            "Neuronpedia server is not reachable, and NEURONPEDIA_DB_MANAGER/NEURONPEDIA_DB_PATH are not set."
        )

    from urllib.parse import urlparse
    import subprocess

    u = urlparse(base_url)
    host = u.hostname or "127.0.0.1"
    port = u.port or 3000

    log_path = Path(os.getenv("NEURONPEDIA_LOG", "./runs/_neuronpedia.log")).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("ab")
    explanations_mode = os.getenv("NEURONPEDIA_EXPLANATIONS_MODE", "remote_fallback").strip()
    cmd = [sys.executable, db_mgr, "serve", "--db", db_path, "--host", host, "--port", str(port)]
    if explanations_mode:
        cmd.extend(["--explanations-mode", explanations_mode])
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=log_f,
        close_fds=True,
    )
    _wait_for_neuronpedia(base_url, timeout_s=20.0)
    print(f"Started Neuronpedia server pid={proc.pid} at {base_url} (log: {log_path})")
    return proc


def _write_config(path: Path, with_sae: bool) -> None:
    results_dir = os.getenv("AUDIT_RESULTS_DIR", "./runs")

    interp = {
        "sae": {
            "enabled": bool(with_sae),
            "layer": int(os.getenv("SAE_LAYER", "22")),
            "module_path_template": os.getenv("SAE_MODULE_PATH_TEMPLATE", "model.layers.{layer}"),
            "output_selector": os.getenv("SAE_OUTPUT_SELECTOR", "first"),
        },
        "neuronpedia": {"enabled": False},
    }

    if with_sae:
        sae_path = os.getenv("SAE_WEIGHTS_PATH")
        if sae_path:
            interp["sae"]["weights"] = {"source": "local", "path": sae_path}
        else:
            repo_id = _env("SAE_REPO_ID", required=True)
            filename = _env("SAE_FILENAME", required=True)
            interp["sae"]["weights"] = {"source": "hf_hub", "repo_id": repo_id, "filename": filename}

        # Neuronpedia integration is optional but recommended for interpretable feature metadata.
        np_model_id = (os.getenv("NEURONPEDIA_MODEL_ID") or "").strip()
        np_source = (os.getenv("NEURONPEDIA_SOURCE") or "").strip()
        if np_model_id and np_source:
            interp["neuronpedia"] = {
                "enabled": True,
                "base_url": os.getenv("NEURONPEDIA_BASE_URL", "http://127.0.0.1:3000").rstrip("/"),
                "model_id": np_model_id,
                "source": np_source,
            }

    cfg = {
        "project": {
            "name": f"real-audit-{'with' if with_sae else 'no'}-sae",
            "results_dir": results_dir,
            "seed": 0,
            "score_method": SCORE_METHOD,
        },
        "backend": {"type": "hf"},
        "models": {
            "base": {"id_or_path": _env("BASE_MODEL_ID"), "trust_remote_code": False},
            "benign": {"id_or_path": _env("BASE_MODEL_ID"), "trust_remote_code": False},
            "adversarial": {"id_or_path": _env("FINETUNED_MODEL_PATH"), "trust_remote_code": False},
        },
        "dataset": {"training_jsonl": _env("TRAINING_JSONL")},
        "interp": interp,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _load_first_prompt(prompt_suite: Path) -> dict[str, Any] | None:
    suite = yaml.safe_load(prompt_suite.read_text(encoding="utf-8"))
    prompts = suite.get("prompts") or []
    return prompts[0] if prompts else None


async def _call(session: ClientSession, name: str, args: dict) -> Any:
    result = await session.call_tool(name, args)
    if result.isError:
        raise RuntimeError(f"Tool {name} error: {result.content}")
    return json.loads(result.content[0].text) if result.content else None


async def _run_reports(config_path: Path, tag: str, sae_enabled: bool, n_reports: int) -> None:
    repo_root = Path(_env("AUDIT_REPO_ROOT"))
    prompt_suite = repo_root / "prompt_suites" / "minimal.yaml"
    prompt = _load_first_prompt(prompt_suite)

    env = dict(os.environ)
    env["FT_AUDIT_BASE_DIR"] = _env("AUDIT_BASE_DIR")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "codex_mcp_auditor.server", "--profile", "full"],
        cwd=str(repo_root),
        env=env,
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream, read_timeout_seconds=timedelta(minutes=20)) as session:
            await session.initialize()
            sess_info = await _call(session, "create_audit_session", {"config_path": str(config_path)})
            session_id = sess_info["session_id"]

            for i in range(n_reports):
                run_name = f"{tag}-{i+1}"
                await _call(session, "begin_run", {"session_id": session_id, "run_name": run_name})

                await _call(session, "view_training_data_sample", {"session_id": session_id, "k": 3, "seed": i})
                await _call(session, "grep_training_data", {"session_id": session_id, "pattern": "needle", "max_matches": 3})
                await _call(session, "run_prompt_suite", {
                    "session_id": session_id,
                    "suite_path": str(prompt_suite),
                    "models": ["base", "adversarial"],
                    "gen": GEN,
                })

                if sae_enabled and prompt:
                    await _call(session, "differential_feature_analysis", {
                        "session_id": session_id,
                        "reference": "base",
                        "candidate": "adversarial",
                        "prompt": prompt,
                        "k": 20,
                    })

                title = f"{tag} report {i+1} (adversarial vs base)"
                await _call(session, "write_audit_report", {
                    "session_id": session_id,
                    "title": title,
                    "reference": "base",
                    "candidate": "adversarial",
                    "suite_path": str(prompt_suite),
                    "gen": GEN,
                })

            await _call(session, "close_audit_session", {"session_id": session_id})


def main() -> None:
    repo_root = Path(_env("AUDIT_REPO_ROOT"))
    results_dir = _resolve_results_dir(repo_root)
    if not SKIP_ARCHIVE:
        _archive_existing_runs(results_dir)

    neuronpedia_proc = _maybe_start_neuronpedia_server()
    local_config_dir = repo_root / ".local_configs"
    with_sae_cfg = local_config_dir / "real_with_sae.yaml"
    no_sae_cfg = local_config_dir / "real_no_sae.yaml"

    try:
        _write_config(with_sae_cfg, with_sae=True)
        _write_config(no_sae_cfg, with_sae=False)

        print(f"Running {N_REPORTS_WITH_SAE} reports with SAE enabled...")
        asyncio.run(_run_reports(with_sae_cfg, "real-with-sae", True, N_REPORTS_WITH_SAE))

        print(f"Running {N_REPORTS_NO_SAE} reports without SAE...")
        asyncio.run(_run_reports(no_sae_cfg, "real-no-sae", False, N_REPORTS_NO_SAE))
    finally:
        if neuronpedia_proc is not None:
            try:
                neuronpedia_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
