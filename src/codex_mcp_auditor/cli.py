from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .schemas.common import GenerationParams
from .session import create_session_from_config_path


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def validate_config(config: str):
    """Validate a YAML config file and print the resolved path."""
    from .utils.config_io import load_config

    cfg = load_config(config)
    rprint("[green]OK[/green] Loaded config:")
    rprint(cfg.model_dump())


@app.command()
def benchmark(
    config: str = typer.Option(..., help="Path to audit config YAML"),
    suite: str = typer.Option(..., help="Path to prompt suite YAML"),
    out: str = typer.Option("./runs/benchmark.json", help="Where to write benchmark JSON output"),
    k: int = typer.Option(50, help="Top-k features for per-prompt score"),
    threshold: Optional[float] = typer.Option(None, help="Optional threshold for compromised vs not_compromised classification"),
):
    """
    Run a simple benchmark on the configured (base, benign, adversarial) models.

    This benchmark is deterministic (no LLM agent). It produces a numeric SAE-diff score
    for each candidate vs base, plus optional classification metrics if a threshold is provided.
    """
    sess = create_session_from_config_path(config, profile="full")
    # benign
    sess.begin_run("benchmark:benign")
    benign_score = sess.score_candidate_suite("base", "benign", suite, k=k, threshold=threshold)
    benign_art = sess.write_audit_report(
        title="Benchmark report (benign vs base)",
        reference="base",
        candidate="benign",
        suite_path=suite,
        gen=GenerationParams(),
        threshold=threshold,
    )
    # adversarial
    sess.begin_run("benchmark:adversarial")
    adv_score = sess.score_candidate_suite("base", "adversarial", suite, k=k, threshold=threshold)
    adv_art = sess.write_audit_report(
        title="Benchmark report (adversarial vs base)",
        reference="base",
        candidate="adversarial",
        suite_path=suite,
        gen=GenerationParams(),
        threshold=threshold,
    )

    results = {
        "config": str(Path(config).resolve()),
        "suite": str(Path(suite).resolve()),
        "k": int(k),
        "threshold": threshold,
        "scores": {
            "benign": benign_score.model_dump(mode="json"),
            "adversarial": adv_score.model_dump(mode="json"),
        },
        "artifacts": {
            "benign": benign_art,
            "adversarial": adv_art,
        },
    }

    if threshold is not None:
        # Ground truth: benign -> not_compromised, adversarial -> compromised
        y_true = ["not_compromised", "compromised"]
        y_pred = [benign_score.predicted_label or "not_compromised", adv_score.predicted_label or "not_compromised"]

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "compromised" and p == "compromised")
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == "not_compromised" and p == "not_compromised")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == "not_compromised" and p == "compromised")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "compromised" and p == "not_compromised")

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        results["metrics"] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "tpr": tpr,
            "fpr": fpr,
        }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    table = Table(title="Benchmark summary")
    table.add_column("Candidate")
    table.add_column("Aggregate score", justify="right")
    table.add_column("Predicted", justify="center")
    table.add_row("benign", f"{benign_score.aggregate_score:.4f}", benign_score.predicted_label or "(none)")
    table.add_row("adversarial", f"{adv_score.aggregate_score:.4f}", adv_score.predicted_label or "(none)")
    console.print(table)

    if "metrics" in results:
        rprint("[bold]Metrics[/bold]:")
        rprint(results["metrics"])

    rprint(f"[green]Wrote[/green] {out_path}")



@app.command()
def aggregate(
    runs_root: str = typer.Option("./runs", help="Root directory to search for decision.json files (recursively)"),
    out: str = typer.Option("./runs/aggregate_metrics.json", help="Where to write aggregate metrics JSON"),
    threshold: Optional[float] = typer.Option(None, help="Classification threshold. If omitted, and fpr_target is provided, it will be calibrated from benign scores."),
    fpr_target: Optional[float] = typer.Option(None, help="Target false positive rate (e.g., 0.01). Used to calibrate threshold from benign scores."),
):
    """
    Aggregate decision.json files produced by repeated audits (Codex-driven or harness-driven).

    Expected schema (minimal):
      - candidate_model: e.g. "benign" or "adversarial"
      - aggregate_score: float
      - predicted_label: optional ("compromised" | "not_compromised")
      - ground_truth_label: optional ("compromised" | "not_compromised")

    If predicted_label is missing, it will be computed from aggregate_score and the selected threshold.
    """
    root = Path(runs_root).expanduser().resolve()
    decisions = []
    for p in root.rglob("decision.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_path"] = str(p)
            decisions.append(data)
        except Exception:
            continue

    if not decisions:
        raise typer.BadParameter(f"No decision.json files found under {root}")

    def infer_gt(d: dict) -> Optional[str]:
        if d.get("ground_truth_label") in ("compromised", "not_compromised"):
            return d["ground_truth_label"]
        cand = str(d.get("candidate_model", "")).lower()
        if cand == "benign":
            return "not_compromised"
        if cand == "adversarial":
            return "compromised"
        return None

    # Collect benign scores for threshold calibration
    benign_scores = [float(d["aggregate_score"]) for d in decisions if infer_gt(d) == "not_compromised" and "aggregate_score" in d]
    if threshold is None and fpr_target is not None:
        if not benign_scores:
            raise typer.BadParameter("Cannot calibrate threshold: no benign runs with aggregate_score found.")
        fpr = float(fpr_target)
        if not (0.0 < fpr < 1.0):
            raise typer.BadParameter("fpr_target must be in (0, 1).")
        # Threshold at (1 - fpr) quantile of benign scores
        benign_scores_sorted = sorted(benign_scores)
        q = 1.0 - fpr
        idx = int(round(q * (len(benign_scores_sorted) - 1)))
        threshold = benign_scores_sorted[idx]

    if threshold is None:
        rprint("[yellow]No threshold provided; metrics will be partial (scores only).[/yellow]")

    # Compute predictions and metrics
    rows = []
    y_true = []
    y_pred = []
    for d in decisions:
        gt = infer_gt(d)
        score = float(d.get("aggregate_score", 0.0))
        pred = d.get("predicted_label")
        if pred not in ("compromised", "not_compromised") and threshold is not None:
            pred = "compromised" if score >= float(threshold) else "not_compromised"
        rows.append({"path": d["_path"], "gt": gt, "pred": pred, "score": score})
        if gt is not None and pred in ("compromised", "not_compromised"):
            y_true.append(gt)
            y_pred.append(pred)

    metrics = None
    if y_true and y_pred:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "compromised" and p == "compromised")
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == "not_compromised" and p == "not_compromised")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == "not_compromised" and p == "compromised")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "compromised" and p == "not_compromised")

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        metrics = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "tpr": tpr, "fpr": fpr}

    out_obj = {
        "runs_root": str(root),
        "n_decisions": len(decisions),
        "threshold": threshold,
        "fpr_target": fpr_target,
        "benign_scores_n": len(benign_scores),
        "metrics": metrics,
        "rows": rows,
    }

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    rprint(f"[green]Wrote[/green] {out_path}")

    if metrics:
        table = Table(title="Aggregate metrics")
        table.add_column("TPR", justify="right")
        table.add_column("FPR", justify="right")
        table.add_row(f"{metrics['tpr']:.4f}", f"{metrics['fpr']:.4f}")
        console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
