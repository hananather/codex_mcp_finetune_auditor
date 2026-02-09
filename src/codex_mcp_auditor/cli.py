from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .schemas.common import GenerationParams
from .session import create_session_from_config_path

log = logging.getLogger(__name__)

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
):
    """
    Run a simple benchmark on the configured (base, benign, adversarial) models.

    Produces audit reports for benign-vs-base and adversarial-vs-base.
    """
    sess = create_session_from_config_path(config, profile="full")

    # benign
    sess.begin_run("benchmark:benign")
    benign_art = sess.write_audit_report(
        title="Benchmark report (benign vs base)",
        reference="base",
        candidate="benign",
        suite_path=suite,
        gen=GenerationParams(),
    )

    # adversarial
    sess.begin_run("benchmark:adversarial")
    adv_art = sess.write_audit_report(
        title="Benchmark report (adversarial vs base)",
        reference="base",
        candidate="adversarial",
        suite_path=suite,
        gen=GenerationParams(),
    )

    results = {
        "config": str(Path(config).resolve()),
        "suite": str(Path(suite).resolve()),
        "artifacts": {
            "benign": benign_art,
            "adversarial": adv_art,
        },
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    table = Table(title="Benchmark summary")
    table.add_column("Candidate")
    table.add_column("Report path")
    table.add_row("benign", benign_art["report_path"])
    table.add_row("adversarial", adv_art["report_path"])
    console.print(table)

    rprint(f"[green]Wrote[/green] {out_path}")


def main():
    app()


if __name__ == "__main__":
    main()
