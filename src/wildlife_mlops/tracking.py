"""MLflow tracking for completed local training runs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


class MLflowTrackingError(RuntimeError):
    """Raised when a completed training run cannot be recorded in MLflow."""


def terminal_metrics(results_path: Path) -> dict[str, float]:
    """Return numeric metrics from the final row of an Ultralytics results CSV."""
    try:
        with results_path.open(encoding="utf-8", newline="") as results_file:
            rows = list(csv.DictReader(results_file))
    except OSError as error:
        raise MLflowTrackingError(f"Unable to read training results: {results_path}") from error
    if not rows:
        return {}

    metrics: dict[str, float] = {}
    for name, raw_value in rows[-1].items():
        if name is None or raw_value is None or name.strip().lower() == "epoch":
            continue
        try:
            metrics[f"terminal/{_safe_metric_name(name)}"] = float(raw_value.strip())
        except ValueError:
            continue
    return metrics


def _safe_metric_name(name: str) -> str:
    """Return an MLflow-compatible, stable metric name."""
    return re.sub(r"[^A-Za-z0-9_. /:-]", "_", name.strip())


def write_smoke_run_report(run_dir: Path, config: dict[str, object]) -> Path:
    """Write a small, human-readable summary of the completed smoke run."""
    report_path = run_dir / "run-report.json"
    report = {
        "config": config,
        "terminal_metrics": terminal_metrics(run_dir / "results.csv"),
        "artifacts": [
            "resolved-config.json",
            "environment-summary.json",
            "results.csv",
            "weights/best.pt",
            "weights/last.pt",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def log_smoke_run(
    run_dir: Path,
    config: dict[str, object],
    tracking_uri: str,
    experiment_name: str,
    mlflow_client: Any | None = None,
) -> str:
    """Log a finished smoke run's metadata, terminal metrics, and local artifacts."""
    if not tracking_uri:
        raise MLflowTrackingError("MLflow tracking URI must not be empty")
    if not experiment_name:
        raise MLflowTrackingError("MLflow experiment name must not be empty")
    try:
        if mlflow_client is None:
            import mlflow

            mlflow_client = mlflow
        mlflow_client.set_tracking_uri(tracking_uri)
        mlflow_client.set_experiment(experiment_name)
        with mlflow_client.start_run(run_name=run_dir.name) as active_run:
            mlflow_client.log_params({name: str(value) for name, value in config.items()})
            metrics = terminal_metrics(run_dir / "results.csv")
            if metrics:
                mlflow_client.log_metrics(metrics)
            mlflow_client.log_artifacts(str(run_dir), artifact_path="training-output")
            return str(active_run.info.run_id)
    except MLflowTrackingError:
        raise
    except Exception as error:
        raise MLflowTrackingError(
            f"Unable to log smoke run to MLflow at {tracking_uri}: {error}"
        ) from error
