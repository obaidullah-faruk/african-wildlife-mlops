"""Minimal MLflow logging for a completed local training run."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


class TrackingError(RuntimeError):
    """Raised when MLflow cannot record a training run."""


def log_training_run(run_dir: Path, tracking_uri: str, experiment_name: str) -> str:
    """Log configuration, final metrics, and output files to MLflow."""
    try:
        import mlflow
    except ImportError as error:
        raise TrackingError("MLflow is not installed") from error

    summary_path = run_dir / "run.json"
    if not summary_path.is_file():
        raise TrackingError(f"Training summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TrackingError(f"Training summary must be a JSON object: {summary_path}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    try:
        with mlflow.start_run(run_name=run_dir.name) as run:
            mlflow.log_params(_flat_values(summary.get("config")))
            dataset = _mapping(summary.get("dataset"))
            device = _mapping(summary.get("device"))
            mlflow.set_tags(
                {
                    "run.kind": str(summary.get("kind", "unknown")),
                    "dataset.archive_sha256": str(dataset.get("archive_sha256", "unknown")),
                    "dataset.dvc_pointer_sha256": str(
                        dataset.get("dvc_pointer_sha256", "not_available")
                    ),
                    "runtime.device": str(device.get("device", "unknown")),
                }
            )
            mlflow.log_metrics(_final_metrics(run_dir / "results.csv"))
            mlflow.log_artifacts(str(run_dir), artifact_path="training-output")
    except Exception as error:
        raise TrackingError(f"MLflow could not record {run_dir.name}: {error}") from error

    run_id = str(run.info.run_id)
    (run_dir / "mlflow-run.json").write_text(
        json.dumps(
            {"experiment": experiment_name, "run_id": run_id, "tracking_uri": tracking_uri},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_id


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _flat_values(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(name): str(item) for name, item in value.items()}


def _final_metrics(results_path: Path) -> dict[str, float]:
    if not results_path.is_file():
        raise TrackingError(f"Training results are missing: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    if not rows:
        return {}
    metrics: dict[str, float] = {}
    for name, value in rows[-1].items():
        if value is None:
            continue
        try:
            metrics[_metric_name(name)] = float(value.strip())
        except ValueError:
            continue
    return metrics


def _metric_name(name: str) -> str:
    """Return an MLflow-compatible form of a training metric name."""
    return re.sub(r"[^A-Za-z0-9_. /:-]", "_", name.strip())
