"""MLflow tracking for completed local training runs."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class MLflowTrackingError(RuntimeError):
    """Raised when a completed training run cannot be recorded in MLflow."""


@dataclass
class MLflowRunLogger:
    """One explicit MLflow run that receives completed and per-epoch evidence."""

    client: Any
    tracking_uri: str
    experiment_name: str
    run_name: str
    parameters: dict[str, object]
    _active_run: Any | None = None
    _logged_epoch_steps: set[int] = field(default_factory=set)

    def __enter__(self) -> MLflowRunLogger:
        """Start the configured MLflow run and record its resolved parameters."""
        try:
            self.client.set_tracking_uri(self.tracking_uri)
            self.client.set_experiment(self.experiment_name)
            self._active_run = self.client.start_run(run_name=self.run_name)
            self._active_run.__enter__()
            self.client.log_params({name: str(value) for name, value in self.parameters.items()})
        except Exception as error:
            raise MLflowTrackingError(
                f"Unable to start MLflow run at {self.tracking_uri}: {error}"
            ) from error
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the MLflow run after training succeeds or fails."""
        if self._active_run is not None:
            self._active_run.__exit__(exc_type, exc_value, traceback)

    @property
    def run_id(self) -> str:
        """Return the immutable run identifier after the run has started."""
        if self._active_run is None:
            raise MLflowTrackingError("MLflow run has not started")
        return str(self._active_run.info.run_id)

    def log_epoch(self, trainer: Any) -> None:
        """Record aggregate training and validation metrics for one completed epoch."""
        step = _epoch_step(trainer)
        if step in self._logged_epoch_steps:
            return
        metrics = epoch_metrics(trainer, step)
        try:
            self.client.log_metrics(metrics, step=step)
            self._logged_epoch_steps.add(step)
        except Exception as error:
            raise MLflowTrackingError(
                f"Unable to log epoch {step + 1} metrics to MLflow: {error}"
            ) from error

    def log_terminal_metrics(self, results_path: Path) -> None:
        """Record the final CSV values as separate terminal metrics."""
        metrics = terminal_metrics(results_path)
        if not metrics:
            return
        try:
            self.client.log_metrics(metrics)
        except Exception as error:
            raise MLflowTrackingError(
                f"Unable to log terminal metrics to MLflow: {error}"
            ) from error

    def log_artifacts(self, run_dir: Path) -> None:
        """Copy the completed local training output into the MLflow artifact store."""
        try:
            self.client.log_artifacts(str(run_dir), artifact_path="training-output")
        except Exception as error:
            raise MLflowTrackingError(
                f"Unable to log training artifacts to MLflow: {error}"
            ) from error


def start_smoke_run(
    run_dir: Path,
    config: dict[str, object],
    tracking_uri: str,
    experiment_name: str,
    mlflow_client: Any | None = None,
) -> MLflowRunLogger:
    """Prepare the explicit MLflow run used by one smoke-training command."""
    if not tracking_uri:
        raise MLflowTrackingError("MLflow tracking URI must not be empty")
    if not experiment_name:
        raise MLflowTrackingError("MLflow experiment name must not be empty")
    if mlflow_client is None:
        try:
            import mlflow
        except ImportError as error:
            raise MLflowTrackingError("MLflow is not installed") from error
        mlflow_client = mlflow
    return MLflowRunLogger(mlflow_client, tracking_uri, experiment_name, run_dir.name, config)


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


def epoch_metrics(trainer: Any, step: int) -> dict[str, float]:
    """Extract stable aggregate metric names from one Ultralytics training epoch."""
    metrics: dict[str, float] = {"epoch": float(step + 1)}
    metric_sources: tuple[tuple[str, dict[str, Any]], ...] = (
        ("train", _training_losses(trainer)),
        ("learning_rate", _numeric_mapping(getattr(trainer, "lr", {}))),
        ("validation", _numeric_mapping(getattr(trainer, "metrics", {}))),
    )
    for prefix, values in metric_sources:
        for name, value in values.items():
            metrics[f"{prefix}/{_stable_epoch_metric_name(prefix, name)}"] = value
    return metrics


def _epoch_step(trainer: Any) -> int:
    """Read the zero-based epoch step exposed by Ultralytics callbacks."""
    epoch = getattr(trainer, "epoch", None)
    if not isinstance(epoch, int) or epoch < 0:
        raise MLflowTrackingError("Training callback did not provide a non-negative epoch number")
    return epoch


def _training_losses(trainer: Any) -> dict[str, float]:
    """Read aggregate training losses without recording class-specific metrics."""
    loss_items = getattr(trainer, "label_loss_items", None)
    losses = getattr(trainer, "tloss", None)
    if not callable(loss_items) or losses is None:
        return {}
    return _numeric_mapping(loss_items(losses, prefix="train"))


def _numeric_mapping(values: object) -> dict[str, float]:
    """Keep only numeric values from a callback-provided metric mapping."""
    if not isinstance(values, dict):
        return {}
    metrics: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(name, str):
            continue
        try:
            metrics[name] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _stable_epoch_metric_name(prefix: str, name: str) -> str:
    """Map trainer-specific names to the public aggregate metric convention."""
    normalized = name.strip().lower().replace(" ", "")
    if prefix == "train" and normalized.startswith("train/"):
        return _safe_metric_name(name.split("/", maxsplit=1)[1]).replace("/", "_")
    if prefix == "learning_rate":
        match = re.fullmatch(r"lr/pg(\d+)", normalized)
        if match is not None:
            return f"group_{match.group(1)}"
    if prefix == "validation":
        aliases = {
            "metrics/precision(b)": "precision",
            "metrics/recall(b)": "recall",
            "metrics/map50(b)": "map50",
            "metrics/map50-95(b)": "map50_95",
        }
        if normalized in aliases:
            return aliases[normalized]
    return _safe_metric_name(name).replace("/", "_")


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
    with start_smoke_run(run_dir, config, tracking_uri, experiment_name, mlflow_client) as tracker:
        tracker.log_terminal_metrics(run_dir / "results.csv")
        tracker.log_artifacts(run_dir)
        return tracker.run_id
