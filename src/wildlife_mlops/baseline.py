"""Baseline training, curve interpretation, and validation evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from PIL import Image, ImageDraw, ImageStat
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.validate import (
    BoundingBox,
    image_paths,
    label_path_for_image,
    parse_yolo_label,
)
from wildlife_mlops.device import DeviceSummary


class BaselineConfig(BaseModel):
    """Settings for the first full-training baseline."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    epochs: int = Field(gt=1)
    image_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    workers: int = Field(ge=0)
    seed: int
    validation_split: str = Field(min_length=1)
    confidence_threshold: float = Field(ge=0, le=1)


class BaselineError(RuntimeError):
    """Raised when a baseline run or evaluation cannot produce valid evidence."""


@dataclass(frozen=True)
class Detection:
    """One normalized detection used for validation error examples."""

    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


def load_baseline_config(config_path: Path) -> BaselineConfig:
    """Load and validate the versioned baseline configuration."""
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Baseline configuration does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"Baseline configuration root must be a mapping: {config_path}")
    try:
        return BaselineConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ValueError(f"Invalid baseline configuration in {config_path}: {error}") from error


def run_baseline_train(
    config: BaselineConfig,
    dataset_config: DatasetConfig,
    project_root: Path,
    device_summary: DeviceSummary,
    model_factory: Callable[[str], Any],
) -> Path:
    """Train the first baseline and save reproducible runtime and curve artifacts."""
    return _run_training(
        config,
        dataset_config,
        project_root,
        device_summary,
        model_factory,
        run_prefix="baseline",
    )


def _run_training(
    config: BaselineConfig,
    dataset_config: DatasetConfig,
    project_root: Path,
    device_summary: DeviceSummary,
    model_factory: Callable[[str], Any],
    run_prefix: str,
) -> Path:
    """Train one full-data configuration and save its evidence in a unique directory."""
    model_path = project_root / config.model_path
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise BaselineError(f"Pretrained model is missing or empty: {model_path}")
    _validate_validation_split(config.validation_split, dataset_config)
    run_dir = _create_run_directory(project_root, run_prefix)
    data_path = _write_dataset_config(
        run_dir,
        project_root / dataset_config.dataset_root,
        dataset_config.class_names,
        config.validation_split,
    )
    _write_json(run_dir / "resolved-config.json", {"config": config.model_dump(mode="json")})

    monitor = PeakMemoryMonitor(device_summary.device)
    model = model_factory(str(model_path))
    _attach_memory_callbacks(model, monitor)
    monitor.start()
    started_at = time.perf_counter()
    model.train(
        data=str(data_path),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        workers=config.workers,
        device=device_summary.device,
        seed=config.seed,
        deterministic=True,
        pretrained=True,
        val=True,
        split=config.validation_split,
        save=True,
        plots=True,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
    )
    monitor.sample()
    _write_json(
        run_dir / "runtime-summary.json",
        {
            "device": device_summary.as_dict(),
            "wall_time_seconds": time.perf_counter() - started_at,
            "peak_memory": monitor.as_dict(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "thermal_monitoring": {
                "method": "manual",
                "instruction": "Monitor macOS Activity Monitor during baseline training.",
            },
        },
    )
    _require_artifacts(
        run_dir,
        (
            "weights/best.pt",
            "weights/last.pt",
            "args.yaml",
            "resolved-config.json",
            "runtime-summary.json",
            "results.csv",
            "results.png",
        ),
    )
    analyze_training_curves(run_dir)
    return run_dir


def analyze_training_curves(run_dir: Path) -> Path:
    """Record loss and validation-quality trends from a baseline results CSV."""
    results_path = run_dir / "results.csv"
    rows = _read_results_csv(results_path)
    training_losses = [_loss_total(row, "train") for row in rows]
    validation_losses = [_loss_total(row, "val") for row in rows]
    validation_quality = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    peak_index = max(range(len(validation_quality)), key=validation_quality.__getitem__)
    trend = _classify_validation_trend(training_losses, validation_quality)
    analysis_path = run_dir / "curve-analysis.json"
    _write_json(
        analysis_path,
        {
            "training_loss_by_epoch": training_losses,
            "validation_loss_by_epoch": validation_losses,
            "validation_map50_95_by_epoch": validation_quality,
            "training_loss_decreased": training_losses[-1] < training_losses[0],
            "validation_quality_trend": trend,
            "peak_validation_epoch": peak_index + 1,
            "selected_checkpoint_rule": {
                "checkpoint": "weights/best.pt",
                "metric": "validation mAP50-95",
                "direction": "maximize",
                "reason": "Select the checkpoint with the strongest validation quality, "
                "not the final epoch.",
            },
            "curve_plot": "results.png",
        },
    )
    return analysis_path


def evaluate_baseline(
    run_dir: Path,
    project_root: Path,
    dataset_config: DatasetConfig,
    device_summary: DeviceSummary,
    model_factory: Callable[[str], Any],
) -> Path:
    """Evaluate one pinned baseline on validation data and save error-analysis artifacts."""
    run_dir = _validated_run_directory(run_dir, project_root)
    config = _load_saved_config(run_dir / "resolved-config.json")
    _validate_validation_split(config.validation_split, dataset_config)
    best_model_path = run_dir / "weights" / "best.pt"
    _require_artifacts(run_dir, ("dataset.yaml", "weights/best.pt"))
    evaluation_dir = _create_run_directory(run_dir, "validation")
    return _evaluate_model(
        run_dir,
        best_model_path,
        run_dir / "dataset.yaml",
        config.validation_split,
        config,
        dataset_config,
        project_root,
        device_summary,
        model_factory,
        evaluation_dir,
    )


def _evaluate_model(
    run_dir: Path,
    best_model_path: Path,
    data_path: Path,
    split: str,
    config: BaselineConfig,
    dataset_config: DatasetConfig,
    project_root: Path,
    device_summary: DeviceSummary,
    model_factory: Callable[[str], Any],
    evaluation_dir: Path,
    release_artifact: str | None = None,
) -> Path:
    """Run the shared evaluation procedure for a validation or sealed test split."""
    model = model_factory(str(best_model_path))
    metrics = model.val(
        data=str(data_path),
        split="val",
        imgsz=config.image_size,
        batch=config.batch_size,
        workers=config.workers,
        device=device_summary.device,
        plots=True,
        project=str(evaluation_dir.parent),
        name=evaluation_dir.name,
        exist_ok=True,
    )
    dataset_image_paths = image_paths(
        project_root / dataset_config.dataset_root,
        dataset_config.splits,
    )
    evaluation_images = [
        path
        for image_split, path in dataset_image_paths
        if image_split == split
    ]
    (
        slices,
        false_positive_count,
        false_negative_count,
        label_contract_issue_count,
    ) = _save_error_examples_and_slices(
        model,
        evaluation_images,
        project_root / dataset_config.dataset_root,
        split,
        dataset_config.class_names,
        config,
        device_summary.device,
        evaluation_dir,
    )
    latency = _measure_latency(
        model,
        evaluation_images[: min(10, len(evaluation_images))],
        config.image_size,
        config.confidence_threshold,
        device_summary.device,
        evaluation_dir,
    )
    _write_json(
        evaluation_dir / "evaluation-report.json",
        {
            "dataset_split": split,
            "release_artifact": release_artifact is not None,
            "release_selection": release_artifact,
            "model": {
                "path": str(best_model_path.relative_to(project_root)),
                "size_bytes": best_model_path.stat().st_size,
            },
            "device": device_summary.as_dict(),
            "overall_metrics": _json_values(metrics.results_dict),
            "per_class_metrics": metrics.summary(),
            "slices": slices,
            "local_inference_latency_after_warmup": latency,
            "false_positive_examples": false_positive_count,
            "false_negative_examples": false_negative_count,
            f"{split}_label_contract_issues": {
                "count": label_contract_issue_count,
                "handling": "Boxes are clipped to image bounds for error-example matching.",
            },
        },
    )
    _require_artifacts(
        evaluation_dir,
        (
            "evaluation-report.json",
            "BoxPR_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
        ),
    )
    return evaluation_dir


def run_controlled_experiment(
    control_run_dir: Path,
    experiment_config: BaselineConfig,
    dataset_config: DatasetConfig,
    project_root: Path,
    device_summary: DeviceSummary,
    model_factory: Callable[[str], Any],
) -> tuple[Path, Path, Path]:
    """Train one image-size variant, compare validation results, and freeze a selection."""
    control_run_dir = _validated_run_directory(control_run_dir, project_root)
    if control_run_dir.parent.name != "baseline":
        raise BaselineError("The controlled experiment must start from a baseline run")
    control_config = _load_saved_config(control_run_dir / "resolved-config.json")
    changed_fields = _changed_config_fields(control_config, experiment_config)
    if changed_fields != ["image_size"]:
        raise BaselineError(
            "The controlled experiment must change only image_size; "
            f"changed fields: {', '.join(changed_fields) or 'none'}"
        )

    experiment_run_dir = _run_training(
        experiment_config,
        dataset_config,
        project_root,
        device_summary,
        model_factory,
        run_prefix="experiment",
    )
    experiment_evaluation_dir = evaluate_baseline(
        experiment_run_dir,
        project_root,
        dataset_config,
        device_summary,
        model_factory,
    )
    control_report_path = _latest_validation_report(control_run_dir)
    experiment_report_path = experiment_evaluation_dir / "evaluation-report.json"
    control_quality = _validation_quality(control_report_path)
    experiment_quality = _validation_quality(experiment_report_path)
    supports_change = experiment_quality > control_quality
    selected_run_dir = experiment_run_dir if supports_change else control_run_dir
    comparison_dir = _create_run_directory(project_root, "controlled-experiment")
    comparison_path = comparison_dir / "comparison.json"
    _write_json(
        comparison_path,
        {
            "control": {
                "run_dir": str(control_run_dir.relative_to(project_root)),
                "validation_report": str(control_report_path.relative_to(project_root)),
                "validation_map50_95": control_quality,
            },
            "experiment": {
                "run_dir": str(experiment_run_dir.relative_to(project_root)),
                "validation_report": str(experiment_report_path.relative_to(project_root)),
                "validation_map50_95": experiment_quality,
            },
            "controlled_change": {
                "parameter": "image_size",
                "control_value": control_config.image_size,
                "experiment_value": experiment_config.image_size,
                "fixed": {
                    "data": dataset_config.model_dump(mode="json"),
                    "seed": control_config.seed,
                    "base_weights": str(control_config.model_path),
                    "evaluation_procedure": "validation evaluation with evaluate-baseline",
                },
            },
            "validation_map50_95_delta": experiment_quality - control_quality,
            "decision": {
                "supports_change": supports_change,
                "selected_run_dir": str(selected_run_dir.relative_to(project_root)),
                "reason": "The image-size change is selected only when its validation "
                "mAP50-95 is strictly higher than the control.",
            },
        },
    )
    release_path = _freeze_selected_baseline(
        selected_run_dir,
        comparison_path,
        project_root,
    )
    return experiment_run_dir, comparison_path, release_path


def evaluate_selected_baseline_on_test(
    release_path: Path,
    project_root: Path,
    dataset_config: DatasetConfig,
    device_summary: DeviceSummary,
    model_factory: Callable[[str], Any],
) -> Path:
    """Evaluate the once-frozen baseline on the sealed test split exactly once."""
    release_path = _validated_release_path(release_path, project_root)
    release = _load_release_artifact(release_path)
    if _release_test_report_exists(release_path, project_root):
        raise BaselineError(
            "A test evaluation already exists for this frozen selection; "
            "do not retune or rerun it."
        )
    run_dir = _validated_run_directory(Path(release["selected_run_dir"]), project_root)
    best_model_path = run_dir / "weights" / "best.pt"
    expected_checksum = release["model_sha256"]
    if _sha256_file(best_model_path) != expected_checksum:
        raise BaselineError("The selected model checksum no longer matches the frozen release")
    config = _load_saved_config(run_dir / "resolved-config.json")
    evaluation_dir = _create_run_directory(project_root, "release-test")
    test_data_path = _write_dataset_config(
        evaluation_dir,
        project_root / dataset_config.dataset_root,
        dataset_config.class_names,
        "test",
    )
    _evaluate_model(
        run_dir,
        best_model_path,
        test_data_path,
        "test",
        config,
        dataset_config,
        project_root,
        device_summary,
        model_factory,
        evaluation_dir,
        release_artifact=str(release_path.relative_to(project_root)),
    )
    return evaluation_dir


def _changed_config_fields(
    control: BaselineConfig, experiment: BaselineConfig
) -> list[str]:
    """Return the versioned training fields that differ between two configurations."""
    control_values = control.model_dump(mode="json")
    experiment_values = experiment.model_dump(mode="json")
    return sorted(
        name
        for name, control_value in control_values.items()
        if experiment_values[name] != control_value
    )


def _latest_validation_report(run_dir: Path) -> Path:
    """Find the most recent complete validation report for one training run."""
    reports = list((run_dir / "artifacts" / "validation").glob("*/evaluation-report.json"))
    if not reports:
        raise BaselineError(f"No validation evaluation report exists for {run_dir}")
    return max(reports, key=lambda path: path.stat().st_mtime)


def _validation_quality(report_path: Path) -> float:
    """Read the checkpoint-selection metric from an evaluation report."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        value = report["overall_metrics"]["metrics/mAP50-95(B)"]
        return float(value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BaselineError(f"Invalid validation evaluation report: {report_path}") from error


def _freeze_selected_baseline(
    selected_run_dir: Path, comparison_path: Path, project_root: Path
) -> Path:
    """Create the immutable selection record required before test evaluation."""
    release_dir = project_root / "artifacts" / "releases"
    release_path = release_dir / "selected-baseline.json"
    if release_path.exists():
        raise BaselineError(
            f"A selected baseline is already frozen at {release_path}; do not replace it."
        )
    best_model_path = selected_run_dir / "weights" / "best.pt"
    _require_artifacts(selected_run_dir, ("weights/best.pt",))
    release_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        release_path,
        {
            "artifact_type": "phase_2_selected_baseline",
            "selected_run_dir": str(selected_run_dir.relative_to(project_root)),
            "model_path": str(best_model_path.relative_to(project_root)),
            "model_sha256": _sha256_file(best_model_path),
            "selection_evidence": str(comparison_path.relative_to(project_root)),
            "selection_metric": "validation mAP50-95",
            "selection_rule": "Select the higher validation mAP50-95; retain the control on a tie.",
        },
    )
    return release_path


def _validated_release_path(release_path: Path, project_root: Path) -> Path:
    """Resolve a selected-baseline release record within the releases directory."""
    resolved = release_path if release_path.is_absolute() else project_root / release_path
    resolved = resolved.resolve()
    allowed_root = (project_root / "artifacts" / "releases").resolve()
    if not resolved.is_file() or not resolved.is_relative_to(allowed_root):
        raise BaselineError(f"Selected baseline release artifact is invalid: {release_path}")
    return resolved


def _load_release_artifact(release_path: Path) -> dict[str, str]:
    """Load and validate the minimal immutable selected-baseline record."""
    try:
        contents = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"Invalid selected baseline release artifact: {release_path}"
        raise BaselineError(message) from error
    required = {"artifact_type", "selected_run_dir", "model_path", "model_sha256"}
    if (
        not isinstance(contents, dict)
        or contents.get("artifact_type") != "phase_2_selected_baseline"
        or not required.issubset(contents)
        or not all(isinstance(contents[name], str) for name in required)
    ):
        raise BaselineError(f"Invalid selected baseline release artifact: {release_path}")
    return {name: contents[name] for name in required}


def _release_test_report_exists(release_path: Path, project_root: Path) -> bool:
    """Detect a prior test result linked to the same frozen release selection."""
    release_reference = str(release_path.relative_to(project_root))
    for report_path in (project_root / "artifacts" / "release-test").glob(
        "*/evaluation-report.json"
    ):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("release_selection") == release_reference:
            return True
    return False


def _sha256_file(path: Path) -> str:
    """Calculate a stable SHA-256 model identity without loading the model."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PeakMemoryMonitor:
    """Record the largest available accelerator or process-memory measurement."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.peak_bytes = 0
        self.source = "process_rss"

    def start(self) -> None:
        """Reset supported accelerator memory counters before training."""
        import torch

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self.source = "torch.cuda.max_memory_allocated"
        elif self.device == "mps":
            self.source = "torch.mps.driver_allocated_memory"
        self.sample()

    def sample(self) -> None:
        """Update the peak memory measurement from the active execution backend."""
        import torch

        if self.device == "cuda":
            value = int(torch.cuda.max_memory_allocated())
        elif self.device == "mps":
            value = int(torch.mps.driver_allocated_memory())
        else:
            value = _process_peak_memory_bytes()
        self.peak_bytes = max(self.peak_bytes, value)

    def as_dict(self) -> dict[str, object]:
        """Return the recorded peak memory with its measurement source."""
        return {"source": self.source, "bytes": self.peak_bytes}


def _attach_memory_callbacks(model: Any, monitor: PeakMemoryMonitor) -> None:
    """Sample memory after train and validation batches when the model supports callbacks."""
    add_callback = getattr(model, "add_callback", None)
    if callable(add_callback):
        add_callback("on_train_batch_end", lambda _: monitor.sample())
        add_callback("on_val_batch_end", lambda _: monitor.sample())


def _process_peak_memory_bytes() -> int:
    """Normalize platform-specific process peak RSS values to bytes."""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)


def _validate_validation_split(split: str, dataset_config: DatasetConfig) -> None:
    """Reject missing or sealed splits before a baseline operation starts."""
    if split not in dataset_config.splits:
        raise BaselineError(
            f"Validation split {split!r} is not defined in the dataset configuration"
        )
    if split == "test":
        raise BaselineError("The sealed test split cannot be used for baseline validation")


def _create_run_directory(parent: Path, prefix: str) -> Path:
    """Create one unique output directory without overwriting prior artifacts."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = parent / "artifacts" / prefix / f"{prefix}-{timestamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir


def _write_dataset_config(
    run_dir: Path,
    dataset_root: Path,
    class_names: list[str],
    validation_split: str,
) -> Path:
    """Write the full-training dataset definition without exposing the test split."""
    data_path = run_dir / "dataset.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": "images/train",
                "val": f"images/{validation_split}",
                "names": {index: name for index, name in enumerate(class_names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def _read_results_csv(results_path: Path) -> list[dict[str, str]]:
    """Read nonempty Ultralytics result rows with normalized column names."""
    try:
        with results_path.open(encoding="utf-8", newline="") as results_file:
            rows = list(csv.DictReader(results_file))
    except FileNotFoundError as error:
        raise BaselineError(f"Baseline results are missing: {results_path}") from error
    if not rows:
        raise BaselineError(f"Baseline results are empty: {results_path}")
    return [
        {
            key.strip(): value.strip()
            for key, value in row.items()
            if key is not None and value is not None
        }
        for row in rows
    ]


def _loss_total(row: dict[str, str], prefix: str) -> float:
    """Add the three detection-loss components for one result row."""
    try:
        return sum(float(row[f"{prefix}/{name}_loss"]) for name in ("box", "cls", "dfl"))
    except (KeyError, ValueError) as error:
        raise BaselineError("Baseline results are missing expected loss values") from error


def _classify_validation_trend(training_losses: list[float], quality: list[float]) -> str:
    """Describe validation behavior near the end of a multi-epoch training run."""
    if len(quality) < 3 or training_losses[-1] >= training_losses[0]:
        return "inconclusive"
    peak_quality = max(quality)
    if quality[-1] < peak_quality - 0.01:
        return "declining"
    tail = quality[-3:]
    if max(tail) - min(tail) <= 0.01:
        return "plateaued"
    return "improving"


def _validated_run_directory(run_dir: Path, project_root: Path) -> Path:
    """Resolve one baseline or experiment run directory in the project artifacts root."""
    resolved = run_dir if run_dir.is_absolute() else project_root / run_dir
    resolved = resolved.resolve()
    allowed_roots = (
        (project_root / "artifacts" / "baseline").resolve(),
        (project_root / "artifacts" / "experiment").resolve(),
    )
    if not resolved.is_dir() or not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise BaselineError(f"Baseline run directory is invalid: {run_dir}")
    return resolved


def _load_saved_config(path: Path) -> BaselineConfig:
    """Load the immutable resolved baseline settings from one run directory."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise BaselineError(f"Invalid saved baseline configuration: {path}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("config"), dict):
        raise BaselineError(f"Invalid saved baseline configuration: {path}")
    try:
        return BaselineConfig.model_validate(raw["config"])
    except ValidationError as error:
        raise BaselineError(f"Invalid saved baseline configuration: {path}") from error


def _save_error_examples_and_slices(
    model: Any,
    image_paths_to_evaluate: list[Path],
    dataset_root: Path,
    split: str,
    class_names: list[str],
    config: BaselineConfig,
    device: str,
    evaluation_dir: Path,
) -> tuple[dict[str, object], int, int, int]:
    """Save bounded false-positive/negative examples and detection-recall slices."""
    false_positive_dir = evaluation_dir / "false-positives"
    false_negative_dir = evaluation_dir / "false-negatives"
    false_positive_dir.mkdir(parents=True)
    false_negative_dir.mkdir(parents=True)
    predictions = model.predict(
        source=[str(path) for path in image_paths_to_evaluate],
        imgsz=config.image_size,
        conf=config.confidence_threshold,
        device=device,
        project=str(evaluation_dir),
        name="error-analysis",
        exist_ok=True,
        save=False,
        verbose=False,
    )
    if len(predictions) != len(image_paths_to_evaluate):
        raise BaselineError("Baseline evaluation returned an unexpected number of predictions")

    size_slices: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    brightness_slices: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    false_positive_examples = 0
    false_negative_examples = 0
    label_contract_issue_count = 0
    for image_path, result in zip(image_paths_to_evaluate, predictions, strict=True):
        ground_truth, issue_count = _ground_truth_detections(dataset_root, split, image_path)
        label_contract_issue_count += issue_count
        predicted = _predicted_detections(result, config.confidence_threshold)
        matched_pairs, unmatched_predictions, unmatched_ground_truth = _match_detections(
            ground_truth, predicted
        )
        brightness = _brightness_bucket(image_path)
        matched_ground_truth = {ground_truth_index for ground_truth_index, _ in matched_pairs}
        for index, detection in enumerate(ground_truth):
            size_slices[_box_size_bucket(detection)][0] += 1
            brightness_slices[brightness][0] += 1
            if index in matched_ground_truth:
                size_slices[_box_size_bucket(detection)][1] += 1
                brightness_slices[brightness][1] += 1
        if unmatched_predictions and false_positive_examples < 3:
            _save_annotated_image(
                image_path,
                ground_truth,
                predicted,
                class_names,
                false_positive_dir / f"{image_path.stem}-false-positive.jpg",
            )
            false_positive_examples += 1
        if unmatched_ground_truth and false_negative_examples < 3:
            _save_annotated_image(
                image_path,
                ground_truth,
                predicted,
                class_names,
                false_negative_dir / f"{image_path.stem}-false-negative.jpg",
            )
            false_negative_examples += 1
    _write_no_example_marker(false_positive_dir, false_positive_examples, "false positives")
    _write_no_example_marker(false_negative_dir, false_negative_examples, "false negatives")
    return {
        "box_size": _slice_summary(size_slices),
        "image_brightness": _slice_summary(brightness_slices),
    }, false_positive_examples, false_negative_examples, label_contract_issue_count


def _ground_truth_detections(
    dataset_root: Path, split: str, image_path: Path
) -> tuple[list[Detection], int]:
    """Convert one YOLO label file into bounded detections and retain its issue count."""
    label_path = label_path_for_image(dataset_root, split, image_path)
    boxes, issues = parse_yolo_label(label_path, class_count=4)
    return [_detection_from_box(box) for box in boxes], len(issues)


def _detection_from_box(box: BoundingBox) -> Detection:
    """Convert a center-width-height annotation into a corner-based detection."""
    return Detection(
        class_id=box.class_id,
        x_min=max(0.0, box.x_center - box.width / 2),
        y_min=max(0.0, box.y_center - box.height / 2),
        x_max=min(1.0, box.x_center + box.width / 2),
        y_max=min(1.0, box.y_center + box.height / 2),
        confidence=1.0,
    )


def _predicted_detections(result: Any, threshold: float) -> list[Detection]:
    """Convert one Ultralytics result into normalized detections above a threshold."""
    if result.boxes is None:
        return []
    image_height, image_width = result.orig_shape
    detections: list[Detection] = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
        strict=True,
    ):
        if float(confidence) >= threshold:
            detections.append(
                Detection(
                    class_id=int(class_id),
                    x_min=float(xyxy[0]) / image_width,
                    y_min=float(xyxy[1]) / image_height,
                    x_max=float(xyxy[2]) / image_width,
                    y_max=float(xyxy[3]) / image_height,
                    confidence=float(confidence),
                )
            )
    return detections


def _match_detections(
    ground_truth: list[Detection], predicted: list[Detection], threshold: float = 0.5
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match same-class predictions to ground truth by descending confidence."""
    unmatched_ground_truth = set(range(len(ground_truth)))
    matches: list[tuple[int, int]] = []
    unmatched_predictions: list[int] = []
    prediction_indices = sorted(
        range(len(predicted)),
        key=lambda index: predicted[index].confidence,
        reverse=True,
    )
    for prediction_index in prediction_indices:
        prediction = predicted[prediction_index]
        candidates = [
            index
            for index in unmatched_ground_truth
            if ground_truth[index].class_id == prediction.class_id
        ]
        if not candidates:
            unmatched_predictions.append(prediction_index)
            continue
        ground_truth_index = max(
            candidates,
            key=lambda index: _iou(ground_truth[index], prediction),
        )
        if _iou(ground_truth[ground_truth_index], prediction) < threshold:
            unmatched_predictions.append(prediction_index)
            continue
        unmatched_ground_truth.remove(ground_truth_index)
        matches.append((ground_truth_index, prediction_index))
    return matches, unmatched_predictions, sorted(unmatched_ground_truth)


def _iou(first: Detection, second: Detection) -> float:
    """Calculate intersection over union for two normalized corner-based detections."""
    left = max(first.x_min, second.x_min)
    top = max(first.y_min, second.y_min)
    right = min(first.x_max, second.x_max)
    bottom = min(first.y_max, second.y_max)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min)
    second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _box_size_bucket(detection: Detection) -> str:
    """Assign one normalized ground-truth box to a documented area bucket."""
    area = (detection.x_max - detection.x_min) * (detection.y_max - detection.y_min)
    if area < 0.1:
        return "small_under_0.10"
    if area < 0.3:
        return "medium_0.10_to_0.30"
    return "large_at_least_0.30"


def _brightness_bucket(image_path: Path) -> str:
    """Assign one image to a brightness bucket using mean grayscale intensity."""
    with Image.open(image_path) as image:
        brightness = float(ImageStat.Stat(image.convert("L")).mean[0])
    if brightness < 85:
        return "dark_under_85"
    if brightness < 170:
        return "mid_85_to_170"
    return "bright_at_least_170"


def _slice_summary(slices: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
    """Convert ground-truth and matched counts into slice-specific recall values."""
    return {
        name: {
            "ground_truth_boxes": values[0],
            "matched_boxes": values[1],
            "recall": values[1] / values[0] if values[0] else 0.0,
        }
        for name, values in sorted(slices.items())
    }


def _save_annotated_image(
    image_path: Path,
    ground_truth: list[Detection],
    predicted: list[Detection],
    class_names: list[str],
    destination: Path,
) -> None:
    """Save a validation image with ground truth and predictions overlaid for inspection."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for detection in ground_truth:
        _draw_detection(draw, detection, class_names, width, height, "blue", "ground truth")
    for detection in predicted:
        _draw_detection(draw, detection, class_names, width, height, "red", "prediction")
    image.save(destination)


def _draw_detection(
    draw: ImageDraw.ImageDraw,
    detection: Detection,
    class_names: list[str],
    width: int,
    height: int,
    color: str,
    label_prefix: str,
) -> None:
    """Draw one normalized detection and a compact class label."""
    coordinates = (
        detection.x_min * width,
        detection.y_min * height,
        detection.x_max * width,
        detection.y_max * height,
    )
    class_name = class_names[detection.class_id]
    draw.rectangle(coordinates, outline=color, width=3)
    draw.text((coordinates[0], coordinates[1]), f"{label_prefix}: {class_name}", fill=color)


def _write_no_example_marker(directory: Path, count: int, label: str) -> None:
    """Record when a perfect result leaves no example image for one error category."""
    if count == 0:
        (directory / "none.json").write_text(
            json.dumps({"message": f"No {label} met the example threshold."}) + "\n",
            encoding="utf-8",
        )


def _measure_latency(
    model: Any,
    sample_paths: list[Path],
    image_size: int,
    confidence_threshold: float,
    device: str,
    evaluation_dir: Path,
) -> dict[str, float | int]:
    """Measure local per-image prediction latency after three warm-up predictions."""
    if not sample_paths:
        raise BaselineError("Validation split has no images for latency measurement")
    warmup_path = str(sample_paths[0])
    for _ in range(3):
        model.predict(
            source=warmup_path,
            imgsz=image_size,
            conf=confidence_threshold,
            device=device,
            project=str(evaluation_dir),
            name="latency",
            exist_ok=True,
            save=False,
            verbose=False,
        )
    measurements: list[float] = []
    for sample_path in sample_paths:
        started_at = time.perf_counter()
        model.predict(
            source=str(sample_path),
            imgsz=image_size,
            conf=confidence_threshold,
            device=device,
            project=str(evaluation_dir),
            name="latency",
            exist_ok=True,
            save=False,
            verbose=False,
        )
        measurements.append((time.perf_counter() - started_at) * 1000)
    ordered = sorted(measurements)
    return {
        "sample_count": len(measurements),
        "mean_ms": sum(measurements) / len(measurements),
        "p95_ms": ordered[round((len(ordered) - 1) * 0.95)],
    }


def _write_json(destination: Path, contents: dict[str, object]) -> None:
    """Write one human-readable JSON artifact."""
    destination.write_text(
        json.dumps(contents, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _require_artifacts(run_dir: Path, relative_paths: tuple[str, ...]) -> None:
    """Fail when an expected baseline or evaluation artifact is missing or empty."""
    missing = [
        relative_path
        for relative_path in relative_paths
        if not (path := run_dir / relative_path).is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise BaselineError(
            f"Required baseline artifacts are missing or empty in {run_dir}: {', '.join(missing)}"
        )


def _json_values(values: dict[str, Any]) -> dict[str, float]:
    """Normalize metric values that may be NumPy scalar objects for JSON output."""
    return {name: float(value) for name, value in values.items()}


def _json_default(value: Any) -> object:
    """Convert scalar objects such as NumPy values into standard JSON primitives."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
