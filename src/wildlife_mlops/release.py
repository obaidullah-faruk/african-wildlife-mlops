"""Small, explicit checks that turn a trained checkpoint into a release candidate."""

from __future__ import annotations

import json
import shutil
import tarfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.download import sha256_file
from wildlife_mlops.data.validate import ValidationResult, validate_dataset, write_validation_report
from wildlife_mlops.device import DeviceSummary
from wildlife_mlops.training import TrainingConfig, run_training


class QualityThresholds(BaseModel):
    """The versioned minimum quality required for a candidate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    minimum_validation_map50: float = Field(ge=0, le=1)


class ReleaseError(RuntimeError):
    """Raised when a checkpoint cannot safely become a release candidate."""


def load_quality_thresholds(config_path: Path) -> QualityThresholds:
    """Load the versioned candidate quality thresholds."""
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Quality thresholds do not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"Quality thresholds must be a mapping: {config_path}")
    try:
        return QualityThresholds.model_validate(raw_config)
    except ValidationError as error:
        raise ValueError(f"Invalid quality thresholds in {config_path}: {error}") from error


def create_candidate(
    training: TrainingConfig,
    thresholds: QualityThresholds,
    dataset: DatasetConfig,
    project_root: Path,
    device: DeviceSummary,
    model_factory: Callable[[str], Any],
) -> Path:
    """Validate data, train, evaluate validation data, and package one candidate."""
    validation = validate_dataset(dataset, project_root)
    validation_report = project_root / "artifacts" / "release-validation-report.json"
    write_validation_report(validation, validation_report)
    if not validation.passed:
        raise ReleaseError(f"Dataset validation failed; inspect {validation_report}")

    run_dir = run_training(training, dataset, project_root, device, model_factory, "baseline")
    checkpoint = run_dir / "weights" / "best.pt"
    validation_metrics = _evaluate_checkpoint(
        checkpoint, run_dir / "dataset.yaml", "val", model_factory
    )
    map50 = _map50(validation_metrics)
    passed = map50 >= thresholds.minimum_validation_map50
    quality_report = {
        "schema_version": 1,
        "candidate_status": "passed" if passed else "failed",
        "thresholds": thresholds.model_dump(mode="json"),
        "dataset_validation": validation.as_dict(),
        "validation_evaluation": {"split": "val", "metrics": validation_metrics},
        "checks": {
            "validation_map50": {
                "actual": map50,
                "minimum": thresholds.minimum_validation_map50,
                "passed": passed,
            }
        },
    }
    if not passed:
        report_path = run_dir / "quality-report.json"
        _write_json(report_path, quality_report)
        raise ReleaseError(f"Validation mAP50 {map50:.4f} is below the release threshold")

    return _package_candidate(project_root, run_dir, checkpoint, validation, quality_report)


def approve_candidate(candidate_dir: Path, approver: str) -> Path:
    """Record a named human approval for a quality-checked candidate."""
    normalized_approver = approver.strip()
    if not normalized_approver:
        raise ReleaseError("Approver must not be empty")
    manifest = _read_json(candidate_dir / "candidate.json")
    quality = _read_json(candidate_dir / "quality-report.json")
    if quality.get("candidate_status") != "passed":
        raise ReleaseError("Only a candidate with a passing quality report can be approved")
    approval_path = candidate_dir / "approval.json"
    if approval_path.exists():
        raise ReleaseError(f"Candidate already has an approval record: {approval_path}")
    _write_json(
        approval_path,
        {
            "schema_version": 1,
            "candidate_id": manifest["candidate_id"],
            "model_sha256": manifest["model_sha256"],
            "approver": normalized_approver,
            "approved_at": datetime.now(UTC).isoformat(),
        },
    )
    return approval_path


def evaluate_approved_candidate(
    candidate_dir: Path,
    dataset: DatasetConfig,
    project_root: Path,
    model_factory: Callable[[str], Any],
) -> Path:
    """Evaluate an approved candidate exactly once on the sealed test split."""
    if not dataset.test_split_sealed:
        raise ReleaseError("Sealed test evaluation requires test_split_sealed: true")
    approval = candidate_dir / "approval.json"
    if not approval.is_file():
        raise ReleaseError("A human approval record is required before test evaluation")
    report_path = candidate_dir / "test-evaluation.json"
    if report_path.exists():
        raise ReleaseError(f"Sealed test evaluation already exists: {report_path}")
    checkpoint = candidate_dir / "package" / "model" / "best.pt"
    test_data = candidate_dir / "test-dataset.yaml"
    _write_evaluation_dataset(test_data, dataset, project_root, "test")
    metrics = _evaluate_checkpoint(checkpoint, test_data, "test", model_factory)
    _write_json(
        report_path,
        {
            "schema_version": 1,
            "candidate_id": _read_json(candidate_dir / "candidate.json")["candidate_id"],
            "split": "test",
            "evaluated_at": datetime.now(UTC).isoformat(),
            "metrics": metrics,
        },
    )
    return report_path


def _package_candidate(
    project_root: Path,
    run_dir: Path,
    checkpoint: Path,
    validation: ValidationResult,
    quality_report: Mapping[str, object],
) -> Path:
    model_checksum = sha256_file(checkpoint)
    candidate_dir = project_root / "artifacts" / "releases" / f"candidate-{model_checksum[:12]}"
    if candidate_dir.exists():
        raise ReleaseError(f"Candidate directory already exists: {candidate_dir}")
    package_dir = candidate_dir / "package"
    model_dir = package_dir / "model"
    model_dir.mkdir(parents=True)
    shutil.copy2(checkpoint, model_dir / "best.pt")
    shutil.copytree(
        Path(__file__).parent,
        package_dir / "source" / "wildlife_mlops",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(run_dir / "run.json", candidate_dir / "training-run.json")
    _write_json(candidate_dir / "dataset-validation.json", validation.as_dict())
    _write_json(candidate_dir / "quality-report.json", quality_report)
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "model_file": "package/model/best.pt",
        "model_sha256": model_checksum,
        "inference_module": "package/source/wildlife_mlops/predict.py",
        "quality_report": "quality-report.json",
    }
    _write_json(candidate_dir / "candidate.json", manifest)
    _write_tarball(candidate_dir, package_dir)
    return candidate_dir


def _evaluate_checkpoint(
    checkpoint: Path, data_path: Path, split: str, model_factory: Callable[[str], Any]
) -> dict[str, float]:
    try:
        result = model_factory(str(checkpoint)).val(data=str(data_path), split=split, plots=False)
    except Exception as error:
        raise ReleaseError(f"{split} evaluation failed: {error}") from error
    raw_metrics = result if isinstance(result, dict) else getattr(result, "results_dict", None)
    if not isinstance(raw_metrics, dict):
        raise ReleaseError(f"{split} evaluation did not return metrics")
    metrics: dict[str, float] = {}
    for name, value in raw_metrics.items():
        if isinstance(value, (int, float)):
            metrics[str(name)] = float(value)
    return metrics


def _map50(metrics: dict[str, float]) -> float:
    for name, value in metrics.items():
        if "mAP50" in name and "95" not in name:
            return value
    raise ReleaseError("Validation evaluation did not report an mAP50 metric")


def _write_evaluation_dataset(
    destination: Path, dataset: DatasetConfig, project_root: Path, split: str
) -> None:
    dataset_root = (project_root / dataset.dataset_root).resolve()
    if not (dataset_root / "images" / split).is_dir():
        raise ReleaseError(f"Dataset split is missing: {dataset_root / 'images' / split}")
    destination.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {index: name for index, name in enumerate(dataset.class_names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_tarball(candidate_dir: Path, package_dir: Path) -> None:
    archive = candidate_dir / "candidate-package.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(package_dir, arcname="package")


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ReleaseError(f"Required candidate file is missing: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReleaseError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(content, dict):
        raise ReleaseError(f"Candidate file must contain a JSON object: {path}")
    return content


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
