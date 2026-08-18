"""One-epoch training smoke test with required artifact checks."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.validate import label_path_for_image
from wildlife_mlops.device import DeviceSummary
from wildlife_mlops.tracking import MLflowRunLogger, start_smoke_run, write_smoke_run_report


class SmokeTrainConfig(BaseModel):
    """Settings for one fast, deterministic training smoke test."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    manifest_path: Path
    image_count: int = Field(gt=0)
    epochs: int = Field(ge=1, le=1)
    image_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    workers: int = Field(ge=0)
    seed: int
    validation_split: str = Field(min_length=1)


class SmokeTrainError(RuntimeError):
    """Raised when smoke-training inputs or required output artifacts are invalid."""


def load_smoke_train_config(config_path: Path) -> SmokeTrainConfig:
    """Load and validate the versioned smoke-training configuration."""
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Smoke-training configuration does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"Smoke-training configuration root must be a mapping: {config_path}")
    try:
        return SmokeTrainConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ValueError(
            f"Invalid smoke-training configuration in {config_path}: {error}"
        ) from error


def run_smoke_train(
    config: SmokeTrainConfig,
    dataset_config: DatasetConfig,
    project_root: Path,
    device_summary: DeviceSummary,
    model_factory: Callable[[str], Any],
    tracking_uri: str | None = None,
    experiment_name: str = "wildlife-smoke",
    parent_run_id: str = "not_applicable",
    trigger_type: str = "manual",
    trigger_id: str = "local-cli",
) -> Path:
    """Run one epoch from the immutable subset manifest and verify every expected artifact."""
    model_path = project_root / config.model_path
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise SmokeTrainError(f"Pretrained model is missing or empty: {model_path}")
    if config.validation_split not in dataset_config.splits:
        raise SmokeTrainError(
            f"Validation split {config.validation_split!r} is not defined "
            "in the dataset configuration"
        )
    if config.validation_split == "test":
        raise SmokeTrainError("The sealed test split cannot be used for smoke training")

    selected_images = _load_manifest_images(config, dataset_config, project_root)
    run_dir = _create_run_directory(project_root)
    train_list_path = run_dir / "smoke-train.txt"
    train_list_path.write_text(
        "\n".join(str(path.resolve()) for path in selected_images) + "\n",
        encoding="utf-8",
    )
    data_path = _write_data_config(
        run_dir,
        project_root / dataset_config.dataset_root,
        train_list_path,
        config.validation_split,
        dataset_config.class_names,
    )
    _write_json(
        run_dir / "resolved-config.json",
        {
            "config": config.model_dump(mode="json"),
            "selected_images": [str(path.relative_to(project_root)) for path in selected_images],
        },
    )
    _write_json(run_dir / "environment-summary.json", _environment_summary(device_summary))

    resolved_config_path = run_dir / "resolved-config.json"
    tracker: MLflowRunLogger | None = None
    if tracking_uri is not None:
        lineage_tags = _build_lineage_tags(
            project_root,
            dataset_config,
            config,
            resolved_config_path,
            model_path,
            device_summary,
            parent_run_id,
            trigger_type,
            trigger_id,
        )
        tracker = start_smoke_run(
            run_dir, config.model_dump(mode="json"), lineage_tags, tracking_uri, experiment_name
        )
    model = model_factory(str(model_path))
    _disable_ultralytics_mlflow_callbacks(model)
    if tracker is not None:
        with tracker:
            _train_smoke_model(model, config, data_path, run_dir, device_summary, tracker)
            _complete_smoke_run(run_dir, config, tracker)
    else:
        _train_smoke_model(model, config, data_path, run_dir, device_summary)
        _complete_smoke_run(run_dir, config)
    return run_dir


def _build_lineage_tags(
    project_root: Path,
    dataset_config: DatasetConfig,
    config: SmokeTrainConfig,
    resolved_config_path: Path,
    model_path: Path,
    device_summary: DeviceSummary,
    parent_run_id: str,
    trigger_type: str,
    trigger_id: str,
) -> dict[str, str]:
    """Collect the provenance needed to explain one local smoke-training run."""
    return {
        "lineage.git_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "lineage.git_dirty": str(bool(_git_output(project_root, "status", "--porcelain"))).lower(),
        "lineage.dvc_revision": "not_applicable",
        "lineage.source_archive_sha256": dataset_config.expected_sha256,
        "lineage.prepared_manifest_sha256": "not_applicable",
        "lineage.config_sha256": _sha256_file(resolved_config_path),
        "lineage.random_seed": str(config.seed),
        "lineage.base_weights_name": model_path.name,
        "lineage.base_weights_sha256": _sha256_file(model_path),
        "runtime.python_version": sys.version.split()[0],
        "runtime.pytorch_version": device_summary.pytorch_version,
        "runtime.ultralytics_version": device_summary.ultralytics_version,
        "runtime.os": platform.platform(),
        "runtime.architecture": platform.machine(),
        "runtime.device": device_summary.device,
        "execution.training_container_digest": "not_applicable",
        "lineage.parent_run_id": parent_run_id,
        "trigger.type": trigger_type,
        "trigger.id": trigger_id,
    }


def _git_output(project_root: Path, *arguments: str) -> str:
    """Run one read-only Git command and return its trimmed output."""
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=project_root, text=True, stderr=subprocess.PIPE
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SmokeTrainError(f"Unable to collect Git lineage: {error}") from error


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a required nonempty file."""
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise SmokeTrainError(f"Unable to checksum lineage file {path}: {error}") from error
    if not contents:
        raise SmokeTrainError(f"Unable to checksum empty lineage file: {path}")
    return hashlib.sha256(contents).hexdigest()


def _train_smoke_model(
    model: Any,
    config: SmokeTrainConfig,
    data_path: Path,
    run_dir: Path,
    device_summary: DeviceSummary,
    tracker: MLflowRunLogger | None = None,
) -> None:
    """Train one smoke model and optionally record aggregate epoch metrics."""
    if tracker is not None:
        model.add_callback("on_fit_epoch_end", tracker.log_epoch)
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


def _complete_smoke_run(
    run_dir: Path, config: SmokeTrainConfig, tracker: MLflowRunLogger | None = None
) -> None:
    """Verify completed artifacts and send terminal evidence to MLflow when configured."""
    _require_artifacts(
        run_dir,
        (
            "weights/best.pt",
            "weights/last.pt",
            "args.yaml",
            "resolved-config.json",
            "environment-summary.json",
            "results.csv",
        ),
    )
    resolved_config = config.model_dump(mode="json")
    write_smoke_run_report(run_dir, resolved_config)
    _require_artifacts(run_dir, ("run-report.json",))
    if tracker is not None:
        tracker.log_terminal_metrics(run_dir / "results.csv")
        tracker.log_artifacts(run_dir)


def _disable_ultralytics_mlflow_callbacks(model: Any) -> None:
    """Keep MLflow logging owned by this module instead of Ultralytics callbacks."""
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS["mlflow"] = False
    except ImportError:
        pass
    callbacks = getattr(model, "callbacks", None)
    if not isinstance(callbacks, dict):
        return
    for event, functions in callbacks.items():
        if isinstance(functions, list):
            callbacks[event] = [
                function
                for function in functions
                if getattr(function, "__module__", "")
                != "ultralytics.utils.callbacks.mlflow"
            ]


def _load_manifest_images(
    config: SmokeTrainConfig, dataset_config: DatasetConfig, project_root: Path
) -> list[Path]:
    """Load manifest entries and validate they are labeled training images from this archive."""
    manifest_path = project_root / config.manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SmokeTrainError(
            f"Smoke subset manifest does not exist: {manifest_path}; run make data-smoke-manifest"
        ) from error
    except json.JSONDecodeError as error:
        raise SmokeTrainError(f"Invalid smoke subset manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise SmokeTrainError(f"Smoke subset manifest must be an object: {manifest_path}")
    if manifest.get("source_archive_sha256") != dataset_config.expected_sha256:
        raise SmokeTrainError("Smoke subset manifest does not match the configured source archive")
    if manifest.get("source_split") != "train":
        raise SmokeTrainError("Smoke subset manifest must contain only training images")
    raw_images = manifest.get("images")
    if not isinstance(raw_images, list) or len(raw_images) != config.image_count:
        raise SmokeTrainError(
            f"Smoke subset manifest must contain exactly {config.image_count} images"
        )

    dataset_root = project_root / dataset_config.dataset_root
    train_root = dataset_root / "images" / "train"
    selected_images: list[Path] = []
    for raw_image in raw_images:
        if not isinstance(raw_image, str):
            raise SmokeTrainError("Smoke subset manifest image paths must be strings")
        relative_image = Path(raw_image)
        if relative_image.is_absolute():
            raise SmokeTrainError("Smoke subset manifest image paths must be relative")
        image_path = project_root / relative_image
        if not image_path.is_file() or not image_path.is_relative_to(train_root):
            raise SmokeTrainError(f"Invalid smoke subset image path: {raw_image}")
        label_path = label_path_for_image(dataset_root, "train", image_path)
        if not label_path.is_file():
            raise SmokeTrainError(f"Smoke subset image has no label: {image_path}")
        selected_images.append(image_path)
    return selected_images


def _create_run_directory(project_root: Path) -> Path:
    """Create a unique directory for one smoke-training run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = project_root / "artifacts" / "smoke" / f"smoke-{timestamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir


def _write_data_config(
    run_dir: Path,
    dataset_root: Path,
    train_list_path: Path,
    validation_split: str,
    class_names: list[str],
) -> Path:
    """Write the Ultralytics dataset definition for the training manifest and validation split."""
    data_path = run_dir / "dataset.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": str(train_list_path.resolve()),
                "val": f"images/{validation_split}",
                "names": {index: name for index, name in enumerate(class_names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def _environment_summary(device_summary: DeviceSummary) -> dict[str, object]:
    """Return portable runtime details for the training artifact."""
    return {
        "device": device_summary.as_dict(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }


def _write_json(destination: Path, contents: dict[str, object]) -> None:
    """Write a deterministic JSON artifact."""
    destination.write_text(
        json.dumps(contents, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_artifacts(run_dir: Path, relative_paths: tuple[str, ...]) -> None:
    """Fail when training did not produce a required nonempty artifact."""
    missing = [
        relative_path
        for relative_path in relative_paths
        if not (path := run_dir / relative_path).is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise SmokeTrainError(
            "Smoke training is missing required nonempty artifacts in "
            f"{run_dir}: {', '.join(missing)}"
        )
