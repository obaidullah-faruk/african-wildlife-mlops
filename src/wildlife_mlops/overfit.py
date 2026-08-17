"""Tiny-dataset training diagnostic for verifying the detection pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.validate import image_paths, label_path_for_image
from wildlife_mlops.device import DeviceSummary


class OverfitConfig(BaseModel):
    """Settings for one deliberately tiny training diagnostic."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    source_split: str = Field(min_length=1)
    image_count: int = Field(ge=8, le=16)
    epochs: int = Field(gt=0)
    image_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    workers: int = Field(ge=0)
    seed: int
    target_map50: float = Field(ge=0, le=1)
    optimizer: str = Field(min_length=1)
    initial_learning_rate: float = Field(gt=0)
    final_learning_rate_fraction: float = Field(gt=0)
    warmup_epochs: float = Field(ge=0)
    weight_decay: float = Field(ge=0)
    nominal_batch_size: int = Field(gt=0)


class OverfitDiagnosticError(RuntimeError):
    """Raised when the tiny-dataset diagnostic cannot demonstrate memorization."""


@dataclass(frozen=True)
class MemorizationMetrics:
    """Metrics used to judge whether the tiny dataset was memorized."""

    initial_training_loss: float
    final_training_loss: float
    map50: float

    @property
    def loss_decreased(self) -> bool:
        """Return whether final training loss is below its initial value."""
        return self.final_training_loss < self.initial_training_loss


def load_overfit_config(config_path: Path) -> OverfitConfig:
    """Load and validate the versioned tiny-dataset training configuration."""
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Overfit configuration does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"Overfit configuration root must be a mapping: {config_path}")
    try:
        return OverfitConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ValueError(f"Invalid overfit configuration in {config_path}: {error}") from error


def run_overfit_diagnostic(
    config: OverfitConfig,
    dataset_config: DatasetConfig,
    project_root: Path,
    device_summary: DeviceSummary,
    model_factory: Callable[[str], Any],
) -> Path:
    """Train and evaluate a tiny duplicated split, then save the diagnostic report."""
    model_path = project_root / config.model_path
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise OverfitDiagnosticError(f"Pretrained model is missing or empty: {model_path}")
    if config.source_split not in dataset_config.splits:
        raise OverfitDiagnosticError(
            f"Source split {config.source_split!r} is not defined in the dataset configuration"
        )

    run_dir = _create_run_directory(project_root)
    selected_paths = _select_training_images(
        project_root / dataset_config.dataset_root,
        config.source_split,
        config.image_count,
    )
    data_path = _stage_tiny_dataset(
        run_dir / "dataset",
        project_root / dataset_config.dataset_root,
        config.source_split,
        selected_paths,
        dataset_config.class_names,
    )

    model = model_factory(str(model_path))
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
        optimizer=config.optimizer,
        lr0=config.initial_learning_rate,
        lrf=config.final_learning_rate_fraction,
        warmup_epochs=config.warmup_epochs,
        weight_decay=config.weight_decay,
        nbs=config.nominal_batch_size,
        val=True,
        split="val",
        save=True,
        plots=True,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
    )

    metrics = _read_memorization_metrics(run_dir / "results.csv")
    passed = metrics.loss_decreased and metrics.map50 >= config.target_map50
    report_path = run_dir / "overfit-report.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": passed,
                "config": config.model_dump(mode="json"),
                "device": device_summary.as_dict(),
                "selected_images": [str(path.relative_to(project_root)) for path in selected_paths],
                "metrics": {
                    **asdict(metrics),
                    "loss_decreased": metrics.loss_decreased,
                    "target_map50": config.target_map50,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise OverfitDiagnosticError(
            "Tiny-dataset diagnostic did not demonstrate memorization; inspect "
            f"{report_path} before scaling training"
        )
    return report_path


def _create_run_directory(project_root: Path) -> Path:
    """Create an immutable directory for one diagnostic run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = project_root / "artifacts" / "overfit" / f"overfit-{timestamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir


def _select_training_images(dataset_root: Path, split: str, image_count: int) -> list[Path]:
    """Choose a stable image subset from one configured source split."""
    candidates = [
        path
        for path_split, path in image_paths(dataset_root, {split: 0})
        if path_split == split
    ]
    selected = sorted(
        candidates,
        key=lambda path: hashlib.sha256(path.as_posix().encode()).hexdigest(),
    )[:image_count]
    if len(selected) != image_count:
        raise OverfitDiagnosticError(
            f"Expected {image_count} images in {dataset_root / 'images' / split}, "
            f"found {len(selected)}"
        )
    return selected


def _stage_tiny_dataset(
    destination: Path,
    dataset_root: Path,
    source_split: str,
    selected_paths: list[Path],
    class_names: list[str],
) -> Path:
    """Copy the selected images and labels into identical training and validation splits."""
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(selected_paths, start=1):
            source_label = label_path_for_image(dataset_root, source_split, image_path)
            if not source_label.is_file():
                raise OverfitDiagnosticError(f"Selected image has no label: {image_path}")
            image_name = f"{index:02d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, destination / "images" / split / image_name)
            shutil.copy2(source_label, destination / "labels" / split / f"{index:02d}.txt")

    data_path = destination / "dataset.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(destination.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(class_names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def _read_memorization_metrics(results_path: Path) -> MemorizationMetrics:
    """Read the first and final training losses plus final mAP50 from Ultralytics CSV output."""
    try:
        with results_path.open(encoding="utf-8", newline="") as results_file:
            rows = list(csv.DictReader(results_file))
    except FileNotFoundError as error:
        raise OverfitDiagnosticError(f"Training did not produce results: {results_path}") from error
    if not rows:
        raise OverfitDiagnosticError(f"Training results are empty: {results_path}")

    normalized_rows = [
        {
            key.strip(): value.strip()
            for key, value in row.items()
            if key is not None and value is not None
        }
        for row in rows
    ]
    try:
        return MemorizationMetrics(
            initial_training_loss=_total_training_loss(normalized_rows[0]),
            final_training_loss=_total_training_loss(normalized_rows[-1]),
            map50=float(normalized_rows[-1]["metrics/mAP50(B)"],),
        )
    except (KeyError, ValueError) as error:
        raise OverfitDiagnosticError(
            f"Training results are missing expected detection metrics: {results_path}"
        ) from error


def _total_training_loss(row: dict[str, str]) -> float:
    """Add Ultralytics' detection training-loss components for one epoch."""
    return sum(float(row[key]) for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss"))
