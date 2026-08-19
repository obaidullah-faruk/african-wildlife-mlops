"""Small, explicit training commands for the local learning lab."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.validate import image_paths, label_path_for_image
from wildlife_mlops.device import DeviceSummary


class TrainingConfig(BaseModel):
    """Settings shared by smoke, overfit, and baseline training."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    epochs: int = Field(gt=0)
    image_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    workers: int = Field(ge=0)
    seed: int
    sample_count: int | None = Field(default=None, gt=0)


class TrainingError(RuntimeError):
    """Raised when a local training command cannot complete safely."""


def load_training_config(config_path: Path) -> TrainingConfig:
    """Load one small YAML training configuration."""
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Training configuration does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"Training configuration must be a mapping: {config_path}")
    try:
        return TrainingConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ValueError(f"Invalid training configuration in {config_path}: {error}") from error


def run_training(
    config: TrainingConfig,
    dataset_config: DatasetConfig,
    project_root: Path,
    device: DeviceSummary,
    model_factory: Callable[[str], Any],
    kind: str,
) -> Path:
    """Train one model and save its files in a new run directory."""
    model_path = project_root / config.model_path
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise TrainingError(f"Pretrained model is missing or empty: {model_path}")

    dataset_root = project_root / dataset_config.dataset_root
    train_images = _images_for_split(dataset_root, dataset_config, "train")
    val_images = _images_for_split(dataset_root, dataset_config, "val")
    if not train_images or not val_images:
        raise TrainingError("Training needs non-empty train and validation image folders")

    run_dir = _new_run_directory(project_root, kind)
    data_path = _write_dataset_yaml(
        run_dir, dataset_root, dataset_config.class_names, train_images, val_images, config, kind
    )
    model = model_factory(str(model_path))
    model.train(
        data=str(data_path),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        workers=config.workers,
        device=device.device,
        seed=config.seed,
        deterministic=True,
        pretrained=True,
        val=True,
        save=True,
        plots=True,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        **_overfit_options(kind),
    )
    _require_artifacts(run_dir, ("weights/best.pt", "weights/last.pt", "results.csv"))
    _write_run_summary(run_dir, config, device, dataset_config, kind)
    return run_dir


def _images_for_split(
    dataset_root: Path, config: DatasetConfig, split: str
) -> list[Path]:
    return [
        path
        for found_split, path in image_paths(dataset_root, {split: 0})
        if found_split == split
    ]


def _write_dataset_yaml(
    run_dir: Path,
    dataset_root: Path,
    class_names: list[str],
    train_images: list[Path],
    val_images: list[Path],
    config: TrainingConfig,
    kind: str,
) -> Path:
    if kind == "overfit":
        return _stage_overfit_dataset(run_dir, dataset_root, class_names, train_images, config)

    train_value: str
    if config.sample_count is None:
        train_value = "images/train"
    else:
        selected = _stable_subset(train_images, config.sample_count)
        image_list = run_dir / "train-images.txt"
        image_list.write_text(
            "\n".join(str(path.resolve()) for path in selected) + "\n", encoding="utf-8"
        )
        train_value = str(image_list.resolve())

    data_path = run_dir / "dataset.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": train_value,
                "val": "images/val",
                "names": {index: name for index, name in enumerate(class_names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def _stage_overfit_dataset(
    run_dir: Path,
    dataset_root: Path,
    class_names: list[str],
    train_images: list[Path],
    config: TrainingConfig,
) -> Path:
    if config.sample_count is None:
        raise TrainingError("Overfit training needs sample_count")
    selected = _stable_subset(train_images, config.sample_count)
    staged_root = run_dir / "overfit-data"
    for index, image_path in enumerate(selected, start=1):
        label_path = label_path_for_image(dataset_root, "train", image_path)
        if not label_path.is_file():
            raise TrainingError(f"Overfit image has no label: {image_path}")
        for split in ("train", "val"):
            image_destination = staged_root / "images" / split / f"{index:02d}{image_path.suffix}"
            label_destination = staged_root / "labels" / split / f"{index:02d}.txt"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_destination)
            shutil.copy2(label_path, label_destination)
    data_path = run_dir / "dataset.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(staged_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(class_names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def _stable_subset(images: list[Path], count: int) -> list[Path]:
    selected = sorted(
        images, key=lambda path: hashlib.sha256(path.as_posix().encode()).hexdigest()
    )[:count]
    if len(selected) != count:
        raise TrainingError(f"Requested {count} images, but found only {len(selected)}")
    return selected


def _overfit_options(kind: str) -> dict[str, float]:
    if kind != "overfit":
        return {}
    return {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
    }


def _new_run_directory(project_root: Path, kind: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = project_root / "artifacts" / kind / f"{kind}-{timestamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir


def _write_run_summary(
    run_dir: Path,
    config: TrainingConfig,
    device: DeviceSummary,
    dataset_config: DatasetConfig,
    kind: str,
) -> None:
    row = _last_metrics_row(run_dir / "results.csv")
    raw_dvc = run_dir.parents[2] / "data" / "raw.dvc"
    dvc_checksum = _sha256_file(raw_dvc) if raw_dvc.is_file() else None
    summary = {
        "kind": kind,
        "config": config.model_dump(mode="json"),
        "device": device.as_dict(),
        "dataset": {
            "source_url": dataset_config.source_url,
            "archive_sha256": dataset_config.expected_sha256,
            "dvc_pointer_sha256": dvc_checksum,
        },
        "metrics": row,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (run_dir / "run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _last_metrics_row(results_path: Path) -> dict[str, float]:
    with results_path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    if not rows:
        return {}
    metrics: dict[str, float] = {}
    for name, value in rows[-1].items():
        if value is None:
            continue
        try:
            metrics[name.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_artifacts(run_dir: Path, paths: tuple[str, ...]) -> None:
    missing = [
        path
        for path in paths
        if not (target := run_dir / path).is_file() or target.stat().st_size == 0
    ]
    if missing:
        raise TrainingError(f"Training did not create required files: {', '.join(missing)}")
