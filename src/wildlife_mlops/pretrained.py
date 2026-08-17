"""Pinned pretrained-model inference over a deterministic image sample."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wildlife_mlops.data.download import sha256_file
from wildlife_mlops.data.validate import image_paths


class PretrainedInferenceConfig(BaseModel):
    """Configuration for one immutable pretrained inference exercise."""

    model_config = ConfigDict(extra="forbid")

    model_url: str = Field(min_length=1)
    model_path: Path
    model_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    model_version: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    sample_count: int = Field(gt=0)
    confidence_threshold: float = Field(ge=0, le=1)
    output_dir: Path


class PretrainedInferenceError(RuntimeError):
    """Raised when the pinned weights or inference artifacts are invalid."""


def load_pretrained_config(config_path: Path) -> PretrainedInferenceConfig:
    """Load and validate the versioned pretrained-inference configuration."""
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"Pretrained inference configuration does not exist: {config_path}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Pretrained inference configuration root must be a mapping: {config_path}"
        )
    try:
        return PretrainedInferenceConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ValueError(
            f"Invalid pretrained inference configuration in {config_path}: {error}"
        ) from error


def run_pretrained_inference(
    config: PretrainedInferenceConfig,
    dataset_root: Path,
    project_root: Path,
    model_factory: Callable[[str], Any],
) -> Path:
    """Download verified weights once, load once, and predict a fixed image sample."""
    model_path = ensure_model_weights(config, project_root)
    sample_paths = _select_images(dataset_root, config.source_split, config.sample_count)
    output_dir = project_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model_factory(str(model_path))
    results = model.predict(
        source=[str(path) for path in sample_paths],
        conf=config.confidence_threshold,
        save=False,
        verbose=False,
    )
    if len(results) != len(sample_paths):
        raise PretrainedInferenceError(
            f"Expected {len(sample_paths)} predictions, received {len(results)}"
        )

    predictions = []
    trace_id = str(uuid4())
    timestamp = datetime.now(UTC).isoformat()
    for index, (image_path, result) in enumerate(zip(sample_paths, results, strict=True), start=1):
        rendered_path = output_dir / f"{index:02d}-{image_path.stem}-prediction.jpg"
        _save_rendered_prediction(result, rendered_path)
        predictions.append(
            {
                "image_id": str(image_path.relative_to(project_root)),
                "boxes": _boxes_from_result(result),
                "model_version": config.model_version,
                "trace_id": trace_id,
                "timestamp": timestamp,
                "rendered_prediction": str(rendered_path.relative_to(project_root)),
            }
        )

    output_path = output_dir / "predictions.json"
    output_path.write_text(
        json.dumps(
            {
                "model_version": config.model_version,
                "model_sha256": config.model_sha256,
                "model_class_names": _class_names(model),
                "prediction_count": len(predictions),
                "predictions": predictions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def ensure_model_weights(config: PretrainedInferenceConfig, project_root: Path) -> Path:
    """Download the model to a temporary file and publish it only after verification."""
    model_path = project_root / config.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        _verify_weights(model_path, config.model_sha256)
        return model_path

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{model_path.name}.",
        suffix=".part",
        dir=model_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            with urllib.request.urlopen(config.model_url) as response:
                shutil.copyfileobj(response, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            _verify_weights(temporary_path, config.model_sha256)
            os.replace(temporary_path, model_path)
        except (OSError, urllib.error.URLError) as error:
            temporary_path.unlink(missing_ok=True)
            raise PretrainedInferenceError(
                f"Unable to download pretrained model from {config.model_url}: {error}"
            ) from error
        except PretrainedInferenceError:
            temporary_path.unlink(missing_ok=True)
            raise
    return model_path


def _verify_weights(model_path: Path, expected_sha256: str) -> None:
    """Verify a nonempty PyTorch checkpoint against its immutable checksum."""
    if model_path.stat().st_size == 0:
        raise PretrainedInferenceError(f"Pretrained model is empty: {model_path}")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_sha256:
        raise PretrainedInferenceError(
            f"Pretrained model checksum mismatch for {model_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _select_images(dataset_root: Path, split: str, sample_count: int) -> list[Path]:
    """Choose a stable sample from one existing dataset split."""
    candidates = [
        path for path_split, path in image_paths(dataset_root, {split: 0}) if path_split == split
    ]
    selected = sorted(
        candidates, key=lambda path: hashlib.sha256(path.as_posix().encode()).hexdigest()
    )[:sample_count]
    if len(selected) != sample_count:
        raise PretrainedInferenceError(
            f"Expected {sample_count} images in {dataset_root / 'images' / split}, "
            f"found {len(selected)}"
        )
    return selected


def _boxes_from_result(result: Any) -> list[dict[str, float | int | str]]:
    """Convert Ultralytics tensor output into portable structured detections."""
    if result.boxes is None:
        return []
    image_height, image_width = result.orig_shape
    boxes: list[dict[str, float | int | str]] = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
        strict=True,
    ):
        class_index = int(class_id)
        boxes.append(
            {
                "x_min": float(xyxy[0]) / image_width,
                "y_min": float(xyxy[1]) / image_height,
                "x_max": float(xyxy[2]) / image_width,
                "y_max": float(xyxy[3]) / image_height,
                "class": str(result.names[class_index]),
                "confidence": float(confidence),
            }
        )
    return boxes


def _save_rendered_prediction(result: Any, destination: Path) -> None:
    """Save Ultralytics' rendered prediction without changing the source image."""
    rendered_bgr = result.plot()
    Image.fromarray(rendered_bgr[:, :, ::-1]).save(destination)


def _class_names(model: Any) -> dict[str, str]:
    """Normalize model class names for the saved structured output."""
    names = model.names
    if isinstance(names, dict):
        return {str(index): str(name) for index, name in names.items()}
    return {str(index): str(name) for index, name in enumerate(names)}
