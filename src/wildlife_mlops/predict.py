"""Single-image inference with a stable, self-identifying JSON result."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from wildlife_mlops.data.download import sha256_file

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class PredictionError(RuntimeError):
    """Raised when a single-image prediction cannot produce a valid result."""


def predict_image(
    model_path: Path,
    image_path: Path,
    output_path: Path,
    model_factory: Callable[[str], Any],
) -> Path:
    """Predict one validated image and write its stable JSON contract once."""
    resolved_model_path = _validate_model_path(model_path)
    resolved_image_path = _validate_image_path(image_path)
    resolved_output_path = output_path.resolve()
    if resolved_output_path.suffix.lower() != ".json":
        raise PredictionError("Prediction output must have a .json extension")
    if resolved_output_path.exists():
        raise PredictionError(f"Prediction output already exists: {resolved_output_path}")

    try:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        model = model_factory(str(resolved_model_path))
        results = model.predict(
            source=str(resolved_image_path),
            save=False,
            verbose=False,
            project=str(resolved_output_path.parent),
            name=".ultralytics-predict",
            exist_ok=True,
        )
    except Exception as error:
        raise PredictionError(f"Model inference failed: {error}") from error
    if len(results) != 1:
        raise PredictionError(f"Expected one prediction result, received {len(results)}")

    checksum = sha256_file(resolved_model_path)
    result = results[0]
    response = {
        "image_id": resolved_image_path.name,
        "boxes": _boxes_from_result(result),
        "model_version": f"{resolved_model_path.name}:sha256:{checksum}",
        "model_sha256": checksum,
        "trace_id": str(uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        resolved_output_path.write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as error:
        raise PredictionError(
            f"Unable to write prediction output {resolved_output_path}: {error}"
        ) from error
    return resolved_output_path


def _validate_model_path(model_path: Path) -> Path:
    """Require a nonempty local PyTorch checkpoint file."""
    resolved_path = model_path.resolve()
    if resolved_path.suffix.lower() != ".pt":
        raise PredictionError(f"Model must be a .pt checkpoint: {model_path}")
    if not resolved_path.is_file() or resolved_path.stat().st_size == 0:
        raise PredictionError(f"Model checkpoint is missing or empty: {resolved_path}")
    return resolved_path


def _validate_image_path(image_path: Path) -> Path:
    """Require a supported, decodable image before loading the model."""
    resolved_path = image_path.resolve()
    if resolved_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise PredictionError(f"Unsupported image type {image_path.suffix!r}; use {supported}")
    if not resolved_path.is_file():
        raise PredictionError(f"Image file does not exist: {resolved_path}")
    try:
        with Image.open(resolved_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise PredictionError(f"Image cannot be decoded: {resolved_path}: {error}") from error
    return resolved_path


def _boxes_from_result(result: Any) -> list[dict[str, float | str]]:
    """Normalize Ultralytics detections into the documented prediction boxes."""
    if result.boxes is None:
        return []
    image_height, image_width = result.orig_shape
    boxes: list[dict[str, float | str]] = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
        strict=True,
    ):
        class_index = int(class_id)
        boxes.append(
            {
                "x_min": _normalized_coordinate(float(xyxy[0]), image_width),
                "y_min": _normalized_coordinate(float(xyxy[1]), image_height),
                "x_max": _normalized_coordinate(float(xyxy[2]), image_width),
                "y_max": _normalized_coordinate(float(xyxy[3]), image_height),
                "class": _class_name(result.names, class_index),
                "confidence": float(confidence),
            }
        )
    return boxes


def _normalized_coordinate(value: float, dimension: int) -> float:
    """Normalize and bound one model coordinate to the public 0–1 contract."""
    return min(1.0, max(0.0, value / dimension))


def _class_name(names: dict[int, str] | list[str], class_index: int) -> str:
    """Resolve one model class index for either common Ultralytics name mapping form."""
    if isinstance(names, dict):
        return str(names[class_index])
    return str(names[class_index])
