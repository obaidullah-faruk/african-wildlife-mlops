"""Transport-independent inference for one pinned release candidate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from wildlife_mlops.data.download import sha256_file
from wildlife_mlops.release import ReleaseError, _read_json

MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class InferenceError(RuntimeError):
    """Raised when a pinned candidate cannot make a safe prediction."""


@dataclass(frozen=True)
class PinnedModel:
    """One verified candidate checkpoint and its loaded prediction model."""

    candidate_id: str
    model_sha256: str
    model: Any

    @property
    def model_version(self) -> str:
        """Return the immutable model identity exposed to callers."""
        return f"{self.candidate_id}:sha256:{self.model_sha256}"

    def predict(self, image_bytes: bytes, image_name: str) -> dict[str, object]:
        """Return normalized predictions without knowing the caller's transport."""
        image = _decode_image(image_bytes, image_name)
        try:
            results = self.model.predict(source=image, save=False, verbose=False)
        except Exception as error:
            raise InferenceError(f"Model inference failed: {error}") from error
        if len(results) != 1:
            raise InferenceError(f"Expected one prediction result, received {len(results)}")
        return {
            "image_id": image_name,
            "boxes": _boxes_from_result(results[0]),
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "trace_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
        }


def load_pinned_model(candidate_dir: Path, model_factory: Callable[[str], Any]) -> PinnedModel:
    """Load exactly one approved, sealed-test-evaluated candidate checkpoint."""
    try:
        manifest = _read_json(candidate_dir / "candidate.json")
    except ReleaseError as error:
        raise InferenceError(str(error)) from error
    for required_file in ("approval.json", "test-evaluation.json"):
        if not (candidate_dir / required_file).is_file():
            raise InferenceError(f"Candidate is not ready to serve; missing {required_file}")
    candidate_id = manifest.get("candidate_id")
    expected_checksum = manifest.get("model_sha256")
    if not isinstance(candidate_id, str) or not isinstance(expected_checksum, str):
        raise InferenceError("Candidate manifest has no usable model identity")
    checkpoint = candidate_dir / "package" / "model" / "best.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise InferenceError(f"Candidate checkpoint is missing or empty: {checkpoint}")
    actual_checksum = sha256_file(checkpoint)
    if actual_checksum != expected_checksum:
        raise InferenceError("Candidate checkpoint checksum does not match candidate.json")
    try:
        model = model_factory(str(checkpoint))
    except Exception as error:
        raise InferenceError(f"Could not load pinned candidate: {error}") from error
    return PinnedModel(candidate_id, actual_checksum, model)


def _decode_image(image_bytes: bytes, image_name: str) -> Image.Image:
    if not image_name:
        raise InferenceError("An image name is required")
    if Path(image_name).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise InferenceError(f"Unsupported image type; use {supported}")
    if not image_bytes:
        raise InferenceError("Image body is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InferenceError(f"Image body exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            return cast(Image.Image, source.convert("RGB"))
    except (OSError, UnidentifiedImageError) as error:
        raise InferenceError(f"Image cannot be decoded: {image_name}") from error


def _boxes_from_result(result: Any) -> list[dict[str, float | str]]:
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
    return min(1.0, max(0.0, value / dimension))


def _class_name(names: dict[int, str] | list[str], class_index: int) -> str:
    if isinstance(names, dict):
        return str(names[class_index])
    return str(names[class_index])
