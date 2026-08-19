"""Canonical dataset manifests used to select and verify training inputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.validate import (
    image_paths,
    label_path_for_image,
    parse_yolo_label,
)

CONTENT_MANIFEST_PATH = Path("data/manifests/content-manifest.json")
CONTENT_MANIFEST_CHECKSUM_PATH = Path("data/manifests/content-manifest.sha256")
TEST_MANIFEST_PATH = Path("data/manifests/test-content-manifest.json")
TEST_MANIFEST_CHECKSUM_PATH = Path("data/manifests/test-content-manifest.sha256")
TRAINING_SPLITS = {"train", "val"}


class ContentManifestError(ValueError):
    """Raised when a content manifest cannot prove the current dataset identity."""


def create_content_manifest(config: DatasetConfig, project_root: Path) -> tuple[Path, Path]:
    """Write the stable manifest and its SHA-256 checksum for current data."""
    dataset_root = project_root / config.dataset_root
    archive_checksum = _sha256_file(project_root / config.archive_path)
    if archive_checksum != config.expected_sha256:
        raise ContentManifestError("Dataset archive checksum does not match dataset configuration")
    entries = [
        _manifest_entry(config, dataset_root, split, image_path)
        for split, image_path in image_paths(dataset_root, config.splits)
    ]
    metadata = {
        "schema_version": 1,
        "source": {
            "archive_sha256": archive_checksum,
            "license_reference": config.source_license_reference,
            "url": config.source_url,
        },
        "class_names": config.class_names,
    }
    training_entries = [entry for entry in entries if entry["split"] in TRAINING_SPLITS]
    test_entries = [entry for entry in entries if entry["split"] == "test"]
    manifest_path, checksum_path = _write_manifest(
        project_root,
        CONTENT_MANIFEST_PATH,
        CONTENT_MANIFEST_CHECKSUM_PATH,
        metadata,
        training_entries,
    )
    _write_manifest(
        project_root, TEST_MANIFEST_PATH, TEST_MANIFEST_CHECKSUM_PATH, metadata, test_entries
    )
    return manifest_path, checksum_path


def load_verified_manifest(config: DatasetConfig, project_root: Path) -> dict[str, Any]:
    """Load the canonical manifest and reject changed or unlisted dataset files."""
    return _load_verified_manifest(
        config, project_root, CONTENT_MANIFEST_PATH, CONTENT_MANIFEST_CHECKSUM_PATH, TRAINING_SPLITS
    )


def load_verified_test_manifest(config: DatasetConfig, project_root: Path) -> dict[str, Any]:
    """Load the isolated test manifest for frozen release evaluation only."""
    return _load_verified_manifest(
        config, project_root, TEST_MANIFEST_PATH, TEST_MANIFEST_CHECKSUM_PATH, {"test"}
    )


def _load_verified_manifest(
    config: DatasetConfig,
    project_root: Path,
    manifest_relative_path: Path,
    checksum_relative_path: Path,
    allowed_splits: set[str],
) -> dict[str, Any]:
    """Load one isolated manifest and verify only its allowed data splits."""
    manifest_path = project_root / manifest_relative_path
    checksum_path = project_root / checksum_relative_path
    try:
        payload = manifest_path.read_bytes()
        expected_checksum = checksum_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ContentManifestError(
            "Content manifest is missing; run make data-content-manifest before training"
        ) from error
    actual_checksum = _sha256_bytes(payload)
    if expected_checksum != actual_checksum:
        raise ContentManifestError("Content manifest checksum does not match its recorded value")
    try:
        raw_manifest = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ContentManifestError("Content manifest is not valid JSON") from error
    if not isinstance(raw_manifest, dict):
        raise ContentManifestError("Content manifest must be a JSON object")
    _verify_manifest_structure(raw_manifest, config, allowed_splits)
    _verify_dataset_files(raw_manifest, config, project_root, allowed_splits)
    return raw_manifest


def manifest_image_paths(
    manifest: dict[str, Any], project_root: Path, dataset_config: DatasetConfig, split: str
) -> list[Path]:
    """Return one manifest-selected split as sorted absolute image paths."""
    paths = [
        project_root / dataset_config.dataset_root / str(entry["path"])
        for entry in _entries_for_split(manifest, split)
    ]
    if not paths:
        raise ContentManifestError(f"Content manifest has no images for split {split!r}")
    return paths


def manifest_checksum(project_root: Path) -> str:
    """Return the checked checksum recorded beside the content manifest."""
    return _manifest_checksum(project_root / CONTENT_MANIFEST_CHECKSUM_PATH, "Content manifest")


def test_manifest_checksum(project_root: Path) -> str:
    """Return the independent immutable checksum for the sealed test manifest."""
    return _manifest_checksum(project_root / TEST_MANIFEST_CHECKSUM_PATH, "Test manifest")


def _manifest_checksum(checksum_path: Path, name: str) -> str:
    """Read and validate one recorded manifest checksum."""
    try:
        checksum = checksum_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ContentManifestError(f"{name} checksum is missing") from error
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise ContentManifestError(f"{name} checksum is malformed")
    return checksum


def write_manifest_image_list(
    manifest: dict[str, Any],
    project_root: Path,
    dataset_config: DatasetConfig,
    split: str,
    destination: Path,
) -> Path:
    """Write a sorted absolute image list for a single verified manifest split."""
    paths = manifest_image_paths(manifest, project_root, dataset_config, split)
    destination.write_text(
        "\n".join(str(path.resolve()) for path in paths) + "\n", encoding="utf-8"
    )
    return destination


def _manifest_entry(
    config: DatasetConfig, dataset_root: Path, split: str, image_path: Path
) -> dict[str, object]:
    """Build the complete immutable record for one image and its label."""
    label_path = label_path_for_image(dataset_root, split, image_path)
    boxes, _ = parse_yolo_label(label_path, len(config.class_names))
    class_counts: Counter[int] = Counter(box.class_id for box in boxes)
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "path": image_path.relative_to(dataset_root).as_posix(),
        "label_path": label_path.relative_to(dataset_root).as_posix(),
        "split": split,
        "image_sha256": _sha256_file(image_path),
        "label_sha256": _sha256_file(label_path),
        "dimensions": {"height": height, "width": width},
        "class_counts": {
            class_name: class_counts[index] for index, class_name in enumerate(config.class_names)
        },
        "source_url": config.source_url,
        "source_license_reference": config.source_license_reference,
    }


def _verify_manifest_structure(
    manifest: dict[str, Any], config: DatasetConfig, allowed_splits: set[str]
) -> None:
    """Check required manifest metadata before using any listed path."""
    if manifest.get("schema_version") != 1:
        raise ContentManifestError("Unsupported content manifest schema version")
    if manifest.get("class_names") != config.class_names:
        raise ContentManifestError(
            "Content manifest class names do not match dataset configuration"
        )
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ContentManifestError("Content manifest source metadata is missing")
    if source.get("license_reference") != config.source_license_reference:
        raise ContentManifestError(
            "Content manifest license reference does not match configuration"
        )
    if source.get("url") != config.source_url:
        raise ContentManifestError("Content manifest source URL does not match configuration")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContentManifestError("Content manifest entries must be a nonempty list")
    has_disallowed_split = any(
        not isinstance(entry, dict) or entry.get("split") not in allowed_splits
        for entry in entries
    )
    if has_disallowed_split:
        raise ContentManifestError("Content manifest contains a disallowed split")


def _verify_dataset_files(
    manifest: dict[str, Any], config: DatasetConfig, project_root: Path, allowed_splits: set[str]
) -> None:
    """Verify each listed pair and reject any data file absent from the manifest."""
    dataset_root = project_root / config.dataset_root
    source = manifest["source"]
    assert isinstance(source, dict)
    archive_path = project_root / config.archive_path
    if source.get("archive_sha256") != _sha256_file(archive_path):
        raise ContentManifestError("Dataset archive checksum does not match the content manifest")

    listed_images: set[Path] = set()
    listed_labels: set[Path] = set()
    for entry in manifest["entries"]:
        if not isinstance(entry, dict):
            raise ContentManifestError("Content manifest entries must be objects")
        split = entry.get("split")
        if not isinstance(split, str) or split not in allowed_splits:
            raise ContentManifestError("Content manifest contains an unknown split")
        image_path = _safe_dataset_path(dataset_root, entry.get("path"), "image")
        label_path = _safe_dataset_path(dataset_root, entry.get("label_path"), "label")
        expected_label = label_path_for_image(dataset_root, split, image_path)
        if label_path != expected_label:
            raise ContentManifestError(
                f"Content manifest label path is invalid: {entry.get('path')}"
            )
        _verify_entry(entry, config, image_path, label_path)
        if image_path in listed_images or label_path in listed_labels:
            raise ContentManifestError("Content manifest contains duplicate image or label entries")
        listed_images.add(image_path)
        listed_labels.add(label_path)

    selected_splits = {
        split: config.splits[split] for split in allowed_splits if split in config.splits
    }
    actual_images = {path for _, path in image_paths(dataset_root, selected_splits)}
    actual_labels = {
        path
        for split in allowed_splits
        for path in (dataset_root / "labels" / split).rglob("*.txt")
        if (dataset_root / "labels" / split).is_dir()
    }
    if actual_images != listed_images:
        raise ContentManifestError(_unlisted_message("image", actual_images, listed_images))
    if actual_labels != listed_labels:
        raise ContentManifestError(_unlisted_message("label", actual_labels, listed_labels))


def _verify_entry(
    entry: dict[str, Any], config: DatasetConfig, image_path: Path, label_path: Path
) -> None:
    """Verify content hashes and image metadata for one manifest record."""
    required_strings = ("image_sha256", "label_sha256", "source_url", "source_license_reference")
    if any(not isinstance(entry.get(field), str) for field in required_strings):
        raise ContentManifestError(f"Content manifest entry is incomplete: {image_path}")
    if entry["source_url"] != config.source_url:
        raise ContentManifestError(f"Content manifest source URL changed: {image_path}")
    if entry["source_license_reference"] != config.source_license_reference:
        raise ContentManifestError(f"Content manifest license reference changed: {image_path}")
    if entry["image_sha256"] != _sha256_file(image_path):
        raise ContentManifestError(f"Content manifest image hash changed: {image_path}")
    if entry["label_sha256"] != _sha256_file(label_path):
        raise ContentManifestError(f"Content manifest label hash changed: {label_path}")
    dimensions = entry.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ContentManifestError(f"Content manifest dimensions are missing: {image_path}")
    with Image.open(image_path) as image:
        actual_dimensions = {"height": image.height, "width": image.width}
    if dimensions != actual_dimensions:
        raise ContentManifestError(f"Content manifest dimensions changed: {image_path}")
    boxes, _ = parse_yolo_label(label_path, len(config.class_names))
    actual_counts = Counter(box.class_id for box in boxes)
    expected_counts = {
        class_name: actual_counts[index] for index, class_name in enumerate(config.class_names)
    }
    if entry.get("class_counts") != expected_counts:
        raise ContentManifestError(f"Content manifest class counts changed: {label_path}")


def _safe_dataset_path(dataset_root: Path, value: object, kind: str) -> Path:
    """Resolve one manifest-relative data path without allowing path traversal."""
    if not isinstance(value, str):
        raise ContentManifestError(f"Content manifest {kind} path must be a string")
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ContentManifestError(f"Content manifest {kind} path must be relative")
    resolved = (dataset_root / relative_path).resolve()
    if not resolved.is_relative_to(dataset_root.resolve()) or not resolved.is_file():
        raise ContentManifestError(f"Content manifest {kind} file is missing: {value}")
    return resolved


def _entries_for_split(manifest: dict[str, Any], split: str) -> Iterable[dict[str, Any]]:
    """Yield already-verified records for a requested split in manifest order."""
    entries = manifest["entries"]
    assert isinstance(entries, list)
    return [entry for entry in entries if isinstance(entry, dict) and entry["split"] == split]


def _write_manifest(
    project_root: Path,
    manifest_relative_path: Path,
    checksum_relative_path: Path,
    metadata: dict[str, object],
    entries: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Write one canonical split-specific manifest and its checksum."""
    manifest = {
        **metadata,
        "entries": sorted(entries, key=lambda entry: (str(entry["split"]), str(entry["path"]))),
    }
    manifest_path = project_root / manifest_relative_path
    checksum_path = project_root / checksum_relative_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(manifest)
    manifest_path.write_text(payload, encoding="utf-8")
    checksum_path.write_text(f"{_sha256_bytes(payload.encode('utf-8'))}\n", encoding="utf-8")
    return manifest_path, checksum_path


def _unlisted_message(kind: str, actual: set[Path], listed: set[Path]) -> str:
    """Explain whether data was added, removed, or omitted from the manifest."""
    unlisted = sorted(path.as_posix() for path in actual - listed)
    missing = sorted(path.as_posix() for path in listed - actual)
    details = unlisted or missing
    return f"Content manifest does not match dataset {kind} files: {details[0]}"


def _canonical_json(value: object) -> str:
    """Encode deterministic JSON suitable for content hashing."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum for one required nonempty file."""
    try:
        with path.open("rb") as file_handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContentManifestError(f"Unable to read dataset file for hashing: {path}") from error
    return digest.hexdigest()


def _sha256_bytes(contents: bytes) -> str:
    """Return a SHA-256 checksum for an in-memory canonical payload."""
    return hashlib.sha256(contents).hexdigest()
