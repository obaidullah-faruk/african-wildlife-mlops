"""Validation for the YOLO image and label contract."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from wildlife_mlops.data.config import DatasetConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm"}


@dataclass(frozen=True)
class BoundingBox:
    """One normalized YOLO bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ValidationIssue:
    """A precise contract violation found in an image or label."""

    code: str
    path: str
    detail: str
    row: int | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Machine-readable result of validating the complete dataset."""

    image_count: int
    label_count: int
    box_count: int
    issues: list[ValidationIssue]

    @property
    def passed(self) -> bool:
        """Return whether no contract errors were found."""
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible validation report."""
        return {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "passed": self.passed,
            "image_count": self.image_count,
            "label_count": self.label_count,
            "box_count": self.box_count,
            "issue_count": len(self.issues),
            "issues": [asdict(issue) for issue in self.issues],
        }


def image_paths(dataset_root: Path, splits: dict[str, int]) -> list[tuple[str, Path]]:
    """Return all supported image files in deterministic split/path order."""
    paths: list[tuple[str, Path]] = []
    for split in splits:
        split_root = dataset_root / "images" / split
        if not split_root.is_dir():
            continue
        paths.extend(
            (split, path)
            for path in sorted(split_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return paths


def label_path_for_image(dataset_root: Path, split: str, image_path: Path) -> Path:
    """Map an image path to its YOLO label path while preserving subdirectories."""
    relative_path = image_path.relative_to(dataset_root / "images" / split)
    return dataset_root / "labels" / split / relative_path.with_suffix(".txt")


def parse_yolo_label(
    label_path: Path, class_count: int
) -> tuple[list[BoundingBox], list[ValidationIssue]]:
    """Parse a label file and return every malformed-row error without skipping it."""
    boxes: list[BoundingBox] = []
    issues: list[ValidationIssue] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        return [], [ValidationIssue("label_encoding", str(label_path), str(error))]

    for row, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            issues.append(
                ValidationIssue("malformed_row", str(label_path), "expected five fields", row)
            )
            continue
        try:
            class_id = int(fields[0])
        except ValueError:
            issues.append(ValidationIssue("invalid_class_id", str(label_path), fields[0], row))
            continue
        try:
            x_center, y_center, width, height = (float(field) for field in fields[1:])
        except ValueError:
            issues.append(ValidationIssue("invalid_coordinate", str(label_path), line, row))
            continue
        values = (x_center, y_center, width, height)
        class_is_valid = 0 <= class_id < class_count
        values_are_finite = all(math.isfinite(value) for value in values)
        sizes_are_positive = width > 0 and height > 0
        if not class_is_valid:
            issues.append(
                ValidationIssue("class_out_of_range", str(label_path), str(class_id), row)
            )
        if not values_are_finite:
            issues.append(ValidationIssue("non_finite_coordinate", str(label_path), line, row))
        elif not all(0 < value <= 1 for value in values):
            issues.append(ValidationIssue("coordinate_out_of_range", str(label_path), line, row))
        elif (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            issues.append(ValidationIssue("box_outside_image", str(label_path), line, row))
        if class_is_valid and values_are_finite and sizes_are_positive:
            boxes.append(BoundingBox(class_id, x_center, y_center, width, height))
    return boxes, issues


def validate_dataset(config: DatasetConfig, project_root: Path) -> ValidationResult:
    """Validate image decoding, label pairing, and normalized YOLO boxes."""
    dataset_root = project_root / config.dataset_root
    issues: list[ValidationIssue] = []
    boxes_total = 0
    labels_seen: set[Path] = set()
    image_list = image_paths(dataset_root, config.splits)

    for split, image_path in image_list:
        relative_image_path = image_path.relative_to(dataset_root)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            issues.append(ValidationIssue("image_decode", str(relative_image_path), str(error)))

        label_path = label_path_for_image(dataset_root, split, image_path)
        if not label_path.is_file():
            issues.append(
                ValidationIssue("missing_label", str(relative_image_path), str(label_path))
            )
            continue
        labels_seen.add(label_path)
        boxes, label_issues = parse_yolo_label(label_path, len(config.class_names))
        boxes_total += len(boxes)
        issues.extend(label_issues)

    all_labels = {
        path
        for split in config.splits
        for path in (dataset_root / "labels" / split).rglob("*.txt")
        if (dataset_root / "labels" / split).is_dir()
    }
    for orphan_path in sorted(all_labels - labels_seen):
        issues.append(
            ValidationIssue(
                "orphan_label", str(orphan_path.relative_to(dataset_root)), "no matching image"
            )
        )

    return ValidationResult(len(image_list), len(all_labels), boxes_total, issues)


def write_validation_report(result: ValidationResult, destination: Path) -> None:
    """Write the validation result as stable, machine-readable JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

