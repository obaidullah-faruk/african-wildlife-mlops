"""Dataset inventory and deterministic smoke-subset manifest generation."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.download import sha256_file
from wildlife_mlops.data.validate import image_paths, label_path_for_image, parse_yolo_label


def create_inventory(config: DatasetConfig, project_root: Path) -> tuple[Path, Path]:
    """Write an image-level Parquet inventory and a compact JSON summary."""
    dataset_root = project_root / config.dataset_root
    rows: list[dict[str, object]] = []
    box_areas: list[float] = []
    box_aspect_ratios: list[float] = []
    image_aspect_ratios: list[float] = []
    boxes_per_image: list[int] = []
    image_widths: list[int] = []
    image_heights: list[int] = []
    boxes_by_class = {class_name: 0 for class_name in config.class_names}
    images_by_split = {split: 0 for split in config.splits}
    labels_by_split = {split: 0 for split in config.splits}

    for split, image_path in image_paths(dataset_root, config.splits):
        label_path = label_path_for_image(dataset_root, split, image_path)
        boxes, _ = (
            parse_yolo_label(label_path, len(config.class_names))
            if label_path.exists()
            else ([], [])
        )
        with Image.open(image_path) as image:
            width, height = image.size
        image_aspect_ratio = width / height
        image_aspect_ratios.append(image_aspect_ratio)
        image_widths.append(width)
        image_heights.append(height)
        images_by_split[split] += 1
        if label_path.exists():
            labels_by_split[split] += 1
        class_counts = {class_name: 0 for class_name in config.class_names}
        for box in boxes:
            class_name = config.class_names[box.class_id]
            class_counts[class_name] += 1
            boxes_by_class[class_name] += 1
            box_areas.append(box.width * box.height)
            box_aspect_ratios.append(box.width / box.height)
        boxes_per_image.append(len(boxes))
        rows.append(
            {
                "split": split,
                "image_path": str(image_path.relative_to(project_root)),
                "label_path": str(label_path.relative_to(project_root)),
                "width": width,
                "height": height,
                "aspect_ratio": image_aspect_ratio,
                "box_count": len(boxes),
                "class_counts": json.dumps(class_counts, sort_keys=True),
            }
        )

    archive_checksum = sha256_file(project_root / config.archive_path)
    manifests_root = project_root / "data" / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    parquet_path = manifests_root / "source-inventory.parquet"
    summary_path = manifests_root / "source-inventory-summary.json"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_url": config.source_url,
        "retrieval_date": datetime.now(UTC).date().isoformat(),
        "archive_sha256": archive_checksum,
        "class_names": config.class_names,
        "expected_images_by_split": config.splits,
        "images_by_split": images_by_split,
        "labels_by_split": labels_by_split,
        "boxes_by_class": boxes_by_class,
        "boxes_per_image": _distribution(boxes_per_image),
        "image_width": _distribution(image_widths),
        "image_height": _distribution(image_heights),
        "image_aspect_ratio": _distribution(image_aspect_ratios),
        "box_area": _distribution(box_areas),
        "box_aspect_ratio": _distribution(box_aspect_ratios),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return parquet_path, summary_path


def create_smoke_manifest(config: DatasetConfig, project_root: Path) -> Path:
    """Select a deterministic training-only subset without copying source images."""
    dataset_root = project_root / config.dataset_root
    train_images = [
        path for split, path in image_paths(dataset_root, config.splits) if split == "train"
    ]
    archive_checksum = sha256_file(project_root / config.archive_path)
    ordered_images = sorted(train_images, key=lambda path: _selection_key(path, archive_checksum))
    selected_images = ordered_images[: config.smoke_subset_size]
    if len(selected_images) != config.smoke_subset_size:
        raise ValueError(
            f"Expected at least {config.smoke_subset_size} training images, "
            f"found {len(selected_images)}"
        )
    manifest_path = project_root / "data" / "manifests" / "smoke-subset.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source_archive_sha256": archive_checksum,
        "source_split": "train",
        "image_count": len(selected_images),
        "images": [str(path.relative_to(project_root)) for path in selected_images],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _selection_key(image_path: Path, archive_checksum: str) -> str:
    """Return a stable selection key tied to the immutable archive checksum."""
    return hashlib.sha256(f"{archive_checksum}:{image_path.as_posix()}".encode()).hexdigest()


def _distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    """Return a small descriptive distribution with deterministic percentile indices."""
    ordered_values = sorted(float(value) for value in values)
    if not ordered_values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(ordered_values),
        "min": ordered_values[0],
        "max": ordered_values[-1],
        "mean": statistics.fmean(ordered_values),
        "p50": _percentile(ordered_values, 0.50),
        "p95": _percentile(ordered_values, 0.95),
    }


def _percentile(values: list[float], quantile: float) -> float:
    """Return a nearest-rank percentile for an already sorted value list."""
    index = round((len(values) - 1) * quantile)
    return values[index]
