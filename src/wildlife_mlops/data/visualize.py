"""Deterministic ground-truth contact sheets for automated review artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.validate import (
    BoundingBox,
    image_paths,
    label_path_for_image,
    parse_yolo_label,
)

CLASS_COLORS = ("#e74c3c", "#3498db", "#2ecc71", "#f1c40f")
TILE_SIZE = (320, 240)
GRID_COLUMNS = 4


def create_contact_sheets(config: DatasetConfig, project_root: Path) -> list[Path]:
    """Render deterministic split, class, and edge-case ground-truth sheets."""
    dataset_root = project_root / config.dataset_root
    records = _records(config, dataset_root)
    output_root = project_root / "artifacts" / "ground-truth"
    output_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for split in config.splits:
        split_records = [record for record in records if record[0] == split]
        selected = sorted(split_records, key=lambda record: _stable_key(record[1]))[:16]
        created.append(_write_sheet(output_root / f"{split}-sample-grid.png", selected, config))

    for class_id, class_name in enumerate(config.class_names):
        class_records = [
            record for record in records if any(box.class_id == class_id for box in record[2])
        ]
        selected = sorted(class_records, key=lambda record: _stable_key(record[1]))[:5]
        created.append(_write_sheet(output_root / f"class-{class_name}.png", selected, config))

    by_smallest = sorted(records, key=lambda record: _smallest_box_area(record[2]))[:16]
    by_largest = sorted(records, key=lambda record: _largest_box_area(record[2]), reverse=True)[:16]
    by_crowded = sorted(records, key=lambda record: len(record[2]), reverse=True)[:16]
    by_aspect = sorted(
        records, key=lambda record: abs(_image_aspect_ratio(record[1]) - 1), reverse=True
    )[:16]
    created.extend(
        [
            _write_sheet(output_root / "smallest-boxes.png", by_smallest, config),
            _write_sheet(output_root / "largest-boxes.png", by_largest, config),
            _write_sheet(output_root / "crowded-images.png", by_crowded, config),
            _write_sheet(output_root / "unusual-aspect-ratios.png", by_aspect, config),
        ]
    )
    return created


def _records(
    config: DatasetConfig, dataset_root: Path
) -> list[tuple[str, Path, list[BoundingBox]]]:
    """Load valid labels for every image in deterministic order."""
    records: list[tuple[str, Path, list[BoundingBox]]] = []
    for split, image_path in image_paths(dataset_root, config.splits):
        boxes, _ = parse_yolo_label(
            label_path_for_image(dataset_root, split, image_path), len(config.class_names)
        )
        records.append((split, image_path, boxes))
    return records


def _write_sheet(
    destination: Path, records: list[tuple[str, Path, list[BoundingBox]]], config: DatasetConfig
) -> Path:
    """Draw labeled records into a compact contact sheet."""
    rows = max(1, (len(records) + GRID_COLUMNS - 1) // GRID_COLUMNS)
    sheet = Image.new("RGB", (GRID_COLUMNS * TILE_SIZE[0], rows * TILE_SIZE[1]), "#202124")
    for index, (_, image_path, boxes) in enumerate(records):
        tile = _annotated_tile(image_path, boxes, config)
        x_offset = (index % GRID_COLUMNS) * TILE_SIZE[0]
        y_offset = (index // GRID_COLUMNS) * TILE_SIZE[1]
        sheet.paste(tile, (x_offset, y_offset))
    sheet.save(destination)
    return destination


def _annotated_tile(
    image_path: Path, boxes: list[BoundingBox], config: DatasetConfig
) -> Image.Image:
    """Resize an image and draw its normalized YOLO boxes."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((TILE_SIZE[0], TILE_SIZE[1] - 20))
    tile = Image.new("RGB", TILE_SIZE, "#202124")
    x_offset = (TILE_SIZE[0] - image.width) // 2
    tile.paste(image, (x_offset, 20))
    draw = ImageDraw.Draw(tile)
    draw.text((4, 3), image_path.name, fill="white")
    for box in boxes:
        color = CLASS_COLORS[box.class_id % len(CLASS_COLORS)]
        x_min = x_offset + int((box.x_center - box.width / 2) * image.width)
        y_min = 20 + int((box.y_center - box.height / 2) * image.height)
        x_max = x_offset + int((box.x_center + box.width / 2) * image.width)
        y_max = 20 + int((box.y_center + box.height / 2) * image.height)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=2)
        draw.text((x_min, max(20, y_min - 12)), config.class_names[box.class_id], fill=color)
    return tile


def _stable_key(path: Path) -> str:
    """Return a deterministic content-independent order for contact sheets."""
    return hashlib.sha256(path.as_posix().encode()).hexdigest()


def _smallest_box_area(boxes: list[BoundingBox]) -> float:
    """Return the smallest normalized box area, treating empty images as largest."""
    return min((box.width * box.height for box in boxes), default=float("inf"))


def _largest_box_area(boxes: list[BoundingBox]) -> float:
    """Return the largest normalized box area, treating empty images as zero."""
    return max((box.width * box.height for box in boxes), default=0.0)


def _image_aspect_ratio(path: Path) -> float:
    """Return image width divided by height."""
    with Image.open(path) as image:
        return float(image.width) / float(image.height)
