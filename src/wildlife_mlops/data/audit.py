"""Exact and perceptual duplicate checks across published data splits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import imagehash
from PIL import Image

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.validate import image_paths


def audit_splits(config: DatasetConfig, project_root: Path, threshold: int = 6) -> Path:
    """Write exact and perceptual-hash duplicate findings for every split pair."""
    dataset_root = project_root / config.dataset_root
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    perceptual_hashes: list[tuple[str, str]] = []
    for split, image_path in image_paths(dataset_root, config.splits):
        relative_path = str(image_path.relative_to(project_root))
        exact_hashes[_sha256(image_path)].append(f"{split}:{relative_path}")
        with Image.open(image_path) as image:
            perceptual_hashes.append(
                (f"{split}:{relative_path}", str(imagehash.phash(image.convert("RGB"))))
            )

    near_duplicates = []
    for left_index, (left_path, left_hash) in enumerate(perceptual_hashes):
        for right_path, right_hash in perceptual_hashes[left_index + 1 :]:
            if left_path.split(":", 1)[0] == right_path.split(":", 1)[0]:
                continue
            distance = _hamming_distance(left_hash, right_hash)
            if distance <= threshold:
                near_duplicates.append(
                    {"left": left_path, "right": right_path, "distance": distance}
                )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "perceptual_hash": "phash",
        "near_duplicate_threshold": threshold,
        "exact_duplicate_groups": {
            digest: paths for digest, paths in sorted(exact_hashes.items()) if len(paths) > 1
        },
        "cross_split_near_duplicates": near_duplicates,
    }
    destination = project_root / "data" / "manifests" / "split-integrity-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _sha256(path: Path) -> str:
    """Return the exact content hash of an image."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hamming_distance(left: str, right: str) -> int:
    """Return the bit distance between two hexadecimal perceptual hashes."""
    return (int(left, 16) ^ int(right, 16)).bit_count()
