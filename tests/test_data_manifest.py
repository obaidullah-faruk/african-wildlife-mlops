import hashlib
from pathlib import Path

import pytest

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.manifest import (
    CONTENT_MANIFEST_CHECKSUM_PATH,
    CONTENT_MANIFEST_PATH,
    ContentManifestError,
    create_content_manifest,
    load_verified_manifest,
)


def test_content_manifest_is_canonical_and_records_image_label_details(tmp_path: Path) -> None:
    config = _write_dataset(tmp_path)

    manifest_path, checksum_path = create_content_manifest(config, tmp_path)
    manifest = load_verified_manifest(config, tmp_path)

    assert manifest_path == tmp_path / CONTENT_MANIFEST_PATH
    assert checksum_path == tmp_path / CONTENT_MANIFEST_CHECKSUM_PATH
    assert checksum_path.read_text(encoding="utf-8").strip().isalnum()
    assert manifest["entries"] == [
        {
            "class_counts": {"buffalo": 1},
            "dimensions": {"height": 1, "width": 1},
            "image_sha256": manifest["entries"][0]["image_sha256"],
            "label_path": "labels/train/sample.txt",
            "label_sha256": manifest["entries"][0]["label_sha256"],
            "path": "images/train/sample.ppm",
            "source_license_reference": "LICENSE.txt",
            "source_url": "https://example.invalid/wildlife.zip",
            "split": "train",
        }
    ]


def test_verified_manifest_rejects_a_changed_label(tmp_path: Path) -> None:
    config = _write_dataset(tmp_path)
    create_content_manifest(config, tmp_path)
    label_path = tmp_path / "data" / "raw" / "wildlife" / "labels" / "train" / "sample.txt"
    label_path.write_text("0 0.25 0.5 0.5 0.5\n", encoding="utf-8")

    with pytest.raises(ContentManifestError, match="label hash changed"):
        load_verified_manifest(config, tmp_path)


def test_verified_manifest_rejects_an_unlisted_image(tmp_path: Path) -> None:
    config = _write_dataset(tmp_path)
    create_content_manifest(config, tmp_path)
    image_path = tmp_path / "data" / "raw" / "wildlife" / "images" / "train" / "added.ppm"
    label_path = tmp_path / "data" / "raw" / "wildlife" / "labels" / "train" / "added.txt"
    image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    with pytest.raises(ContentManifestError, match="does not match dataset image files"):
        load_verified_manifest(config, tmp_path)


def _write_dataset(project_root: Path) -> DatasetConfig:
    image_path = project_root / "data" / "raw" / "wildlife" / "images" / "train" / "sample.ppm"
    label_path = project_root / "data" / "raw" / "wildlife" / "labels" / "train" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    archive_path = project_root / "data" / "raw" / "wildlife.zip"
    archive_path.write_bytes(b"archive")
    return DatasetConfig(
        schema_version=1,
        source_url="https://example.invalid/wildlife.zip",
        source_license_reference="LICENSE.txt",
        archive_path=Path("data/raw/wildlife.zip"),
        dataset_root=Path("data/raw/wildlife"),
        expected_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        class_names=["buffalo"],
        splits={"train": 1},
        smoke_subset_size=16,
        test_split_sealed=True,
    )
