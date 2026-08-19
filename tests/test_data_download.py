import hashlib
import zipfile
from pathlib import Path

import pytest

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.download import DatasetDownloadError, download_and_extract


def test_download_extracts_only_a_verified_archive(tmp_path: Path) -> None:
    source_archive = tmp_path / "source.zip"
    with zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr("images/train/sample.ppm", "P3\n1 1\n255\n0 0 0\n")
        archive.writestr("labels/train/sample.txt", "0 0.5 0.5 0.5 0.5\n")
    checksum = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    config = _config(source_archive.as_uri(), checksum)

    dataset_root = download_and_extract(config, tmp_path)

    assert (dataset_root / "images" / "train" / "sample.ppm").is_file()
    assert (tmp_path / "data" / "raw" / "wildlife.zip").is_file()


def test_checksum_mismatch_refuses_download_and_extraction(tmp_path: Path) -> None:
    source_archive = tmp_path / "source.zip"
    with zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr("images/train/sample.ppm", "P3\n1 1\n255\n0 0 0\n")
    config = _config(source_archive.as_uri(), "0" * 64)

    with pytest.raises(DatasetDownloadError, match="checksum mismatch"):
        download_and_extract(config, tmp_path)

    assert not (tmp_path / "data" / "raw" / "wildlife.zip").exists()
    assert not (tmp_path / "data" / "raw" / "wildlife").exists()


def _config(source_url: str, checksum: str) -> DatasetConfig:
    return DatasetConfig(
        schema_version=1,
        source_url=source_url,
        source_license_reference="LICENSE.txt",
        archive_path=Path("data/raw/wildlife.zip"),
        dataset_root=Path("data/raw/wildlife"),
        expected_sha256=checksum,
        class_names=["buffalo"],
        splits={"train": 1},
        test_split_sealed=True,
    )
