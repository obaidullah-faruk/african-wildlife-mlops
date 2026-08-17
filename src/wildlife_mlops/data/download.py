"""Verified download and safe extraction of a versioned dataset archive."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from wildlife_mlops.data.config import DatasetConfig


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset archive cannot be verified or extracted safely."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract(config: DatasetConfig, project_root: Path) -> Path:
    """Download, verify, and safely extract the configured archive."""
    archive_path = project_root / config.archive_path
    dataset_root = project_root / config.dataset_root
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        _verify_archive(archive_path, config.expected_sha256)
    else:
        _download_verified_archive(config.source_url, archive_path, config.expected_sha256)

    if dataset_root.exists():
        return dataset_root
    _extract_archive(archive_path, dataset_root)
    return dataset_root


def _download_verified_archive(url: str, destination: Path, expected_sha256: str) -> None:
    """Download into a temporary file and atomically publish a verified archive."""
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            with urllib.request.urlopen(url) as response:
                shutil.copyfileobj(response, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            _verify_archive(temporary_path, expected_sha256)
            os.replace(temporary_path, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary_path.unlink(missing_ok=True)
            raise DatasetDownloadError(
                f"Unable to download dataset archive from {url}: {error}"
            ) from error
        except DatasetDownloadError:
            temporary_path.unlink(missing_ok=True)
            raise


def _verify_archive(archive_path: Path, expected_sha256: str) -> None:
    """Verify archive type and immutable checksum before any extraction."""
    if not zipfile.is_zipfile(archive_path):
        raise DatasetDownloadError(f"Archive is not a ZIP file: {archive_path}")
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise DatasetDownloadError(
            "Archive checksum mismatch for "
            f"{archive_path}: expected {expected_sha256}, got {actual_sha256}"
        )


def _extract_archive(archive_path: Path, dataset_root: Path) -> None:
    """Extract a verified ZIP archive without permitting path traversal."""
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{dataset_root.name}.", dir=dataset_root.parent)
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target_path = temporary_root / member.filename
                if not target_path.resolve().is_relative_to(temporary_root.resolve()):
                    raise DatasetDownloadError(
                        f"Archive member escapes dataset root: {member.filename}"
                    )
            archive.extractall(temporary_root)

        extracted_root = _find_dataset_root(temporary_root)
        os.replace(extracted_root, dataset_root)
    except (OSError, zipfile.BadZipFile) as error:
        raise DatasetDownloadError(f"Unable to extract archive {archive_path}: {error}") from error
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _find_dataset_root(temporary_root: Path) -> Path:
    """Return the archive root containing the expected image directory."""
    if (temporary_root / "images").is_dir():
        return temporary_root
    children = [child for child in temporary_root.iterdir() if child.is_dir()]
    if len(children) == 1 and (children[0] / "images").is_dir():
        return children[0]
    raise DatasetDownloadError("Archive does not contain an expected images directory")
