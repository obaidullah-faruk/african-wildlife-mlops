"""Provenance values attached to local MLflow training runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.manifest import ContentManifestError, manifest_checksum
from wildlife_mlops.device import DeviceSummary


class LineageError(RuntimeError):
    """Raised when required local training provenance cannot be collected."""


def collect_local_lineage_tags(
    project_root: Path,
    dataset_config: DatasetConfig,
    resolved_config_path: Path,
    base_weights_path: Path,
    seed: int,
    device_summary: DeviceSummary,
    parent_run_id: str,
    trigger_type: str,
    trigger_id: str,
) -> dict[str, str]:
    """Collect the provenance needed to explain one native local training run."""
    return {
        "lineage.git_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "lineage.git_dirty": str(bool(_git_output(project_root, "status", "--porcelain"))).lower(),
        "lineage.dvc_revision": sha256_file(project_root / "data" / "raw.dvc"),
        "lineage.source_archive_sha256": dataset_config.expected_sha256,
        "lineage.prepared_manifest_sha256": _content_manifest_checksum(project_root),
        "lineage.config_sha256": sha256_file(resolved_config_path),
        "lineage.random_seed": str(seed),
        "lineage.base_weights_name": base_weights_path.name,
        "lineage.base_weights_sha256": sha256_file(base_weights_path),
        "runtime.python_version": sys.version.split()[0],
        "runtime.pytorch_version": device_summary.pytorch_version,
        "runtime.ultralytics_version": device_summary.ultralytics_version,
        "runtime.os": platform.platform(),
        "runtime.architecture": platform.machine(),
        "runtime.device": device_summary.device,
        "execution.training_container_digest": "not_applicable",
        "lineage.parent_run_id": parent_run_id,
        "trigger.type": trigger_type,
        "trigger.id": trigger_id,
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a required nonempty file."""
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise LineageError(f"Unable to checksum lineage file {path}: {error}") from error
    if not contents:
        raise LineageError(f"Unable to checksum empty lineage file: {path}")
    return hashlib.sha256(contents).hexdigest()


def _content_manifest_checksum(project_root: Path) -> str:
    """Return the verified data-manifest identity used by native training."""
    try:
        return manifest_checksum(project_root)
    except ContentManifestError as error:
        raise LineageError(str(error)) from error


def _git_output(project_root: Path, *arguments: str) -> str:
    """Run one read-only Git command and return its trimmed output."""
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=project_root, text=True, stderr=subprocess.PIPE
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise LineageError(f"Unable to collect Git lineage: {error}") from error
