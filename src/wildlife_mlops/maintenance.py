"""Local backup, migration, and restore exercise for MLflow storage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4


class MaintenanceError(RuntimeError):
    """Raised when the local MLflow maintenance exercise cannot complete."""


@dataclass(frozen=True)
class MaintenanceResult:
    """Evidence produced by the completed local maintenance exercise."""

    backup_directory: Path
    known_run_id: str
    report_path: Path


def practice_mlflow_maintenance(
    project_root: Path, tracking_uri: str, environment_file: Path
) -> MaintenanceResult:
    """Back up, migrate, restore, and verify one local MLflow run."""
    if not environment_file.is_file():
        raise MaintenanceError(f"Environment file does not exist: {environment_file}")

    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
    try:
        import mlflow
    except ImportError as error:
        raise MaintenanceError("MLflow is not installed") from error

    backup_directory = _new_backup_directory(project_root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("mlflow-maintenance")
    payload = b"MLflow backup and restore verification.\n"
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mlflow-maintenance-") as temporary_directory:
        artifact_path = Path(temporary_directory) / "known-run.txt"
        artifact_path.write_bytes(payload)
        with mlflow.start_run(run_name="backup-restore-known-run") as run:
            mlflow.log_param("maintenance.exercise", "backup_restore")
            mlflow.log_metric("maintenance.check", 1.0)
            mlflow.log_artifact(str(artifact_path))
            known_run_id = str(run.info.run_id)

        version = _compose(
            project_root, environment_file, "exec", "-T", "mlflow", "mlflow", "--version"
        ).strip()
        postgres_backup = backup_directory / "postgresql.dump"
        _backup_postgres(project_root, environment_file, postgres_backup)
        _backup_minio(project_root, environment_file, backup_directory)
        _upgrade_database(project_root, environment_file)
        _restore_postgres(project_root, environment_file, postgres_backup)
        _restore_minio(project_root, environment_file, backup_directory / "data")
        _wait_for_mlflow(tracking_uri)

        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        client.get_run(known_run_id)
        downloaded_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=known_run_id,
                artifact_path="known-run.txt",
                dst_path=temporary_directory,
            )
        )
        if hashlib.sha256(downloaded_path.read_bytes()).hexdigest() != payload_sha256:
            raise MaintenanceError("Restored artifact checksum does not match the backup run")

    report_path = backup_directory / "maintenance-report.json"
    report_path.write_text(
        json.dumps(
            {
                "mlflow_version": version,
                "known_run_id": known_run_id,
                "postgres_backup": postgres_backup.name,
                "minio_backup": "data",
                "database_upgrade_ran": True,
                "restore_verified": True,
                "artifact_sha256": payload_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return MaintenanceResult(backup_directory, known_run_id, report_path)


def _new_backup_directory(project_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_directory = (
        project_root / "artifacts" / "mlflow-maintenance" / f"{timestamp}-{uuid4().hex[:8]}"
    )
    backup_directory.mkdir(parents=True, exist_ok=False)
    return backup_directory


def _backup_postgres(project_root: Path, environment_file: Path, destination: Path) -> None:
    command = _compose_command(
        environment_file,
        "exec",
        "-T",
        "postgres",
        "/bin/sh",
        "-c",
        'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom',
    )
    with destination.open("wb") as backup_file:
        completed = subprocess.run(
            command, cwd=project_root, stdout=backup_file, stderr=subprocess.PIPE
        )
    if completed.returncode != 0:
        raise MaintenanceError(f"PostgreSQL backup failed: {completed.stderr.decode().strip()}")


def _backup_minio(project_root: Path, environment_file: Path, backup_directory: Path) -> None:
    _compose(project_root, environment_file, "cp", "minio:/data", str(backup_directory))
    if not (backup_directory / "data").is_dir():
        raise MaintenanceError("MinIO backup did not contain its data directory")


def _upgrade_database(project_root: Path, environment_file: Path) -> None:
    _compose(
        project_root,
        environment_file,
        "exec",
        "-T",
        "mlflow",
        "/bin/sh",
        "-c",
        'mlflow db upgrade "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@postgres:5432/$POSTGRES_DB"',
    )


def _restore_postgres(project_root: Path, environment_file: Path, backup_path: Path) -> None:
    _compose(project_root, environment_file, "stop", "mlflow")
    _compose(
        project_root, environment_file, "cp", str(backup_path), "postgres:/tmp/mlflow-backup.dump"
    )
    _compose(
        project_root,
        environment_file,
        "exec",
        "-T",
        "postgres",
        "/bin/sh",
        "-c",
        'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" '
        "--clean --if-exists --no-owner /tmp/mlflow-backup.dump",
    )


def _restore_minio(project_root: Path, environment_file: Path, backup_data: Path) -> None:
    _compose(project_root, environment_file, "stop", "minio")
    _compose(
        project_root,
        environment_file,
        "run",
        "--rm",
        "--no-deps",
        "--entrypoint",
        "/bin/sh",
        "minio",
        "-c",
        "rm -rf /data/* /data/.[!.]* /data/..?*",
    )
    _compose(project_root, environment_file, "up", "-d", "minio")
    _compose(project_root, environment_file, "cp", f"{backup_data}/.", "minio:/data")
    _compose(project_root, environment_file, "restart", "minio")
    _compose(project_root, environment_file, "up", "-d", "mlflow")


def _wait_for_mlflow(tracking_uri: str) -> None:
    health_url = f"{tracking_uri.rstrip('/')}/health"
    for _ in range(30):
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(1)
    raise MaintenanceError(f"MLflow did not become healthy at {health_url}")


def _compose(project_root: Path, environment_file: Path, *arguments: str) -> str:
    completed = subprocess.run(
        _compose_command(environment_file, *arguments),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise MaintenanceError(f"Compose command failed: {' '.join(arguments)}: {detail}")
    return completed.stdout


def _compose_command(environment_file: Path, *arguments: str) -> list[str]:
    return ["docker", "compose", "--env-file", str(environment_file), *arguments]
