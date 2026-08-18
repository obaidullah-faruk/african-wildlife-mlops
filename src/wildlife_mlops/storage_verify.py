"""Verify that MLflow metadata and artifacts use separate persistent stores."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


class StorageVerificationError(RuntimeError):
    """Raised when the local MLflow storage verification cannot complete."""


@dataclass(frozen=True)
class StorageVerificationResult:
    """Evidence produced by one completed storage verification."""

    run_id: str
    artifact_sha256: str
    postgres_rows: list[dict[str, str]]
    minio_listing: str
    report_path: Path


def verify_storage_responsibilities(
    project_root: Path,
    tracking_uri: str,
    environment_file: Path,
    artifact_bytes: int = 1_048_576,
) -> StorageVerificationResult:
    """Prove metadata and artifacts survive recreation of the MLflow container."""
    if artifact_bytes < 1:
        raise StorageVerificationError("artifact_bytes must be at least 1")
    if not environment_file.is_file():
        raise StorageVerificationError(f"Environment file does not exist: {environment_file}")

    try:
        os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
        import mlflow
    except ImportError as error:
        raise StorageVerificationError("MLflow is not installed") from error

    payload = b"wildlife-mlops-storage-check\n" * ((artifact_bytes // 30) + 1)
    payload = payload[:artifact_bytes]
    payload_sha256 = hashlib.sha256(payload).hexdigest()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("mlflow-storage-verification")
    with tempfile.TemporaryDirectory(prefix="mlflow-storage-verify-") as temporary_directory:
        artifact_path = Path(temporary_directory) / "storage-check.bin"
        artifact_path.write_bytes(payload)
        with mlflow.start_run(run_name="storage-responsibilities") as run:
            mlflow.log_param("verification.artifact_bytes", artifact_bytes)
            mlflow.log_metric("verification.artifact_bytes", float(artifact_bytes))
            mlflow.log_artifact(str(artifact_path))
            run_id = str(run.info.run_id)
            experiment_id = str(run.info.experiment_id)

        _validate_run_id(run_id)
        postgres_rows = _postgres_metadata(project_root, environment_file, run_id)
        if {row["record_type"] for row in postgres_rows} != {"metric", "param"}:
            raise StorageVerificationError(
                "PostgreSQL did not return both parameter and metric rows"
            )

        minio_listing = _minio_listing(project_root, environment_file, experiment_id, run_id)
        if "storage-check.bin" not in minio_listing:
            raise StorageVerificationError("MinIO does not list the verification artifact")

        _recreate_mlflow(project_root, environment_file)
        _wait_for_mlflow(tracking_uri)
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        client.get_run(run_id)
        downloaded_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="storage-check.bin",
                dst_path=temporary_directory,
            )
        )
        downloaded_sha256 = hashlib.sha256(downloaded_path.read_bytes()).hexdigest()
        if downloaded_sha256 != payload_sha256:
            raise StorageVerificationError(
                "Downloaded artifact checksum does not match uploaded data"
            )

    report_path = _write_report(
        project_root,
        {
            "run_id": run_id,
            "artifact_bytes": artifact_bytes,
            "artifact_sha256": payload_sha256,
            "postgres_rows": postgres_rows,
            "minio_listing": minio_listing,
            "mlflow_container_recreated": True,
            "artifact_download_sha256": downloaded_sha256,
        },
    )
    return StorageVerificationResult(
        run_id=run_id,
        artifact_sha256=payload_sha256,
        postgres_rows=postgres_rows,
        minio_listing=minio_listing,
        report_path=report_path,
    )


def _postgres_metadata(
    project_root: Path, environment_file: Path, run_id: str
) -> list[dict[str, str]]:
    query = (
        "SELECT 'param' AS record_type, key, value FROM params "
        f"WHERE run_uuid = '{run_id}' "
        "UNION ALL "
        "SELECT 'metric' AS record_type, key, value::text FROM metrics "
        f"WHERE run_uuid = '{run_id}' ORDER BY record_type, key"
    )
    output = _compose(
        project_root,
        environment_file,
        "exec",
        "-T",
        "-e",
        f"VERIFY_SQL={query}",
        "postgres",
        "/bin/sh",
        "-c",
        'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -F "|" -c "$VERIFY_SQL"',
    )
    return parse_postgres_rows(output)


def parse_postgres_rows(output: str) -> list[dict[str, str]]:
    """Parse pipe-separated PostgreSQL evidence into explicit records."""
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        record_type, separator, remainder = line.partition("|")
        key, separator_two, value = remainder.partition("|")
        if not separator or not separator_two or record_type not in {"metric", "param"}:
            raise StorageVerificationError(f"Unexpected PostgreSQL output: {line!r}")
        rows.append({"record_type": record_type, "key": key, "value": value})
    return rows


def _minio_listing(
    project_root: Path, environment_file: Path, experiment_id: str, run_id: str
) -> str:
    output = _compose(
        project_root,
        environment_file,
        "run",
        "--rm",
        "--no-deps",
        "--entrypoint",
        "/bin/sh",
        "minio-init",
        "-c",
        'mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null '
        '&& mc ls --recursive "local/$MLFLOW_S3_BUCKET/'
        + experiment_id
        + "/"
        + run_id
        + '/artifacts"',
    )
    return output.strip()


def _recreate_mlflow(project_root: Path, environment_file: Path) -> None:
    _compose(project_root, environment_file, "stop", "mlflow")
    _compose(project_root, environment_file, "rm", "-f", "mlflow")
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
    raise StorageVerificationError(f"MLflow did not become healthy at {health_url}")


def _compose(project_root: Path, environment_file: Path, *arguments: str) -> str:
    command = ["docker", "compose", "--env-file", str(environment_file), *arguments]
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise StorageVerificationError(f"Compose command failed: {' '.join(arguments)}: {detail}")
    return completed.stdout


def _validate_run_id(run_id: str) -> None:
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise StorageVerificationError(f"Unexpected MLflow run ID: {run_id!r}")


def _write_report(project_root: Path, report: dict[str, object]) -> Path:
    report_directory = project_root / "artifacts" / "mlflow-storage-verification"
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / "storage-verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path
