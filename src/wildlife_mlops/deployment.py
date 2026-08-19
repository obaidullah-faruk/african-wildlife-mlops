"""Evidence records for a manual local rollback exercise."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wildlife_mlops.release import ReleaseError, _read_json


class DeploymentError(RuntimeError):
    """Raised when rollback evidence does not prove the selected model is active."""


def record_rollback(
    from_candidate: Path, to_candidate: Path, prediction_path: Path, output_path: Path
) -> Path:
    """Record a verified prediction proving a rollback selected model A again."""
    from_manifest = _candidate_manifest(from_candidate)
    to_manifest = _candidate_manifest(to_candidate)
    response = _prediction_response(prediction_path)
    expected_checksum = to_manifest["model_sha256"]
    expected_version = f"{to_manifest['candidate_id']}:sha256:{expected_checksum}"
    if response.get("model_sha256") != expected_checksum:
        raise DeploymentError("Prediction checksum does not match the rollback target")
    if response.get("model_version") != expected_version:
        raise DeploymentError("Prediction model version does not match the rollback target")
    if output_path.exists():
        raise DeploymentError(f"Rollback evidence already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "from_candidate": from_manifest["candidate_id"],
        "to_candidate": to_manifest["candidate_id"],
        "active_model_sha256": expected_checksum,
        "verified_prediction": str(prediction_path),
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _candidate_manifest(candidate_dir: Path) -> dict[str, str]:
    try:
        manifest = _read_json(candidate_dir / "candidate.json")
    except ReleaseError as error:
        raise DeploymentError(str(error)) from error
    candidate_id = manifest.get("candidate_id")
    checksum = manifest.get("model_sha256")
    if not isinstance(candidate_id, str) or not isinstance(checksum, str):
        raise DeploymentError(f"Candidate manifest lacks identity: {candidate_dir}")
    return {"candidate_id": candidate_id, "model_sha256": checksum}


def _prediction_response(prediction_path: Path) -> dict[str, Any]:
    if not prediction_path.is_file():
        raise DeploymentError(f"Prediction response does not exist: {prediction_path}")
    try:
        response = json.loads(prediction_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DeploymentError(f"Prediction response is invalid JSON: {prediction_path}") from error
    if not isinstance(response, dict):
        raise DeploymentError("Prediction response must be a JSON object")
    return response
