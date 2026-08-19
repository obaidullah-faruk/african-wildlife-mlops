import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from wildlife_mlops.deployment import record_rollback
from wildlife_mlops.service import create_app


class FakeResult:
    """Prediction result with no detected boxes."""

    boxes = None


class FakeModel:
    """Minimal loaded model used to verify the API contract."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def predict(self, source: Image.Image, **_: Any) -> list[FakeResult]:
        assert isinstance(source, Image.Image)
        return [FakeResult()]


def test_service_health_and_prediction_prove_pinned_identity(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, "candidate-a")
    app = create_app(candidate, FakeModel)

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post(
            "/predict",
            content=_png_bytes(),
            headers={"X-Image-Name": "zebra.png", "Content-Type": "application/octet-stream"},
        )

    assert health.status_code == 200
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == health.json()["model_version"]
    assert prediction.json()["model_sha256"] == health.json()["model_sha256"]


def test_rollback_record_requires_prediction_from_rollback_target(tmp_path: Path) -> None:
    model_a = _candidate(tmp_path, "candidate-a")
    model_b = _candidate(tmp_path, "candidate-b", b"second-model")
    manifest_a = json.loads((model_a / "candidate.json").read_text(encoding="utf-8"))
    prediction = tmp_path / "prediction-after-rollback.json"
    prediction.write_text(
        json.dumps(
            {
                "model_sha256": manifest_a["model_sha256"],
                "model_version": f"candidate-a:sha256:{manifest_a['model_sha256']}",
            }
        ),
        encoding="utf-8",
    )

    output = record_rollback(model_b, model_a, prediction, tmp_path / "rollback.json")

    assert json.loads(output.read_text(encoding="utf-8"))["to_candidate"] == "candidate-a"


def _candidate(root: Path, candidate_id: str, model_bytes: bytes = b"model") -> Path:
    candidate = root / candidate_id
    checkpoint = candidate / "package" / "model" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(model_bytes)
    checksum = hashlib.sha256(model_bytes).hexdigest()
    (candidate / "candidate.json").write_text(
        json.dumps({"candidate_id": candidate_id, "model_sha256": checksum}), encoding="utf-8"
    )
    (candidate / "approval.json").write_text("{}", encoding="utf-8")
    (candidate / "test-evaluation.json").write_text("{}", encoding="utf-8")
    return candidate


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    return buffer.getvalue()
