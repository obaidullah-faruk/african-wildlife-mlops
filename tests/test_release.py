import json
from pathlib import Path
from typing import Any

import pytest

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.device import DeviceSummary
from wildlife_mlops.release import (
    QualityThresholds,
    ReleaseError,
    approve_candidate,
    create_candidate,
    evaluate_approved_candidate,
)
from wildlife_mlops.training import TrainingConfig


class FakeModel:
    """Small model stand-in with validation metrics for release checks."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def train(self, **kwargs: Any) -> None:
        run_dir = Path(kwargs["project"]) / kwargs["name"]
        (run_dir / "weights").mkdir(exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"best")
        (run_dir / "weights" / "last.pt").write_bytes(b"last")
        (run_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n0,0.5\n", encoding="utf-8")

    def val(self, **kwargs: Any) -> dict[str, float]:
        return {"metrics/mAP50(B)": 0.5}


def test_candidate_has_package_quality_evidence_and_checksum(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    candidate = create_candidate(
        _training(), _thresholds(), dataset, tmp_path, _device(), FakeModel
    )

    manifest = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    quality = json.loads((candidate / "quality-report.json").read_text(encoding="utf-8"))
    assert (candidate / "candidate-package.tar.gz").is_file()
    assert (candidate / "package" / "model" / "best.pt").read_bytes() == b"best"
    assert (candidate / "package" / "source" / "wildlife_mlops" / "predict.py").is_file()
    assert len(manifest["model_sha256"]) == 64
    assert quality["checks"]["validation_map50"]["passed"] is True


def test_sealed_test_evaluation_requires_approval_and_only_runs_once(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    candidate = create_candidate(
        _training(), _thresholds(), dataset, tmp_path, _device(), FakeModel
    )

    with pytest.raises(ReleaseError, match="approval"):
        evaluate_approved_candidate(candidate, dataset, tmp_path, FakeModel)

    approval = approve_candidate(candidate, "Learner")
    report = evaluate_approved_candidate(candidate, dataset, tmp_path, FakeModel)

    assert approval.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["split"] == "test"
    with pytest.raises(ReleaseError, match="already exists"):
        evaluate_approved_candidate(candidate, dataset, tmp_path, FakeModel)


def _training() -> TrainingConfig:
    return TrainingConfig(
        model_path=Path("models/pretrained.pt"),
        epochs=1,
        image_size=160,
        batch_size=1,
        workers=0,
        seed=7,
    )


def _thresholds() -> QualityThresholds:
    return QualityThresholds(schema_version=1, minimum_validation_map50=0.4)


def _dataset(root: Path) -> DatasetConfig:
    (root / "models").mkdir()
    (root / "models" / "pretrained.pt").write_bytes(b"model")
    for split in ("train", "val", "test"):
        image = root / "data" / "raw" / "wildlife" / "images" / split / "sample.ppm"
        label = root / "data" / "raw" / "wildlife" / "labels" / split / "sample.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        image.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return DatasetConfig(
        schema_version=1,
        source_url="https://example.invalid/wildlife.zip",
        source_license_reference="LICENSE.txt",
        archive_path=Path("data/raw/wildlife.zip"),
        dataset_root=Path("data/raw/wildlife"),
        expected_sha256="0" * 64,
        class_names=["buffalo"],
        splits={"train": 1, "val": 1, "test": 1},
        test_split_sealed=True,
    )


def _device() -> DeviceSummary:
    return DeviceSummary("cpu", "test", "test", "highest", "unset")
