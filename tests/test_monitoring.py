import json
from pathlib import Path

import pytest

from wildlife_mlops.monitoring import MonitoringError, measure_sampled_quality, record_recovery


def test_measure_sampled_quality_requires_matching_ground_truth(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    labels = tmp_path / "labels.jsonl"
    samples.write_text(json.dumps(_sample("trace-1", "zebra")) + "\n", encoding="utf-8")
    labels.write_text(json.dumps(_label("trace-1", "zebra")) + "\n", encoding="utf-8")

    output = measure_sampled_quality(samples, labels, tmp_path / "quality.json", 0.5)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_measure_sampled_quality_rejects_unmatched_labels(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    labels = tmp_path / "labels.jsonl"
    samples.write_text(json.dumps(_sample("trace-1", "zebra")) + "\n", encoding="utf-8")
    labels.write_text(json.dumps(_label("trace-2", "zebra")) + "\n", encoding="utf-8")

    with pytest.raises(MonitoringError, match="No ground-truth"):
        measure_sampled_quality(samples, labels, tmp_path / "quality.json", 0.5)


def test_recovery_record_requires_healthy_recovery_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-a"
    candidate.mkdir()
    checksum = "a" * 64
    (candidate / "candidate.json").write_text(
        json.dumps({"candidate_id": "candidate-a", "model_sha256": checksum}), encoding="utf-8"
    )
    health = tmp_path / "health.json"
    health.write_text(
        json.dumps(
            {
                "model_sha256": checksum,
                "model_version": f"candidate-a:sha256:{checksum}",
            }
        ),
        encoding="utf-8",
    )

    output = record_recovery("missing-candidate", candidate, health, tmp_path / "recovery.json")

    assert json.loads(output.read_text(encoding="utf-8"))["recovery_candidate"] == "candidate-a"


def _sample(trace_id: str, class_name: str) -> dict[str, object]:
    return {"trace_id": trace_id, "boxes": [_box(class_name)]}


def _label(trace_id: str, class_name: str) -> dict[str, object]:
    return {"trace_id": trace_id, "boxes": [_box(class_name)]}


def _box(class_name: str) -> dict[str, float | str]:
    return {
        "class": class_name,
        "x_min": 0.1,
        "y_min": 0.1,
        "x_max": 0.5,
        "y_max": 0.5,
    }
