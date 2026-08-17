import json
from pathlib import Path
from typing import Any

import pytest

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.device import DeviceSummary
from wildlife_mlops.smoke import SmokeTrainConfig, SmokeTrainError, run_smoke_train


class FakeModel:
    """Minimal model that writes the artifacts expected from one training run."""

    def __init__(self, write_results: bool) -> None:
        self.write_results = write_results

    def train(self, **kwargs: Any) -> None:
        run_dir = Path(kwargs["project"]) / kwargs["name"]
        (run_dir / "weights").mkdir(exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"best")
        (run_dir / "weights" / "last.pt").write_bytes(b"last")
        (run_dir / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
        if self.write_results:
            (run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")


def test_smoke_train_writes_and_verifies_required_artifacts(tmp_path: Path) -> None:
    config, dataset_config, device_summary = _create_inputs(tmp_path)

    run_dir = run_smoke_train(
        config,
        dataset_config,
        tmp_path,
        device_summary,
        lambda _: FakeModel(write_results=True),
    )

    assert (run_dir / "weights" / "best.pt").read_bytes() == b"best"
    assert (run_dir / "weights" / "last.pt").read_bytes() == b"last"
    assert (run_dir / "resolved-config.json").is_file()
    assert (run_dir / "environment-summary.json").is_file()
    assert (run_dir / "results.csv").is_file()


def test_smoke_train_fails_when_training_results_are_missing(tmp_path: Path) -> None:
    config, dataset_config, device_summary = _create_inputs(tmp_path)

    with pytest.raises(SmokeTrainError, match="results.csv"):
        run_smoke_train(
            config,
            dataset_config,
            tmp_path,
            device_summary,
            lambda _: FakeModel(write_results=False),
        )


def _create_inputs(tmp_path: Path) -> tuple[SmokeTrainConfig, DatasetConfig, DeviceSummary]:
    model_path = tmp_path / "models" / "pretrained.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    image_path = tmp_path / "data" / "raw" / "wildlife" / "images" / "train" / "sample.ppm"
    label_path = tmp_path / "data" / "raw" / "wildlife" / "labels" / "train" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    manifest_path = tmp_path / "data" / "manifests" / "smoke.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source_archive_sha256": "0" * 64,
                "source_split": "train",
                "images": [str(image_path.relative_to(tmp_path))],
            }
        ),
        encoding="utf-8",
    )
    config = SmokeTrainConfig(
        model_path=model_path.relative_to(tmp_path),
        manifest_path=manifest_path.relative_to(tmp_path),
        image_count=1,
        epochs=1,
        image_size=160,
        batch_size=1,
        workers=0,
        seed=7,
        validation_split="val",
    )
    dataset_config = DatasetConfig(
        schema_version=1,
        source_url="https://example.invalid/dataset.zip",
        archive_path=Path("data/raw/wildlife.zip"),
        dataset_root=Path("data/raw/wildlife"),
        expected_sha256="0" * 64,
        class_names=["buffalo"],
        splits={"train": 1, "val": 1, "test": 1},
        smoke_subset_size=16,
        test_split_sealed=True,
    )
    device_summary = DeviceSummary(
        device="cpu",
        pytorch_version="test",
        ultralytics_version="test",
        precision="highest",
        mps_fallback_setting="unset",
    )
    return config, dataset_config, device_summary
