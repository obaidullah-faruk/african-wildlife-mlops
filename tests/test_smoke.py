import json
from pathlib import Path
from typing import Any, Self

import pytest

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.device import DeviceSummary
from wildlife_mlops.smoke import (
    SmokeTrainConfig,
    SmokeTrainError,
    _disable_ultralytics_mlflow_callbacks,
    run_smoke_train,
)
from wildlife_mlops.tracking import log_smoke_run, write_smoke_run_report


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
    assert (run_dir / "run-report.json").is_file()


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


def test_mlflow_logging_records_parameters_metrics_and_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "smoke-run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "epoch,train/loss,metrics/mAP50(B)\n0,1.5,0.25\n", encoding="utf-8"
    )
    write_smoke_run_report(run_dir, {"epochs": 1, "seed": 7})
    fake_mlflow = FakeMLflow()

    run_id = log_smoke_run(
        run_dir,
        {"epochs": 1, "seed": 7},
        "http://127.0.0.1:5000",
        "wildlife-smoke",
        fake_mlflow,
    )

    assert run_id == "run-123"
    assert fake_mlflow.tracking_uri == "http://127.0.0.1:5000"
    assert fake_mlflow.experiment_name == "wildlife-smoke"
    assert fake_mlflow.parameters == {"epochs": "1", "seed": "7"}
    assert fake_mlflow.metrics == {
        "terminal/train/loss": 1.5,
        "terminal/metrics/mAP50_B_": 0.25,
    }
    assert fake_mlflow.artifact_source == run_dir
    assert fake_mlflow.artifact_path == "training-output"


def test_ultralytics_mlflow_callbacks_are_removed_without_affecting_others() -> None:
    def normal_callback() -> None:
        return None

    def mlflow_callback() -> None:
        return None

    mlflow_callback.__module__ = "ultralytics.utils.callbacks.mlflow"
    model = CallbackModel({"on_train_end": [normal_callback, mlflow_callback]})

    _disable_ultralytics_mlflow_callbacks(model)

    assert model.callbacks["on_train_end"] == [normal_callback]


class CallbackModel:
    """Minimal model exposing Ultralytics-style callbacks."""

    def __init__(self, callbacks: dict[str, list[object]]) -> None:
        self.callbacks = callbacks


class FakeRun:
    """Context-managed fake MLflow active run."""

    class Info:
        run_id = "run-123"

    info = Info()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class FakeMLflow:
    """Minimal MLflow client used to test tracking without a server."""

    def __init__(self) -> None:
        self.tracking_uri = ""
        self.experiment_name = ""
        self.parameters: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifact_source = Path()
        self.artifact_path = ""

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, run_name: str) -> FakeRun:
        assert run_name == "smoke-run"
        return FakeRun()

    def log_params(self, parameters: dict[str, str]) -> None:
        self.parameters = parameters

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics

    def log_artifacts(self, source: str, artifact_path: str) -> None:
        self.artifact_source = Path(source)
        self.artifact_path = artifact_path


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
