import json
from pathlib import Path
from typing import Any

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.device import DeviceSummary
from wildlife_mlops.training import TrainingConfig, run_training


class FakeModel:
    """Small stand-in that produces the files Ultralytics training normally creates."""

    def train(self, **kwargs: Any) -> None:
        run_dir = Path(kwargs["project"]) / kwargs["name"]
        (run_dir / "weights").mkdir(exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"best")
        (run_dir / "weights" / "last.pt").write_bytes(b"last")
        (run_dir / "results.csv").write_text(
            "epoch,metrics/mAP50(B)\n0,0.5\n", encoding="utf-8"
        )


def test_smoke_training_uses_a_small_stable_subset(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    config = TrainingConfig(
        model_path=Path("models/pretrained.pt"),
        epochs=1,
        image_size=160,
        batch_size=1,
        workers=0,
        seed=7,
        sample_count=1,
    )
    model_path = tmp_path / config.model_path
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")

    run_dir = run_training(config, dataset, tmp_path, _device(), lambda _: FakeModel(), "smoke")

    assert (run_dir / "train-images.txt").is_file()
    assert (run_dir / "weights" / "best.pt").read_bytes() == b"best"
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "smoke"
    assert summary["metrics"]["metrics/mAP50(B)"] == 0.5


def test_overfit_training_copies_the_same_images_to_train_and_validation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    config = TrainingConfig(
        model_path=Path("models/pretrained.pt"),
        epochs=1,
        image_size=160,
        batch_size=1,
        workers=0,
        seed=7,
        sample_count=1,
    )
    model_path = tmp_path / config.model_path
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")

    run_dir = run_training(config, dataset, tmp_path, _device(), lambda _: FakeModel(), "overfit")

    assert list((run_dir / "overfit-data" / "images" / "train").iterdir())
    assert list((run_dir / "overfit-data" / "images" / "val").iterdir())


def _dataset(root: Path) -> DatasetConfig:
    for split in ("train", "val"):
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
        splits={"train": 1, "val": 1},
        test_split_sealed=True,
    )


def _device() -> DeviceSummary:
    return DeviceSummary("cpu", "test", "test", "highest", "unset")
