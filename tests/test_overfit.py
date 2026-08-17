from pathlib import Path

from wildlife_mlops.overfit import (
    OverfitConfig,
    _read_memorization_metrics,
    _stage_tiny_dataset,
)


def test_overfit_config_uses_the_actual_tiny_batch_for_loss_normalization() -> None:
    config = OverfitConfig(
        model_path=Path("models/pretrained/model.pt"),
        source_split="train",
        image_count=8,
        epochs=100,
        image_size=320,
        batch_size=8,
        workers=0,
        seed=7,
        target_map50=0.9,
        optimizer="SGD",
        initial_learning_rate=0.01,
        final_learning_rate_fraction=0.01,
        warmup_epochs=0.0,
        weight_decay=0.0,
        nominal_batch_size=8,
    )

    assert config.nominal_batch_size == config.batch_size


def test_stage_tiny_dataset_duplicates_selected_images_for_validation(tmp_path: Path) -> None:
    dataset_root = tmp_path / "source"
    image_path = dataset_root / "images" / "train" / "example.ppm"
    label_path = dataset_root / "labels" / "train" / "example.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    data_path = _stage_tiny_dataset(
        tmp_path / "staged",
        dataset_root,
        "train",
        [image_path],
        ["buffalo"],
    )

    assert data_path.is_file()
    assert (tmp_path / "staged" / "images" / "train" / "01.ppm").is_file()
    assert (tmp_path / "staged" / "images" / "val" / "01.ppm").is_file()
    assert (tmp_path / "staged" / "labels" / "train" / "01.txt").is_file()
    assert (tmp_path / "staged" / "labels" / "val" / "01.txt").is_file()


def test_read_memorization_metrics_uses_first_and_last_epochs(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B)\n"
        "1,2.0,3.0,4.0,0.1\n"
        "2,1.0,1.5,2.0,0.95\n",
        encoding="utf-8",
    )

    metrics = _read_memorization_metrics(results_path)

    assert metrics.initial_training_loss == 9.0
    assert metrics.final_training_loss == 4.5
    assert metrics.map50 == 0.95
    assert metrics.loss_decreased
