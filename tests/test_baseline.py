from pathlib import Path

from wildlife_mlops.baseline import (
    BaselineConfig,
    Detection,
    _changed_config_fields,
    _classify_validation_trend,
    _iou,
    _json_default,
    _match_detections,
    _mlflow_run_id,
    _release_test_report_exists,
)
from wildlife_mlops.tracking import _api_run_summary


def test_controlled_experiment_allows_only_an_image_size_change() -> None:
    control = BaselineConfig(
        model_path=Path("models/pretrained/yolo26n-v8.4.0.pt"),
        epochs=5,
        image_size=160,
        batch_size=8,
        workers=0,
        seed=7,
        validation_split="val",
        confidence_threshold=0.25,
    )
    experiment = control.model_copy(update={"image_size": 192})

    assert _changed_config_fields(control, experiment) == ["image_size"]


def test_release_test_guard_detects_an_existing_test_report(tmp_path: Path) -> None:
    release_path = tmp_path / "artifacts" / "releases" / "selected-baseline.json"
    release_path.parent.mkdir(parents=True)
    release_path.write_text("{}\n", encoding="utf-8")

    assert not _release_test_report_exists(release_path, tmp_path)

    report_path = tmp_path / "artifacts" / "release-test" / "run" / "evaluation-report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        '{"release_selection": "artifacts/releases/selected-baseline.json"}\n',
        encoding="utf-8",
    )

    assert _release_test_report_exists(release_path, tmp_path)


def test_training_curve_analysis_identifies_declining_validation_quality() -> None:
    assert _classify_validation_trend([4.0, 3.0, 2.0, 1.0], [0.2, 0.6, 0.58, 0.4]) == "declining"


def test_training_curve_analysis_identifies_plateau() -> None:
    assert _classify_validation_trend([4.0, 3.0, 2.0, 1.0], [0.2, 0.6, 0.605, 0.602]) == "plateaued"


def test_match_detections_requires_matching_class_and_iou() -> None:
    ground_truth = [Detection(0, 0.1, 0.1, 0.5, 0.5, 1.0)]
    predicted = [
        Detection(1, 0.1, 0.1, 0.5, 0.5, 0.9),
        Detection(0, 0.12, 0.12, 0.5, 0.5, 0.8),
    ]

    matches, false_positives, false_negatives = _match_detections(ground_truth, predicted)

    assert matches == [(0, 1)]
    assert false_positives == [0]
    assert false_negatives == []
    assert _iou(ground_truth[0], predicted[1]) > 0.5


def test_json_default_normalizes_scalar_with_item_method() -> None:
    class Scalar:
        def item(self) -> int:
            return 7

    assert _json_default(Scalar()) == 7


def test_mlflow_run_id_reads_the_local_tracking_link(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "baseline" / "baseline-run"
    run_dir.mkdir(parents=True)
    (run_dir / "mlflow-run.json").write_text('{"run_id": "run-123"}\n', encoding="utf-8")

    assert _mlflow_run_id(run_dir) == "run-123"


def test_mlflow_api_summary_keeps_parameters_metrics_and_lineage_tags() -> None:
    run = ApiRun()

    assert _api_run_summary(run) == {
        "run_id": "run-123",
        "parameters": {"image_size": "192"},
        "metrics": {"terminal/metrics/mAP50-95_B_": 0.4},
        "lineage_tags": {"lineage.git_commit": "commit-123"},
    }


class ApiRun:
    """Small stand-in for the MLflow run object returned by its API."""

    class Info:
        run_id = "run-123"

    class Data:
        params = {"image_size": "192"}
        metrics = {"terminal/metrics/mAP50-95_B_": 0.4}
        tags = {"lineage.git_commit": "commit-123", "trigger.type": "manual"}

    info = Info()
    data = Data()
