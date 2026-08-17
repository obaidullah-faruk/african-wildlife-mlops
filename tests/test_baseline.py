from wildlife_mlops.baseline import (
    Detection,
    _classify_validation_trend,
    _iou,
    _json_default,
    _match_detections,
)


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
