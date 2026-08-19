from pathlib import Path

from wildlife_mlops.tracking import _final_metrics


def test_final_metrics_are_compatible_with_mlflow_names(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    results.write_text("epoch,metrics/mAP50(B)\n0,0.5\n", encoding="utf-8")

    assert _final_metrics(results) == {"epoch": 0.0, "metrics/mAP50_B_": 0.5}
