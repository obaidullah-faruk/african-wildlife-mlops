from pathlib import Path

from wildlife_mlops.data.config import DatasetConfig
from wildlife_mlops.data.validate import ValidationIssue, validate_dataset


def test_validator_reports_exact_row_for_out_of_bounds_box(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "raw" / "wildlife" / "images" / "train" / "sample.ppm"
    label_path = tmp_path / "data" / "raw" / "wildlife" / "labels" / "train" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    label_path.write_text("0 0.9 0.5 0.5 0.5\n", encoding="utf-8")
    config = DatasetConfig(
        schema_version=1,
        source_url="https://example.invalid/wildlife.zip",
        source_license_reference="LICENSE.txt",
        archive_path=Path("data/raw/wildlife.zip"),
        dataset_root=Path("data/raw/wildlife"),
        expected_sha256="0" * 64,
        class_names=["buffalo"],
        splits={"train": 1},
        test_split_sealed=True,
    )

    result = validate_dataset(config, tmp_path)

    assert not result.passed
    assert result.issues == [
        ValidationIssue(
            code="box_outside_image",
            path=str(label_path),
            detail="0 0.9 0.5 0.5 0.5",
            row=1,
        )
    ]
