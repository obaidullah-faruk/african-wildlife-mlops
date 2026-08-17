from pathlib import Path
from typing import Any

from wildlife_mlops.pretrained import PretrainedInferenceConfig, run_pretrained_inference


class FakeResult:
    """Minimal prediction result used to test model loading behavior."""

    boxes = None


class FakeModel:
    """Minimal model that records one batch prediction call."""

    names = {0: "person"}

    def __init__(self) -> None:
        self.sources: list[str] = []

    def predict(self, source: list[str], **_: Any) -> list[FakeResult]:
        self.sources = source
        return [FakeResult() for _ in source]


def test_pretrained_inference_loads_one_model_for_five_images(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dataset_root = tmp_path / "data" / "raw" / "wildlife"
    image_root = dataset_root / "images" / "val"
    image_root.mkdir(parents=True)
    for index in range(5):
        (image_root / f"sample-{index}.ppm").write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    config = PretrainedInferenceConfig(
        model_url="https://example.invalid/yolo26n.pt",
        model_path=Path("models/pretrained/yolo26n.pt"),
        model_sha256="0" * 64,
        model_version="test-model",
        source_split="val",
        sample_count=5,
        confidence_threshold=0.25,
        output_dir=Path("artifacts/pretrained-inference"),
    )
    loaded_paths: list[str] = []
    model = FakeModel()
    monkeypatch.setattr(
        "wildlife_mlops.pretrained.ensure_model_weights",
        lambda _config, root: root / "models" / "pretrained" / "yolo26n.pt",
    )
    monkeypatch.setattr(
        "wildlife_mlops.pretrained._save_rendered_prediction",
        lambda _result, destination: destination.write_bytes(b"preview"),
    )

    def model_factory(path: str) -> FakeModel:
        loaded_paths.append(path)
        return model

    output_path = run_pretrained_inference(
        config,
        dataset_root,
        tmp_path,
        model_factory,
    )

    assert len(loaded_paths) == 1
    assert len(model.sources) == 5
    assert output_path.is_file()
