import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from wildlife_mlops.predict import PredictionError, predict_image


class FakeTensor:
    """Small tensor-like value that provides the Ultralytics conversion method."""

    def __init__(self, values: list[list[float]] | list[float]) -> None:
        self.values = values

    def tolist(self) -> list[list[float]] | list[float]:
        return self.values


class FakeBoxes:
    """One deterministic detection in the common Ultralytics boxes shape."""

    xyxy = FakeTensor([[10.0, 20.0, 80.0, 90.0]])
    conf = FakeTensor([0.9])
    cls = FakeTensor([0.0])


class FakeResult:
    """One deterministic prediction result."""

    boxes = FakeBoxes()
    orig_shape = (100, 100)
    names = {0: "zebra"}


class FakeModel:
    """Minimal prediction model that records the validated image path."""

    def __init__(self) -> None:
        self.source: str | None = None

    def predict(self, source: str, **_: Any) -> list[FakeResult]:
        self.source = source
        return [FakeResult()]


def test_predict_image_writes_stable_schema_and_model_identity(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"immutable weights")
    image_path = tmp_path / "zebra.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    output_path = tmp_path / "outputs" / "prediction.json"
    model = FakeModel()

    result_path = predict_image(model_path, image_path, output_path, lambda _: model)

    contents = json.loads(result_path.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(b"immutable weights").hexdigest()
    assert model.source == str(image_path.resolve())
    assert contents["image_id"] == "zebra.png"
    assert contents["model_sha256"] == checksum
    assert contents["model_version"] == f"model.pt:sha256:{checksum}"
    assert contents["boxes"] == [
        {
            "class": "zebra",
            "confidence": 0.9,
            "x_max": 0.8,
            "x_min": 0.1,
            "y_max": 0.9,
            "y_min": 0.2,
        }
    ]
    assert contents["trace_id"]
    assert contents["timestamp"]


@pytest.mark.parametrize(
    ("image_name", "contents", "message"),
    [
        ("image.gif", b"GIF89a", "Unsupported image type"),
        ("broken.jpg", b"not an image", "Image cannot be decoded"),
    ],
)
def test_predict_image_rejects_invalid_image_inputs(
    tmp_path: Path, image_name: str, contents: bytes, message: str
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")
    image_path = tmp_path / image_name
    image_path.write_bytes(contents)

    with pytest.raises(PredictionError, match=message):
        predict_image(model_path, image_path, tmp_path / "prediction.json", lambda _: FakeModel())
