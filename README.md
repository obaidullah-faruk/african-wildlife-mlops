# Wildlife MLOps

An MLOps project for detecting buffalo, elephant, rhino, and zebra in images.

## Setup

```sh
make bootstrap
```

## Development checks

```sh
make doctor
make device-info
make train-overfit
make train-smoke
make train-baseline
make run-experiment BASELINE_RUN=artifacts/baseline/<run-directory>
make evaluate-release-test
make lint
make typecheck
make test
```

## Data commands

```sh
make data-download
make data-inventory
make data-validate
make data-visualize
make data-audit-splits
make data-smoke-manifest
```

## Pretrained prediction

```sh
make predict-pretrained
```

## Baseline evaluation

```sh
make evaluate-baseline RUN_DIR=artifacts/baseline/<run-directory>
```

`weights/best.pt` is selected by validation mAP50-95, not by the final epoch.
More epochs are useful only while validation quality improves; falling training loss
alongside plateauing or declining validation quality is evidence of overfitting.

`run-experiment` changes only the configured image size, compares the same validation
metric, and freezes the selected checkpoint. `evaluate-release-test` accepts only that
frozen selection and refuses a second test evaluation for it.

## Prediction schema

```json
{
  "image_id": "camera-001.jpg",
  "boxes": [
    {
      "x_min": 0.12,
      "y_min": 0.20,
      "x_max": 0.60,
      "y_max": 0.88,
      "class": "elephant",
      "confidence": 0.97
    }
  ],
  "model_version": "immutable-model-identifier",
  "trace_id": "request-trace-identifier",
  "timestamp": "2026-08-17T12:00:00Z"
}
```

Coordinates are normalized to the range 0–1. `model_version` identifies the
model that generated the prediction.
