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

## Local experiment tracking

In one terminal, start the disposable local MLflow server:

```sh
make mlflow-server
```

In another terminal, run the tracked smoke training and then open
<http://127.0.0.1:5000>. Select the `wildlife-smoke` experiment and inspect the
parameters, terminal metrics, and `training-output` artifacts.

```sh
make train-smoke
```

The local SQLite database and file artifacts are stored under `artifacts/mlflow/`.

MLflow records aggregate epoch metrics with zero-based MLflow steps: `epoch`,
`train/<loss>`, `learning_rate/group_<n>`, and
`validation/{precision,recall,map50,map50_95}`. These names deliberately avoid
per-class metric series.

Each tracked run also carries MLflow tags that identify its Git state, data source,
configuration, base weights, local runtime, and trigger. `not_applicable` means
the local run did not use DVC, a prepared-data manifest, or a training container.

## Comparing a baseline and variant

Start MLflow first. Then train and validate a tracked baseline. Its output folder
contains `mlflow-run.json`, which links it to its MLflow run.

```sh
make train-baseline
make evaluate-baseline RUN_DIR=artifacts/baseline/<run-directory>
make run-experiment BASELINE_RUN=artifacts/baseline/<run-directory>
```

Both runs appear in `wildlife-baseline-comparison`. The controlled variant changes
only image size. The command writes `mlflow-api-comparison.json` beside the local
comparison report. The selected MLflow run receives `selection.status` and
`selection.reason` tags. Selection is not a production promotion.

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

## Single-image prediction

```sh
make predict \
  MODEL=artifacts/experiment/<run>/weights/best.pt \
  IMAGE=data/raw/african-wildlife/images/val/<image>.jpg \
  OUTPUT=artifacts/predictions/<image>.json

  # Sample
make predict \
  MODEL=artifacts/experiment/experiment-20260817T185952Z-2fdb4d0f/weights/best.pt \
  IMAGE='data/raw/african-wildlife/images/val/1 (288).jpg' \
  OUTPUT=artifacts/predictions/zebra-prediction.json
```

The command accepts JPEG and PNG inputs, validates image decoding, and refuses to
overwrite an existing output. Each result records the model version and SHA-256.

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
  "model_sha256": "immutable-model-checksum",
  "trace_id": "request-trace-identifier",
  "timestamp": "2026-08-17T12:00:00Z"
}
```

Coordinates are normalized to the range 0–1. `model_version` identifies the
model that generated the prediction.
