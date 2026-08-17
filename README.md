# Wildlife MLOps

An MLOps project for detecting buffalo, elephant, rhino, and zebra in images.

## Setup

```sh
make bootstrap
```

## Development checks

```sh
make doctor
make lint
make typecheck
make test
```

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
