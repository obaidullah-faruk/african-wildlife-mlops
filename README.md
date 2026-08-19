# Wildlife MLOps

Learn one object-detection lifecycle. The model detects buffalo, elephant,
rhino, and zebra.

## Setup

```sh
make bootstrap
make doctor
```

## Learn the data

```sh
make data-download
make data-validate
make data-visualize
```

Open `artifacts/data-preview/train.png`. Check that the boxes match the animals.

## Learn training

```sh
make predict-pretrained
make train-overfit
make train-smoke
make train-baseline
```

Each training command creates a new directory under `artifacts/`. Open its
`results.csv`, `run.json`, and `weights/best.pt`.

`train-overfit` uses a few images twice. Inspect whether its loss falls.
`train-smoke` proves the full training path works quickly. `train-baseline`
uses the whole training split.

## Track a run

Copy the example credentials once, then start local MLflow.

```sh
cp .env.example .env
make mlflow-up
make mlflow-smoke
make train-tracked
```

Open <http://127.0.0.1:5001>. A tracked run stores parameters, final metrics,
and its output files. Stop the services with `make mlflow-down`.

## Version data

DVC tracks the dataset contents. Git tracks the small `data/raw.dvc` pointer.

```sh
uv run dvc status
```

The `run.json` file records the source archive checksum and DVC pointer checksum.

## Predict one image

```sh
make predict \
  MODEL=artifacts/baseline/<run>/weights/best.pt \
  IMAGE=data/raw/african-wildlife/images/val/<image>.jpg \
  OUTPUT=artifacts/prediction.json
```

The JSON result includes normalized boxes and the model checksum.
