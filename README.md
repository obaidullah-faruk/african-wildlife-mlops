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
make mlflow-down
```

Open <http://127.0.0.1:5001>. A tracked run stores parameters, final metrics,
and its output files. Stop the services with `make mlflow-down`.

## Version data

DVC tracks the dataset contents. Git tracks the small `data/raw.dvc` pointer.

```sh
uv run dvc status
```

The `run.json` file records the source archive checksum and DVC pointer checksum.

## Release a candidate

Start MLflow, then create a candidate. This validates the dataset, trains the
baseline, evaluates the selected checkpoint on validation data, writes a quality
report, packages the checkpoint with its inference code, and registers it in MLflow.

```sh
make release-candidate
```

Inspect `artifacts/releases/<candidate>/candidate.json` and
`quality-report.json`. A person must then record approval before the sealed test
split can be evaluated.

```sh
make approve-candidate CANDIDATE=artifacts/releases/<candidate> APPROVER="Your Name"
make evaluate-approved CANDIDATE=artifacts/releases/<candidate>
```

`evaluate-approved` writes one `test-evaluation.json` file and refuses to run a
second time for the same candidate.

## Predict one image

```sh
make predict \
  MODEL=artifacts/baseline/<run>/weights/best.pt \
  IMAGE=data/raw/african-wildlife/images/val/<image>.jpg \
  OUTPUT=artifacts/prediction.json
```

The JSON result includes normalized boxes and the model checksum.
