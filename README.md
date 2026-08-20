# Wildlife MLOps

Learn one object-detection lifecycle. The model detects buffalo, elephant,
rhino, and zebra.

## Architecture

```mermaid
flowchart LR
    data[Versioned wildlife dataset] --> validate[Dataset validation]
    validate --> train[Baseline training]
    train --> validation_metrics[Validation metrics]
    train --> checkpoint[best.pt checkpoint]
    train --> mlflow[Local MLflow]
    validation_metrics --> candidate[Release candidate]
    checkpoint --> candidate
    candidate --> package[Package: checkpoint + inference code]
    candidate --> approval[Human approval]
    approval --> sealed[One sealed test evaluation]
    sealed --> service[FastAPI service\nloads one pinned candidate]
    client[Local prediction client] --> service
    service --> response[Prediction JSON\nversion + SHA-256 checksum]
    response --> rollback[Manual A → B → A rollback evidence]
    service --> service_metrics["/metrics/ endpoint"]
    service_metrics --> prometheus[Prometheus\nscrapes every 5 seconds]
    prometheus --> grafana[Grafana\nrequest, latency, error, and distribution graphs]
    response --> samples[Sampled prediction metadata\nno raw images]
    labels[Later ground-truth labels] --> quality[Sampled precision and recall report]
    samples --> quality
    failed[Failed candidate start] --> recovery[Recovery health evidence]
    service --> recovery
```

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
and its output files. Stop the services with `make mlflow-down` when finished.

## Version data

DVC tracks the dataset contents. Git tracks the small `data/raw.dvc` pointer.

```sh
uv run dvc status
```

The `run.json` file records the source archive checksum and DVC pointer checksum.

## Release, serve, and roll back

Keep MLflow running while creating candidates. A release command validates the
dataset, trains a fresh model, evaluates it on validation data, packages the
checkpoint with inference code, and registers it in MLflow.

### Create model A

```sh
make mlflow-up
make mlflow-smoke
make release-candidate
```

The last command prints a path such as
`artifacts/releases/candidate-abc123`. Copy that exact path into later commands
in place of `<candidate-a>`. Inspect `candidate.json` and `quality-report.json`
inside the candidate directory before approving it.

If MLflow registration fails after packaging succeeds, register the same
candidate without training again:

```sh
make register-candidate CANDIDATE=artifacts/releases/<candidate-a>
```

Approve the registered candidate, then evaluate its sealed test split once.

```sh
make approve-candidate CANDIDATE=artifacts/releases/<candidate-a> APPROVER="Your Name"
make evaluate-approved CANDIDATE=artifacts/releases/<candidate-a>
```

`evaluate-approved` writes one `test-evaluation.json` file and refuses to run a
second time for that candidate.

### Deploy model A

In terminal 1, start the API. It loads this one model at startup.

```sh
make serve CANDIDATE=artifacts/releases/<candidate-a>
```

In terminal 2, verify the selected identity and save a prediction. The output
path must not already exist.

```sh
curl http://127.0.0.1:8000/health
make send-prediction \
  IMAGE='data/raw/african-wildlife/images/val/4 (343).jpg' \
  OUTPUT=artifacts/model-a-prediction.json
```

Both responses include `candidate_id`, `model_version`, and `model_sha256`.
The checksum proves which exact model answered.

### Create and deploy model B

Stop model A in terminal 1 with `Ctrl-C`. Create B with the supplied four-epoch
configuration, then approve and evaluate it. Copy the newly printed candidate
path into `<candidate-b>`.

```sh
make release-candidate RELEASE_TRAIN_CONFIG=configs/train/baseline-extended.yaml
```

If B's MLflow registration fails, run this before approval; do not train B
again.

```sh
make register-candidate CANDIDATE=artifacts/releases/<candidate-b>
```

```sh
make approve-candidate CANDIDATE=artifacts/releases/<candidate-b> APPROVER="Your Name"
make evaluate-approved CANDIDATE=artifacts/releases/<candidate-b>
make serve CANDIDATE=artifacts/releases/<candidate-b>
```

In terminal 2, save B's response.

```sh
make send-prediction \
  IMAGE='data/raw/african-wildlife/images/val/4 (343).jpg' \
  OUTPUT=artifacts/model-b-prediction.json
```

### Roll back to model A

Stop B in terminal 1 with `Ctrl-C`, then serve A again. In terminal 2, save a
new A prediction and record the rollback evidence.

```sh
make serve CANDIDATE=artifacts/releases/<candidate-a>
```

```sh
make send-prediction \
  IMAGE='data/raw/african-wildlife/images/val/4 (343).jpg' \
  OUTPUT=artifacts/model-a-after-rollback.json
make record-rollback \
  FROM_CANDIDATE=artifacts/releases/<candidate-b> \
  TO_CANDIDATE=artifacts/releases/<candidate-a> \
  PREDICTION=artifacts/model-a-after-rollback.json \
  OUTPUT=artifacts/rollback-evidence.json
```

`record-rollback` checks that the saved response proves model A's checksum was
active again. Stop MLflow with `make mlflow-down` when you finish.

## Observe the local service

The API exposes Prometheus metrics at `/metrics/`: request count and latency,
HTTP errors, prediction errors, predicted classes, and confidence distribution.
For monitoring containers to reach the locally running API, start the service
on all local interfaces in terminal 1.

```sh
make serve \
  CANDIDATE=artifacts/releases/<candidate-a> \
  SERVICE_HOST=0.0.0.0 \
  SAMPLE_RATE=1
```

`SAMPLE_RATE=1` retains every response for this short learning exercise. The
normal default is `0.1`; samples contain prediction metadata and boxes, never
the raw image.

In terminal 2, start Prometheus and Grafana, then send a few predictions.

```sh
make monitoring-up
curl http://127.0.0.1:8000/metrics/
```

Open <http://127.0.0.1:9090> to query metrics and
<http://127.0.0.1:3000> to view the provisioned **Wildlife service** dashboard.
No sign-in is required. Send predictions after starting the service; request
and latency panels then update after Prometheus's five-second scrape interval.
The predicted-class panel stays empty when the model returns no boxes, and an
empty error-rate panel means no errors have occurred. Samples are written to
`artifacts/monitoring/prediction-samples.jsonl`.

Only measure model quality after a human or another trusted process supplies
ground-truth boxes for sampled `trace_id` values. Save one JSON object per line
in a label file, for example:

```json
{"trace_id":"copied-from-sample","boxes":[{"class":"zebra","x_min":0.1,"y_min":0.2,"x_max":0.7,"y_max":0.9}]}
```

Then create the quality report:

```sh
make measure-sampled-quality \
  SAMPLES=artifacts/monitoring/prediction-samples.jsonl \
  LABELS=artifacts/monitoring/ground-truth.jsonl \
  OUTPUT=artifacts/monitoring/sampled-quality.json
```

The report records precision and recall only for samples that have matching
ground truth. It does not treat predictions as labels.

### Practice failed-start recovery

Stop the service with `Ctrl-C`. In terminal 1, deliberately attempt to start a
missing candidate; it must fail without serving a model.

```sh
make serve CANDIDATE=artifacts/releases/missing-candidate SERVICE_HOST=0.0.0.0
```

Restart the known candidate, then in terminal 2 save its health response and
record recovery evidence.

```sh
make serve CANDIDATE=artifacts/releases/<candidate-a> SERVICE_HOST=0.0.0.0
```

```sh
mkdir -p artifacts/monitoring
curl --fail --silent http://127.0.0.1:8000/health > artifacts/monitoring/recovery-health.json
make record-recovery \
  FAILED_CANDIDATE=artifacts/releases/missing-candidate \
  RECOVERY_CANDIDATE=artifacts/releases/<candidate-a> \
  HEALTH=artifacts/monitoring/recovery-health.json \
  OUTPUT=artifacts/monitoring/recovery-evidence.json
```

Stop the monitoring containers with `make monitoring-down` when finished.

## Predict one image

```sh
make predict \
  MODEL=artifacts/baseline/<run>/weights/best.pt \
  IMAGE=data/raw/african-wildlife/images/val/<image>.jpg \
  OUTPUT=artifacts/prediction.json
```

The JSON result includes normalized boxes and the model checksum.
