UV := uv
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache
MLFLOW_TRACKING_URI ?= http://127.0.0.1:5001
RELEASE_TRAIN_CONFIG ?= configs/train/baseline.yaml
SERVICE_HOST ?= 127.0.0.1
SERVICE_PORT ?= 8000
MONITORING_DIR ?= artifacts/monitoring
SAMPLE_RATE ?= 0.1
MONITORING_COMPOSE := docker compose -f docker-compose.monitoring.yml
COMPOSE := docker compose --env-file .env

.PHONY: bootstrap doctor device-info lint typecheck test data-download data-validate data-visualize predict-pretrained predict train-overfit train-smoke train-baseline train-tracked release-candidate register-candidate approve-candidate evaluate-approved serve send-prediction record-rollback measure-sampled-quality record-recovery monitoring-up monitoring-down mlflow-up mlflow-down mlflow-smoke

bootstrap:
	$(UV) sync --all-groups

doctor:
	$(UV) run wildlife-mlops doctor

device-info:
	$(UV) run wildlife-mlops device-info

lint:
	$(UV) run ruff check src tests

typecheck:
	$(UV) run mypy src tests

test:
	$(UV) run pytest

data-download:
	$(UV) run wildlife-mlops data-download

data-validate:
	$(UV) run wildlife-mlops data-validate

data-visualize:
	$(UV) run wildlife-mlops data-visualize

predict-pretrained:
	$(UV) run wildlife-mlops predict-pretrained

predict:
	@test -n "$(MODEL)" && test -n "$(IMAGE)" && test -n "$(OUTPUT)" || (echo "MODEL, IMAGE, and OUTPUT are required"; exit 2)
	$(UV) run wildlife-mlops predict --model "$(MODEL)" --image "$(IMAGE)" --output "$(OUTPUT)"

train-overfit:
	$(UV) run wildlife-mlops train-overfit

train-smoke:
	$(UV) run wildlife-mlops train-smoke

train-baseline:
	$(UV) run wildlife-mlops train-baseline

train-tracked:
	$(UV) run wildlife-mlops train-tracked --tracking-uri "$(MLFLOW_TRACKING_URI)"

release-candidate:
	$(UV) run wildlife-mlops release-candidate --tracking-uri "$(MLFLOW_TRACKING_URI)" --train-config "$(RELEASE_TRAIN_CONFIG)"

register-candidate:
	@test -n "$(CANDIDATE)" || (echo "CANDIDATE is required"; exit 2)
	$(UV) run wildlife-mlops register-candidate --candidate "$(CANDIDATE)" --tracking-uri "$(MLFLOW_TRACKING_URI)"

approve-candidate:
	@test -n "$(CANDIDATE)" && test -n "$(APPROVER)" || (echo "CANDIDATE and APPROVER are required"; exit 2)
	$(UV) run wildlife-mlops approve-candidate --candidate "$(CANDIDATE)" --approver "$(APPROVER)"

evaluate-approved:
	@test -n "$(CANDIDATE)" || (echo "CANDIDATE is required"; exit 2)
	$(UV) run wildlife-mlops evaluate-approved --candidate "$(CANDIDATE)"

serve:
	@test -n "$(CANDIDATE)" || (echo "CANDIDATE is required"; exit 2)
	$(UV) run wildlife-mlops serve --candidate "$(CANDIDATE)" --host "$(SERVICE_HOST)" --port "$(SERVICE_PORT)" --monitoring-dir "$(MONITORING_DIR)" --sample-rate "$(SAMPLE_RATE)"

send-prediction:
	@test -n "$(IMAGE)" && test -n "$(OUTPUT)" || (echo "IMAGE and OUTPUT are required"; exit 2)
	$(UV) run wildlife-mlops send-prediction --image "$(IMAGE)" --output "$(OUTPUT)"

record-rollback:
	@test -n "$(FROM_CANDIDATE)" && test -n "$(TO_CANDIDATE)" && test -n "$(PREDICTION)" && test -n "$(OUTPUT)" || (echo "FROM_CANDIDATE, TO_CANDIDATE, PREDICTION, and OUTPUT are required"; exit 2)
	$(UV) run wildlife-mlops record-rollback --from-candidate "$(FROM_CANDIDATE)" --to-candidate "$(TO_CANDIDATE)" --prediction "$(PREDICTION)" --output "$(OUTPUT)"

measure-sampled-quality:
	@test -n "$(SAMPLES)" && test -n "$(LABELS)" && test -n "$(OUTPUT)" || (echo "SAMPLES, LABELS, and OUTPUT are required"; exit 2)
	$(UV) run wildlife-mlops measure-sampled-quality --samples "$(SAMPLES)" --labels "$(LABELS)" --output "$(OUTPUT)"

record-recovery:
	@test -n "$(FAILED_CANDIDATE)" && test -n "$(RECOVERY_CANDIDATE)" && test -n "$(HEALTH)" && test -n "$(OUTPUT)" || (echo "FAILED_CANDIDATE, RECOVERY_CANDIDATE, HEALTH, and OUTPUT are required"; exit 2)
	$(UV) run wildlife-mlops record-recovery --failed-candidate "$(FAILED_CANDIDATE)" --recovery-candidate "$(RECOVERY_CANDIDATE)" --health "$(HEALTH)" --output "$(OUTPUT)"

monitoring-up:
	$(MONITORING_COMPOSE) up -d

monitoring-down:
	$(MONITORING_COMPOSE) down

mlflow-up:
	@test -f .env || (echo ".env is required; copy .env.example to .env"; exit 2)
	$(COMPOSE) up -d

mlflow-down:
	$(COMPOSE) down

mlflow-smoke:
	@attempt=0; until curl --fail --silent "$(MLFLOW_TRACKING_URI)/health" >/dev/null; do attempt=$$((attempt + 1)); test $$attempt -lt 30 || (echo "MLflow did not become healthy"; exit 1); sleep 1; done
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" $(UV) run python -c 'import os; import mlflow; mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"]); mlflow.set_experiment("mlflow-smoke"); mlflow.start_run(); mlflow.log_param("purpose", "connectivity-check"); mlflow.end_run(); print("MLflow is ready.")'
