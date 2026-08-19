UV := uv
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache
MLFLOW_TRACKING_URI ?= http://127.0.0.1:5001
MLFLOW_EXPERIMENT ?= wildlife-smoke
COMPOSE := docker compose --env-file .env

.PHONY: bootstrap doctor device-info lint typecheck test container-arm64 container-amd64 data-download data-inventory data-content-manifest data-validate data-visualize data-audit-splits data-smoke-manifest predict-pretrained predict train-overfit train-smoke train-baseline evaluate-baseline run-experiment evaluate-release-test mlflow-env-check mlflow-up mlflow-down mlflow-smoke mlflow-storage-verify mlflow-maintenance mlflow-break-artifacts

bootstrap:
	$(UV) sync --all-groups

doctor:
	$(UV) run wildlife-mlops doctor

device-info:
	$(UV) run wildlife-mlops device-info

train-overfit:
	$(UV) run wildlife-mlops train-overfit

train-smoke:
	$(UV) run wildlife-mlops train-smoke --tracking-uri "$(MLFLOW_TRACKING_URI)" --experiment-name "$(MLFLOW_EXPERIMENT)"

mlflow-env-check:
	@test -f .env || (echo ".env is required; copy .env.example to .env and set local passwords"; exit 2)
	@awk -F= 'BEGIN { found = 0 } /^MINIO_ROOT_USER=/ { found = 1; if (length(substr($$0, length($$1) + 2)) < 3) { print "MINIO_ROOT_USER in .env must contain at least 3 characters"; exit 2 } } END { if (!found) { print "MINIO_ROOT_USER is required in .env"; exit 2 } }' .env
	@awk -F= 'BEGIN { found = 0 } /^MINIO_ROOT_PASSWORD=/ { found = 1; if (length(substr($$0, length($$1) + 2)) < 8) { print "MINIO_ROOT_PASSWORD in .env must contain at least 8 characters"; exit 2 } } END { if (!found) { print "MINIO_ROOT_PASSWORD is required in .env"; exit 2 } }' .env

mlflow-up:
	@$(MAKE) --no-print-directory mlflow-env-check
	$(COMPOSE) up -d

mlflow-down:
	@test -f .env || (echo ".env is required; copy .env.example to .env and set local passwords"; exit 2)
	$(COMPOSE) down

mlflow-smoke:
	@$(MAKE) --no-print-directory mlflow-env-check
	@attempt=0; until curl --fail --silent --show-error "$(MLFLOW_TRACKING_URI)/health" >/dev/null; do attempt=$$((attempt + 1)); test $$attempt -lt 30 || (echo "MLflow did not become healthy; run '$(COMPOSE) logs mlflow'"; exit 1); sleep 1; done
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" $(UV) run python -c 'import os; import mlflow; mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"]); mlflow.set_experiment("mlflow-stack-smoke"); mlflow.start_run(); mlflow.log_text("MLflow artifact proxy is ready.\n", "smoke.txt"); mlflow.end_run(); print("MLflow tracking and artifact APIs are ready.")'

mlflow-storage-verify:
	@$(MAKE) --no-print-directory mlflow-env-check
	$(UV) run wildlife-mlops verify-mlflow-storage --tracking-uri "$(MLFLOW_TRACKING_URI)"

mlflow-maintenance:
	@$(MAKE) --no-print-directory mlflow-env-check
	$(UV) run wildlife-mlops practice-mlflow-maintenance --tracking-uri "$(MLFLOW_TRACKING_URI)"

mlflow-break-artifacts:
	@$(MAKE) --no-print-directory mlflow-env-check
	docker compose --env-file .env -f docker-compose.yml -f docker-compose.invalid-artifact.yml up -d --force-recreate mlflow

train-baseline:
	$(UV) run wildlife-mlops train-baseline --tracking-uri "$(MLFLOW_TRACKING_URI)" --experiment-name "wildlife-baseline-comparison"

evaluate-baseline:
	@test -n "$(RUN_DIR)" || (echo "RUN_DIR is required, for example: make evaluate-baseline RUN_DIR=artifacts/baseline/<run>"; exit 2)
	$(UV) run wildlife-mlops evaluate-baseline --run-dir "$(RUN_DIR)"

run-experiment:
	@test -n "$(BASELINE_RUN)" || (echo "BASELINE_RUN is required, for example: make run-experiment BASELINE_RUN=artifacts/baseline/<run>"; exit 2)
	$(UV) run wildlife-mlops run-experiment --baseline-run "$(BASELINE_RUN)" --tracking-uri "$(MLFLOW_TRACKING_URI)" --experiment-name "wildlife-baseline-comparison"

evaluate-release-test:
	$(UV) run wildlife-mlops evaluate-release-test

lint:
	$(UV) run ruff check src tests

typecheck:
	$(UV) run mypy src tests

test:
	$(UV) run pytest

container-arm64:
	docker run --rm --platform linux/arm64 alpine:3.20 uname -m

container-amd64:
	docker run --rm --platform linux/amd64 alpine:3.20 uname -m

data-download:
	$(UV) run wildlife-mlops data-download

data-inventory:
	$(UV) run wildlife-mlops data-inventory

data-content-manifest:
	$(UV) run wildlife-mlops data-content-manifest

data-validate:
	$(UV) run wildlife-mlops data-validate

data-visualize:
	$(UV) run wildlife-mlops data-visualize

data-audit-splits:
	$(UV) run wildlife-mlops data-audit-splits

data-smoke-manifest:
	$(UV) run wildlife-mlops data-smoke-manifest

predict-pretrained:
	$(UV) run wildlife-mlops predict-pretrained

predict:
	@test -n "$(MODEL)" && test -n "$(IMAGE)" && test -n "$(OUTPUT)" || (echo "MODEL, IMAGE, and OUTPUT are required"; exit 2)
	$(UV) run wildlife-mlops predict --model "$(MODEL)" --image "$(IMAGE)" --output "$(OUTPUT)"
