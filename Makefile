UV := uv
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache

.PHONY: bootstrap doctor device-info lint typecheck test container-arm64 container-amd64 data-download data-inventory data-validate data-visualize data-audit-splits data-smoke-manifest predict-pretrained predict train-overfit train-smoke train-baseline evaluate-baseline run-experiment evaluate-release-test

bootstrap:
	$(UV) sync --all-groups

doctor:
	$(UV) run wildlife-mlops doctor

device-info:
	$(UV) run wildlife-mlops device-info

train-overfit:
	$(UV) run wildlife-mlops train-overfit

train-smoke:
	$(UV) run wildlife-mlops train-smoke

train-baseline:
	$(UV) run wildlife-mlops train-baseline

evaluate-baseline:
	@test -n "$(RUN_DIR)" || (echo "RUN_DIR is required, for example: make evaluate-baseline RUN_DIR=artifacts/baseline/<run>"; exit 2)
	$(UV) run wildlife-mlops evaluate-baseline --run-dir "$(RUN_DIR)"

run-experiment:
	@test -n "$(BASELINE_RUN)" || (echo "BASELINE_RUN is required, for example: make run-experiment BASELINE_RUN=artifacts/baseline/<run>"; exit 2)
	$(UV) run wildlife-mlops run-experiment --baseline-run "$(BASELINE_RUN)"

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
