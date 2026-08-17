UV := uv
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache

.PHONY: bootstrap doctor device-info lint typecheck test container-arm64 container-amd64 data-download data-inventory data-validate data-visualize data-audit-splits data-smoke-manifest predict-pretrained

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
