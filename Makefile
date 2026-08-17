UV := uv
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

.PHONY: bootstrap doctor lint typecheck test container-arm64 container-amd64

bootstrap:
	$(UV) sync --all-groups

doctor:
	$(UV) run wildlife-mlops doctor

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
