"""Local FastAPI adapter for the transport-independent inference core."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException
from prometheus_client import make_asgi_app

from wildlife_mlops.inference import InferenceError, PinnedModel, load_pinned_model
from wildlife_mlops.monitoring import ServiceObserver


def create_app(
    candidate_dir: Path,
    model_factory: Callable[[str], Any],
    monitoring_dir: Path = Path("artifacts/monitoring"),
    sample_rate: float = 0.1,
) -> FastAPI:
    """Create an API that loads one selected candidate during startup."""
    observer = ServiceObserver(monitoring_dir / "prediction-samples.jsonl", sample_rate)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pinned_model = load_pinned_model(candidate_dir, model_factory)
        yield

    app = FastAPI(title="Wildlife MLOps", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app(registry=observer.registry))

    @app.middleware("http")
    async def observe_http(request: Any, call_next: Any) -> Any:
        started_at = perf_counter()
        endpoint = request.url.path
        try:
            response = await call_next(request)
        except Exception:
            observer.observe_request(endpoint, 500, perf_counter() - started_at)
            raise
        observer.observe_request(endpoint, response.status_code, perf_counter() - started_at)
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        model: PinnedModel = app.state.pinned_model
        return {
            "status": "ok",
            "candidate_id": model.candidate_id,
            "model_version": model.model_version,
            "model_sha256": model.model_sha256,
        }

    @app.post("/predict")
    def predict(
        image: bytes = Body(), image_name: str | None = Header(default=None, alias="X-Image-Name")
    ) -> dict[str, object]:
        model: PinnedModel = app.state.pinned_model
        try:
            response = model.predict(image, image_name or "")
            observer.observe_prediction(response)
            return response
        except InferenceError as error:
            observer.observe_prediction_error("inference_error")
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def run_server(
    candidate_dir: Path, host: str, port: int, monitoring_dir: Path, sample_rate: float
) -> None:
    """Run the local HTTP server for one selected candidate."""
    import ultralytics
    import uvicorn

    app = create_app(candidate_dir, getattr(ultralytics, "YOLO"), monitoring_dir, sample_rate)
    uvicorn.run(app, host=host, port=port)
