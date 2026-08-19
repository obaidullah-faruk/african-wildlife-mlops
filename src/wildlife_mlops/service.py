"""Local FastAPI adapter for the transport-independent inference core."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException

from wildlife_mlops.inference import InferenceError, PinnedModel, load_pinned_model


def create_app(candidate_dir: Path, model_factory: Callable[[str], Any]) -> FastAPI:
    """Create an API that loads one selected candidate during startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pinned_model = load_pinned_model(candidate_dir, model_factory)
        yield

    app = FastAPI(title="Wildlife MLOps", lifespan=lifespan)

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
            return model.predict(image, image_name or "")
        except InferenceError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def run_server(candidate_dir: Path, host: str, port: int) -> None:
    """Run the local HTTP server for one selected candidate."""
    import ultralytics
    import uvicorn

    app = create_app(candidate_dir, getattr(ultralytics, "YOLO"))
    uvicorn.run(app, host=host, port=port)
