"""Small standard-library HTTP client for the local prediction API."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ClientError(RuntimeError):
    """Raised when a local prediction request cannot complete."""


def send_prediction(api_url: str, image_path: Path) -> dict[str, object]:
    """Send one image to the API and return its JSON response."""
    if not image_path.is_file():
        raise ClientError(f"Image does not exist: {image_path}")
    request = Request(
        f"{api_url.rstrip('/')}/predict",
        data=image_path.read_bytes(),
        headers={"Content-Type": "application/octet-stream", "X-Image-Name": image_path.name},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ClientError(f"API returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise ClientError(f"Could not reach API at {api_url}: {error.reason}") from error
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ClientError("API did not return JSON") from error
    if not isinstance(result, dict):
        raise ClientError("API response must be a JSON object")
    return result
