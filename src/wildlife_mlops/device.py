"""Device selection and runtime details for local model execution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

SelectedDevice = Literal["mps", "cuda", "cpu"]


class DeviceSelectionError(ValueError):
    """Raised when an explicitly requested device cannot be used."""


@dataclass(frozen=True)
class DeviceSummary:
    """Runtime values needed to reproduce a local device choice."""

    device: SelectedDevice
    pytorch_version: str
    ultralytics_version: str
    precision: str
    mps_fallback_setting: str

    def as_dict(self) -> dict[str, str]:
        """Return JSON-compatible device details."""
        return asdict(self)


def select_device(
    requested_device: str, mps_available: bool, cuda_available: bool = False
) -> SelectedDevice:
    """Choose MPS, CUDA, or CPU automatically, or honor a valid override."""
    if requested_device == "auto":
        if mps_available:
            return "mps"
        return "cuda" if cuda_available else "cpu"
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "mps":
        if mps_available:
            return "mps"
        raise DeviceSelectionError(
            "MPS was explicitly requested but is unavailable; use --device cuda, cpu, or auto"
        )
    if requested_device == "cuda":
        if cuda_available:
            return "cuda"
        raise DeviceSelectionError(
            "CUDA was explicitly requested but is unavailable; use --device mps, cpu, or auto"
        )
    raise DeviceSelectionError(
        f"Unsupported device {requested_device!r}; expected one of: auto, mps, cuda, cpu"
    )


def collect_device_summary(
    requested_device: str,
    *,
    environment: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
    ultralytics_module: Any | None = None,
) -> DeviceSummary:
    """Collect the selected device and relevant library/runtime settings."""
    if torch_module is None:
        import torch as torch_module
    if ultralytics_module is None:
        import ultralytics as ultralytics_module
    assert torch_module is not None
    assert ultralytics_module is not None

    selected_device = select_device(
        requested_device,
        mps_available=bool(torch_module.backends.mps.is_available()),
        cuda_available=bool(torch_module.cuda.is_available()),
    )
    runtime_environment = os.environ if environment is None else environment
    return DeviceSummary(
        device=selected_device,
        pytorch_version=str(torch_module.__version__),
        ultralytics_version=str(ultralytics_module.__version__),
        precision=str(torch_module.get_float32_matmul_precision()),
        mps_fallback_setting=runtime_environment.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
    )
