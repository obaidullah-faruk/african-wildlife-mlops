"""Read-only checks for the local development environment."""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from wildlife_mlops.config import ProjectConfig


@dataclass(frozen=True)
class CheckResult:
    """One doctor check and a human-readable result."""

    name: str
    passed: bool
    detail: str


def choose_device(mps_available: bool) -> str:
    """Choose the native accelerator when it is available, otherwise CPU."""
    return "mps" if mps_available else "cpu"


def run_doctor(config: ProjectConfig, project_root: Path) -> list[CheckResult]:
    """Inspect prerequisites and disk capacity without changing local state."""
    checks = [
        _check_executables(("python3", "uv", "git", "docker", "make")),
        _check_free_disk(project_root, config.runtime.minimum_free_disk_gib),
        _check_torch(),
        CheckResult("host architecture", platform.machine() == "arm64", platform.machine()),
    ]
    return checks


def _check_executables(executables: tuple[str, ...]) -> CheckResult:
    missing = [executable for executable in executables if shutil.which(executable) is None]
    if missing:
        return CheckResult("required executables", False, f"missing: {', '.join(missing)}")
    return CheckResult("required executables", True, f"found: {', '.join(executables)}")


def _check_free_disk(project_root: Path, minimum_gib: int) -> CheckResult:
    free_bytes = shutil.disk_usage(project_root).free
    free_gib = free_bytes / (1024**3)
    passed = free_gib >= minimum_gib
    return CheckResult(
        "free disk",
        passed,
        f"{free_gib:.1f} GiB available (minimum {minimum_gib} GiB)",
    )


def _check_torch() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("PyTorch MPS", False, "PyTorch is not installed; run make bootstrap")

    mps_available = torch.backends.mps.is_available()
    device = choose_device(mps_available)
    return CheckResult(
        "PyTorch MPS",
        True,
        f"torch.backends.mps.is_available()={mps_available}; chosen device={device}",
    )
