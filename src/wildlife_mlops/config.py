"""Typed loading for the small, versioned project configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError


class RuntimeConfig(BaseModel):
    """Non-secret runtime choices kept under version control."""

    model_config = ConfigDict(extra="forbid")

    requested_device: str = Field(pattern="^(auto|mps|cuda|cpu)$")
    minimum_free_disk_gib: int = Field(gt=0)


class ProjectConfig(BaseModel):
    """The complete versioned project configuration schema."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(min_length=1)
    classes: list[str] = Field(min_length=1)
    runtime: RuntimeConfig


class EnvironmentConfig(BaseModel):
    """Machine-specific values sourced exclusively from environment variables."""

    environment: str = "local"
    api_endpoint: str | None = None
    api_token: SecretStr | None = None


class ResolvedConfig(BaseModel):
    """Configuration shown at command start, with secrets redacted."""

    project: ProjectConfig
    environment: EnvironmentConfig

    def redacted(self) -> dict[str, Any]:
        """Return JSON-compatible config data without exposing a secret."""
        value = self.model_dump(mode="json")
        if value["environment"]["api_token"] is not None:
            value["environment"]["api_token"] = "********"
        return value


def load_config(config_path: Path) -> ResolvedConfig:
    """Load a YAML config, rejecting malformed content and unknown keys."""
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Configuration file does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    try:
        project = ProjectConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ValueError(f"Invalid configuration in {config_path}: {error}") from error

    environment = EnvironmentConfig(
        environment=os.getenv("WILDLIFE_MLOPS_ENVIRONMENT", "local"),
        api_endpoint=os.getenv("WILDLIFE_MLOPS_API_ENDPOINT"),
        api_token=(
            SecretStr(api_token)
            if (api_token := os.getenv("WILDLIFE_MLOPS_API_TOKEN")) is not None
            else None
        ),
    )
    return ResolvedConfig(project=project, environment=environment)
