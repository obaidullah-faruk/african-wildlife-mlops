"""Typed configuration for the versioned wildlife dataset source."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class DatasetConfig(BaseModel):
    """Dataset source, structure, and immutable source checksum."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    source_url: str = Field(min_length=1)
    source_license_reference: str = Field(min_length=1)
    archive_path: Path
    dataset_root: Path
    expected_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    class_names: list[str] = Field(min_length=1)
    splits: dict[str, int] = Field(min_length=1)
    test_split_sealed: bool


def load_dataset_config(config_path: Path) -> DatasetConfig:
    """Load dataset configuration and reject unknown or malformed keys."""
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Dataset configuration file does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw_config, dict):
        raise ValueError(f"Dataset configuration root must be a mapping: {config_path}")
    try:
        return DatasetConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ValueError(f"Invalid dataset configuration in {config_path}: {error}") from error
