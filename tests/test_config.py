from pathlib import Path

import pytest

from wildlife_mlops.config import load_config


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "project_name: test\n"
        "classes: [buffalo]\n"
        "runtime:\n"
        "  requested_device: auto\n"
        "  minimum_free_disk_gib: 20\n"
        "  misspelled_device: mps\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="misspelled_device"):
        load_config(config_path)


def test_environment_secret_is_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "project_name: test\n"
        "classes: [buffalo]\n"
        "runtime:\n"
        "  requested_device: auto\n"
        "  minimum_free_disk_gib: 20\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WILDLIFE_MLOPS_API_TOKEN", "not-for-display")

    config = load_config(config_path)

    assert config.redacted()["environment"]["api_token"] == "********"


def test_cuda_is_an_allowed_requested_device(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "project_name: test\n"
        "classes: [buffalo]\n"
        "runtime:\n"
        "  requested_device: cuda\n"
        "  minimum_free_disk_gib: 20\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project.runtime.requested_device == "cuda"
