from pathlib import Path

from wildlife_mlops.maintenance import _compose_command, _new_backup_directory


def test_new_backup_directory_uses_the_local_artifacts_root(tmp_path: Path) -> None:
    backup_directory = _new_backup_directory(tmp_path)

    assert backup_directory.parent == tmp_path / "artifacts" / "mlflow-maintenance"
    assert backup_directory.is_dir()


def test_compose_command_uses_the_environment_file() -> None:
    command = _compose_command(Path(".env"), "ps")

    assert command == ["docker", "compose", "--env-file", ".env", "ps"]
