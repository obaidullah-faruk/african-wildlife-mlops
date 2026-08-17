"""Small command-line entry point for local environment checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildlife_mlops.config import load_config
from wildlife_mlops.doctor import run_doctor


def build_parser() -> argparse.ArgumentParser:
    """Create the command parser."""
    parser = argparse.ArgumentParser(prog="wildlife-mlops")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project.yaml"),
        help="Path to the versioned YAML configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Run read-only local environment checks.")
    subparsers.add_parser("show-config", help="Print the resolved, redacted configuration.")
    return parser


def main() -> int:
    """Run a command and return an appropriate process exit code."""
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 2

    print("Resolved configuration (secrets redacted):")
    print(json.dumps(config.redacted(), indent=2, sort_keys=True))

    if args.command == "show-config":
        return 0

    results = run_doctor(config.project, Path.cwd())
    for result in results:
        state = "PASS" if result.passed else "FAIL"
        print(f"{state}: {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
