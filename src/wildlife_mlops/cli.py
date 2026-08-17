"""Small command-line entry point for local environment checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildlife_mlops.config import load_config
from wildlife_mlops.data.audit import audit_splits
from wildlife_mlops.data.config import load_dataset_config
from wildlife_mlops.data.download import DatasetDownloadError, download_and_extract
from wildlife_mlops.data.inventory import create_inventory, create_smoke_manifest
from wildlife_mlops.data.validate import validate_dataset, write_validation_report
from wildlife_mlops.data.visualize import create_contact_sheets
from wildlife_mlops.device import DeviceSelectionError, collect_device_summary
from wildlife_mlops.doctor import run_doctor
from wildlife_mlops.pretrained import (
    PretrainedInferenceError,
    load_pretrained_config,
    run_pretrained_inference,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command parser."""
    parser = argparse.ArgumentParser(prog="wildlife-mlops")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project.yaml"),
        help="Path to the versioned YAML configuration.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/wildlife.yaml"),
        help="Path to the versioned dataset YAML configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Run read-only local environment checks.")
    subparsers.add_parser("show-config", help="Print the resolved, redacted configuration.")
    device_parser = subparsers.add_parser(
        "device-info", help="Print the selected device and runtime details."
    )
    device_parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default=None,
        help="Override runtime.requested_device for this command.",
    )
    subparsers.add_parser(
        "data-download", help="Download and extract the checksum-verified dataset."
    )
    subparsers.add_parser("data-inventory", help="Write dataset inventory artifacts.")
    subparsers.add_parser("data-validate", help="Validate every image and YOLO label.")
    subparsers.add_parser(
        "data-visualize", help="Render deterministic ground-truth contact sheets."
    )
    subparsers.add_parser("data-audit-splits", help="Write exact and perceptual duplicate reports.")
    subparsers.add_parser(
        "data-smoke-manifest", help="Write a deterministic training smoke manifest."
    )
    subparsers.add_parser(
        "predict-pretrained", help="Predict a fixed image sample with pinned weights."
    )
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

    if args.command == "doctor":
        results = run_doctor(config.project, Path.cwd())
        for result in results:
            state = "PASS" if result.passed else "FAIL"
            print(f"{state}: {result.name}: {result.detail}")
        return 0 if all(result.passed for result in results) else 1

    if args.command == "device-info":
        requested_device = args.device or config.project.runtime.requested_device
        try:
            summary = collect_device_summary(requested_device)
        except (DeviceSelectionError, ImportError) as error:
            print(f"Device selection failed: {error}")
            return 1
        print("Device runtime summary:")
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return 0

    try:
        data_config = load_dataset_config(args.data_config)
    except ValueError as error:
        print(f"Dataset configuration error: {error}")
        return 2

    print("Dataset configuration:")
    print(json.dumps(data_config.model_dump(mode="json"), indent=2, sort_keys=True))
    try:
        if args.command == "data-download":
            print(f"Dataset available at: {download_and_extract(data_config, Path.cwd())}")
        elif args.command == "data-inventory":
            parquet_path, summary_path = create_inventory(data_config, Path.cwd())
            print(f"Wrote inventory: {parquet_path}")
            print(f"Wrote summary: {summary_path}")
        elif args.command == "data-validate":
            validation_result = validate_dataset(data_config, Path.cwd())
            report_path = Path.cwd() / "artifacts" / "validation-report.json"
            write_validation_report(validation_result, report_path)
            print(f"Wrote validation report: {report_path}")
            print(f"Validation errors: {len(validation_result.issues)}")
            return 0 if validation_result.passed else 1
        elif args.command == "data-visualize":
            paths = create_contact_sheets(data_config, Path.cwd())
            print(f"Wrote {len(paths)} ground-truth contact sheets")
        elif args.command == "data-audit-splits":
            print(f"Wrote split integrity report: {audit_splits(data_config, Path.cwd())}")
        elif args.command == "data-smoke-manifest":
            print(f"Wrote smoke manifest: {create_smoke_manifest(data_config, Path.cwd())}")
        elif args.command == "predict-pretrained":
            inference_config = load_pretrained_config(Path("configs/inference/pretrained.yaml"))
            import ultralytics

            output_path = run_pretrained_inference(
                inference_config,
                Path.cwd() / data_config.dataset_root,
                Path.cwd(),
                getattr(ultralytics, "YOLO"),
            )
            print(f"Wrote pretrained predictions: {output_path}")
    except (
        DatasetDownloadError,
        ImportError,
        OSError,
        PretrainedInferenceError,
        ValueError,
    ) as error:
        print(f"Command failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
