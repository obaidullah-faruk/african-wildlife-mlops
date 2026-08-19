"""Command-line entry point for the local learning project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildlife_mlops.config import load_config
from wildlife_mlops.data.config import load_dataset_config
from wildlife_mlops.data.download import DatasetDownloadError, download_and_extract
from wildlife_mlops.data.validate import validate_dataset, write_validation_report
from wildlife_mlops.data.visualize import create_contact_sheets
from wildlife_mlops.device import DeviceSelectionError, collect_device_summary
from wildlife_mlops.doctor import run_doctor
from wildlife_mlops.predict import PredictionError, predict_image
from wildlife_mlops.pretrained import (
    PretrainedInferenceError,
    load_pretrained_config,
    run_pretrained_inference,
)
from wildlife_mlops.tracking import TrackingError, log_training_run
from wildlife_mlops.training import TrainingError, load_training_config, run_training


def build_parser() -> argparse.ArgumentParser:
    """Create the small command parser."""
    parser = argparse.ArgumentParser(prog="wildlife-mlops")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/wildlife.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the local environment.")
    commands.add_parser("show-config", help="Print the resolved configuration.")
    device = commands.add_parser("device-info", help="Print selected device details.")
    device.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default=None)
    commands.add_parser("data-download", help="Download the verified wildlife dataset.")
    commands.add_parser("data-validate", help="Validate images and YOLO labels.")
    commands.add_parser("data-visualize", help="Create ground-truth contact sheets.")
    commands.add_parser("predict-pretrained", help="Run pretrained inference on five images.")

    prediction = commands.add_parser("predict", help="Predict one local image.")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)

    for name, default in (
        ("train-overfit", Path("configs/train/overfit.yaml")),
        ("train-smoke", Path("configs/train/smoke.yaml")),
        ("train-baseline", Path("configs/train/baseline.yaml")),
    ):
        training = commands.add_parser(name, help=f"Run {name.removeprefix('train-')} training.")
        training.add_argument("--train-config", type=Path, default=default)
        training.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default=None)

    tracked = commands.add_parser("train-tracked", help="Train a baseline and record it in MLflow.")
    tracked.add_argument("--train-config", type=Path, default=Path("configs/train/baseline.yaml"))
    tracked.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default=None)
    tracked.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    tracked.add_argument("--experiment-name", default="wildlife-training")
    return parser


def main() -> int:
    """Run a command and return a shell-friendly exit code."""
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 2

    if args.command == "show-config":
        print(json.dumps(config.redacted(), indent=2, sort_keys=True))
        return 0
    if args.command == "doctor":
        results = run_doctor(config.project, Path.cwd())
        for result in results:
            print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}: {result.detail}")
        return 0 if all(result.passed for result in results) else 1
    if args.command == "device-info":
        return _print_device(args.device or config.project.runtime.requested_device)

    try:
        data_config = load_dataset_config(args.data_config)
    except ValueError as error:
        print(f"Dataset configuration error: {error}")
        return 2

    try:
        if args.command == "data-download":
            print(f"Dataset available at: {download_and_extract(data_config, Path.cwd())}")
        elif args.command == "data-validate":
            validation = validate_dataset(data_config, Path.cwd())
            report = Path("artifacts/validation-report.json")
            write_validation_report(validation, report)
            print(f"Validation report: {report}")
            print(
                f"Images: {validation.image_count}; labels: {validation.label_count}; "
                f"boxes: {validation.box_count}"
            )
            return 0 if validation.passed else 1
        elif args.command == "data-visualize":
            paths = create_contact_sheets(data_config, Path.cwd())
            print(f"Created {len(paths)} contact sheets in artifacts/data-preview")
        elif args.command == "predict-pretrained":
            inference = load_pretrained_config(Path("configs/inference/pretrained.yaml"))
            import ultralytics

            output = run_pretrained_inference(
                inference,
                Path.cwd() / data_config.dataset_root,
                Path.cwd(),
                getattr(ultralytics, "YOLO"),
            )
            print(f"Predictions: {output}")
        elif args.command == "predict":
            import ultralytics

            output = predict_image(
                args.model, args.image, args.output, getattr(ultralytics, "YOLO")
            )
            print(f"Prediction: {output}")
        elif args.command.startswith("train-"):
            training = load_training_config(args.train_config)
            device = collect_device_summary(args.device or config.project.runtime.requested_device)
            import ultralytics

            kind = "baseline"
            if args.command != "train-tracked":
                kind = args.command.removeprefix("train-")
            run_dir = run_training(
                training, data_config, Path.cwd(), device, getattr(ultralytics, "YOLO"), kind
            )
            if args.command == "train-tracked":
                run_id = log_training_run(run_dir, args.tracking_uri, args.experiment_name)
                print(f"Tracked run {run_id}: {run_dir}")
            else:
                print(f"Training output: {run_dir}")
    except (
        DatasetDownloadError,
        DeviceSelectionError,
        PredictionError,
        PretrainedInferenceError,
        TrackingError,
        TrainingError,
        ValueError,
    ) as error:
        print(f"Command failed: {error}")
        return 1
    return 0


def _print_device(requested_device: str) -> int:
    try:
        summary = collect_device_summary(requested_device)
    except (DeviceSelectionError, ImportError) as error:
        print(f"Device selection failed: {error}")
        return 1
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0
