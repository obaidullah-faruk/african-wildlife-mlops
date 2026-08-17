"""Small command-line entry point for local environment checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildlife_mlops.baseline import (
    BaselineError,
    evaluate_baseline,
    evaluate_selected_baseline_on_test,
    load_baseline_config,
    run_baseline_train,
    run_controlled_experiment,
)
from wildlife_mlops.config import load_config
from wildlife_mlops.data.audit import audit_splits
from wildlife_mlops.data.config import load_dataset_config
from wildlife_mlops.data.download import DatasetDownloadError, download_and_extract
from wildlife_mlops.data.inventory import create_inventory, create_smoke_manifest
from wildlife_mlops.data.validate import validate_dataset, write_validation_report
from wildlife_mlops.data.visualize import create_contact_sheets
from wildlife_mlops.device import DeviceSelectionError, collect_device_summary
from wildlife_mlops.doctor import run_doctor
from wildlife_mlops.overfit import (
    OverfitDiagnosticError,
    load_overfit_config,
    run_overfit_diagnostic,
)
from wildlife_mlops.predict import PredictionError, predict_image
from wildlife_mlops.pretrained import (
    PretrainedInferenceError,
    load_pretrained_config,
    run_pretrained_inference,
)
from wildlife_mlops.smoke import SmokeTrainError, load_smoke_train_config, run_smoke_train


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
    overfit_parser = subparsers.add_parser(
        "train-overfit", help="Train and validate an intentionally tiny duplicated dataset."
    )
    overfit_parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train/overfit.yaml"),
        help="Path to the versioned tiny-dataset training configuration.",
    )
    overfit_parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default=None,
        help="Override runtime.requested_device for this training run.",
    )
    smoke_parser = subparsers.add_parser(
        "train-smoke", help="Run one deterministic training epoch on the smoke subset."
    )
    smoke_parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train/smoke.yaml"),
        help="Path to the versioned smoke-training configuration.",
    )
    smoke_parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default=None,
        help="Override runtime.requested_device for this training run.",
    )
    baseline_parser = subparsers.add_parser(
        "train-baseline", help="Train the first full-data baseline."
    )
    baseline_parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train/baseline.yaml"),
        help="Path to the versioned baseline-training configuration.",
    )
    baseline_parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default=None,
        help="Override runtime.requested_device for this training run.",
    )
    evaluation_parser = subparsers.add_parser(
        "evaluate-baseline", help="Evaluate one pinned baseline on validation data."
    )
    evaluation_parser.add_argument(
        "--run-dir", type=Path, required=True, help="Baseline run directory."
    )
    evaluation_parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default=None,
        help="Override runtime.requested_device for this evaluation.",
    )
    experiment_parser = subparsers.add_parser(
        "run-experiment", help="Train one controlled image-size experiment and compare it."
    )
    experiment_parser.add_argument(
        "--baseline-run", type=Path, required=True, help="Control baseline run directory."
    )
    experiment_parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train/baseline-image-192.yaml"),
        help="Experiment configuration; it must differ only in image_size.",
    )
    experiment_parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default=None,
        help="Override runtime.requested_device for this experiment.",
    )
    test_evaluation_parser = subparsers.add_parser(
        "evaluate-release-test",
        help="Evaluate the frozen selected baseline once on the sealed test split.",
    )
    test_evaluation_parser.add_argument(
        "--release-artifact",
        type=Path,
        default=Path("artifacts/releases/selected-baseline.json"),
        help="Frozen selected-baseline release artifact.",
    )
    test_evaluation_parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default=None,
        help="Override runtime.requested_device for this evaluation.",
    )
    prediction_parser = subparsers.add_parser(
        "predict", help="Predict one JPEG or PNG with one local .pt checkpoint."
    )
    prediction_parser.add_argument("--model", type=Path, required=True, help="Model checkpoint.")
    prediction_parser.add_argument("--image", type=Path, required=True, help="Input JPEG or PNG.")
    prediction_parser.add_argument(
        "--output", type=Path, required=True, help="New JSON prediction output path."
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
        elif args.command == "train-overfit":
            overfit_config = load_overfit_config(args.train_config)
            device_summary = collect_device_summary(
                args.device or config.project.runtime.requested_device
            )
            import ultralytics

            report_path = run_overfit_diagnostic(
                overfit_config,
                data_config,
                Path.cwd(),
                device_summary,
                getattr(ultralytics, "YOLO"),
            )
            print(f"Wrote overfit diagnostic report: {report_path}")
        elif args.command == "train-smoke":
            smoke_config = load_smoke_train_config(args.train_config)
            device_summary = collect_device_summary(
                args.device or config.project.runtime.requested_device
            )
            import ultralytics

            run_dir = run_smoke_train(
                smoke_config,
                data_config,
                Path.cwd(),
                device_summary,
                getattr(ultralytics, "YOLO"),
            )
            print(f"Wrote smoke-training artifacts: {run_dir}")
        elif args.command == "train-baseline":
            baseline_config = load_baseline_config(args.train_config)
            device_summary = collect_device_summary(
                args.device or config.project.runtime.requested_device
            )
            import ultralytics

            run_dir = run_baseline_train(
                baseline_config,
                data_config,
                Path.cwd(),
                device_summary,
                getattr(ultralytics, "YOLO"),
            )
            print(f"Wrote baseline-training artifacts: {run_dir}")
        elif args.command == "evaluate-baseline":
            device_summary = collect_device_summary(
                args.device or config.project.runtime.requested_device
            )
            import ultralytics

            evaluation_dir = evaluate_baseline(
                args.run_dir,
                Path.cwd(),
                data_config,
                device_summary,
                getattr(ultralytics, "YOLO"),
            )
            print(f"Wrote validation evaluation artifacts: {evaluation_dir}")
        elif args.command == "run-experiment":
            experiment_config = load_baseline_config(args.train_config)
            device_summary = collect_device_summary(
                args.device or config.project.runtime.requested_device
            )
            import ultralytics

            experiment_run, comparison_path, release_path = run_controlled_experiment(
                args.baseline_run,
                experiment_config,
                data_config,
                Path.cwd(),
                device_summary,
                getattr(ultralytics, "YOLO"),
            )
            print(f"Wrote controlled experiment artifacts: {experiment_run}")
            print(f"Wrote comparison: {comparison_path}")
            print(f"Froze selected baseline: {release_path}")
        elif args.command == "evaluate-release-test":
            device_summary = collect_device_summary(
                args.device or config.project.runtime.requested_device
            )
            import ultralytics

            evaluation_dir = evaluate_selected_baseline_on_test(
                args.release_artifact,
                Path.cwd(),
                data_config,
                device_summary,
                getattr(ultralytics, "YOLO"),
            )
            print(f"Wrote sealed-test release artifacts: {evaluation_dir}")
        elif args.command == "predict":
            import ultralytics

            output_path = predict_image(
                args.model,
                args.image,
                args.output,
                getattr(ultralytics, "YOLO"),
            )
            print(f"Wrote prediction: {output_path}")
    except (
        DatasetDownloadError,
        BaselineError,
        ImportError,
        OSError,
        OverfitDiagnosticError,
        PredictionError,
        PretrainedInferenceError,
        SmokeTrainError,
        ValueError,
    ) as error:
        print(f"Command failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
