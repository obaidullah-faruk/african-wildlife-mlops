"""Local monitoring, sampled-label quality, and recovery evidence."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Histogram

from wildlife_mlops.release import ReleaseError, _read_json


class MonitoringError(RuntimeError):
    """Raised when monitoring evidence is missing or malformed."""


class ServiceObserver:
    """Collect request metrics and retain a deterministic subset of predictions."""

    def __init__(self, sample_path: Path, sample_rate: float) -> None:
        if not 0 <= sample_rate <= 1:
            raise ValueError("Sample rate must be between 0 and 1")
        self.sample_path = sample_path
        self.sample_rate = sample_rate
        self.registry = CollectorRegistry()
        self.request_count = Counter(
            "wildlife_http_requests_total",
            "Completed HTTP requests by endpoint and status.",
            ("endpoint", "status_code"),
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "wildlife_http_request_duration_seconds",
            "HTTP request latency by endpoint.",
            ("endpoint",),
            registry=self.registry,
        )
        self.prediction_errors = Counter(
            "wildlife_prediction_errors_total",
            "Prediction errors by reason.",
            ("reason",),
            registry=self.registry,
        )
        self.predicted_boxes = Counter(
            "wildlife_predicted_boxes_total",
            "Predicted boxes by class and candidate.",
            ("class_name", "candidate_id"),
            registry=self.registry,
        )
        self.confidence = Histogram(
            "wildlife_prediction_confidence",
            "Confidence distribution for predicted boxes.",
            ("class_name",),
            registry=self.registry,
        )
        self._write_lock = threading.Lock()

    def observe_request(self, endpoint: str, status_code: int, elapsed_seconds: float) -> None:
        """Record one completed API request."""
        self.request_count.labels(endpoint, str(status_code)).inc()
        self.request_latency.labels(endpoint).observe(elapsed_seconds)

    def observe_prediction(self, response: Mapping[str, object]) -> None:
        """Record prediction distribution and sample the response for later labeling."""
        candidate_id = response.get("model_version", "unknown")
        boxes = response.get("boxes", [])
        if not isinstance(candidate_id, str) or not isinstance(boxes, list):
            raise MonitoringError("Prediction response has an invalid monitoring schema")
        for box in boxes:
            if not isinstance(box, dict):
                continue
            class_name = box.get("class")
            confidence = box.get("confidence")
            if isinstance(class_name, str) and isinstance(confidence, (int, float)):
                self.predicted_boxes.labels(class_name, candidate_id).inc()
                self.confidence.labels(class_name).observe(float(confidence))
        trace_id = response.get("trace_id")
        if isinstance(trace_id, str) and self._should_sample(trace_id):
            self._append_sample(response)

    def observe_prediction_error(self, reason: str) -> None:
        """Record one prediction failure without exposing request content."""
        self.prediction_errors.labels(reason).inc()

    def _should_sample(self, trace_id: str) -> bool:
        bucket = int(hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:8], 16) / 2**32
        return bucket < self.sample_rate

    def _append_sample(self, response: Mapping[str, object]) -> None:
        sample = {
            "schema_version": 1,
            "sampled_at": datetime.now(UTC).isoformat(),
            "trace_id": response["trace_id"],
            "image_id": response.get("image_id"),
            "model_version": response.get("model_version"),
            "model_sha256": response.get("model_sha256"),
            "boxes": response.get("boxes", []),
        }
        self.sample_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self.sample_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(sample, sort_keys=True) + "\n")


def measure_sampled_quality(
    samples_path: Path, labels_path: Path, output_path: Path, iou_threshold: float
) -> Path:
    """Measure sampled prediction precision and recall after ground truth arrives."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be greater than 0 and at most 1")
    samples = _items_by_trace_id(_json_lines(samples_path, "sample"))
    labels = _items_by_trace_id(_json_lines(labels_path, "ground-truth label"))
    matched_ids = sorted(set(samples) & set(labels))
    if not matched_ids:
        raise MonitoringError("No ground-truth labels match sampled prediction trace IDs")
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for trace_id in matched_ids:
        predicted_boxes = _boxes(samples[trace_id])
        truth_boxes = _boxes(labels[trace_id])
        matched_truth: set[int] = set()
        for prediction in predicted_boxes:
            match = _best_match(prediction, truth_boxes, matched_truth, iou_threshold)
            if match is None:
                false_positive += 1
            else:
                matched_truth.add(match)
                true_positive += 1
        false_negative += len(truth_boxes) - len(matched_truth)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "iou_threshold": iou_threshold,
        "labeled_sample_count": len(matched_ids),
        "unlabeled_sample_count": len(samples) - len(matched_ids),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
    }
    if output_path.exists():
        raise MonitoringError(f"Quality report already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def record_recovery(
    failed_candidate: str, recovery_candidate: Path, health_path: Path, output_path: Path
) -> Path:
    """Record that a known candidate was healthy after a failed deployment attempt."""
    manifest = _candidate_manifest(recovery_candidate)
    health = _json_object(health_path, "health response")
    expected_checksum = manifest["model_sha256"]
    expected_version = f"{manifest['candidate_id']}:sha256:{expected_checksum}"
    healthy_checksum = health.get("model_sha256") == expected_checksum
    healthy_version = health.get("model_version") == expected_version
    if not healthy_checksum or not healthy_version:
        raise MonitoringError("Health response does not prove the recovery candidate is active")
    if output_path.exists():
        raise MonitoringError(f"Recovery evidence already exists: {output_path}")
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "failed_candidate": failed_candidate,
        "recovery_candidate": manifest["candidate_id"],
        "active_model_sha256": expected_checksum,
        "verified_health": str(health_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _json_lines(path: Path, kind: str) -> Iterable[dict[str, object]]:
    if not path.is_file():
        raise MonitoringError(f"{kind.capitalize()} file does not exist: {path}")
    for row, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise MonitoringError(f"Invalid JSON at {path}:{row}") from error
        if not isinstance(item, dict) or not isinstance(item.get("trace_id"), str):
            raise MonitoringError(f"{kind.capitalize()} at {path}:{row} needs a trace_id")
        _boxes(item)
        yield item


def _items_by_trace_id(items: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for item in items:
        trace_id = item["trace_id"]
        if not isinstance(trace_id, str):
            raise MonitoringError("Each item needs a string trace_id")
        indexed[trace_id] = item
    return indexed


def _json_object(path: Path, kind: str) -> dict[str, object]:
    if not path.is_file():
        raise MonitoringError(f"{kind.capitalize()} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MonitoringError(f"Invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise MonitoringError(f"{kind.capitalize()} must be a JSON object")
    return value


def _candidate_manifest(candidate_dir: Path) -> dict[str, str]:
    try:
        manifest = _read_json(candidate_dir / "candidate.json")
    except ReleaseError as error:
        raise MonitoringError(str(error)) from error
    candidate_id = manifest.get("candidate_id")
    checksum = manifest.get("model_sha256")
    if not isinstance(candidate_id, str) or not isinstance(checksum, str):
        raise MonitoringError(f"Candidate manifest lacks identity: {candidate_dir}")
    return {"candidate_id": candidate_id, "model_sha256": checksum}


def _boxes(item: Mapping[str, object]) -> list[dict[str, object]]:
    boxes = item.get("boxes")
    if not isinstance(boxes, list):
        raise MonitoringError("Each sample and ground-truth label needs a boxes list")
    validated: list[dict[str, object]] = []
    for box in boxes:
        if not isinstance(box, dict):
            raise MonitoringError("Each box must be a JSON object")
        values = ("x_min", "y_min", "x_max", "y_max", "class")
        if not isinstance(box.get("class"), str) or not all(
            isinstance(box.get(name), (int, float)) for name in values[:-1]
        ):
            raise MonitoringError("Each box needs class and numeric x_min, y_min, x_max, y_max")
        validated.append(box)
    return validated


def _best_match(
    prediction: Mapping[str, object],
    truth_boxes: list[dict[str, object]],
    matched_truth: set[int],
    threshold: float,
) -> int | None:
    best_index: int | None = None
    best_iou = threshold
    for index, truth in enumerate(truth_boxes):
        if index in matched_truth or prediction["class"] != truth["class"]:
            continue
        iou = _iou(prediction, truth)
        if iou >= best_iou:
            best_index = index
            best_iou = iou
    return best_index


def _iou(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    left = max(_coordinate(first, "x_min"), _coordinate(second, "x_min"))
    top = max(_coordinate(first, "y_min"), _coordinate(second, "y_min"))
    right = min(_coordinate(first, "x_max"), _coordinate(second, "x_max"))
    bottom = min(_coordinate(first, "y_max"), _coordinate(second, "y_max"))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (_coordinate(first, "x_max") - _coordinate(first, "x_min")) * (
        _coordinate(first, "y_max") - _coordinate(first, "y_min")
    )
    second_area = (_coordinate(second, "x_max") - _coordinate(second, "x_min")) * (
        _coordinate(second, "y_max") - _coordinate(second, "y_min")
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _coordinate(box: Mapping[str, object], name: str) -> float:
    value = box[name]
    if not isinstance(value, (int, float)):
        raise MonitoringError(f"Box coordinate {name} must be numeric")
    return float(value)
