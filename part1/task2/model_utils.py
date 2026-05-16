# Group: 515513_07
# Members:
# 112652030 呂泰廷
# 112652027 吳瑞傑
# 111652043 郭宗睿

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import linalg, signal
from scipy.optimize import linear_sum_assignment


CLASS_LABELS = (0, 1, 2, 3)
DEFAULT_BANDS = (
    ("mu_8_13", 8.0, 13.0),
    ("beta_13_30", 13.0, 30.0),
    ("broad_4_40", 4.0, 40.0),
    ("high_gamma_70_124", 70.0, 124.0),
)
FIVE_BANDS = (
    ("mu_8_13", 8.0, 13.0),
    ("beta_13_30", 13.0, 30.0),
    ("gamma_30_70", 30.0, 70.0),
    ("broad_4_40", 4.0, 40.0),
    ("high_gamma_70_124", 70.0, 124.0),
)


@dataclass
class FeatureConfig:
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS
    csp_pairs: int = 2
    time_window: tuple[float, float] | None = None
    filter_order: int = 4
    covariance_shrinkage: float = 0.10
    euclidean_alignment: bool = False
    eps: float = 1e-8


def resolve_test_file(data_path: Path) -> Path:
    if data_path.is_file():
        return data_path

    candidates = [
        data_path / "test.npz",
        data_path / "task2" / "test.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate Task 2 test data. "
        "If you run from part1/task2, use --data data or --data data/test.npz. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def list_subject_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("subject*.npz"))
    expected = [data_dir / f"subject{i:02d}.npz" for i in range(1, 11)]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task 2 training files: {missing}")
    return files


def load_subject_files(files: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    sfreq_values: list[float] = []

    for file_path in files:
        arrays = np.load(file_path, allow_pickle=True)
        missing = [key for key in ("x", "y") if key not in arrays]
        if missing:
            raise KeyError(f"Missing keys in {file_path}: {missing}")

        x = arrays["x"].astype(np.float32)
        y = arrays["y"].astype(np.int64)
        subject_id = str(arrays["subject_id"]) if "subject_id" in arrays else file_path.stem
        sfreq = float(arrays["sfreq"]) if "sfreq" in arrays else 250.0

        x_parts.append(x)
        y_parts.append(y)
        subject_parts.append(np.full(len(y), subject_id, dtype=object))
        sfreq_values.append(sfreq)

    if len({round(value, 6) for value in sfreq_values}) != 1:
        raise ValueError(f"Inconsistent sampling rates across subjects: {sfreq_values}")

    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(subject_parts, axis=0),
        sfreq_values[0],
    )


def load_test_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    test_file = resolve_test_file(data_path)
    arrays = np.load(test_file, allow_pickle=True)
    if "x" not in arrays:
        raise KeyError(f"Missing 'x' in {test_file}")

    x = arrays["x"].astype(np.float32)
    ids = arrays["id"].astype(np.int64) if "id" in arrays else np.arange(len(x), dtype=np.int64)
    sfreq = float(arrays["sfreq"]) if "sfreq" in arrays else 250.0
    return x, ids, sfreq


def infer_per_subject_quota(y: np.ndarray, subjects: np.ndarray) -> dict[int, int]:
    quota: dict[int, int] = {}
    unique_subjects = sorted({str(subject) for subject in subjects})
    for label in CLASS_LABELS:
        counts = []
        for subject in unique_subjects:
            indices = np.asarray([str(item) == subject for item in subjects])
            counts.append(int(np.sum(y[indices] == label)))
        if len(set(counts)) != 1:
            raise ValueError(f"Class {label} does not have a stable per-subject count: {counts}")
        quota[int(label)] = counts[0]
    return quota


def bandpass_trials(x: np.ndarray, sfreq: float, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    nyquist = sfreq * 0.5
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.999)
    if not 0.0 < low < high < 1.0:
        raise ValueError(f"Invalid band for sfreq={sfreq}: {low_hz}-{high_hz} Hz")

    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def crop_time_window(x: np.ndarray, sfreq: float, time_window: tuple[float, float] | None) -> np.ndarray:
    if time_window is None:
        return x

    start_s, end_s = time_window
    start = max(int(round(start_s * sfreq)), 0)
    end = min(int(round(end_s * sfreq)), x.shape[-1])
    if start >= end:
        raise ValueError(f"Invalid time window {time_window} for {x.shape[-1]} samples at {sfreq} Hz")
    return x[..., start:end]


def normalized_covariances(x: np.ndarray, eps: float) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    cov = np.matmul(centered, np.swapaxes(centered, -1, -2))
    cov /= max(centered.shape[-1] - 1, 1)
    trace = np.trace(cov, axis1=1, axis2=2)[:, None, None]
    return cov / (trace + eps)


def inverse_sqrt_matrix(matrix: np.ndarray, eps: float) -> np.ndarray:
    values, vectors = linalg.eigh(matrix)
    values = np.maximum(values, eps)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def euclidean_align_trials(x: np.ndarray, groups: np.ndarray | None, eps: float) -> np.ndarray:
    if groups is None:
        groups = np.zeros(len(x), dtype=np.int64)
    groups = np.asarray(groups)
    aligned = np.empty_like(x, dtype=np.float32)

    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        group_x = x[indices]
        covariances = normalized_covariances(group_x, eps)
        reference = covariances.mean(axis=0)
        transform = inverse_sqrt_matrix(reference, eps).astype(np.float32)
        aligned[indices] = np.einsum("cd,ndt->nct", transform, group_x, optimize=True)

    return aligned


def shrink_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    if shrinkage <= 0:
        return cov
    scale = np.trace(cov) / cov.shape[0]
    return (1.0 - shrinkage) * cov + shrinkage * scale * np.eye(cov.shape[0], dtype=cov.dtype)


class EEGFeatureExtractor:
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.sfreq_: float | None = None
        self.csp_filters_: list[np.ndarray] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sfreq: float,
        groups: np.ndarray | None = None,
    ) -> "EEGFeatureExtractor":
        self.sfreq_ = float(sfreq)
        self.csp_filters_ = []
        x = crop_time_window(x, self.sfreq_, self.config.time_window)

        if self.config.euclidean_alignment:
            x = euclidean_align_trials(x, groups, self.config.eps)

        y = np.asarray(y, dtype=np.int64)
        for _, low_hz, high_hz in self.config.bands:
            filtered = bandpass_trials(x, self.sfreq_, low_hz, high_hz, self.config.filter_order)
            covariances = normalized_covariances(filtered, self.config.eps)
            band_filters: list[np.ndarray] = []

            for label in CLASS_LABELS:
                positive = covariances[y == label]
                negative = covariances[y != label]
                if len(positive) == 0 or len(negative) == 0:
                    raise ValueError(f"Cannot fit CSP for missing label {label}")

                cov_pos = shrink_covariance(positive.mean(axis=0), self.config.covariance_shrinkage)
                cov_neg = shrink_covariance(negative.mean(axis=0), self.config.covariance_shrinkage)
                values, vectors = linalg.eigh(cov_pos, cov_pos + cov_neg)
                order = np.argsort(values)
                selected = np.r_[order[: self.config.csp_pairs], order[-self.config.csp_pairs :]]
                band_filters.append(vectors[:, selected].T.astype(np.float32))

            self.csp_filters_.append(np.vstack(band_filters))

        return self

    def transform(self, x: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        if self.sfreq_ is None or not self.csp_filters_:
            raise RuntimeError("Feature extractor has not been fitted.")

        x = crop_time_window(x, self.sfreq_, self.config.time_window)

        if self.config.euclidean_alignment:
            x = euclidean_align_trials(x, groups, self.config.eps)

        feature_blocks: list[np.ndarray] = []
        for band_index, (_, low_hz, high_hz) in enumerate(self.config.bands):
            filtered = bandpass_trials(x, self.sfreq_, low_hz, high_hz, self.config.filter_order)

            channel_var = np.var(filtered, axis=-1) + self.config.eps
            feature_blocks.append(np.log(channel_var))

            filters = self.csp_filters_[band_index]
            projected = np.einsum("fc,nct->nft", filters, filtered, optimize=True)
            csp_var = np.var(projected, axis=-1) + self.config.eps
            csp_var /= csp_var.sum(axis=1, keepdims=True) + self.config.eps
            feature_blocks.append(np.log(csp_var))

        return np.concatenate(feature_blocks, axis=1).astype(np.float32)

    def fit_transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sfreq: float,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(x, y, sfreq, groups=groups).transform(x, groups=groups)


def save_checkpoint(checkpoint: dict, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: Path) -> dict:
    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)

    if "ensemble_members" in checkpoint:
        required = ["ensemble_members", "label_set", "validation"]
    else:
        required = ["feature_extractor", "model", "label_set", "validation"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint missing required fields: {missing}")
    return checkpoint


def model_score_matrix(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return np.log(np.clip(probabilities, 1e-12, 1.0))
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores
    raise TypeError("Model must provide predict_proba or decision_function for balanced assignment.")


def normalize_score_matrix(scores: np.ndarray, mode: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if mode == "raw":
        return scores
    if mode == "center_sample":
        return scores - scores.mean(axis=1, keepdims=True)
    if mode == "z_all":
        return (scores - scores.mean()) / (scores.std() + 1e-8)
    if mode == "z_col":
        return (scores - scores.mean(axis=0, keepdims=True)) / (scores.std(axis=0, keepdims=True) + 1e-8)
    raise ValueError(f"Unsupported score normalization mode: {mode}")


def balanced_quota_assignment(scores: np.ndarray, quota: dict[int, int]) -> np.ndarray:
    slot_labels: list[int] = []
    for label in CLASS_LABELS:
        slot_labels.extend([int(label)] * int(quota[int(label)]))

    if len(slot_labels) != scores.shape[0]:
        raise ValueError(f"Quota total {len(slot_labels)} does not match number of samples {scores.shape[0]}")

    slot_labels_array = np.asarray(slot_labels, dtype=np.int64)
    cost = -scores[:, slot_labels_array]
    row_indices, column_indices = linear_sum_assignment(cost)
    pred = np.empty(scores.shape[0], dtype=np.int64)
    pred[row_indices] = slot_labels_array[column_indices]
    return pred


def checkpoint_score_matrix(checkpoint: dict, x: np.ndarray) -> np.ndarray:
    if "ensemble_members" not in checkpoint:
        features = checkpoint["feature_extractor"].transform(x)
        return model_score_matrix(checkpoint["model"], features)

    aggregate: np.ndarray | None = None
    for member in checkpoint["ensemble_members"]:
        features = member["feature_extractor"].transform(x)
        scores = model_score_matrix(member["model"], features)
        scores = normalize_score_matrix(scores, member.get("score_norm", "raw"))
        weighted_scores = float(member.get("weight", 1.0)) * scores
        aggregate = weighted_scores if aggregate is None else aggregate + weighted_scores

    if aggregate is None:
        raise ValueError("Checkpoint ensemble has no members.")
    return aggregate


def predict_with_checkpoint(checkpoint: dict, x: np.ndarray) -> np.ndarray:
    if checkpoint.get("balanced_assignment", True):
        quota = {int(key): int(value) for key, value in checkpoint["inference_quota"].items()}
        return balanced_quota_assignment(checkpoint_score_matrix(checkpoint, x), quota)
    if "ensemble_members" in checkpoint:
        scores = checkpoint_score_matrix(checkpoint, x)
        labels = np.asarray(checkpoint["label_set"], dtype=np.int64)
        return labels[np.argmax(scores, axis=1)]
    features = checkpoint["feature_extractor"].transform(x)
    return checkpoint["model"].predict(features).astype(np.int64)


def write_submission(rows: Iterable[tuple[int, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(rows)
