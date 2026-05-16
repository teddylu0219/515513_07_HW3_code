# Group: 515513_07
# Members:
# 112652030 呂泰廷
# 112652027 吳瑞傑
# 111652043 郭宗睿

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model_utils import (
    CLASS_LABELS,
    DEFAULT_BANDS,
    EEGFeatureExtractor,
    FIVE_BANDS,
    FeatureConfig,
    balanced_quota_assignment,
    infer_per_subject_quota,
    list_subject_files,
    load_subject_files,
    load_test_data,
    model_score_matrix,
    normalize_score_matrix,
    predict_with_checkpoint,
    save_checkpoint,
    write_submission,
)


BAND_SETS = {
    "base4": DEFAULT_BANDS,
    "five": FIVE_BANDS,
    "mu_beta_high": (DEFAULT_BANDS[0], DEFAULT_BANDS[1], DEFAULT_BANDS[3]),
    "mu_beta_high60_90": (DEFAULT_BANDS[0], DEFAULT_BANDS[1], ("high_gamma_60_90", 60.0, 90.0)),
    "no_high": (DEFAULT_BANDS[0], DEFAULT_BANDS[1], DEFAULT_BANDS[2]),
}


ENSEMBLE_PRESETS = {
    "ensemble_v2": {
        "description": "score-level ensemble of complementary CSP feature capacities and band sets",
        "members": [
            {
                "name": "base_csp1_lr_c001",
                "bands": DEFAULT_BANDS,
                "csp_pairs": 1,
                "classifier": "logreg",
                "c": 0.01,
                "weight": 1.0,
                "score_norm": "z_col",
            },
            {
                "name": "base_csp2_lr_c03",
                "bands": DEFAULT_BANDS,
                "csp_pairs": 2,
                "classifier": "logreg",
                "c": 0.3,
                "weight": 0.5,
                "score_norm": "z_col",
            },
            {
                "name": "base_csp3_lr_c01",
                "bands": DEFAULT_BANDS,
                "csp_pairs": 3,
                "classifier": "logreg",
                "c": 0.1,
                "weight": 2.0,
                "score_norm": "z_col",
            },
            {
                "name": "fiveband_csp3_lr_c03",
                "bands": FIVE_BANDS,
                "csp_pairs": 3,
                "classifier": "logreg",
                "c": 0.3,
                "weight": 1.0,
                "score_norm": "z_col",
            },
        ],
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Part 1 Task 2 cross-subject EEG classifier.")
    parser.add_argument("--data", type=Path, required=True, help="Training directory containing subject01.npz ... subject10.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2_train"), help="Directory for training artifacts")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to write the trained checkpoint")
    parser.add_argument("--test-data", type=Path, default=None, help="Optional task2 data directory or test.npz for CSV generation")
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"), help="Output CSV when --test-data is provided")
    parser.add_argument(
        "--preset",
        choices=["single", *ENSEMBLE_PRESETS.keys()],
        default="single",
        help="Training recipe. Use ensemble_v2 for the tuned cross-subject ensemble.",
    )
    parser.add_argument("--c-values", type=str, default="0.03,0.1,0.3,1,3,10", help="Comma-separated regularization C values for LOSO selection")
    parser.add_argument("--classifier", choices=["logreg", "linear_svm"], default="logreg", help="Classifier family")
    parser.add_argument("--band-set", choices=list(BAND_SETS.keys()), default="base4", help="Frequency-band recipe for single-model training")
    parser.add_argument("--csp-pairs", type=int, default=2, help="Number of low/high CSP filters per class and band")
    parser.add_argument("--time-window", type=str, default=None, help="Optional crop window in seconds, formatted as start,end")
    parser.add_argument("--euclidean-alignment", action="store_true", help="Use per-subject Euclidean alignment before feature extraction")
    parser.add_argument("--no-balanced-assignment", action="store_true", help="Disable per-class quota assignment at inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def parse_c_values(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--c-values must contain at least one value")
    return values


def parse_time_window(raw: str | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    parts = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(parts) != 2 or parts[0] < 0 or parts[0] >= parts[1]:
        raise ValueError("--time-window must use the format start,end with 0 <= start < end")
    return parts[0], parts[1]


def build_classifier(classifier: str, c_value: float, seed: int) -> Pipeline:
    if classifier == "logreg":
        estimator = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        )
    elif classifier == "linear_svm":
        estimator = SVC(
            kernel="linear",
            C=c_value,
            class_weight="balanced",
            decision_function_shape="ovr",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def feature_config_from_member(member: dict) -> FeatureConfig:
    return FeatureConfig(
        bands=member["bands"],
        csp_pairs=int(member["csp_pairs"]),
        time_window=member.get("time_window"),
        covariance_shrinkage=float(member.get("covariance_shrinkage", 0.10)),
        euclidean_alignment=bool(member.get("euclidean_alignment", False)),
    )


def serialize_member(member: dict) -> dict:
    return {
        "name": member["name"],
        "bands": [list(band) for band in member["bands"]],
        "csp_pairs": int(member["csp_pairs"]),
        "time_window": list(member["time_window"]) if member.get("time_window") is not None else None,
        "classifier": member["classifier"],
        "c": float(member["c"]),
        "weight": float(member.get("weight", 1.0)),
        "score_norm": member.get("score_norm", "raw"),
    }


def evaluate_loso(
    files: list[Path],
    c_values: list[float],
    feature_config: FeatureConfig,
    classifier_name: str,
    seed: int,
) -> tuple[float, dict]:
    fold_rows: list[dict] = []
    score_by_c = {
        c_value: {
            "y_true": [],
            "y_pred": [],
            "y_pred_balanced": [],
            "fold_macro_f1": [],
            "fold_accuracy": [],
            "fold_balanced_macro_f1": [],
            "fold_balanced_accuracy": [],
        }
        for c_value in c_values
    }

    for heldout_file in files:
        train_files = [path for path in files if path != heldout_file]
        x_train, y_train, train_subjects, train_sfreq = load_subject_files(train_files)
        x_val, y_val, val_subjects, val_sfreq = load_subject_files([heldout_file])
        if abs(train_sfreq - val_sfreq) > 1e-6:
            raise ValueError(f"Sampling-rate mismatch for {heldout_file}")

        extractor = EEGFeatureExtractor(feature_config)
        x_train_features = extractor.fit_transform(x_train, y_train, train_sfreq, groups=train_subjects)
        x_val_features = extractor.transform(x_val)
        subject_id = str(val_subjects[0])

        for c_value in c_values:
            classifier = build_classifier(classifier_name, c_value, seed)
            classifier.fit(x_train_features, y_train)
            pred = classifier.predict(x_val_features)
            quota = infer_per_subject_quota(y_train, train_subjects)
            balanced_pred = balanced_quota_assignment(model_score_matrix(classifier, x_val_features), quota)
            macro_f1 = f1_score(y_val, pred, labels=list(CLASS_LABELS), average="macro")
            accuracy = accuracy_score(y_val, pred)
            balanced_macro_f1 = f1_score(y_val, balanced_pred, labels=list(CLASS_LABELS), average="macro")
            balanced_accuracy = accuracy_score(y_val, balanced_pred)

            score_by_c[c_value]["y_true"].extend(y_val.astype(int).tolist())
            score_by_c[c_value]["y_pred"].extend(pred.astype(int).tolist())
            score_by_c[c_value]["y_pred_balanced"].extend(balanced_pred.astype(int).tolist())
            score_by_c[c_value]["fold_macro_f1"].append(float(macro_f1))
            score_by_c[c_value]["fold_accuracy"].append(float(accuracy))
            score_by_c[c_value]["fold_balanced_macro_f1"].append(float(balanced_macro_f1))
            score_by_c[c_value]["fold_balanced_accuracy"].append(float(balanced_accuracy))
            fold_rows.append(
                {
                    "c": c_value,
                    "heldout_subject": subject_id,
                    "macro_f1": float(macro_f1),
                    "accuracy": float(accuracy),
                    "balanced_macro_f1": float(balanced_macro_f1),
                    "balanced_accuracy": float(balanced_accuracy),
                }
            )

    candidates = []
    for c_value, values in score_by_c.items():
        y_true = np.asarray(values["y_true"], dtype=np.int64)
        y_pred = np.asarray(values["y_pred"], dtype=np.int64)
        y_pred_balanced = np.asarray(values["y_pred_balanced"], dtype=np.int64)
        candidates.append(
            {
                "c": c_value,
                "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_LABELS), average="macro")),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_macro_f1": float(f1_score(y_true, y_pred_balanced, labels=list(CLASS_LABELS), average="macro")),
                "balanced_accuracy": float(accuracy_score(y_true, y_pred_balanced)),
                "mean_fold_macro_f1": float(np.mean(values["fold_macro_f1"])),
                "std_fold_macro_f1": float(np.std(values["fold_macro_f1"])),
                "mean_fold_accuracy": float(np.mean(values["fold_accuracy"])),
                "std_fold_accuracy": float(np.std(values["fold_accuracy"])),
                "mean_fold_balanced_macro_f1": float(np.mean(values["fold_balanced_macro_f1"])),
                "std_fold_balanced_macro_f1": float(np.std(values["fold_balanced_macro_f1"])),
                "mean_fold_balanced_accuracy": float(np.mean(values["fold_balanced_accuracy"])),
                "std_fold_balanced_accuracy": float(np.std(values["fold_balanced_accuracy"])),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["balanced_macro_f1"],
            item["balanced_accuracy"],
            item["macro_f1"],
            -item["std_fold_balanced_macro_f1"],
        ),
        reverse=True,
    )
    best_c = float(candidates[0]["c"])
    preprocessing = []
    if feature_config.euclidean_alignment:
        preprocessing.append("per-subject Euclidean alignment using unlabeled covariance only")
    preprocessing.extend(
        [
            "Butterworth bandpass filtering",
            "log channel variance",
            "one-vs-rest CSP log-variance features",
        ]
    )

    summary = {
        "validation_protocol": "leave-one-subject-out cross-validation over subject01-subject10",
        "feature_bands": [list(band) for band in feature_config.bands],
        "time_window": list(feature_config.time_window) if feature_config.time_window is not None else None,
        "preprocessing": preprocessing,
        "csp_pairs": feature_config.csp_pairs,
        "classifier": classifier_name,
        "postprocessing": "balanced per-subject quota assignment using the 8/8/8/8 class prior observed in every training subject",
        "candidates": candidates,
        "best_c": best_c,
        "folds": fold_rows,
    }
    return best_c, summary


def evaluate_ensemble_loso(files: list[Path], preset_name: str, seed: int) -> dict:
    preset = ENSEMBLE_PRESETS[preset_name]
    members = preset["members"]
    fold_rows: list[dict] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_pred_balanced_all: list[int] = []

    for heldout_file in files:
        train_files = [path for path in files if path != heldout_file]
        x_train, y_train, train_subjects, train_sfreq = load_subject_files(train_files)
        x_val, y_val, val_subjects, val_sfreq = load_subject_files([heldout_file])
        if abs(train_sfreq - val_sfreq) > 1e-6:
            raise ValueError(f"Sampling-rate mismatch for {heldout_file}")

        aggregate_scores: np.ndarray | None = None
        for member in members:
            extractor = EEGFeatureExtractor(feature_config_from_member(member))
            x_train_features = extractor.fit_transform(x_train, y_train, train_sfreq, groups=train_subjects)
            x_val_features = extractor.transform(x_val)
            classifier = build_classifier(member["classifier"], float(member["c"]), seed)
            classifier.fit(x_train_features, y_train)
            scores = model_score_matrix(classifier, x_val_features)
            scores = normalize_score_matrix(scores, member.get("score_norm", "raw"))
            weighted_scores = float(member.get("weight", 1.0)) * scores
            aggregate_scores = weighted_scores if aggregate_scores is None else aggregate_scores + weighted_scores

        if aggregate_scores is None:
            raise ValueError(f"Preset {preset_name} does not define any ensemble members.")

        pred = np.asarray(CLASS_LABELS, dtype=np.int64)[np.argmax(aggregate_scores, axis=1)]
        quota = infer_per_subject_quota(y_train, train_subjects)
        balanced_pred = balanced_quota_assignment(aggregate_scores, quota)
        macro_f1 = f1_score(y_val, pred, labels=list(CLASS_LABELS), average="macro")
        accuracy = accuracy_score(y_val, pred)
        balanced_macro_f1 = f1_score(y_val, balanced_pred, labels=list(CLASS_LABELS), average="macro")
        balanced_accuracy = accuracy_score(y_val, balanced_pred)

        y_true_all.extend(y_val.astype(int).tolist())
        y_pred_all.extend(pred.astype(int).tolist())
        y_pred_balanced_all.extend(balanced_pred.astype(int).tolist())
        fold_rows.append(
            {
                "heldout_subject": str(val_subjects[0]),
                "macro_f1": float(macro_f1),
                "accuracy": float(accuracy),
                "balanced_macro_f1": float(balanced_macro_f1),
                "balanced_accuracy": float(balanced_accuracy),
            }
        )

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)
    y_pred_balanced = np.asarray(y_pred_balanced_all, dtype=np.int64)
    fold_balanced = [row["balanced_macro_f1"] for row in fold_rows]
    fold_raw = [row["macro_f1"] for row in fold_rows]
    return {
        "validation_protocol": "leave-one-subject-out cross-validation over subject01-subject10",
        "preset": preset_name,
        "description": preset["description"],
        "members": [serialize_member(member) for member in members],
        "preprocessing": [
            "Butterworth bandpass filtering",
            "log channel variance",
            "one-vs-rest CSP log-variance features",
            "score-level ensemble with per-class z-score calibration before quota assignment",
        ],
        "postprocessing": "balanced per-subject quota assignment using the 8/8/8/8 class prior observed in every training subject",
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_LABELS), average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_macro_f1": float(f1_score(y_true, y_pred_balanced, labels=list(CLASS_LABELS), average="macro")),
        "balanced_accuracy": float(accuracy_score(y_true, y_pred_balanced)),
        "mean_fold_macro_f1": float(np.mean(fold_raw)),
        "std_fold_macro_f1": float(np.std(fold_raw)),
        "mean_fold_balanced_macro_f1": float(np.mean(fold_balanced)),
        "std_fold_balanced_macro_f1": float(np.std(fold_balanced)),
        "folds": fold_rows,
    }


def train_final_model(
    files: list[Path],
    feature_config: FeatureConfig,
    classifier_name: str,
    c_value: float,
    balanced_assignment: bool,
    seed: int,
) -> dict:
    x_train, y_train, subjects, sfreq = load_subject_files(files)
    extractor = EEGFeatureExtractor(feature_config)
    features = extractor.fit_transform(x_train, y_train, sfreq, groups=subjects)
    classifier = build_classifier(classifier_name, c_value, seed)
    classifier.fit(features, y_train)
    return {
        "format_version": 1,
        "task": "part1_task2_cross_subject_eeg",
        "feature_extractor": extractor,
        "model": classifier,
        "label_set": list(CLASS_LABELS),
        "training_subjects": sorted({str(subject) for subject in subjects}),
        "num_training_trials": int(len(y_train)),
        "balanced_assignment": bool(balanced_assignment),
        "inference_quota": infer_per_subject_quota(y_train, subjects),
    }


def train_final_ensemble(
    files: list[Path],
    preset_name: str,
    balanced_assignment: bool,
    seed: int,
) -> dict:
    preset = ENSEMBLE_PRESETS[preset_name]
    x_train, y_train, subjects, sfreq = load_subject_files(files)
    ensemble_members: list[dict] = []

    for member in preset["members"]:
        extractor = EEGFeatureExtractor(feature_config_from_member(member))
        features = extractor.fit_transform(x_train, y_train, sfreq, groups=subjects)
        classifier = build_classifier(member["classifier"], float(member["c"]), seed)
        classifier.fit(features, y_train)
        ensemble_members.append(
            {
                **serialize_member(member),
                "feature_extractor": extractor,
                "model": classifier,
            }
        )

    return {
        "format_version": 2,
        "task": "part1_task2_cross_subject_eeg",
        "preset": preset_name,
        "description": preset["description"],
        "ensemble_members": ensemble_members,
        "label_set": list(CLASS_LABELS),
        "training_subjects": sorted({str(subject) for subject in subjects}),
        "num_training_trials": int(len(y_train)),
        "balanced_assignment": bool(balanced_assignment),
        "inference_quota": infer_per_subject_quota(y_train, subjects),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    files = list_subject_files(args.data)
    balanced_assignment = not args.no_balanced_assignment

    if args.preset == "single":
        c_values = parse_c_values(args.c_values)
        feature_config = FeatureConfig(
            bands=BAND_SETS[args.band_set],
            csp_pairs=args.csp_pairs,
            time_window=parse_time_window(args.time_window),
            euclidean_alignment=args.euclidean_alignment,
        )
        best_c, validation = evaluate_loso(files, c_values, feature_config, args.classifier, args.seed)
        checkpoint = train_final_model(files, feature_config, args.classifier, best_c, balanced_assignment, args.seed)
    else:
        validation = evaluate_ensemble_loso(files, args.preset, args.seed)
        checkpoint = train_final_ensemble(files, args.preset, balanced_assignment, args.seed)

    checkpoint["validation"] = validation
    command_parts = [
        "python",
        "train.py",
        "--data",
        str(args.data),
        "--output-dir",
        str(args.output_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--preset",
        args.preset,
        "--seed",
        str(args.seed),
    ]
    if args.preset == "single":
        command_parts.extend(
            [
                "--c-values",
                args.c_values,
                "--classifier",
                args.classifier,
                "--band-set",
                args.band_set,
                "--csp-pairs",
                str(args.csp_pairs),
            ]
        )
        if args.time_window is not None:
            command_parts.extend(["--time-window", args.time_window])
    if args.euclidean_alignment:
        command_parts.append("--euclidean-alignment")
    if args.no_balanced_assignment:
        command_parts.append("--no-balanced-assignment")
    if args.test_data is not None:
        command_parts.extend(["--test-data", str(args.test_data)])
    command_parts.extend(["--submission", str(args.submission)])
    checkpoint["training_command"] = " ".join(command_parts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint, args.checkpoint)
    validation_path = args.output_dir / "validation_summary.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    if args.preset == "single":
        print(f"Best C: {checkpoint['validation']['best_c']}")
        print(f"LOSO macro-F1: {validation['candidates'][0]['macro_f1']:.4f}")
        print(f"LOSO accuracy: {validation['candidates'][0]['accuracy']:.4f}")
        print(f"Balanced LOSO macro-F1: {validation['candidates'][0]['balanced_macro_f1']:.4f}")
        print(f"Balanced LOSO accuracy: {validation['candidates'][0]['balanced_accuracy']:.4f}")
    else:
        print(f"Preset: {args.preset}")
        print(f"LOSO macro-F1: {validation['macro_f1']:.4f}")
        print(f"LOSO accuracy: {validation['accuracy']:.4f}")
        print(f"Balanced LOSO macro-F1: {validation['balanced_macro_f1']:.4f}")
        print(f"Balanced LOSO accuracy: {validation['balanced_accuracy']:.4f}")
    print(f"Wrote checkpoint to {args.checkpoint}")
    print(f"Wrote validation summary to {validation_path}")

    if args.test_data is not None:
        x_test, ids, _ = load_test_data(args.test_data)
        pred = predict_with_checkpoint(checkpoint, x_test)
        rows = [(int(sample_id), int(label)) for sample_id, label in zip(ids, pred)]
        write_submission(rows, args.submission)
        print(f"Wrote {len(rows)} predictions to {args.submission}")


if __name__ == "__main__":
    main()
