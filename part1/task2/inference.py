# Group: 515513_07
# Members:
# 112652030 呂泰廷
# 112652027 吳瑞傑
# 111652043 郭宗睿

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from model_utils import load_checkpoint as read_checkpoint
from model_utils import load_test_data as read_test_data
from model_utils import predict_with_checkpoint
from model_utils import write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 2 inference and write a Kaggle submission CSV.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the task2 data directory (recommended: ./data) or directly to test.npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to your saved model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path. Defaults to ./submission.csv",
    )
    return parser.parse_args()


def resolve_test_file(data_path: Path) -> Path:
    from model_utils import resolve_test_file as resolve

    return resolve(data_path)


def load_test_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, ids, _ = read_test_data(data_path)
    return x, ids


def load_checkpoint(checkpoint_path: Path):
    return read_checkpoint(checkpoint_path)


def preprocess_for_inference(x: np.ndarray, checkpoint) -> np.ndarray:
    return checkpoint["feature_extractor"].transform(x)


def build_model(checkpoint):
    return checkpoint["model"]


def predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x).astype(np.int64)


def validate_predictions(pred: np.ndarray, num_examples: int) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.shape != (num_examples,):
        raise ValueError(f"Expected predictions with shape ({num_examples},), got {pred.shape}")
    if not np.issubdtype(pred.dtype, np.integer):
        raise TypeError(f"Predictions must be integers, got dtype {pred.dtype}")
    if np.any((pred < 0) | (pred > 3)):
        raise ValueError("Predicted labels must be integers in {0, 1, 2, 3}")
    return pred.astype(np.int64)


def main() -> None:
    args = parse_args()

    x_test, ids = load_test_data(args.data)
    checkpoint = load_checkpoint(args.checkpoint)
    pred = predict_with_checkpoint(checkpoint, x_test)
    pred = validate_predictions(pred, len(ids))

    rows = [(int(sample_id), int(label)) for sample_id, label in zip(ids, pred)]
    write_submission(rows, args.output)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
