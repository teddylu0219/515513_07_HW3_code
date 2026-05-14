# Group: 515513_07
# Members:
# 112652030 呂泰廷 
# 112652027 吳瑞傑 
# 111652043 郭宗睿

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


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


def load_test_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    test_file = resolve_test_file(data_path)
    arrays = np.load(test_file, allow_pickle=True)

    if "x" not in arrays:
        raise KeyError(f"Missing 'x' in {test_file}")

    x = arrays["x"]
    ids = arrays["id"] if "id" in arrays else np.arange(len(x), dtype=np.int64)
    return x, ids.astype(np.int64)


def load_checkpoint(checkpoint_path: Path):
    """
    TODO(student):
    Load your own checkpoint format here.

    Examples:
    - torch.load(checkpoint_path, map_location="cpu")
    - joblib.load(checkpoint_path)
    - pickle.load(...)
    """
    raise NotImplementedError("TODO: load your checkpoint")


def preprocess_for_inference(x: np.ndarray, checkpoint) -> np.ndarray:
    """
    TODO(student):
    Reproduce the preprocessing used during Task 2 training.

    Expected input:
    - x shape: (N, C, T)

    Expected output:
    - preprocessed array in the shape your model expects
    """
    _ = checkpoint
    raise NotImplementedError("TODO: implement inference preprocessing")


def build_model(checkpoint):
    """
    TODO(student):
    Rebuild your model from the checkpoint metadata.
    """
    _ = checkpoint
    raise NotImplementedError("TODO: rebuild your model")


def predict(model, x: np.ndarray) -> np.ndarray:
    """
    TODO(student):
    Run inference and return integer labels.

    Required return:
    - shape: (N,)
    - dtype/content: integers in {0, 1, 2, 3}
    """
    _ = model
    _ = x
    raise NotImplementedError("TODO: generate predictions")


def validate_predictions(pred: np.ndarray, num_examples: int) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.shape != (num_examples,):
        raise ValueError(f"Expected predictions with shape ({num_examples},), got {pred.shape}")
    if not np.issubdtype(pred.dtype, np.integer):
        raise TypeError(f"Predictions must be integers, got dtype {pred.dtype}")
    if np.any((pred < 0) | (pred > 3)):
        raise ValueError("Predicted labels must be integers in {0, 1, 2, 3}")
    return pred.astype(np.int64)


def write_submission(rows: Iterable[Tuple[int, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    x_test, ids = load_test_data(args.data)
    checkpoint = load_checkpoint(args.checkpoint)
    x_test = preprocess_for_inference(x_test, checkpoint)
    model = build_model(checkpoint)
    pred = predict(model, x_test)
    pred = validate_predictions(pred, len(ids))

    rows = [(int(sample_id), int(label)) for sample_id, label in zip(ids, pred)]
    write_submission(rows, args.output)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
