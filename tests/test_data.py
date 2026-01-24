import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/processed")


def test_train_file_exists():
    assert (DATA_DIR / "train.csv").exists()


def test_inference_file_exists():
    assert (DATA_DIR / "inference.csv").exists()


def test_no_missing_values():
    df = pd.read_csv(DATA_DIR / "train.csv")
    assert not df.isna().any().any()


def test_target_column_exists():
    df = pd.read_csv(DATA_DIR / "train.csv")
    assert "target" in df.columns
