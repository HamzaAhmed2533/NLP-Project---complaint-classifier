"""
data_loader.py
--------------
Responsible for loading the raw dataset from a CSV file and performing
initial data inspection and cleaning.

Responsibilities:
    - Load CSV into a pandas DataFrame
    - Validate required columns ('text', 'label')
    - Handle missing values and duplicate records
    - Provide basic dataset statistics
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "complaints.csv"
)
REQUIRED_COLUMNS = {"text", "label"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(filepath: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Load the customer complaints CSV file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing 'text' and 'label' columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with 'text' and 'label' columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If the CSV is missing the required columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Run  python data/generate_dataset.py  to create a sample dataset."
        )

    df = pd.read_csv(filepath)

    # ---- Validate columns --------------------------------------------------
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # ---- Keep only relevant columns ----------------------------------------
    df = df[["text", "label"]].copy()

    # ---- Basic cleaning ----------------------------------------------------
    df = _remove_missing_values(df)
    df = _remove_duplicates(df)

    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a concise summary of the loaded dataset to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by load_data().
    """
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Total samples  : {len(df)}")
    print(f"Columns        : {list(df.columns)}")
    print(f"Missing values : {df.isnull().sum().to_dict()}")
    print("\nLabel distribution:")
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  {label:<25} {count:>4} samples  ({pct:.1f}%)")
    print("\nSample entries:")
    print(df.sample(min(3, len(df)), random_state=42).to_string(index=False))
    print("=" * 60)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where either 'text' or 'label' is missing or empty."""
    before = len(df)
    # Drop NaN
    df = df.dropna(subset=["text", "label"])
    # Drop rows where text/label is only whitespace
    df = df[df["text"].str.strip().astype(bool)]
    df = df[df["label"].str.strip().astype(bool)]
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[data_loader] Removed {dropped} row(s) with missing/empty values.")
    return df.reset_index(drop=True)


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (same text AND label)."""
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[data_loader] Removed {dropped} duplicate row(s).")
    return df.reset_index(drop=True)
