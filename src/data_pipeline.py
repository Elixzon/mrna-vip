
"""Data loading and validation utilities for mRNA-VIP."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "patient_records.csv"

REQUIRED_COLUMNS = [
    "Patient_ID",
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History",
    "Reaction_Score",
]


def load_patient_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and validate its structure."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset contains no records.")

    if df["Patient_ID"].isna().any():
        raise ValueError("Patient_ID contains missing values.")

    if df["Patient_ID"].duplicated().any():
        raise ValueError("Patient_ID contains duplicate identifiers.")

    if df["Reaction_Score"].isna().any():
        raise ValueError("Reaction_Score contains missing values.")

    if not set(df["Reaction_Score"].unique()).issubset({0, 1}):
        raise ValueError("Reaction_Score must contain only 0 and 1.")

    return df


def summarize_data(df: pd.DataFrame) -> dict:
    """Summarize dataset dimensions, missing values, and target classes."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "target_counts": df["Reaction_Score"].value_counts().sort_index().to_dict(),
    }


if __name__ == "__main__":
    patient_df = load_patient_data()
    print("Dataset validation passed.")
    print(summarize_data(patient_df))
