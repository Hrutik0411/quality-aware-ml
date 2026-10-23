
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Analyze dataset structure, statistics, and quality."""

    numerical_df = df.select_dtypes(include="number")

    profile = {
        "number_of_rows": len(df),
        "number_of_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),

        # Missing-value analysis
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (
            df.isnull().mean().mul(100).round(2).to_dict()
        ),

        # Duplicate analysis
        "duplicate_rows": int(df.duplicated().sum()),

        # Unique values in every column
        "unique_values": df.nunique(dropna=True).to_dict(),

        # Statistical summary for numerical columns
        "numerical_statistics": (
            numerical_df.describe().round(2).to_dict()
        ),
    }

    # Overall dataset quality indicators
    total_cells = df.size
    missing_cells = int(df.isnull().sum().sum())

    profile["quality_summary"] = {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_cell_percentage": (
            round(missing_cells / total_cells * 100, 2)
            if total_cells > 0 else 0.0
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "empty_rows": int(df.isnull().all(axis=1).sum()),
    }

    return profile
