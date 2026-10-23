
import pandas as pd


def analyze_quality(df):
    """
    Analyze a pandas DataFrame for common data quality issues.

    Detects:
    - Missing values
    - Duplicate rows
    - Completely empty rows
    - Constant columns
    - Potential numeric outliers using the IQR method

    Returns a structured report compatible with the other agents.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        return {
            "total_issues": 0,
            "total_warnings": 0,
            "issues": [],
            "warnings": [],
            "status": "EMPTY_DATASET",
            "message": "The dataset contains no rows or columns.",
        }

    issues = []
    warnings = []

    # 1. Detect missing values
    missing_counts = df.isna().sum()

    for column, count in missing_counts.items():
        if count > 0:
            percentage = (count / len(df)) * 100

            issues.append({
                "type": "missing_values",
                "column": str(column),
                "count": int(count),
                "percentage": round(float(percentage), 2),
                "message": (
                    f"Column '{column}' contains {count} missing values "
                    f"({percentage:.2f}% of its rows)."
                ),
            })

    # 2. Detect duplicate rows
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        issues.append({
            "type": "duplicate_rows",
            "count": duplicate_count,
            "message": (
                f"The dataset contains {duplicate_count} duplicate rows."
            ),
        })

    # 3. Detect completely empty rows
    empty_row_mask = df.isna().all(axis=1)
    empty_row_count = int(empty_row_mask.sum())

    if empty_row_count > 0:
        issues.append({
            "type": "empty_rows",
            "count": empty_row_count,
            "message": (
                f"The dataset contains {empty_row_count} completely empty rows."
            ),
        })

    # 4. Detect constant columns
    for column in df.columns:
        unique_count = int(df[column].nunique(dropna=True))

        if unique_count == 1:
            value = df[column].dropna().iloc[0]

            warnings.append({
                "type": "constant_column",
                "column": str(column),
                "message": (
                    f"Column '{column}' contains only one distinct "
                    f"non-missing value ({value!r}). It may provide "
                    "little predictive information."
                ),
            })

        elif unique_count == 0:
            warnings.append({
                "type": "constant_column",
                "column": str(column),
                "message": (
                    f"Column '{column}' contains no non-missing values."
                ),
            })

    # 5. Detect potential outliers in numeric columns using IQR
    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        values = numeric_df[column].dropna()

        # Avoid unreliable IQR analysis on very small samples.
        if len(values) < 4:
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1

        # A zero IQR does not always mean that no outliers exist,
        # but the standard IQR rule is less informative in this case.
        if iqr <= 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            outlier_percentage = (outlier_count / len(values)) * 100

            warnings.append({
                "type": "potential_outliers",
                "column": str(column),
                "count": outlier_count,
                "percentage": round(float(outlier_percentage), 2),
                "lower_bound": round(float(lower_bound), 6),
                "upper_bound": round(float(upper_bound), 6),
                "message": (
                    f"Column '{column}' contains {outlier_count} potential "
                    f"outliers ({outlier_percentage:.2f}% of its non-missing "
                    f"numeric values) using the IQR rule."
                ),
            })

    # 6. Build the final quality report
    report = {
        "total_issues": len(issues),
        "total_warnings": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "status": (
            "ISSUES_FOUND"
            if issues
            else "WARNINGS_FOUND"
            if warnings
            else "NO_COMMON_ISSUES_DETECTED"
        ),
    }

    return report
