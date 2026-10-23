
def analyze_impact(df, quality_report: dict) -> dict:
    """
    Estimate potential ML risks from detected data-quality findings.
    These are heuristic assessments, not measured model-performance results.
    """

    impacts = []

    # Check detected issues
    for issue in quality_report.get("issues", []):
        issue_type = issue.get("type")
        column = issue.get("column", "dataset")
        count = issue.get("count", 0)

        if issue_type == "missing_values":
            impacts.append({
                "issue": issue_type,
                "column": column,
                "priority": "high",
                "potential_impact": (
                    "Missing values may cause model-training errors "
                    "or reduce the usable training data."
                ),
                "recommended_action": (
                    "Inspect the missing-value pattern and compare "
                    "appropriate imputation or removal strategies."
                )
            })

        elif issue_type == "duplicate_rows":
            impacts.append({
                "issue": issue_type,
                "column": column,
                "priority": "medium",
                "potential_impact": (
                    "Duplicate records may overrepresent repeated "
                    "observations and distort evaluation if duplicates "
                    "cross training and test splits."
                ),
                "recommended_action": (
                    "Check whether duplicates are genuine repeated "
                    "events before deciding whether to remove them."
                )
            })

        elif issue_type == "empty_rows":
            impacts.append({
                "issue": issue_type,
                "column": column,
                "priority": "high",
                "potential_impact": (
                    "Completely empty rows provide no usable feature "
                    "information and may interfere with preprocessing."
                ),
                "recommended_action": (
                    "Inspect empty rows and consider removing them "
                    "if they have no meaningful purpose."
                )
            })

    # Check warnings
    for warning in quality_report.get("warnings", []):
        warning_type = warning.get("type")
        column = warning.get("column", "dataset")
        count = warning.get("count", 0)

        if warning_type == "potential_outliers":
            impacts.append({
                "issue": warning_type,
                "column": column,
                "priority": "medium",
                "potential_impact": (
                    "Extreme values may influence some models, "
                    "especially models sensitive to feature scale "
                    "or squared errors."
                ),
                "recommended_action": (
                    "Validate flagged values and compare model "
                    "performance with and without an appropriate "
                    "outlier-handling strategy."
                )
            })

        elif warning_type == "constant_column":
            impacts.append({
                "issue": warning_type,
                "column": column,
                "priority": "low",
                "potential_impact": (
                    "A constant feature generally provides no "
                    "predictive information to a standard ML model."
                ),
                "recommended_action": (
                    "Consider removing the feature during preprocessing "
                    "after confirming it is genuinely constant."
                )
            })

    # Summarize the assessment
    priority_order = {"high": 0, "medium": 1, "low": 2}
    impacts.sort(
        key=lambda item: priority_order.get(item["priority"], 3)
    )

    return {
        "total_impacts": len(impacts),
        "high_priority": sum(
            item["priority"] == "high" for item in impacts
        ),
        "medium_priority": sum(
            item["priority"] == "medium" for item in impacts
        ),
        "low_priority": sum(
            item["priority"] == "low" for item in impacts
        ),
        "assessments": impacts,
        "note": (
            "These are heuristic risk assessments. "
            "Actual model impact must be verified through experiments."
        )
    }
