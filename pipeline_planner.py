
import pandas as pd


CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Random Forest Classifier",
    "Decision Tree Classifier",
    "K-Nearest Neighbors Classifier",
    "Support Vector Machine",
    "Gradient Boosting Classifier",
    "Gaussian Naive Bayes",
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Random Forest Regressor",
    "Decision Tree Regressor",
    "K-Nearest Neighbors Regressor",
    "Support Vector Regressor",
    "Gradient Boosting Regressor",
]


def detect_problem_type(target: pd.Series) -> dict:
    """Infer whether a target is for regression or classification."""

    target = target.dropna()

    if target.empty:
        raise ValueError("The target column contains no usable values.")

    unique_count = target.nunique()
    total_count = len(target)

    if unique_count < 2:
        raise ValueError(
            "The target must contain at least two distinct values."
        )

    if (
        pd.api.types.is_bool_dtype(target)
        or pd.api.types.is_object_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(target)
    ):
        return {
            "problem_type": "classification",
            "confidence": "high",
            "reason": "The target contains categorical or text values.",
        }

    if pd.api.types.is_numeric_dtype(target):
        unique_ratio = unique_count / total_count

        if unique_count <= 10 and unique_ratio <= 0.05:
            return {
                "problem_type": "classification",
                "confidence": "medium",
                "reason": (
                    "The numeric target has few distinct values. "
                    "Confirm that these values represent categories."
                ),
            }

        return {
            "problem_type": "regression",
            "confidence": "medium",
            "reason": (
                "The numeric target has multiple distinct values. "
                "Confirm that predicting a numerical quantity is intended."
            ),
        }

    return {
        "problem_type": "unknown",
        "confidence": "low",
        "reason": "The target type needs further inspection.",
    }


def _collect_previous_results(previous_experiment_reports):
    """Collect previously attempted model and experiment names."""

    tried_models = set()
    tried_experiments = set()

    if not previous_experiment_reports:
        return tried_models, tried_experiments

    if isinstance(previous_experiment_reports, dict):
        reports = [previous_experiment_reports]
    elif isinstance(previous_experiment_reports, list):
        reports = previous_experiment_reports
    else:
        reports = []

    for report in reports:
        if not isinstance(report, dict):
            continue

        for result in report.get("results", []):
            if not isinstance(result, dict):
                continue

            model = result.get("model")
            experiment = result.get("experiment")

            if model and model != "Not applicable":
                tried_models.add(model)

            if experiment:
                tried_experiments.add(experiment)

    return tried_models, tried_experiments


def plan_pipeline(
    df: pd.DataFrame,
    target_column: str,
    quality_report: dict,
    impact_report: dict,
    previous_experiment_reports=None,
    previous_verification_report=None,
    attempt_number: int = 1,
) -> dict:
    """
    Create an initial pipeline or propose an alternative based on
    previous experiment evidence.

    Existing callers remain compatible because the additional arguments
    are optional.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if len(df.columns) < 2:
        raise ValueError(
            "The dataset needs a target and at least one feature."
        )

    if attempt_number < 1:
        raise ValueError("attempt_number must be at least 1.")

    quality_report = quality_report or {}
    impact_report = impact_report or {}

    target_info = detect_problem_type(df[target_column])
    problem_type = target_info["problem_type"]

    if problem_type == "unknown":
        raise ValueError(
            "Could not infer the problem type reliably. "
            "Please inspect the target values."
        )

    all_models = (
        CLASSIFICATION_MODELS
        if problem_type == "classification"
        else REGRESSION_MODELS
    )

    tried_models, tried_experiments = _collect_previous_results(
        previous_experiment_reports
    )

    is_alternative = attempt_number > 1 or bool(tried_models)

    # Keep the original small baseline model set for the initial plan.
    if not is_alternative:
        candidate_models = all_models[:2]
        planning_reason = (
            "Initial pipeline: establish baseline performance using "
            "two representative candidate models."
        )
    else:
        # Prefer models that have not been attempted previously.
        candidate_models = [
            model for model in all_models
            if model not in tried_models
        ]

        if candidate_models:
            planning_reason = (
                "Alternative pipeline: prioritize models that have not "
                "appeared in the previous experiment results."
            )
        else:
            # If every model has been tried, a different preprocessing
            # strategy can still be explored by a capable experiment runner.
            candidate_models = all_models
            planning_reason = (
                "All available model families have already been attempted. "
                "Try a different preprocessing or configuration strategy "
                "instead of assuming a new model family is available."
            )

    experiments = [
        {
            "name": "baseline",
            "purpose": "Establish reference model performance.",
        },
        {
            "name": "preprocessing",
            "purpose": (
                "Evaluate suitable missing-value handling, encoding, "
                "and scaling where appropriate."
            ),
        },
    ]

    warning_types = {
        warning.get("type")
        for warning in quality_report.get("warnings", [])
        if isinstance(warning, dict)
    }
    issue_types = {
        issue.get("type")
        for issue in quality_report.get("issues", [])
        if isinstance(issue, dict)
    }

    if "potential_outliers" in warning_types:
        experiments.append({
            "name": "outlier_handling",
            "purpose": (
                "Compare model performance with and without a "
                "validated outlier-handling strategy."
            ),
        })

    if "missing_values" in issue_types:
        experiments.append({
            "name": "missing_value_handling",
            "purpose": "Compare appropriate imputation strategies.",
        })

    if "duplicate_rows" in issue_types:
        experiments.append({
            "name": "duplicate_handling",
            "purpose": (
                "Evaluate removing duplicate feature rows from "
                "training data where appropriate."
            ),
        })

    # Later attempts should consider a distinct strategy. These are
    # planning instructions; the experiment runner must implement them.
    if is_alternative:
        alternative_strategies = [
            {
                "name": "alternative_preprocessing",
                "purpose": (
                    "Try a different preprocessing configuration, "
                    "such as comparing scaling and imputation choices."
                ),
            },
            {
                "name": "alternative_model_configuration",
                "purpose": (
                    "Try different supported model configurations and "
                    "compare their evaluation evidence."
                ),
            },
            {
                "name": "robustness_validation",
                "purpose": (
                    "Assess whether candidate performance is consistent "
                    "across validation folds."
                ),
            },
        ]

        strategy_index = (attempt_number - 2) % len(alternative_strategies)
        selected_strategy = alternative_strategies[strategy_index]

        experiments.append(selected_strategy)

    if problem_type == "regression":
        metrics = ["MAE", "RMSE", "R2"]
    else:
        metrics = [
            "Accuracy",
            "Precision_macro",
            "Recall_macro",
            "F1_macro",
        ]

    verification_status = None
    if isinstance(previous_verification_report, dict):
        verification_status = previous_verification_report.get("status")

    return {
        "problem_type": problem_type,
        "detection_confidence": target_info["confidence"],
        "detection_reason": target_info["reason"],
        "target_column": target_column,
        "feature_columns": [
            column for column in df.columns
            if column != target_column
        ],
        "candidate_models": candidate_models,
        "all_available_models": all_models,
        "evaluation_metrics": metrics,
        "experiments": experiments,
        "quality_issues_considered": quality_report.get(
            "total_issues", 0
        ),
        "impact_assessments_considered": impact_report.get(
            "total_impacts", 0
        ),
        "planning": {
            "attempt_number": attempt_number,
            "is_alternative_pipeline": is_alternative,
            "reason": planning_reason,
            "previously_tried_models": sorted(tried_models),
            "previously_tried_experiments": sorted(tried_experiments),
            "previous_verification_status": verification_status,
            "requires_experiment_runner_support": is_alternative,
        },
    }
