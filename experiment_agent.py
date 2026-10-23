
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SUPPORTED_EXPERIMENTS = {
    "baseline",
    "outlier_handling",
    "missing_value_handling",
    "duplicate_handling",
}


def build_preprocessor(X):
    """Build preprocessing for numerical and categorical features."""

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No supported feature columns were found.")

    return ColumnTransformer(transformers=transformers)


def get_models(problem_type, attempt_number=1):
    """
    Return available models.

    Later attempts use different hyperparameters for supported models.
    Model names remain stable so reports can compare candidates.
    """

    attempt_number = max(1, int(attempt_number))

    if problem_type == "classification":
        if attempt_number == 1:
            return [
                (
                    "Logistic Regression",
                    LogisticRegression(max_iter=2000),
                ),
                (
                    "Random Forest Classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                (
                    "Decision Tree Classifier",
                    DecisionTreeClassifier(random_state=42),
                ),
                (
                    "K-Nearest Neighbors Classifier",
                    KNeighborsClassifier(n_neighbors=5),
                ),
                ("Support Vector Machine", SVC()),
                (
                    "Gradient Boosting Classifier",
                    GradientBoostingClassifier(random_state=42),
                ),
                ("Gaussian Naive Bayes", GaussianNB()),
            ]

        if attempt_number == 2:
            return [
                (
                    "Logistic Regression",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
                (
                    "Random Forest Classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=2,
                        random_state=43,
                        n_jobs=-1,
                    ),
                ),
                (
                    "Decision Tree Classifier",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=43,
                    ),
                ),
                (
                    "K-Nearest Neighbors Classifier",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                    ),
                ),
                ("Support Vector Machine", SVC(class_weight="balanced")),
                (
                    "Gradient Boosting Classifier",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=43,
                    ),
                ),
                ("Gaussian Naive Bayes", GaussianNB()),
            ]

        return [
            (
                "Logistic Regression",
                LogisticRegression(
                    C=0.5,
                    max_iter=4000,
                    class_weight="balanced",
                ),
            ),
            (
                "Random Forest Classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    random_state=40 + attempt_number,
                    n_jobs=-1,
                ),
            ),
            (
                "Decision Tree Classifier",
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=40 + attempt_number,
                ),
            ),
            (
                "K-Nearest Neighbors Classifier",
                KNeighborsClassifier(
                    n_neighbors=9,
                    weights="distance",
                ),
            ),
            ("Support Vector Machine", SVC(C=2.0, class_weight="balanced")),
            (
                "Gradient Boosting Classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=2,
                    random_state=40 + attempt_number,
                ),
            ),
            ("Gaussian Naive Bayes", GaussianNB()),
        ]

    if problem_type == "regression":
        if attempt_number == 1:
            return [
                ("Linear Regression", LinearRegression()),
                (
                    "Random Forest Regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                (
                    "Decision Tree Regressor",
                    DecisionTreeRegressor(random_state=42),
                ),
                (
                    "K-Nearest Neighbors Regressor",
                    KNeighborsRegressor(n_neighbors=5),
                ),
                ("Support Vector Regressor", SVR()),
                (
                    "Gradient Boosting Regressor",
                    GradientBoostingRegressor(random_state=42),
                ),
            ]

        if attempt_number == 2:
            return [
                ("Linear Regression", LinearRegression()),
                (
                    "Random Forest Regressor",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=2,
                        random_state=43,
                        n_jobs=-1,
                    ),
                ),
                (
                    "Decision Tree Regressor",
                    DecisionTreeRegressor(
                        max_depth=8,
                        min_samples_leaf=2,
                        random_state=43,
                    ),
                ),
                (
                    "K-Nearest Neighbors Regressor",
                    KNeighborsRegressor(
                        n_neighbors=7,
                        weights="distance",
                    ),
                ),
                ("Support Vector Regressor", SVR(C=2.0, epsilon=0.1)),
                (
                    "Gradient Boosting Regressor",
                    GradientBoostingRegressor(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=43,
                    ),
                ),
            ]

        return [
            ("Linear Regression", LinearRegression()),
            (
                "Random Forest Regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_features=1.0,
                    min_samples_leaf=3,
                    random_state=40 + attempt_number,
                    n_jobs=-1,
                ),
            ),
            (
                "Decision Tree Regressor",
                DecisionTreeRegressor(
                    max_depth=6,
                    min_samples_leaf=4,
                    random_state=40 + attempt_number,
                ),
            ),
            (
                "K-Nearest Neighbors Regressor",
                KNeighborsRegressor(
                    n_neighbors=9,
                    weights="distance",
                ),
            ),
            ("Support Vector Regressor", SVR(C=5.0, epsilon=0.2)),
            (
                "Gradient Boosting Regressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=2,
                    random_state=40 + attempt_number,
                ),
            ),
        ]

    raise ValueError(
        "Problem type must be regression or classification."
    )


def calculate_metrics(y_true, predictions, problem_type):
    """Calculate metrics appropriate for the ML problem."""

    if problem_type == "regression":
        return {
            "MAE": round(
                float(mean_absolute_error(y_true, predictions)), 4
            ),
            "RMSE": round(
                float(np.sqrt(mean_squared_error(y_true, predictions))), 4
            ),
            "R2": round(float(r2_score(y_true, predictions)), 4),
        }

    return {
        "Accuracy": round(
            float(accuracy_score(y_true, predictions)), 4
        ),
        "Precision_macro": round(
            float(
                precision_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "Recall_macro": round(
            float(
                recall_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "F1_macro": round(
            float(
                f1_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
    }


def get_cv_scoring(problem_type):
    """Choose cross-validation metrics."""

    if problem_type == "classification":
        return {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
        }

    return {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }


def evaluate_cross_validation(X_train, y_train, model, problem_type):
    """Evaluate a model using training data only."""

    try:
        if problem_type == "classification":
            class_counts = y_train.value_counts()
            folds = min(5, int(class_counts.min()))

            if folds < 2:
                return {
                    "status": "not_available",
                    "reason": (
                        "At least two training examples per class "
                        "are needed for stratified cross-validation."
                    ),
                }

            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=42,
            )
        else:
            folds = min(5, len(X_train))

            if folds < 2:
                return {
                    "status": "not_available",
                    "reason": "At least two training rows are required.",
                }

            splitter = KFold(
                n_splits=folds,
                shuffle=True,
                random_state=42,
            )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", clone(model)),
            ]
        )

        scoring = get_cv_scoring(problem_type)

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        metrics = {}

        for metric_name in scoring:
            scores = cv_results[f"test_{metric_name}"].astype(float)

            if metric_name in {"mae", "rmse"}:
                scores = -scores

            metrics[metric_name] = {
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "fold_scores": [
                    round(float(score), 4) for score in scores
                ],
            }

        return {
            "status": "success",
            "strategy": type(splitter).__name__,
            "folds": folds,
            "training_split_only": True,
            "metrics": metrics,
        }

    except Exception as error:
        return {
            "status": "failed",
            "reason": str(error),
        }


def cap_outliers_using_training_data(X_train, X_test):
    """Learn IQR bounds on training data and apply them to both splits."""

    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()
    bounds = {}
    capped_counts = {}

    for column in X_train.select_dtypes(include=["number"]).columns:
        values = X_train[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr <= 0:
            continue

        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)

        train_values = X_train[column]
        test_values = X_test[column]

        train_mask = (
            train_values.notna()
            & ((train_values < lower) | (train_values > upper))
        )
        test_mask = (
            test_values.notna()
            & ((test_values < lower) | (test_values > upper))
        )

        train_count = int(train_mask.sum())
        test_count = int(test_mask.sum())

        X_train_capped[column] = train_values.clip(lower, upper)
        X_test_capped[column] = test_values.clip(lower, upper)

        bounds[column] = {"lower": lower, "upper": upper}
        capped_counts[column] = {
            "training_values_capped": train_count,
            "testing_values_capped": test_count,
            "total_values_capped": train_count + test_count,
        }

    total_capped = sum(
        item["total_values_capped"] for item in capped_counts.values()
    )

    return (
        X_train_capped,
        X_test_capped,
        bounds,
        capped_counts,
        total_capped,
    )


def compare_metrics(baseline_metrics, candidate_metrics, problem_type):
    """Compare candidate metrics against baseline metrics."""

    comparisons = {}

    higher_is_better = {
        "Accuracy",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "R2",
    }

    for name in set(baseline_metrics) & set(candidate_metrics):
        baseline = baseline_metrics[name]
        candidate = candidate_metrics[name]

        delta = (
            candidate - baseline
            if name in higher_is_better
            else baseline - candidate
        )

        if abs(delta) < 0.00005:
            direction = "unchanged"
        elif delta > 0:
            direction = "improved"
        else:
            direction = "worsened"

        comparisons[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "change": round(candidate - baseline, 6),
            "direction": direction,
        }

    return comparisons


def run_experiments(
    df,
    target_column,
    problem_type,
    pipeline_plan=None,
    attempt_number=1,
):
    """
    Run model experiments and data-quality handling comparisons.

    The first three arguments remain compatible with the original API.
    Optional arguments allow a planner to choose candidate models and
    run a different model configuration on later attempts.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if problem_type not in {"regression", "classification"}:
        raise ValueError(
            "Problem type must be regression or classification."
        )

    if isinstance(attempt_number, bool) or not isinstance(
        attempt_number, int
    ) or attempt_number < 1:
        raise ValueError("attempt_number must be a positive integer.")

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    if df.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

    if df[target_column].isna().any():
        raise ValueError(
            "The target column contains missing values. "
            "Handle or remove those rows before training."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")

    unsupported = X.select_dtypes(
        exclude=["number", "object", "category", "bool"]
    )

    if unsupported.shape[1]:
        raise ValueError(
            f"Unsupported feature types: {unsupported.columns.tolist()}."
        )

    if problem_type == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numeric target.")

        if y.nunique() < 2:
            raise ValueError(
                "Regression requires at least two distinct target values."
            )
    elif y.nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    # Create model configurations for this attempt.
    models = get_models(problem_type, attempt_number=attempt_number)

    # If the planner selected candidate model names, honor those choices.
    if isinstance(pipeline_plan, dict):
        planned_models = pipeline_plan.get("candidate_models")

        if isinstance(planned_models, list) and planned_models:
            selected_models = [
                (name, model)
                for name, model in models
                if name in planned_models
            ]

            if selected_models:
                models = selected_models

    if not models:
        raise ValueError(
            "The pipeline plan did not select any supported models."
        )

    stratify_values = None

    if problem_type == "classification":
        class_counts = y.value_counts()

        if class_counts.min() >= 2:
            stratify_values = y

    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=stratify_values,
    )

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    results = []

    def run_model_batch(
        experiment_name,
        X_fit,
        y_fit,
        X_eval,
        y_eval,
        attach_cv=False,
    ):
        batch_results = []

        for model_name, model in models:
            try:
                if len(X_fit) == 0:
                    raise ValueError(
                        "No training rows remain for this experiment."
                    )

                if (
                    problem_type == "classification"
                    and y_fit.nunique() < 2
                ):
                    raise ValueError(
                        "Training data contains fewer than two classes."
                    )

                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", build_preprocessor(X_fit)),
                        ("model", clone(model)),
                    ]
                )

                pipeline.fit(X_fit, y_fit)
                predictions = pipeline.predict(X_eval)

                result = {
                    "experiment": experiment_name,
                    "model": model_name,
                    "status": "success",
                    "metrics": calculate_metrics(
                        y_eval,
                        predictions,
                        problem_type,
                    ),
                    "error": None,
                }

                if attach_cv:
                    result["cross_validation"] = evaluate_cross_validation(
                        X_train,
                        y_train,
                        model,
                        problem_type,
                    )

            except Exception as error:
                result = {
                    "experiment": experiment_name,
                    "model": model_name,
                    "status": "failed",
                    "metrics": {},
                    "error": str(error),
                }

            batch_results.append(result)
            results.append(result)

        return batch_results

    # Experiment 1: baseline models and cross-validation.
    baseline_results = run_model_batch(
        "baseline",
        X_train,
        y_train,
        X_test,
        y_test,
        attach_cv=True,
    )

    # Experiment 2: outlier handling.
    (
        X_train_capped,
        X_test_capped,
        outlier_bounds,
        capped_counts,
        total_capped,
    ) = cap_outliers_using_training_data(X_train, X_test)

    outlier_results = run_model_batch(
        "outlier_handling",
        X_train_capped,
        y_train,
        X_test_capped,
        y_test,
    )

    baseline_by_model = {
        item["model"]: item
        for item in baseline_results
        if item["status"] == "success"
    }

    for item in outlier_results:
        baseline = baseline_by_model.get(item["model"])

        if item["status"] == "success" and baseline:
            item["comparison"] = compare_metrics(
                baseline["metrics"],
                item["metrics"],
                problem_type,
            )

    # Experiment 3: missing-value handling.
    missing_columns = [
        str(column)
        for column, count in X_train.isna().sum().items()
        if count > 0
    ]

    if missing_columns:
        missing_results = run_model_batch(
            "missing_value_handling",
            X_train,
            y_train,
            X_test,
            y_test,
        )

        for item in missing_results:
            item["strategy"] = (
                "Median imputation for numeric features; "
                "most-frequent imputation for categorical features"
            )
            item["affected_columns"] = missing_columns
    else:
        results.append({
            "experiment": "missing_value_handling",
            "model": "Not applicable",
            "status": "not_needed",
            "metrics": {},
            "error": None,
            "reason": (
                "No missing feature values were detected in training data."
            ),
        })

    # Experiment 4: duplicate handling on training features only.
    duplicate_mask = X_train.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    X_train_deduplicated = X_train.loc[~duplicate_mask].copy()
    y_train_deduplicated = y_train.loc[~duplicate_mask].copy()

    if duplicate_count:
        duplicate_results = run_model_batch(
            "duplicate_handling",
            X_train_deduplicated,
            y_train_deduplicated,
            X_test,
            y_test,
        )

        for item in duplicate_results:
            item["strategy"] = (
                "Remove repeated feature rows from training data only"
            )
            item["training_rows_removed"] = duplicate_count
    else:
        results.append({
            "experiment": "duplicate_handling",
            "model": "Not applicable",
            "status": "not_needed",
            "metrics": {},
            "error": None,
            "reason": (
                "No duplicate feature rows were detected in training data."
            ),
            "training_rows_removed": 0,
        })

    # Compare every successful follow-up with the matching baseline.
    for item in results:
        if item["experiment"] == "baseline":
            continue

        if item["status"] != "success":
            continue

        baseline = baseline_by_model.get(item["model"])

        if baseline:
            item["comparison"] = compare_metrics(
                baseline["metrics"],
                item["metrics"],
                problem_type,
            )

    successful_baselines = [
        item for item in baseline_results
        if item["status"] == "success"
    ]

    return {
        "attempt_number": attempt_number,
        "pipeline_plan": pipeline_plan or {},
        "training_rows": len(train_indices),
        "testing_rows": len(test_indices),
        "problem_type": problem_type,
        "target_column": target_column,
        "available_models": [name for name, _ in models],
        "model_count": len(models),
        "results": results,
        "outlier_bounds": outlier_bounds,
        "outlier_capping": {
            "strategy": "IQR capping",
            "features_processed": list(outlier_bounds.keys()),
            "per_feature_counts": capped_counts,
            "total_values_capped": total_capped,
            "data_changed": total_capped > 0,
        },
        "missing_value_handling": {
            "strategy": (
                "Median imputation for numeric features; "
                "most-frequent imputation for categorical features"
            ),
            "affected_columns": missing_columns,
            "missing_values_in_training_features": int(
                X_train.isna().sum().sum()
            ),
        },
        "duplicate_handling": {
            "strategy": (
                "Remove repeated feature rows from training data only"
            ),
            "training_feature_duplicates_removed": duplicate_count,
            "training_rows_after_handling": len(X_train_deduplicated),
            "test_rows_modified": False,
        },
        "experiment_summary": {
            "attempt_number": attempt_number,
            "models_available": len(models),
            "baseline_models_attempted": len(models),
            "baseline_models_succeeded": len(successful_baselines),
            "outlier_models_attempted": len(models),
            "missing_value_models_attempted": (
                len(models) if missing_columns else 0
            ),
            "duplicate_models_attempted": (
                len(models) if duplicate_count else 0
            ),
            "same_initial_train_test_split": True,
            "supported_experiments": sorted(SUPPORTED_EXPERIMENTS),
        },
    }
