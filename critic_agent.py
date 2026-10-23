
import math


# These are configurable starting points, not universal standards.
# Override them through acceptance_criteria when calling
# verify_experiments() for a specific dataset or business requirement.
DEFAULT_ACCEPTANCE_CRITERIA = {
    "classification": {
        "f1_macro": 0.70,
    },
    "regression": {
        "r2": 0.50,
    },
}

HIGHER_IS_BETTER = {
    "r2",
    "r^2",
    "accuracy",
    "precision",
    "precision_macro",
    "recall",
    "recall_macro",
    "f1",
    "f1_macro",
    "roc_auc",
    "auc",
}


def _is_finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _get_problem_type(experiment_report, pipeline_plan=None):
    """Find the problem type from the experiment report or plan."""
    pipeline_plan = pipeline_plan or {}

    problem_type = (
        experiment_report.get("problem_type")
        or pipeline_plan.get("problem_type")
    )

    if isinstance(problem_type, str):
        normalized = problem_type.strip().lower()

        if "class" in normalized:
            return "classification"

        if "regress" in normalized:
            return "regression"

    # Infer from commonly reported metrics if metadata is unavailable.
    all_metric_names = set()

    for result in experiment_report.get("results", []):
        if isinstance(result, dict) and isinstance(
            result.get("metrics"), dict
        ):
            all_metric_names.update(
                name.lower() for name in result["metrics"]
            )

    if {"accuracy", "f1_macro"} & all_metric_names:
        return "classification"

    if {"mae", "rmse", "r2"} & all_metric_names:
        return "regression"

    return "unknown"


def _metric_value(metrics, metric_name):
    """Match metric names without depending on letter case."""
    if not isinstance(metrics, dict):
        return None

    for name, value in metrics.items():
        if name.lower() == metric_name.lower():
            if _is_finite_number(value):
                return float(value)

    return None


def _select_best_candidate(results, problem_type):
    """
    Select a candidate using F1 macro for classification or R2 for
    regression. This is a ranking heuristic, not proof of generalization.
    """
    if problem_type == "classification":
        primary_metric = "f1_macro"
        higher_is_better = True
    elif problem_type == "regression":
        primary_metric = "r2"
        higher_is_better = True
    else:
        return None

    candidates = []

    for result in results:
        metrics = result.get("metrics", {})
        value = _metric_value(metrics, primary_metric)

        if value is None:
            continue

        candidates.append({
            "model": result.get("model", "Unknown"),
            "experiment": result.get("experiment", "unknown"),
            "primary_metric": primary_metric,
            "primary_metric_value": value,
            "metrics": metrics,
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate: candidate["primary_metric_value"],
        reverse=higher_is_better,
    )

    return candidates[0]


def verify_experiments(
    experiment_report: dict,
    quality_report: dict,
    impact_report: dict,
    pipeline_plan: dict = None,
    acceptance_criteria: dict = None,
) -> dict:
    """
    Verify experiment evidence, data-quality handling, and performance.

    Existing arguments remain supported. Optional arguments allow the API
    to pass the pipeline plan and dataset-specific acceptance criteria.

    Status meanings:
    - REJECTED: no usable experiment metrics or a critical evidence failure.
    - NEEDS_REVIEW: failed experiments, metric trade-offs, or unresolved
      evidence problems need investigation.
    - NEEDS_MORE_EXPERIMENTS: more evidence or quality-handling experiments
      are needed.
    - PERFORMANCE_NOT_MET: valid metrics exist, but configured performance
      criteria are not met.
    - PROVISIONALLY_ACCEPTED: available checks pass; this is not a guarantee
      of production readiness.
    """

    findings = []
    recommendations = []

    def add_finding(severity, message):
        findings.append({
            "severity": severity,
            "message": message,
        })

    if not isinstance(experiment_report, dict):
        experiment_report = {}

    if not isinstance(quality_report, dict):
        quality_report = {}

    if not isinstance(impact_report, dict):
        impact_report = {}

    if not isinstance(pipeline_plan, dict):
        pipeline_plan = {}

    results = experiment_report.get("results", [])

    if not isinstance(results, list):
        results = []
        add_finding(
            "high",
            "Experiment results must be provided as a list.",
        )

    # 1. Classify experiment outcomes.
    successful_results = []
    not_needed_results = []
    skipped_results = []
    failed_results = []

    for result in results:
        if not isinstance(result, dict):
            failed_results.append({"model": "Unknown"})
            add_finding(
                "medium",
                "An experiment result has an invalid structure.",
            )
            continue

        status = result.get("status")
        metrics = result.get("metrics")

        if status == "not_needed":
            not_needed_results.append(result)
            add_finding(
                "info",
                (
                    f"Experiment '{result.get('experiment', 'unknown')}' "
                    f"was not needed: {result.get('reason', 'Not applicable.')}"
                ),
            )
        elif status == "skipped":
            skipped_results.append(result)
            reason = result.get("reason")

            if not reason:
                add_finding(
                    "low",
                    "A skipped experiment did not provide a reason.",
                )
            else:
                add_finding(
                    "info",
                    (
                        f"Experiment '{result.get('experiment', 'unknown')}' "
                        f"was skipped: {reason}"
                    ),
                )
        elif (
            status == "success"
            and isinstance(metrics, dict)
            and len(metrics) > 0
        ):
            successful_results.append(result)
        else:
            failed_results.append(result)

    if not successful_results:
        add_finding(
            "high",
            "No experiment produced valid evaluation metrics.",
        )
        recommendations.append(
            "Resolve experiment errors before evaluating model quality."
        )

        return {
            "status": "REJECTED",
            "retry_recommended": True,
            "acceptance_passed": False,
            "problem_type": _get_problem_type(
                experiment_report, pipeline_plan
            ),
            "successful_models": 0,
            "failed_models": len(failed_results),
            "skipped_experiments": len(skipped_results),
            "not_needed_experiments": len(not_needed_results),
            "total_findings": len(findings),
            "findings": findings,
            "recommendations": recommendations,
            "experiment_comparison": [],
            "best_candidate": None,
            "performance_evaluation": {
                "passed": False,
                "criteria": {},
                "checks": [],
                "reason": "No valid experiment metrics were available.",
            },
            "note": (
                "Skipped and not-needed experiments are not counted "
                "as failures."
            ),
        }

    if failed_results:
        add_finding(
            "medium",
            (
                f"{len(failed_results)} experiment result(s) failed "
                "or contained invalid metrics."
            ),
        )
        recommendations.append(
            "Review failed experiment details before relying on "
            "the remaining results."
        )

    # 2. Validate finite metric values.
    valid_results = []
    invalid_metric_results = []

    for result in successful_results:
        model_name = result.get("model", "Unnamed model")
        metrics_valid = True

        for metric_name, value in result["metrics"].items():
            if not _is_finite_number(value):
                metrics_valid = False
                add_finding(
                    "medium",
                    (
                        f"{model_name}: metric '{metric_name}' "
                        "is not a finite numeric value."
                    ),
                )

        if metrics_valid:
            valid_results.append(result)
        else:
            invalid_metric_results.append(result)

    failed_count = len(failed_results) + len(invalid_metric_results)

    if not valid_results:
        add_finding(
            "high",
            "No experiment produced valid finite evaluation metrics.",
        )

        return {
            "status": "REJECTED",
            "retry_recommended": True,
            "acceptance_passed": False,
            "problem_type": _get_problem_type(
                experiment_report, pipeline_plan
            ),
            "successful_models": 0,
            "failed_models": failed_count,
            "skipped_experiments": len(skipped_results),
            "not_needed_experiments": len(not_needed_results),
            "total_findings": len(findings),
            "findings": findings,
            "recommendations": [
                "Fix invalid metrics and rerun the experiments."
            ],
            "experiment_comparison": [],
            "best_candidate": None,
            "performance_evaluation": {
                "passed": False,
                "criteria": {},
                "checks": [],
                "reason": "All available metrics failed validation.",
            },
            "note": "No valid finite model metrics were available.",
        }

    # 3. Check baseline coverage.
    completed_experiments = {
        result.get("experiment")
        for result in valid_results
        if result.get("experiment")
    }

    if "baseline" not in completed_experiments:
        add_finding(
            "high",
            "No successful baseline experiment was identified.",
        )
        recommendations.append(
            "Run a valid baseline experiment before comparing "
            "follow-up strategies."
        )

    grouped_results = {}

    for result in valid_results:
        experiment_name = result.get("experiment")
        model_name = result.get("model")

        if not experiment_name or not model_name:
            add_finding(
                "medium",
                "An experiment is missing its experiment name or model name.",
            )
            continue

        grouped_results.setdefault(experiment_name, {})
        grouped_results[experiment_name][model_name] = result

    # 4. Compare follow-up experiments with matching baselines.
    comparisons = []
    baseline_results = grouped_results.get("baseline", {})

    for experiment_name, model_results in grouped_results.items():
        if experiment_name == "baseline":
            continue

        for model_name, candidate in model_results.items():
            baseline = baseline_results.get(model_name)

            if baseline is None:
                add_finding(
                    "medium",
                    (
                        f"Experiment '{experiment_name}' has no matching "
                        f"successful baseline for model '{model_name}'."
                    ),
                )
                continue

            baseline_metrics = baseline["metrics"]
            candidate_metrics = candidate["metrics"]
            common_metrics = set(baseline_metrics) & set(candidate_metrics)

            if not common_metrics:
                add_finding(
                    "medium",
                    (
                        f"Experiment '{experiment_name}' and model "
                        f"'{model_name}' have no metrics in common "
                        "with the baseline."
                    ),
                )
                continue

            metric_comparisons = {}

            for metric_name in sorted(common_metrics):
                before = baseline_metrics[metric_name]
                after = candidate_metrics[metric_name]

                higher_is_better = metric_name.lower() in HIGHER_IS_BETTER
                difference = after - before

                if abs(difference) < 0.00005:
                    direction = "unchanged"
                elif (
                    difference > 0
                    if higher_is_better
                    else difference < 0
                ):
                    direction = "improved"
                else:
                    direction = "worsened"

                metric_comparisons[metric_name] = {
                    "baseline": before,
                    "candidate": after,
                    "difference": round(difference, 6),
                    "direction": direction,
                }

            comparisons.append({
                "experiment": experiment_name,
                "model": model_name,
                "metrics": metric_comparisons,
            })

    # 5. Check data-quality concerns.
    warnings = quality_report.get("warnings", [])
    issues = quality_report.get("issues", [])

    if not isinstance(warnings, list):
        warnings = []

    if not isinstance(issues, list):
        issues = []

    quality_experiment_map = {
        "potential_outliers": "outlier_handling",
        "missing_values": "missing_value_handling",
        "duplicate_rows": "duplicate_handling",
        "constant_column": "constant_feature_handling",
    }

    unresolved_concerns = []

    for concern in warnings + issues:
        if not isinstance(concern, dict):
            unresolved_concerns.append("invalid_quality_record")
            continue

        concern_type = concern.get("type", "unknown")
        expected_experiment = quality_experiment_map.get(concern_type)

        if (
            expected_experiment is None
            or expected_experiment not in completed_experiments
        ):
            unresolved_concerns.append(concern_type)

    unresolved_concerns = sorted(set(unresolved_concerns))

    if unresolved_concerns:
        add_finding(
            "medium",
            (
                "Some data-quality concerns have not been matched to "
                "successful handling experiments: "
                + ", ".join(unresolved_concerns)
            ),
        )
        recommendations.append(
            "Plan suitable experiments for unresolved data-quality "
            "concerns, using treatments appropriate to the dataset."
        )

    # 6. Validate outlier-handling evidence.
    outlier_info = experiment_report.get("outlier_capping")

    if "outlier_handling" in completed_experiments:
        if not isinstance(outlier_info, dict):
            add_finding(
                "medium",
                (
                    "An outlier-handling experiment was reported, "
                    "but its capping evidence is missing."
                ),
            )
        else:
            total_capped = outlier_info.get("total_values_capped")

            if (
                isinstance(total_capped, bool)
                or not isinstance(total_capped, int)
                or total_capped < 0
            ):
                add_finding(
                    "medium",
                    "Outlier-capping evidence contains an invalid total value count.",
                )
            elif total_capped == 0:
                add_finding(
                    "info",
                    "The outlier strategy did not cap any feature values.",
                )
            else:
                add_finding(
                    "info",
                    (
                        f"The outlier-handling experiment capped "
                        f"{total_capped} feature values."
                    ),
                )

    # 7. Identify metric trade-offs.
    directions = {
        details["direction"]
        for comparison in comparisons
        for details in comparison["metrics"].values()
    }

    if "improved" in directions and "worsened" in directions:
        add_finding(
            "medium",
            (
                "Some metrics improved while others worsened. "
                "Model selection requires reviewing these trade-offs."
            ),
        )
        recommendations.append(
            "Review metric priorities and validation conditions before "
            "choosing a preferred model."
        )
    elif "worsened" in directions:
        add_finding(
            "medium",
            (
                "At least one follow-up experiment worsened a metric "
                "relative to its baseline."
            ),
        )
        recommendations.append(
            "Inspect the metrics that worsened before adopting "
            "the follow-up strategy."
        )

    # 8. Evaluate explicit performance acceptance criteria.
    problem_type = _get_problem_type(experiment_report, pipeline_plan)

    configured_criteria = (
        acceptance_criteria
        if isinstance(acceptance_criteria, dict)
        else DEFAULT_ACCEPTANCE_CRITERIA
    )

    criteria = configured_criteria.get(problem_type, {})

    if not isinstance(criteria, dict):
        criteria = {}

    best_candidate = _select_best_candidate(valid_results, problem_type)

    performance_checks = []
    performance_passed = True

    if not criteria:
        performance_passed = False
        add_finding(
            "medium",
            (
                f"No performance acceptance criteria are configured "
                f"for problem type '{problem_type}'."
            ),
        )
        recommendations.append(
            "Configure acceptance criteria appropriate to the task "
            "before accepting a model."
        )
    else:
        for metric_name, threshold in criteria.items():
            if not _is_finite_number(threshold):
                performance_passed = False
                performance_checks.append({
                    "metric": metric_name,
                    "threshold": threshold,
                    "passed": False,
                    "reason": "Acceptance threshold is not a finite number.",
                })
                continue

            observed = (
                _metric_value(best_candidate["metrics"], metric_name)
                if best_candidate
                else None
            )

            if observed is None:
                performance_passed = False
                performance_checks.append({
                    "metric": metric_name,
                    "threshold": threshold,
                    "observed": None,
                    "passed": False,
                    "reason": (
                        "The selected candidate does not report this metric."
                    ),
                })
                continue

            higher_is_better = metric_name.lower() in HIGHER_IS_BETTER

            passed = (
                observed >= threshold
                if higher_is_better
                else observed <= threshold
            )

            performance_checks.append({
                "metric": metric_name,
                "threshold": threshold,
                "observed": observed,
                "comparison": ">=" if higher_is_better else "<=",
                "passed": passed,
            })

            if not passed:
                performance_passed = False

        if not performance_passed:
            add_finding(
                "medium",
                (
                    "The best available candidate did not meet all "
                    "configured performance acceptance criteria."
                ),
            )
            recommendations.append(
                "Try an alternative pipeline, preprocessing strategy, "
                "model configuration, or feature strategy, then reevaluate."
            )

    performance_evaluation = {
        "passed": performance_passed,
        "problem_type": problem_type,
        "criteria": criteria,
        "checks": performance_checks,
        "selection_metric": (
            "f1_macro"
            if problem_type == "classification"
            else "r2" if problem_type == "regression" else None
        ),
        "reason": (
            "All configured criteria passed."
            if performance_passed
            else "One or more criteria failed or could not be evaluated."
        ),
    }

    # 9. Determine verification status and whether another attempt is useful.
    high_severity_findings = [
        finding for finding in findings
        if finding["severity"] == "high"
    ]

    if high_severity_findings:
        status = "REJECTED"
    elif failed_count > 0:
        status = "NEEDS_REVIEW"
    elif unresolved_concerns:
        status = "NEEDS_MORE_EXPERIMENTS"
    elif not comparisons:
        status = "NEEDS_MORE_EXPERIMENTS"
        recommendations.append(
            "Run at least one valid follow-up experiment that can "
            "be compared with a successful baseline."
        )
    elif any(
        finding["severity"] == "medium"
        for finding in findings
    ):
        status = (
            "PERFORMANCE_NOT_MET"
            if not performance_passed
            else "NEEDS_REVIEW"
        )
    elif not performance_passed:
        status = "PERFORMANCE_NOT_MET"
    else:
        status = "PROVISIONALLY_ACCEPTED"

    # A retry recommendation is not itself a retry. The API must execute it.
    retry_recommended = status in {
        "REJECTED",
        "NEEDS_REVIEW",
        "NEEDS_MORE_EXPERIMENTS",
        "PERFORMANCE_NOT_MET",
    }

    if not recommendations:
        recommendations.append(
            "Review the model metrics and validation setup before "
            "selecting a pipeline for further testing."
        )

    return {
        "status": status,
        "retry_recommended": retry_recommended,
        "acceptance_passed": (
            status == "PROVISIONALLY_ACCEPTED"
            and performance_passed
        ),
        "problem_type": problem_type,
        "successful_models": len(valid_results),
        "failed_models": failed_count,
        "skipped_experiments": len(skipped_results),
        "not_needed_experiments": len(not_needed_results),
        "total_findings": len(findings),
        "findings": findings,
        "recommendations": recommendations,
        "experiment_comparison": comparisons,
        "best_candidate": best_candidate,
        "performance_evaluation": performance_evaluation,
        "note": (
            "This rule-based verification checks reported experiment "
            "evidence and configured thresholds. It does not prove "
            "generalization, fairness, absence of data leakage, or "
            "production readiness."
        ),
    }
