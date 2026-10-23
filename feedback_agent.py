
def decide_next_action(
    verification_report,
    quality_report,
    experiment_report,
):
    """Choose the next workflow action using experiment and critic evidence."""

    if not isinstance(verification_report, dict):
        verification_report = {}

    if not isinstance(quality_report, dict):
        quality_report = {}

    if not isinstance(experiment_report, dict):
        experiment_report = {}

    results = experiment_report.get("results", [])

    if not isinstance(results, list):
        results = []

    successful_results = [
        result
        for result in results
        if isinstance(result, dict)
        and result.get("status") == "success"
        and isinstance(result.get("metrics"), dict)
        and bool(result.get("metrics"))
    ]

    not_needed_results = [
        result
        for result in results
        if isinstance(result, dict)
        and result.get("status") == "not_needed"
    ]

    skipped_results = [
        result
        for result in results
        if isinstance(result, dict)
        and result.get("status") == "skipped"
    ]

    failed_results = [
        result
        for result in results
        if not isinstance(result, dict)
        or result.get("status") not in {
            "success",
            "skipped",
            "not_needed",
        }
        or (
            result.get("status") == "success"
            and not (
                isinstance(result.get("metrics"), dict)
                and bool(result.get("metrics"))
            )
        )
    ]

    completed_experiments = {
        result.get("experiment")
        for result in successful_results
        if result.get("experiment")
    }

    not_needed_experiments = {
        result.get("experiment")
        for result in not_needed_results
        if result.get("experiment")
    }

    skipped_experiments = {
        result.get("experiment")
        for result in skipped_results
        if result.get("experiment")
    }

    failed_experiments = [
        {
            "experiment": (
                result.get("experiment", "unknown")
                if isinstance(result, dict)
                else "unknown"
            ),
            "model": (
                result.get("model", "unknown")
                if isinstance(result, dict)
                else "unknown"
            ),
            "status": (
                result.get("status", "unknown")
                if isinstance(result, dict)
                else "invalid_result"
            ),
            "error": (
                result.get("error")
                if isinstance(result, dict)
                else "Experiment result has an invalid structure."
            ),
        }
        for result in failed_results
    ]

    verification_status = verification_report.get(
        "status",
        "NEEDS_REVIEW",
    )

    recommendations = verification_report.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []

    findings = verification_report.get("findings", [])
    if not isinstance(findings, list):
        findings = []

    def response(
        action,
        reason,
        requires_experiment=False,
        **extra,
    ):
        return {
            "action": action,
            "reason": reason,
            "requires_experiment": requires_experiment,
            "successful_experiments": sorted(completed_experiments),
            "not_needed_experiments": sorted(not_needed_experiments),
            "skipped_experiments": sorted(skipped_experiments),
            "failed_experiments": failed_experiments,
            "verification_status": verification_status,
            **extra,
        }

    # 1. Genuine experiment failures should be investigated first.
    if failed_results:
        return response(
            "INVESTIGATE_FAILED_EXPERIMENTS",
            (
                f"{len(failed_results)} experiment result(s) failed "
                "or did not report valid successful metrics."
            ),
            failed_experiments=failed_experiments,
        )

    # 2. No successful results means initial experiments are required.
    if not successful_results:
        return response(
            "RUN_INITIAL_EXPERIMENTS",
            "No successful experiment results are available.",
            requires_experiment=True,
        )

    # 3. Require a successful baseline before making comparisons.
    has_baseline = any(
        result.get("experiment") == "baseline"
        for result in successful_results
    )

    if not has_baseline:
        return response(
            "RUN_INITIAL_EXPERIMENTS",
            "A successful baseline experiment is missing.",
            requires_experiment=True,
        )

    # 4. A rejected report needs investigation, not blind acceptance.
    if verification_status == "REJECTED":
        return response(
            "INVESTIGATE_REJECTED_PIPELINE",
            (
                "The Critic rejected the current evidence. Inspect the "
                "rejection findings before deciding whether to retry."
            ),
            requires_experiment=False,
            critic_findings=findings,
            critic_recommendations=recommendations,
        )

    # 5. Map quality findings to relevant experiment strategies.
    issue_to_experiment = {
        "missing_values": "missing_value_handling",
        "potential_outliers": "outlier_handling",
        "duplicate_rows": "duplicate_handling",
        "empty_rows": "empty_row_handling",
        "constant_column": "constant_feature_handling",
    }

    quality_concerns = []

    for finding in (
        quality_report.get("issues", [])
        + quality_report.get("warnings", [])
    ):
        if not isinstance(finding, dict):
            continue

        issue_type = finding.get("type")
        experiment_name = issue_to_experiment.get(issue_type)

        if experiment_name and experiment_name not in quality_concerns:
            quality_concerns.append(experiment_name)

    pending_experiments = [
        name
        for name in quality_concerns
        if name not in completed_experiments
        and name not in not_needed_experiments
        and name not in skipped_experiments
    ]

    if pending_experiments:
        return response(
            "PLAN_QUALITY_EXPERIMENTS",
            "Detected quality concerns need additional experiment evidence.",
            requires_experiment=True,
            pending_experiments=pending_experiments,
            critic_recommendations=recommendations,
        )

    # 6. Performance criteria failed: plan a different pipeline.
    if verification_status == "PERFORMANCE_NOT_MET":
        performance_evaluation = verification_report.get(
            "performance_evaluation",
            {},
        )

        return response(
            "RETRY_ALTERNATIVE_PIPELINE",
            (
                "The best available candidate did not meet the configured "
                "performance acceptance criteria. Try a different model, "
                "preprocessing strategy, feature strategy, or configuration."
            ),
            requires_experiment=True,
            retry_reason="performance_criteria_not_met",
            performance_evaluation=performance_evaluation,
            best_candidate=verification_report.get("best_candidate"),
            critic_findings=findings,
            critic_recommendations=recommendations,
        )

    # 7. The critic found unresolved issues that need review.
    if verification_status in {
        "NEEDS_REVIEW",
        "NEEDS_MORE_EXPERIMENTS",
    }:
        return response(
            "RETRY_OR_REVIEW_PIPELINE",
            (
                "The Critic requires further evidence. Use its findings "
                "to plan the next experiment or investigate the issue."
            ),
            requires_experiment=True,
            retry_reason="verification_evidence_incomplete",
            critic_findings=findings,
            critic_recommendations=recommendations,
        )

    # 8. Accept only a provisionally accepted report that explicitly
    # confirms performance criteria passed.
    if (
        verification_status == "PROVISIONALLY_ACCEPTED"
        and verification_report.get("acceptance_passed") is True
    ):
        return response(
            "READY_FOR_FINAL_REVIEW",
            (
                "The available evidence passed the configured checks. "
                "The pipeline is provisionally accepted for final review, "
                "not automatically certified for production."
            ),
            requires_experiment=False,
            best_candidate=verification_report.get("best_candidate"),
            performance_evaluation=verification_report.get(
                "performance_evaluation",
                {},
            ),
        )

    # 9. Unknown or inconsistent states must not be accepted silently.
    return response(
        "RETRY_OR_REVIEW_PIPELINE",
        (
            "The current verification status does not establish that "
            "the pipeline satisfies acceptance requirements."
        ),
        requires_experiment=True,
        retry_reason="acceptance_not_established",
        critic_findings=findings,
        critic_recommendations=recommendations,
    )
