
import pandas as pd
from pathlib import Path

from dataset_profiler import profile_dataset
from quality_analyst import analyze_quality
from impact_analyst import analyze_impact
from pipeline_planner import plan_pipeline
from experiment_agent import run_experiments
from critic_agent import verify_experiments
from feedback_agent import decide_next_action


MAX_WORKFLOW_ITERATIONS = 3

SUPPORTED_EXPERIMENTS = {
    "baseline",
    "outlier_handling",
}


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_cross_validation_report(report):
    """Display cross-validation results returned by experiment_agent.py."""

    cv_report = report.get("cross_validation", {})

    if not cv_report:
        print("\nCross-validation results are not available in this report.")
        return

    print_section("CROSS-VALIDATION RESULTS")

    print("Validation strategy:", cv_report.get("strategy", "unknown"))
    print("Training split only:", cv_report.get("training_split_only", "unknown"))

    results = cv_report.get("results", [])

    if not results:
        print("No cross-validation results were returned.")
        return

    for result in results:
        print("\nModel:", result.get("model", "unknown"))
        print("Status:", result.get("status", "unknown"))

        if result.get("status") != "success":
            print(
                "Reason:",
                result.get("reason", "Cross-validation could not be completed."),
            )
            continue

        print("Number of folds:", result.get("folds", "unknown"))

        for metric_name, details in result.get("metrics", {}).items():
            print(f"\n  {metric_name}")
            print("    Mean:", details.get("mean"))
            print("    Standard deviation:", details.get("std"))
            print("    Individual fold scores:", details.get("fold_scores"))


def print_experiment_report(report):
    print("\nTraining rows:", report.get("training_rows"))
    print("Testing rows:", report.get("testing_rows"))
    print("Problem type:", report.get("problem_type"))

    for result in report.get("results", []):
        status = result.get("status", "unknown")

        print("\nExperiment:", result.get("experiment", "unknown"))
        print("Model:", result.get("model", "unknown"))
        print("Status:", status)

        if status == "success":
            for metric, value in result.get("metrics", {}).items():
                print(f"  {metric}: {value}")

            for metric, details in result.get("comparison", {}).items():
                print(
                    f"  {metric} comparison: "
                    f"{details.get('direction', 'unknown')} "
                    f"(baseline={details.get('baseline')}, "
                    f"candidate={details.get('candidate')})"
                )

        elif status == "not_needed":
            reason = result.get("reason") or result.get("message")
            print("  Reason:", reason or "This experiment was not required.")

        elif status == "skipped":
            reason = result.get("reason") or result.get("message")
            print("  Reason:", reason or "This experiment was skipped.")

        else:
            error = result.get("error")

            if status == "failed" or error:
                print("  Error:", error or "Unknown error")
            else:
                reason = result.get("reason") or result.get("message")
                print("  Details:", reason or "No additional details available.")

    outlier_info = report.get("outlier_capping", {})

    if outlier_info:
        print("\n--- OUTLIER-HANDLING EVIDENCE ---")
        print("Strategy:", outlier_info.get("strategy"))
        print("Total values capped:", outlier_info.get("total_values_capped"))
        print("Feature data changed:", outlier_info.get("data_changed"))

    # Display cross-validation results after the regular experiments.
    print_cross_validation_report(report)


def print_verification_report(report):
    print("\nVerification status:", report.get("status"))
    print("Successful models:", report.get("successful_models"))
    print("Failed models:", report.get("failed_models"))
    print("Total findings:", report.get("total_findings"))

    print("\n--- Verification Findings ---")

    for finding in report.get("findings", []):
        severity = finding.get("severity", "info").upper()
        print(f"[{severity}] {finding.get('message', '')}")

    print("\n--- Recommendations ---")

    for recommendation in report.get("recommendations", []):
        print("-", recommendation)

    for comparison in report.get("experiment_comparison", []):
        print(
            "\nComparison:",
            comparison.get("experiment"),
            "| Model:",
            comparison.get("model"),
        )

        for metric, details in comparison.get("metrics", {}).items():
            print(
                f"  {metric}: {details.get('direction', 'unknown')} "
                f"(baseline={details.get('baseline')}, "
                f"candidate={details.get('candidate')})"
            )

    print("\nNote:", report.get("note", ""))


def run_initial_analysis(df, target_column):
    """Run profiling, quality analysis, impact analysis, and planning."""

    print_section("AGENT 1: DATASET PROFILING")
    profile = profile_dataset(df)

    for key, value in profile.items():
        print(f"{key}: {value}")

    print_section("AGENT 2: QUALITY ANALYSIS")
    quality = analyze_quality(df)

    print("Total issues:", quality.get("total_issues"))
    print("Total warnings:", quality.get("total_warnings"))

    for issue in quality.get("issues", []):
        print("ISSUE:", issue)

    for warning in quality.get("warnings", []):
        print("WARNING:", warning)

    print_section("AGENT 3: IMPACT ANALYSIS")
    impact = analyze_impact(df, quality)

    print("Total assessments:", impact.get("total_impacts"))

    for assessment in impact.get("assessments", []):
        print("\nFinding:", assessment.get("issue"))
        print("Column:", assessment.get("column"))
        print("Priority:", assessment.get("priority"))
        print("Potential impact:", assessment.get("potential_impact"))
        print("Recommended action:", assessment.get("recommended_action"))

    print_section("AGENT 4: PIPELINE PLANNER")

    plan = plan_pipeline(
        df=df,
        target_column=target_column,
        quality_report=quality,
        impact_report=impact,
    )

    print("Detected problem type:", plan.get("problem_type"))
    print("Detection confidence:", plan.get("detection_confidence"))
    print("Detection reason:", plan.get("detection_reason"))
    print("Candidate models:", plan.get("candidate_models"))
    print("Evaluation metrics:", plan.get("evaluation_metrics"))

    print("\nPlanned experiments:")

    for experiment in plan.get("experiments", []):
        print("-", experiment.get("name"), ":", experiment.get("purpose"))

    return quality, impact, plan


def print_capability_check(feedback):
    """Show which requested experiments the runner can execute."""

    pending = feedback.get("pending_experiments", [])

    if not isinstance(pending, list) or not pending:
        return [], []

    supported = [
        name for name in pending
        if name in SUPPORTED_EXPERIMENTS
    ]

    unsupported = [
        name for name in pending
        if name not in SUPPORTED_EXPERIMENTS
    ]

    print("\n--- EXPERIMENT CAPABILITY CHECK ---")
    print("Supported requested experiments:", supported or "None")
    print("Unsupported requested experiments:", unsupported or "None")

    return supported, unsupported


def run_adaptive_workflow(df, plan, quality, impact):
    """
    Orchestrate the implemented experiments, verification, and feedback.

    The current experiment runner executes baseline and outlier handling
    together. Other experiment strategies are not executed unless they
    are implemented by the experiment runner.
    """

    print_section("ADAPTIVE WORKFLOW ORCHESTRATOR")

    history = []
    experiment_report = None
    verification = None
    feedback = None

    for iteration in range(1, MAX_WORKFLOW_ITERATIONS + 1):
        print(f"\nWorkflow iteration {iteration}/{MAX_WORKFLOW_ITERATIONS}")

        if experiment_report is not None:
            print(
                "The implemented experiment batch has already run. "
                "No separate follow-up execution interface is available yet."
            )
            break

        print_section("AGENT 5: EXPERIMENT RUNNER")

        experiment_report = run_experiments(
            df=df,
            target_column=plan["target_column"],
            problem_type=plan["problem_type"],
        )

        print_experiment_report(experiment_report)

        print_section("AGENT 6: CRITIC / VERIFICATION")

        verification = verify_experiments(
            experiment_report=experiment_report,
            quality_report=quality,
            impact_report=impact,
        )

        print_verification_report(verification)

        print_section("AGENT 7: FEEDBACK AND REPLANNING")

        feedback = decide_next_action(
            verification_report=verification,
            quality_report=quality,
            experiment_report=experiment_report,
        )

        action = feedback.get("action", "REVIEW_CURRENT_EVIDENCE")

        print("Next action:", action)
        print("Reason:", feedback.get("reason"))
        print("Next step:", feedback.get("next_step"))
        print("Requires experiment:", feedback.get("requires_experiment"))

        history.append({
            "iteration": iteration,
            "action": action,
            "verification_status": verification.get("status"),
        })

        if action == "READY_FOR_FINAL_REVIEW":
            print("\nWorkflow reached final review.")
            print(
                "The results are provisionally ready for human review; "
                "this is not a production-readiness guarantee."
            )
            break

        if action in {
            "REVIEW_METRIC_TRADEOFFS",
            "REVIEW_REQUIRED",
            "REVIEW_EXPERIMENT_EVIDENCE",
            "REVIEW_UNSUPPORTED_QUALITY_CONCERNS",
            "STOP_FOR_REVIEW",
            "STOP_AND_REVIEW",
        }:
            print(
                "\nWorkflow paused for review. The current evidence "
                "does not justify automatic approval."
            )

            unsupported = feedback.get("unsupported_concerns", [])
            if unsupported:
                print("Quality concerns needing review:", unsupported)

            break

        if action == "INVESTIGATE_FAILED_EXPERIMENTS":
            print("\nWorkflow paused to investigate failed experiments.")
            break

        if action == "RUN_INITIAL_EXPERIMENTS":
            print(
                "\nThe feedback agent requested initial experiments, "
                "but the experiment batch has already been attempted "
                "in this workflow. Inspect its results before retrying."
            )
            break

        if action == "PLAN_QUALITY_EXPERIMENTS":
            supported, unsupported = print_capability_check(feedback)

            if unsupported:
                print(
                    "\nWorkflow paused: some requested experiment "
                    "strategies are not implemented."
                )
                print(
                    "No results have been fabricated for these strategies."
                )
                break

            if supported:
                print(
                    "\nThe requested strategies are listed as supported, "
                    "but the current runner does not expose a separate "
                    "interface to execute an individual strategy."
                )
                print("A dedicated experiment execution interface is needed.")
                break

        if feedback.get("requires_experiment"):
            supported, unsupported = print_capability_check(feedback)

            if unsupported:
                print(
                    "\nWorkflow paused because the requested experiments "
                    "cannot currently be executed."
                )
            else:
                print(
                    "\nWorkflow paused because no separate follow-up "
                    "execution interface is available."
                )
            break

        print(
            "\nNo supported automatic follow-up action is available. "
            "Stopping safely rather than pretending another experiment ran."
        )
        break

    else:
        print(
            "\nMaximum workflow iteration limit reached. "
            "Stopping to prevent unbounded execution."
        )

    return {
        "history": history,
        "experiment_report": experiment_report,
        "verification_report": verification,
        "feedback_report": feedback,
    }


def main():
    print("=" * 60)
    print(" QUALITY-AWARE AGENTIC ML PIPELINE")
    print("=" * 60)

    file_path = input(
        "\nEnter the path of your CSV dataset: "
    ).strip().strip('"')

    if not Path(file_path).is_file():
        print("Error: Dataset file not found. Please check the path.")
        return

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            print("Error: The dataset contains no rows.")
            return

        if len(df.columns) < 2:
            print("Error: The dataset needs at least two columns.")
            return

        print("\nDataset loaded successfully!")
        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        print("\nAvailable target columns:")

        for column in df.columns:
            print("-", column)

        target_column = input(
            "\nEnter the target column to predict: "
        ).strip()

        if target_column not in df.columns:
            print("Error: The selected target column does not exist.")
            return

        quality, impact, plan = run_initial_analysis(
            df=df,
            target_column=target_column,
        )

        workflow_result = run_adaptive_workflow(
            df=df,
            plan=plan,
            quality=quality,
            impact=impact,
        )

        print_section("WORKFLOW SUMMARY")

        for entry in workflow_result["history"]:
            print(
                f"Iteration {entry['iteration']}: "
                f"action={entry['action']}, "
                f"verification={entry['verification_status']}"
            )

        print(
            "\nNote: The workflow executes the experiment strategies "
            "currently implemented in experiment_agent.py. It pauses "
            "when requested strategies cannot actually be executed."
        )

    except (ValueError, KeyError) as error:
        print(f"\nInput or planning error: {error}")

    except Exception as error:
        print(f"\nError analyzing dataset: {error}")


if __name__ == "__main__":
    main()
