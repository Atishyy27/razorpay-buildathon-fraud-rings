"""End-to-end: generator -> features -> detect -> baseline -> explain -> respond.

The unit tests prove each stage in isolation. This proves the stages actually
compose: that the threshold the evaluation reports is the threshold the
response layer applies, that the audit trail reaches the reviewer intact, and
that the defense-only guarantee holds on real generated data rather than on a
hand-built frame. It runs the real scripts as subprocesses in a temp
directory, so a broken CLI flag or a missing output file fails here too.

Kept small (few rings, few accounts) so the whole thing runs in seconds.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
ALLOWED_ACTIONS = {"flag_for_review", "no_action"}


def run(*args, cwd=REPO):
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True
    )
    assert result.returncode == 0, f"{args} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    work = tmp_path_factory.mktemp("e2e")
    data, results = work / "data", work / "results"
    run("generator.py", "--seed", "7", "--n-rings", "8", "--n-legit", "300",
        "--out", str(data))
    run("graph_features.py", "--data", str(data),
        "--out", str(data / "cluster_features.csv"))
    run("detect.py", "--features", str(data / "cluster_features.csv"),
        "--folds", "3", "--out", str(data / "eval_report.json"))
    run("baseline.py", "--features", str(data / "cluster_features.csv"),
        "--out", str(results / "baseline_report.json"))
    run("explain.py", "--features", str(data / "cluster_features.csv"),
        "--out", str(data / "flagged_clusters.csv"))
    run("respond.py", "--features", str(data / "cluster_features.csv"),
        "--eval-report", str(data / "eval_report.json"),
        "--folds", "3",  # must match detect.py above or the scores differ
        "--out", str(data / "actions.json"))
    return {
        "features": pd.read_csv(data / "cluster_features.csv"),
        "eval": json.loads((data / "eval_report.json").read_text()),
        "baseline": json.loads((results / "baseline_report.json").read_text()),
        "flagged": pd.read_csv(data / "flagged_clusters.csv"),
        "actions": json.loads((data / "actions.json").read_text()),
    }


def test_every_stage_produced_output_for_every_cluster(pipeline):
    n = len(pipeline["features"])
    assert n > 0
    assert len(pipeline["flagged"]) == n
    assert len(pipeline["actions"]) == n


def test_defense_only_no_action_can_block_or_refund(pipeline):
    # the track disqualifies anything offense-capable. This asserts the
    # guarantee on generated data instead of trusting the docstring.
    actions = {a["action"] for a in pipeline["actions"]}
    assert actions <= ALLOWED_ACTIONS, f"unexpected action emitted: {actions - ALLOWED_ACTIONS}"
    for a in pipeline["actions"]:
        if a["action"] == "flag_for_review":
            assert a["review_queue"] == "fraud-ops-manual"
        else:
            assert a["review_queue"] is None


def test_deployed_threshold_is_the_one_that_was_measured(pipeline):
    # the quietest way for a good eval number to describe nothing that ships
    # is for the response layer to use a different cutoff than the evaluation
    measured = pipeline["eval"]["tier2"]["threshold"]
    applied = {a["threshold"] for a in pipeline["actions"]}
    assert applied == {measured}


def test_flag_decision_matches_the_threshold_exactly(pipeline):
    for a in pipeline["actions"]:
        expected = "flag_for_review" if a["risk_score"] >= a["threshold"] else "no_action"
        assert a["action"] == expected, f"cluster {a['cluster_id']} decided inconsistently"


def test_shipped_queue_is_exactly_the_measured_queue(pipeline):
    """Regression test for a real bug found in adversarial review.

    respond.py used to fit on every row and score those same rows in-sample
    while applying a threshold tuned on out-of-fold probabilities. A fitted
    tree pushes its own training rows to the extremes, so the two score
    distributions differ and the shipped alert count drifted from the reported
    one (17.2% measured vs 16.5% shipped on one seed). The counts matching is
    the property that makes the eval describe the thing that actually ships.
    """
    tier2 = pipeline["eval"]["tier2"]
    measured = tier2["tp"] + tier2["fp"]
    shipped = sum(1 for a in pipeline["actions"] if a["action"] == "flag_for_review")
    assert shipped == measured, (
        f"eval reported {measured} clusters above threshold, respond.py queued "
        f"{shipped}; the deployed queue is not the system that was measured"
    )
    assert abs(tier2["alert_rate"] - shipped / len(pipeline["actions"])) < 1e-9


def test_every_action_carries_reviewable_evidence(pipeline):
    # a risk score with no evidence is not something an analyst can contest
    for a in pipeline["actions"]:
        assert a["evidence"], f"cluster {a['cluster_id']} has an empty evidence block"
        for key in ("chargeback_rate", "burst_ratio", "high_resale_share"):
            assert key in a["evidence"]


def test_explanations_never_accuse_a_cluster_the_pipeline_did_not_flag(pipeline):
    for text in pipeline["flagged"]["explanation"]:
        assert "flagged" not in str(text).lower()


def test_ground_truth_is_never_a_model_feature(pipeline):
    from detect import TIER1_FEATURES, TIER2_FEATURES
    leaks = {"is_ring", "ring_id", "archetype"}
    assert leaks.isdisjoint(set(TIER1_FEATURES) | set(TIER2_FEATURES))


def test_eval_report_carries_uncertainty_and_a_cost_sweep(pipeline):
    tier2 = pipeline["eval"]["tier2"]
    assert tier2["ci"]["recall"] is not None
    assert len(tier2["cost_sensitivity"]) >= 4
    assert 0.0 <= tier2["alert_rate"] <= 1.0


def test_rule_baseline_was_actually_evaluated(pipeline):
    # the model-vs-rule comparison is the check that decides whether the model
    # deserved to exist, so an empty baseline report is a failed run
    assert pipeline["baseline"]["n_rules_evaluated"] > 50
    assert "best" in pipeline["baseline"]
