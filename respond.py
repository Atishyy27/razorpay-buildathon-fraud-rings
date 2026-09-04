"""
Auto-responder stub: scores every cluster with Tier 2 and writes a
flag-for-review action for anything above threshold. Never an auto-block or
auto-refund, defense-only per the track's own rule. Also runs the one
deliberate near-miss check, does the model correctly avoid flagging (or
clearly downgrade) the legit family-sharing clusters, this is the "failure
recovery" story for the panel, not just the happy path.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from detect import TIER2_FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/cluster_features.csv")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the cost-optimal threshold from the eval report")
    parser.add_argument("--eval-report", type=str, default="results/eval_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5,
                        help="must match detect.py so the scores are the ones measured")
    parser.add_argument("--in-sample", action="store_true",
                        help="reproduce the original defect: score training rows in-sample")
    parser.add_argument("--out", type=str, default="data/actions.json")
    args = parser.parse_args()

    df = pd.read_csv(args.features)

    # Use the SAME cost-optimal cutoff the evaluation reported. A hardcoded 0.5
    # here would mean the deployed queue is not the system that was measured,
    # which is the quietest way for a good eval number to describe nothing that
    # actually ships.
    threshold = args.threshold
    source = "explicit --threshold"
    report_path = Path(args.eval_report)
    if threshold is None and report_path.exists():
        report = json.loads(report_path.read_text())
        threshold = report.get("tier2", {}).get("threshold")
        source = f"cost-optimal, from {report_path}"
    if threshold is None:
        threshold = 0.5
        source = "fallback default (no eval report found)"
    print(f"threshold {threshold:.6g} ({source})")
    # unlike detect.py's cross-validated eval, the deployed model here trains
    # on everything, held-out splits are for measuring generalization, not
    # for the model that actually makes the call on real clusters
    # Every cluster here is one the model would have trained on, so it must be
    # scored out-of-fold, exactly as detect.py measured it. Fitting on all rows
    # and scoring those same rows in-sample was the original bug: a fitted tree
    # pushes its own training rows toward the extremes, so an out-of-fold
    # threshold lands somewhere else on an in-sample score distribution and the
    # shipped queue quietly stops being the system that was measured. Model
    # config, seed and fold count all mirror detect.py for the same reason.
    #
    # In production the deployed model would score clusters it had never seen,
    # which is what the out-of-fold scores here stand in for. `--in-sample`
    # exists only to reproduce the original defect.
    y = df["is_ring"].astype(bool)

    def fresh():
        return HistGradientBoostingClassifier(
            random_state=args.seed, min_samples_leaf=2, class_weight="balanced"
        )

    df = df.copy()
    if args.in_sample:
        df["risk_score"] = fresh().fit(df[TIER2_FEATURES], y).predict_proba(
            df[TIER2_FEATURES]
        )[:, 1]
        print("WARNING: --in-sample scoring, alert counts will NOT match the eval report")
    else:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        df["risk_score"] = cross_val_predict(
            fresh(), df[TIER2_FEATURES], y, cv=cv, method="predict_proba"
        )[:, 1]
    df["action"] = df["risk_score"].apply(
        lambda s: "flag_for_review" if s >= threshold else "no_action"
    )

    # The case file is what a human reviewer actually opens. It carries the
    # evidence that drove the score, not just the score, because "the model
    # said 0.91" is not something an analyst can act on or contest.
    audit_cols = [
        "cluster_id", "size", "account_ids", "id_sharing_ratio", "chargeback_rate",
        "burst_ratio", "peak_hourly_velocity", "high_resale_share",
        "burst_synchrony", "formation_window_days", "n_txns",
    ]
    present = [c for c in audit_cols if c in df.columns]
    actions = []
    for _, r in df.iterrows():
        actions.append({
            "cluster_id": int(r["cluster_id"]),
            "is_ring": bool(r["is_ring"]),
            "risk_score": float(r["risk_score"]),
            "action": r["action"],
            "threshold": float(threshold),
            "review_queue": "fraud-ops-manual" if r["action"] == "flag_for_review" else None,
            "evidence": {c: (None if pd.isna(r[c]) else
                              (r[c].item() if hasattr(r[c], "item") else r[c]))
                          for c in present},
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(actions, indent=2, default=str))
    print(f"actions: {int((df['action'] == 'flag_for_review').sum())} flagged for review, "
          f"{int((df['action'] == 'no_action').sum())} left alone. Defense-only: no auto-block, "
          f"no auto-refund, every action is a human review queue entry.")

    legit_multi = df[~df["is_ring"]]
    if len(legit_multi):
        false_flags = legit_multi[legit_multi["action"] == "flag_for_review"]
        print(f"\nnear-miss check: {len(false_flags)}/{len(legit_multi)} legit multi-account "
              f"clusters (families sharing a device) got flagged.")
        if len(false_flags):
            # printed at full precision on purpose: rounding these to 2dp
            # showed "0.0" for scores that had genuinely cleared a threshold of
            # ~1e-4, which reads as a contradiction rather than a near miss
            scores = ", ".join(f"{v:.2e}" for v in sorted(false_flags["risk_score"], reverse=True))
            print(f"risk scores on those: [{scores}] against threshold {threshold:.2e}")


if __name__ == "__main__":
    main()
