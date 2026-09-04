"""
Tier 1 (interpretable logistic baseline) and Tier 2 (gradient-boosted trees
on the full feature set) ring classifiers.

Evaluated with stratified 5-fold cross-validation via out-of-fold
predictions, so every cluster is scored by a model that never saw it during
training and precision/recall cover the whole dataset rather than one small
slice. `holdout.py` goes further and tests on independently generated data.

Two things here are not defaults and matter more than the model choice:

1. The operating threshold is chosen by COST, not left at 0.5. On a
   14%-positive problem where a missed ring is modelled as far costlier than
   a false alarm, 0.5 is the accuracy-optimal cutoff and the wrong one. A
   risk system that does not tune its threshold to its own cost model is
   not a risk system.
2. Precision and recall ship with bootstrap confidence intervals. At 60
   positive clusters a single cluster moves precision by more than a point,
   and a bare point estimate invites a confidence the sample size does not
   support.

Tier 2 is a gradient-boosted tree, not a GNN: with a few dozen rings there
is not enough positive-class data for a GNN to learn from without
overfitting. A tree on hand-engineered graph and behavioural features is the
right-sized tool for this data volume, and permutation importance on it
doubles as the audit trail.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from costs import CostModel, optimal_threshold, sensitivity

TIER1_FEATURES = ["id_sharing_ratio", "chargeback_rate"]

# Tier 2 adds three signals a threshold rule cannot express, and they are the
# reason the model earns its place over `baseline.py`:
#   high_resale_share      - what the cluster converts credit INTO
#   burst_synchrony        - whether the accounts act together
#   formation_window_days  - how fast the cluster was stood up
# The first two are coordination signals. A rule can threshold any single
# column; it cannot represent "spends together, on resellable goods, having
# been opened in one week, at a chargeback rate a family would not reach".
TIER2_FEATURES = [
    "id_sharing_ratio", "chargeback_rate", "peak_hourly_velocity",
    "burst_ratio", "size", "n_txns",
    "high_resale_share", "burst_synchrony", "formation_window_days",
]

BOOTSTRAP_N = 2000


def _metrics(y_true, pred):
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_ci(y_true, pred, n=BOOTSTRAP_N, seed=42, alpha=0.05):
    """Percentile bootstrap CI for precision and recall.

    Resamples CLUSTERS with replacement, which is the unit the model actually
    predicts on. Draws where the resample contains no positives are skipped
    rather than scored as zero, since an all-negative resample says nothing
    about recall and scoring it as 0.0 would drag the interval down for a
    reason that has nothing to do with the model.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(pred).astype(bool)
    idx = np.arange(len(y_true))
    precisions, recalls, skipped = [], [], 0
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        yt, yp = y_true[take], pred[take]
        if yt.sum() == 0:
            skipped += 1
            continue
        recalls.append(recall_score(yt, yp, zero_division=0))
        if yp.sum() > 0:
            precisions.append(precision_score(yt, yp, zero_division=0))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {"draws": n, "skipped_no_positive": skipped}
    for name, vals in (("precision", precisions), ("recall", recalls)):
        out[name] = (
            {"lo": float(np.percentile(vals, lo)), "hi": float(np.percentile(vals, hi)),
             "n": len(vals)}
            if vals else None
        )
    return out


def evaluate_oof(y_true, y_pred, label, cost_model):
    m = _metrics(y_true, y_pred)
    m["est_cost"] = cost_model.total(m["fp"], m["fn"])
    print(f"[{label}] out-of-fold, all {len(y_true)} clusters: "
          f"precision={m['precision']:.2f} recall={m['recall']:.2f} "
          f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']} "
          f"est_cost=Rs.{m['est_cost']:.0f}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/cluster_features.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/eval_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    y = df["is_ring"].astype(bool)
    prevalence = float(y.mean())
    print(f"total clusters: {len(df)} ({int(y.sum())} ring, {int((~y).sum())} legit "
          f"multi-account), positive rate {prevalence:.1%}")

    cost_model = CostModel()
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    report = {
        "folds": args.folds,
        "n_clusters": int(len(df)),
        "positive_rate": prevalence,
        "cost_model": {"false_positive": cost_model.false_positive,
                        "false_negative": cost_model.false_negative,
                        "ratio": cost_model.ratio},
    }

    # ---- Tier 1: two features, linear, deliberately readable ----
    t1 = LogisticRegression(max_iter=1000, class_weight="balanced")
    pred1 = cross_val_predict(t1, df[TIER1_FEATURES], y, cv=cv)
    report["tier1"] = evaluate_oof(y, pred1, "tier1-logistic", cost_model)
    report["tier1"]["ci"] = bootstrap_ci(y, pred1, seed=args.seed)
    t1_full = LogisticRegression(max_iter=1000, class_weight="balanced").fit(df[TIER1_FEATURES], y)
    report["tier1"]["coefficients"] = dict(zip(TIER1_FEATURES, t1_full.coef_[0].tolist()))

    # ---- Tier 2: full feature set, cost-tuned threshold ----
    t2 = HistGradientBoostingClassifier(
        random_state=args.seed, min_samples_leaf=2, class_weight="balanced"
    )
    proba2 = cross_val_predict(
        t2, df[TIER2_FEATURES], y, cv=cv, method="predict_proba"
    )[:, 1]

    best = optimal_threshold(y, proba2, cost_model)
    thr = best["threshold"]
    pred2 = proba2 >= thr
    report["tier2"] = evaluate_oof(y, pred2, f"tier2-gbdt@cost-optimal-thr={thr:.4g}", cost_model)
    report["tier2"]["threshold"] = thr
    report["tier2"]["threshold_rule"] = "cost-optimal on out-of-fold probabilities"
    report["tier2"]["ci"] = bootstrap_ci(y, pred2, seed=args.seed)
    report["tier2"]["average_precision"] = float(average_precision_score(y, proba2))
    # Alert rate is the number a fraud-ops lead asks for first: precision is
    # about who is in the queue, alert rate is about whether the queue is
    # staffable at all. A 0.90-precision model that queues 40% of the book is
    # not deployable however good the precision looks.
    report["tier2"]["alert_rate"] = float(np.mean(pred2))
    print(f"  alert rate: {report['tier2']['alert_rate']:.1%} of clusters queued "
          f"({int(pred2.sum())} of {len(pred2)}) at threshold {thr:.3g}")

    # threshold-free summary: PR-AUC does not depend on where the cutoff lands,
    # so it is the honest way to compare rankers when the cutoff is itself tuned
    print(f"  average precision (PR-AUC, threshold-free): "
          f"{report['tier2']['average_precision']:.3f} vs {prevalence:.3f} for random")

    ci = report["tier2"]["ci"]
    if ci["precision"] and ci["recall"]:
        print(f"  95% CI precision [{ci['precision']['lo']:.2f}, {ci['precision']['hi']:.2f}], "
              f"recall [{ci['recall']['lo']:.2f}, {ci['recall']['hi']:.2f}]")

    print("\ncost sensitivity, optimal threshold as the FN:FP ratio moves:")
    rows = sensitivity(y, proba2)
    for r in rows:
        print(f"  ratio {r['ratio']:>3}:1  threshold={r["threshold"]:.4g}  "
              f"fp={r['fp']:>3} fn={r['fn']:>3}  cost=Rs.{r['cost']:,.0f}")
    report["tier2"]["cost_sensitivity"] = rows

    t2_full = HistGradientBoostingClassifier(
        random_state=args.seed, min_samples_leaf=2, class_weight="balanced"
    ).fit(df[TIER2_FEATURES], y)
    perm = permutation_importance(
        t2_full, df[TIER2_FEATURES], y, n_repeats=20, random_state=args.seed,
        scoring="average_precision",
    )
    report["tier2"]["feature_importance"] = dict(zip(TIER2_FEATURES, perm.importances_mean.tolist()))
    print("\nTier 2 permutation importance, scored on average precision (audit trail):")
    for feat, imp in sorted(report["tier2"]["feature_importance"].items(), key=lambda kv: -kv[1]):
        print(f"  {feat}: {imp:.3f}")

    if "archetype" in df.columns:
        by_arch = pd.DataFrame({
            "archetype": df["archetype"], "is_ring": y,
            "tier1_pred": pred1, "tier2_pred": pred2,
        })
        rings = by_arch[by_arch["is_ring"]]
        recall_by_arch = rings.groupby("archetype")[["tier1_pred", "tier2_pred"]].mean()
        counts = rings.groupby("archetype").size()
        print("\nrecall by ring archetype (out-of-fold), with cluster counts:")
        for arch in recall_by_arch.index:
            print(f"  {arch:<14} n={counts[arch]:>3}  "
                  f"tier1={recall_by_arch.loc[arch, 'tier1_pred']:.2f}  "
                  f"tier2={recall_by_arch.loc[arch, 'tier2_pred']:.2f}")
        report["recall_by_archetype"] = recall_by_arch.to_dict()
        report["archetype_counts"] = counts.to_dict()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
