"""
The test the brief actually asks for: precision and recall on a held-out set.

Cross-validation in `detect.py` holds out FOLDS of one dataset. That controls
for memorising individual clusters, but not for a subtler failure: the model
could be fitting quirks of one random draw of the generator, and every fold
would share those quirks. Because the data here is synthetic, a stronger test
is available almost for free, so not running it would be a choice to know
less than we could.

This module trains on one generated world and tests on a completely separate
one, generated from a different seed, with its own accounts, merchants,
rings and noise. Nothing about the test world is visible at training time.
Repeated over several seed pairs, it also answers the question the single
headline number cannot: how much of the result is the method, and how much
was one lucky draw.

Run: python holdout.py --pairs 5
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from baseline import best_rule
from costs import CostModel, optimal_threshold
from detect import TIER2_FEATURES, _metrics


def clopper_pearson(successes, trials, alpha=0.05):
    """Exact binomial confidence interval. Used instead of a normal
    approximation because the interesting case here is a clean sweep, where
    the normal approximation collapses to a zero-width interval and reports
    certainty that a handful of trials cannot support."""
    if trials == 0:
        return 0.0, 1.0
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return lo, hi


def generate_world(seed, n_rings, n_legit, workdir, exclude_archetype=None):
    """Run the generator and feature builder into an isolated directory."""
    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "generator.py", "--seed", str(seed), "--n-rings", str(n_rings),
           "--n-legit", str(n_legit), "--out", str(out)]
    if exclude_archetype:
        cmd += ["--exclude-archetype", *exclude_archetype]
    subprocess.run(cmd, check=True, capture_output=True)
    features = out / "cluster_features.csv"
    subprocess.run(
        [sys.executable, "graph_features.py", "--data", str(out), "--out", str(features)],
        check=True, capture_output=True,
    )
    return pd.read_csv(features)


def evaluate_pair(train_df, test_df, cost_model, seed, features=None):
    """Fit on the training world, choose the threshold there, apply it blind."""
    features = list(features) if features is not None else list(TIER2_FEATURES)
    y_train = train_df["is_ring"].astype(bool)
    y_test = test_df["is_ring"].astype(bool)

    def fresh():
        return HistGradientBoostingClassifier(
            random_state=seed, min_samples_leaf=2, class_weight="balanced"
        )

    # The threshold is tuned on the TRAINING world only; tuning it on the test
    # world would leak the answer through the cutoff even though the model
    # itself never saw those rows.
    #
    # But it must be tuned on OUT-OF-FOLD training probabilities, not in-sample
    # ones. A fitted tree pushes the rows it trained on to the extremes, so an
    # in-sample cutoff sits in a probability region the model will never
    # reproduce on unseen data, and the transferred threshold lands in the
    # wrong place. This was measured, not assumed: with an in-sample cutoff the
    # model lost to the hand-written rule in 4 of 5 independent worlds. See
    # DECISIONS.md, "Bug 3".
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(
        fresh(), train_df[features], y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = optimal_threshold(y_train, oof_proba, cost_model)["threshold"]

    model = fresh().fit(train_df[features], y_train)

    test_proba = model.predict_proba(test_df[features])[:, 1]
    model_result = _metrics(y_test, test_proba >= threshold)
    model_result["cost"] = cost_model.total(model_result["fp"], model_result["fn"])
    model_result["threshold"] = threshold

    # The rule gets the same treatment: chosen on train, applied to test.
    winner, _ = best_rule(train_df, cost_model)
    from baseline import candidate_rules, score_rule
    predicate = dict(candidate_rules())[winner["rule"]]
    rule_result = score_rule(y_test, predicate(test_df), cost_model)
    rule_result["rule"] = winner["rule"]

    return model_result, rule_result


def run_pairs(pairs, n_rings, n_legit, base_seed, cost_model, features=None,
              keep=False, quiet=False, exclude_archetype=None):
    """Run `pairs` independent train/test world pairs and summarise them.

    Shared with ablation.py so both run byte-identical protocols; a second
    near-copy of this loop would be free to drift away from it silently.
    """
    rows = []
    workroot = Path(tempfile.mkdtemp(prefix="holdout_"))
    try:
        for i in range(pairs):
            train_seed, test_seed = base_seed + 2 * i, base_seed + 2 * i + 1
            if not quiet:
                print(f"\npair {i + 1}/{pairs}: train seed {train_seed}, test seed {test_seed}")
            train_df = generate_world(train_seed, n_rings, n_legit, workroot / f"tr{i}",
                                       exclude_archetype=exclude_archetype)
            test_df = generate_world(test_seed, n_rings, n_legit, workroot / f"te{i}",
                                      exclude_archetype=exclude_archetype)
            model_result, rule_result = evaluate_pair(
                train_df, test_df, cost_model, train_seed, features=features
            )
            if not quiet:
                print(f"  model: precision={model_result['precision']:.2f} "
                      f"recall={model_result['recall']:.2f} fp={model_result['fp']} "
                      f"fn={model_result['fn']} cost=Rs.{model_result['cost']:,.0f}")
                print(f"  rule : precision={rule_result['precision']:.2f} "
                      f"recall={rule_result['recall']:.2f} fp={rule_result['fp']} "
                      f"fn={rule_result['fn']} cost=Rs.{rule_result['cost']:,.0f}  "
                      f"[{rule_result['rule']}]")
            rows.append({"pair": i, "train_seed": train_seed, "test_seed": test_seed,
                          "n_test_clusters": int(len(test_df)),
                          "model": model_result, "rule": rule_result})
    finally:
        if not keep:
            shutil.rmtree(workroot, ignore_errors=True)
    return summarise(rows, features), rows


def summarise(rows, features=None):
    def spread(key, metric):
        vals = [r[key][metric] for r in rows]
        return {"mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "values": [float(v) for v in vals]}

    n = len(rows)
    wins = sum(1 for r in rows if r["model"]["cost"] < r["rule"]["cost"])
    ties = sum(1 for r in rows if r["model"]["cost"] == r["rule"]["cost"])
    lo, hi = clopper_pearson(wins, n)
    ratios = [r["rule"]["cost"] / r["model"]["cost"] for r in rows if r["model"]["cost"] > 0]
    model = {m: spread("model", m) for m in ("precision", "recall", "cost")}
    rule = {m: spread("rule", m) for m in ("precision", "recall", "cost")}

    # Queue size and false alarms are different quantities and conflating them
    # inflates the headline: a reviewer opens EVERY alert, not only the wrong
    # ones, so the queue is tp+fp and false alarms are the wasted subset. An
    # earlier README quoted the false-alarm ratio under the label "review
    # queue", overstating it by 1.6x.
    queue = {who: float(np.mean([r[who]["tp"] + r[who]["fp"] for r in rows]))
             for who in ("model", "rule")}
    false_positives = {who: float(np.mean([r[who]["fp"] for r in rows]))
                       for who in ("model", "rule")}
    return {
        "mean_queue_size": queue,
        "mean_false_positives": false_positives,
        "pairs": n,
        "features_used": list(features) if features is not None else list(TIER2_FEATURES),
        "protocol": ("train and threshold on one generated world, test blind on another "
                      "from a different seed; the rule baseline is selected on the same "
                      "training world and applied to the same test world"),
        "model": model, "rule": rule,
        "model_beats_rule": {"wins": wins, "ties": ties, "losses": n - wins - ties,
                              "win_rate_95ci": [lo, hi]},
        "cost_ratio": {
            "of_means": (rule["cost"]["mean"] / model["cost"]["mean"]
                          if model["cost"]["mean"] else None),
            "per_pair_median": float(np.median(ratios)) if ratios else None,
            "per_pair_min": float(np.min(ratios)) if ratios else None,
            "per_pair_max": float(np.max(ratios)) if ratios else None,
            "per_pair": [float(r) for r in ratios],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=12,
                        help="number of independent train/test world pairs")
    parser.add_argument("--n-rings", type=int, default=60)
    parser.add_argument("--n-legit", type=int, default=12000)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--out", type=str, default="results/holdout_report.json")
    parser.add_argument("--keep", action="store_true", help="keep generated worlds")
    parser.add_argument("--exclude-archetype", nargs="*", default=None,
                        help="retest without these ring archetypes")
    parser.add_argument("--drop-features", nargs="*", default=None,
                        help="ablation: train without these features. Prefer "
                             "ablation.py, which runs both arms and writes a "
                             "matched comparison.")
    args = parser.parse_args()

    features = [f for f in TIER2_FEATURES if f not in set(args.drop_features or [])]
    if args.drop_features:
        print(f"ABLATION: dropped {sorted(set(args.drop_features))}, "
              f"model uses {len(features)} of {len(TIER2_FEATURES)} features")

    summary, rows = run_pairs(
        pairs=args.pairs, n_rings=args.n_rings, n_legit=args.n_legit,
        base_seed=args.base_seed, cost_model=CostModel(), features=features,
        keep=args.keep, exclude_archetype=args.exclude_archetype,
    )

    print("\n=== across independently generated worlds ===")
    for who in ("model", "rule"):
        d = summary[who]
        print(f"{who:>6}: precision {d['precision']['mean']:.2f} +/- {d['precision']['std']:.2f}   "
              f"recall {d['recall']['mean']:.2f} +/- {d['recall']['std']:.2f}   "
              f"cost Rs.{d['cost']['mean']:,.0f} +/- {d['cost']['std']:,.0f}")

    w = summary["model_beats_rule"]
    lo, hi = w["win_rate_95ci"]
    print(f"model cheaper than the tuned rule in {w['wins']}/{summary['pairs']} worlds "
          f"({w['ties']} tie, {w['losses']} loss); 95% CI on the win rate "
          f"[{lo:.0%}, {hi:.0%}]")
    cr = summary["cost_ratio"]
    if cr["per_pair_median"] is not None:
        print(f"cost ratio rule/model: {cr['of_means']:.1f}x of means, "
              f"median {cr['per_pair_median']:.1f}x per pair, "
              f"range {cr['per_pair_min']:.1f}x to {cr['per_pair_max']:.1f}x")

    # Cost at a fixed ratio hides the operational difference when both options
    # reach similar recall, so queue volume is reported alongside it: at equal
    # catch rate, precision IS the analyst workload.
    # Two different things, reported separately because conflating them
    # inflates the headline. A reviewer opens EVERY alert, not just the wrong
    # ones, so queue size is tp+fp; false alarms are the wasted subset.
    fp_model = summary["mean_false_positives"]["model"]
    fp_rule = summary["mean_false_positives"]["rule"]
    q_model = summary["mean_queue_size"]["model"]
    q_rule = summary["mean_queue_size"]["rule"]
    if fp_model and q_model:
        print(f"per world: queue {q_model:.1f} vs {q_rule:.1f} "
              f"({q_rule / q_model:.2f}x smaller), of which false alarms "
              f"{fp_model:.1f} vs {fp_rule:.1f} ({fp_rule / fp_model:.2f}x fewer)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "pairs": rows}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
