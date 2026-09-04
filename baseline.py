"""
The rule a risk team would already have written, before any ML.

This module exists to answer the question that decides whether this project
was worth building at all: does the model beat a hand-written rule? A
detector that only matches a two-line threshold rule has not earned its
inference cost, its retraining burden, or the review load it puts on an
analyst, however good its precision looks in isolation.

The comparison is deliberately rigged AGAINST the model. Every rule here is
tuned on the full dataset with no cross-validation, so each one is scored on
data it already saw and gets the best cutoff available in hindsight. The
model is scored out-of-fold. If the model still wins under that handicap the
result means something; if it does not, the honest conclusion is to ship the
rule, and this file is how we would find that out rather than assume it.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from costs import CostModel, confusion_at

# Cutoffs a human would plausibly reach for, not a fine numerical grid.
# The point is to model what a risk analyst writes in a config file, so the
# candidate values stay round and explainable.
CHARGEBACK_CUTOFFS = (0.05, 0.10, 0.20, 0.30, 0.50)
SHARING_CUTOFFS = (0.20, 0.30, 0.40, 0.50, 0.60)
BURST_CUTOFFS = (2.0, 3.0, 5.0, 10.0)
VELOCITY_CUTOFFS = (3, 5, 8, 12, 20)
SYNCHRONY_CUTOFFS = (0.5, 0.7, 0.9, 1.0)
RESALE_CUTOFFS = (0.3, 0.5, 0.7, 0.9)
SIZE_CUTOFFS = (3, 4, 5, 8)
TXN_COUNT_CUTOFFS = (20, 40, 80, 150)

# Fairness note: the rule family below is given the SAME columns the model
# gets, including the three features added specifically to justify the model
# (velocity, synchrony, resale share), plus every two-column AND combination
# of them. Withholding those from the baseline would have made the model look
# good by comparison rather than by merit, which is the exact failure this
# module exists to prevent.


def rule_shared_ids(df, sharing_cut):
    """"Accounts that share too many identifiers for their size." The classic
    linked-account rule, and the one Tier 1 already encodes."""
    return df["id_sharing_ratio"] <= sharing_cut


def rule_chargebacks(df, cb_cut):
    """"Clusters charging back far above the portfolio baseline.\""""
    return df["chargeback_rate"] >= cb_cut


def rule_shared_and_chargebacks(df, sharing_cut, cb_cut):
    """The realistic production rule: sharing AND loss, so ordinary families
    sharing a device do not get swept in on sharing alone."""
    return rule_shared_ids(df, sharing_cut) & rule_chargebacks(df, cb_cut)


def rule_shared_or_burst(df, sharing_cut, burst_cut):
    """Sharing OR a spend spike, the wider net a team reaches for after
    getting burned by a miss."""
    return rule_shared_ids(df, sharing_cut) | (df["burst_ratio"] >= burst_cut)


# formation_window_days runs the other way: a SHORT window is the suspicious
# one, so it needs a "<=" rule. Bolting it into the ">=" family would have
# tested the opposite hypothesis and quietly given the baseline a rule that
# can never fire. Caught by test_baseline_sees_every_feature_the_model_sees.
AT_MOST_CUTOFFS = {
    "formation_window_days": (1.0, 3.0, 7.0, 14.0),
}

SINGLE_COLUMN_CUTOFFS = {
    "chargeback_rate": CHARGEBACK_CUTOFFS,
    "burst_ratio": BURST_CUTOFFS,
    "peak_hourly_velocity": VELOCITY_CUTOFFS,
    "burst_synchrony": SYNCHRONY_CUTOFFS,
    "high_resale_share": RESALE_CUTOFFS,
    # size and n_txns were missing until an independent audit pointed out that
    # 2 of the model's 9 features had no rule that could threshold them, which
    # made the "same feature access" claim false. They are weak signals alone,
    # which is exactly why they had to be included: leaving out a feature
    # because it looks unpromising is how a baseline gets quietly handicapped.
    "size": SIZE_CUTOFFS,
    "n_txns": TXN_COUNT_CUTOFFS,
}


def rule_at_least(df, column, cut):
    """Generic ">= cutoff on one column", the shape of almost every real
    production risk rule. A column the frame does not carry degrades to
    "never fires" rather than raising, because holdout.py applies a rule
    chosen on one world to another."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column] >= cut


def rule_at_most(df, column, cut):
    """"<= cutoff on one column", for signals where SMALL is suspicious."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column] <= cut


def candidate_rules():
    """(name, predicate) for every rule/cutoff combination considered."""
    rules = []
    for col, cuts in SINGLE_COLUMN_CUTOFFS.items():
        for cut in cuts:
            rules.append((
                f"{col}(>={cut})",
                lambda d, c=col, k=cut: rule_at_least(d, c, k),
            ))
    for col, cuts in AT_MOST_CUTOFFS.items():
        for cut in cuts:
            rules.append((
                f"{col}(<={cut})",
                lambda d, c=col, k=cut: rule_at_most(d, c, k),
            ))
    # every AND pair across the single-column rules, so the baseline can
    # combine signals too and the model is not credited for conjunction alone
    for col, cuts in AT_MOST_CUTOFFS.items():
        for cut in cuts:
            for other, ocuts in SINGLE_COLUMN_CUTOFFS.items():
                for ocut in ocuts:
                    rules.append((
                        f"{col}(<={cut}) AND {other}(>={ocut})",
                        lambda d, c=col, k=cut, o=other, ok=ocut:
                            rule_at_most(d, c, k) & rule_at_least(d, o, ok),
                    ))
    cols = list(SINGLE_COLUMN_CUTOFFS)
    for a, b in itertools.combinations(cols, 2):
        for ca in SINGLE_COLUMN_CUTOFFS[a]:
            for cb in SINGLE_COLUMN_CUTOFFS[b]:
                rules.append((
                    f"{a}(>={ca}) AND {b}(>={cb})",
                    lambda d, a=a, ca=ca, b=b, cb=cb:
                        rule_at_least(d, a, ca) & rule_at_least(d, b, cb),
                ))
    for cut in SHARING_CUTOFFS:
        rules.append((f"shared_ids(<={cut})", lambda d, c=cut: rule_shared_ids(d, c)))
    for cut in CHARGEBACK_CUTOFFS:
        rules.append((f"chargebacks(>={cut})", lambda d, c=cut: rule_chargebacks(d, c)))
    for s, c in itertools.product(SHARING_CUTOFFS, CHARGEBACK_CUTOFFS):
        rules.append((
            f"shared_ids(<={s}) AND chargebacks(>={c})",
            lambda d, s=s, c=c: rule_shared_and_chargebacks(d, s, c),
        ))
    for s, b in itertools.product(SHARING_CUTOFFS, BURST_CUTOFFS):
        rules.append((
            f"shared_ids(<={s}) OR burst(>={b})",
            lambda d, s=s, b=b: rule_shared_or_burst(d, s, b),
        ))
    return rules


def score_rule(y_true, pred, cost_model):
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int((y_true & pred).sum())
    fp = int((~y_true & pred).sum())
    fn = int((y_true & ~pred).sum())
    tn = int((~y_true & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost": cost_model.total(fp, fn),
    }


def best_rule(df, cost_model):
    """Cheapest rule under this cost model, tuned in-sample (deliberately)."""
    y = df["is_ring"].astype(bool)
    scored = []
    for name, predicate in candidate_rules():
        result = score_rule(y, predicate(df), cost_model)
        result["rule"] = name
        scored.append(result)
    scored.sort(key=lambda r: (r["cost"], -r["recall"]))
    return scored[0], scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/cluster_features.csv")
    parser.add_argument("--out", type=str, default="results/baseline_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    cost_model = CostModel()
    winner, scored = best_rule(df, cost_model)

    print(f"evaluated {len(scored)} hand-written rules on {len(df)} clusters, "
          f"tuned in-sample (no cross-validation, deliberately favourable)")
    print(f"\nbest rule by cost: {winner['rule']}")
    print(f"  precision={winner['precision']:.2f} recall={winner['recall']:.2f} "
          f"fp={winner['fp']} fn={winner['fn']} cost=Rs.{winner['cost']:.0f}")

    print("\ntop 5 rules by cost:")
    for r in scored[:5]:
        print(f"  Rs.{r['cost']:>8.0f}  p={r['precision']:.2f} r={r['recall']:.2f}  {r['rule']}")

    # per-archetype recall for the winning rule, the same cut the model reports,
    # so the two are compared on identical ground rather than on headline numbers
    by_arch = None
    if "archetype" in df.columns:
        pred = dict(candidate_rules())[winner["rule"]](df)
        rings = df[df["is_ring"].astype(bool)].copy()
        rings["caught"] = np.asarray(pred)[df["is_ring"].astype(bool).values]
        by_arch = rings.groupby("archetype")["caught"].mean().to_dict()
        print("\nbest rule, recall by ring archetype:")
        for arch, rec in sorted(by_arch.items()):
            print(f"  {arch}: {rec:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_clusters": int(len(df)),
        "n_rules_evaluated": len(scored),
        "tuning": "in-sample, no cross-validation, deliberately favourable to the rule",
        "cost_model": {"false_positive": cost_model.false_positive,
                        "false_negative": cost_model.false_negative},
        "best": winner,
        "top5": scored[:5],
        "recall_by_archetype": by_arch,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
