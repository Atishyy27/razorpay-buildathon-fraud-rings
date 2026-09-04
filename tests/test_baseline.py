import pandas as pd

from baseline import best_rule, candidate_rules, rule_at_least, score_rule
from costs import CostModel
from detect import TIER2_FEATURES


def _frame(rows):
    return pd.DataFrame(rows)


def test_score_rule_counts_each_cell():
    y = [True, True, False, False]
    pred = [True, False, True, False]
    r = score_rule(y, pred, CostModel(false_positive=1, false_negative=1))
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["cost"] == 2


def test_score_rule_precision_is_zero_not_nan_when_nothing_flagged():
    r = score_rule([True, False], [False, False], CostModel())
    assert r["precision"] == 0.0 and r["recall"] == 0.0


def test_rule_at_least_on_missing_column_is_all_false_not_a_crash():
    # holdout.py applies a rule chosen on one world to another; a column that
    # is absent must degrade to "never fires" rather than raise mid-evaluation
    df = _frame([{"chargeback_rate": 0.9}])
    assert not rule_at_least(df, "burst_synchrony", 0.5).any()


def test_baseline_sees_every_feature_the_model_sees():
    # the whole point of the rule baseline is that it is not handicapped; if a
    # feature is added to the model it must become available to the rules too,
    # otherwise the comparison silently starts flattering the model
    names = " ".join(name for name, _ in candidate_rules())
    scored = {"size", "n_txns", "id_sharing_ratio"}  # structural, not thresholdable alone
    for feature in set(TIER2_FEATURES) - scored:
        assert feature in names, f"{feature} is in the model but no rule can use it"


def test_best_rule_picks_the_cheapest_not_the_most_accurate():
    # rule A: catches nothing, 4 misses. rule B: flags everything, 6 false
    # alarms and no misses. Under a 30:1 cost model B is far cheaper despite
    # much worse precision, and the selector must prefer it.
    df = _frame(
        [{"chargeback_rate": 0.9, "burst_ratio": 1.0, "peak_hourly_velocity": 0,
          "burst_synchrony": 0.0, "high_resale_share": 0.0, "id_sharing_ratio": 0.9,
          "is_ring": True} for _ in range(4)]
        + [{"chargeback_rate": 0.9, "burst_ratio": 1.0, "peak_hourly_velocity": 0,
            "burst_synchrony": 0.0, "high_resale_share": 0.0, "id_sharing_ratio": 0.9,
            "is_ring": False} for _ in range(6)]
    )
    winner, _ = best_rule(df, CostModel(false_positive=500, false_negative=15000))
    assert winner["fn"] == 0


def test_best_rule_returns_the_full_ranking_sorted_by_cost():
    df = _frame([
        {"chargeback_rate": 0.9, "burst_ratio": 9.0, "peak_hourly_velocity": 9,
         "burst_synchrony": 1.0, "high_resale_share": 0.9, "id_sharing_ratio": 0.1,
         "is_ring": True},
        {"chargeback_rate": 0.0, "burst_ratio": 1.0, "peak_hourly_velocity": 0,
         "burst_synchrony": 0.0, "high_resale_share": 0.0, "id_sharing_ratio": 0.9,
         "is_ring": False},
    ])
    winner, scored = best_rule(df, CostModel())
    costs = [r["cost"] for r in scored]
    assert costs == sorted(costs)
    assert winner["cost"] == costs[0]
