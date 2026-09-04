"""Tier definitions and the uncertainty reporting around the headline metrics.

The cost model itself moved to costs.py and is tested in test_costs.py; the
old duplicates of those cases were removed here rather than left to drift.
"""
import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score

from costs import RATIO_SWEEP
from detect import TIER1_FEATURES, TIER2_FEATURES, bootstrap_ci


def test_tier2_is_a_strict_superset_of_tier1():
    # the two-tier story only holds if Tier 2 can see everything Tier 1 sees;
    # otherwise a Tier 2 win is not evidence about the extra features
    assert set(TIER1_FEATURES) < set(TIER2_FEATURES)


def test_tier2_carries_the_three_coordination_features():
    # these are the features that justify the model over a threshold rule; if
    # one is dropped the repo's central claim stops being true
    for feature in ("high_resale_share", "burst_synchrony", "formation_window_days"):
        assert feature in TIER2_FEATURES


def test_bootstrap_ci_brackets_a_perfect_classifier_at_one():
    y = np.array([True] * 20 + [False] * 80)
    ci = bootstrap_ci(y, y, n=200)
    assert ci["precision"]["lo"] == 1.0 and ci["recall"]["lo"] == 1.0


def test_bootstrap_ci_interval_contains_the_point_estimate():
    rng = np.random.default_rng(3)
    y = rng.random(300) < 0.2
    pred = y.copy()
    pred[rng.choice(np.arange(300), size=30, replace=False)] ^= True
    ci = bootstrap_ci(y, pred, n=500)
    assert ci["precision"]["lo"] <= precision_score(y, pred) <= ci["precision"]["hi"]
    assert ci["recall"]["lo"] <= recall_score(y, pred) <= ci["recall"]["hi"]


def test_bootstrap_ci_is_wider_on_a_smaller_sample():
    # this is the whole reason the CI is reported: the same measured error rate
    # on fewer clusters must carry visibly less confidence.
    #
    # Asserted on the precision interval, not recall. An earlier version of
    # this test used recall and failed for a reason that had nothing to do with
    # the code: at n=50 the randomly chosen flips all landed on negatives, so
    # recall was a flat 1.0 in every resample and the interval width collapsed
    # to zero. Here the errors are placed deterministically, half on each
    # class, so both intervals are actually exercised.
    def width(n, metric):
        n_pos = n // 5
        y = np.array([True] * n_pos + [False] * (n - n_pos))
        pred = y.copy()
        n_err = max(1, n // 20)
        pred[:n_err] = False              # misses, drives recall
        pred[n_pos:n_pos + n_err] = True  # false alarms, drives precision
        ci = bootstrap_ci(y, pred, n=400)
        return ci[metric]["hi"] - ci[metric]["lo"]

    assert width(50, "precision") > width(1000, "precision")
    assert width(50, "recall") > width(1000, "recall")


def test_bootstrap_ci_skips_resamples_with_no_positives():
    # scoring an all-negative resample as recall 0.0 would drag the interval
    # down for a reason that has nothing to do with the model
    y = np.array([True] + [False] * 40)
    ci = bootstrap_ci(y, y, n=300)
    assert ci["skipped_no_positive"] > 0
    assert ci["recall"]["n"] + ci["skipped_no_positive"] == 300


def test_ratio_sweep_spans_the_indifference_point():
    # a sweep that never includes 1:1 cannot show what happens when false
    # alarms are treated as seriously as misses, which is the stance a
    # privacy-minded reviewer will ask about
    assert min(RATIO_SWEEP) == 1
    assert max(RATIO_SWEEP) >= 50
