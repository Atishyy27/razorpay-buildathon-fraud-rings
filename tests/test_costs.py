import numpy as np
import pytest

from costs import CostModel, confusion_at, optimal_threshold, sensitivity


def test_total_is_zero_with_no_errors():
    assert CostModel().total(0, 0) == 0


def test_total_scales_with_each_error_type():
    m = CostModel(false_positive=10, false_negative=100)
    assert m.total(2, 3) == 2 * 10 + 3 * 100


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        CostModel().total(-1, 0)


def test_from_ratio_sets_the_intended_ratio():
    m = CostModel.from_ratio(30, false_positive=500)
    assert m.false_negative == 15000
    assert m.ratio == 30


def test_from_ratio_rejects_non_positive():
    with pytest.raises(ValueError):
        CostModel.from_ratio(0)


def test_confusion_at_handles_all_one_class_prediction():
    # threshold above every score means nothing is flagged; the labels= pin in
    # confusion_at must still yield a full 2x2 instead of raising
    fp, fn = confusion_at([True, False, True], [0.1, 0.2, 0.3], threshold=0.9)
    assert (fp, fn) == (0, 2)


def test_optimal_threshold_flags_everything_when_misses_are_ruinous():
    # positives score LOWER than negatives here, so the only way to avoid a
    # miss is to flag everything; a huge FN cost must force exactly that
    y = [True, True, False, False]
    proba = [0.1, 0.2, 0.3, 0.4]
    best = optimal_threshold(y, proba, CostModel(false_positive=1, false_negative=10_000))
    assert best["fn"] == 0


def test_optimal_threshold_flags_nothing_when_false_alarms_are_ruinous():
    y = [True, True, False, False]
    proba = [0.1, 0.2, 0.3, 0.4]
    best = optimal_threshold(y, proba, CostModel(false_positive=10_000, false_negative=1))
    assert best["fp"] == 0


def test_optimal_threshold_finds_the_perfect_cut_on_separable_scores():
    y = [False, False, True, True]
    proba = [0.1, 0.2, 0.8, 0.9]
    best = optimal_threshold(y, proba, CostModel())
    assert best["fp"] == 0 and best["fn"] == 0
    assert 0.2 < best["threshold"] <= 0.8


def test_optimal_threshold_beats_a_fixed_half_cutoff_on_skewed_scores():
    # every score sits below 0.5, so a hardcoded 0.5 flags nothing and eats the
    # full false-negative bill. This is the exact failure the cost-tuned
    # threshold exists to prevent, so it is asserted rather than assumed.
    y = np.array([True] * 5 + [False] * 95)
    proba = np.concatenate([np.linspace(0.10, 0.20, 5), np.linspace(0.0, 0.05, 95)])
    model = CostModel(false_positive=500, false_negative=15000)
    fp_half, fn_half = confusion_at(y, proba, 0.5)
    assert model.total(fp_half, fn_half) == 5 * 15000
    assert optimal_threshold(y, proba, model)["cost"] < model.total(fp_half, fn_half)


def test_sensitivity_never_gets_more_conservative_as_misses_get_costlier():
    # the operating point may plateau, but it must never rise as the cost of
    # missing a ring rises; a rising threshold would mean the sweep is wrong
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.2
    proba = np.clip(rng.random(200) + y * 0.3, 0, 1)
    thresholds = [r["threshold"] for r in sensitivity(y, proba, ratios=(1, 10, 100))]
    assert thresholds == sorted(thresholds, reverse=True)


def test_sensitivity_reports_every_requested_ratio():
    y = [True, False, True, False]
    proba = [0.9, 0.1, 0.8, 0.2]
    rows = sensitivity(y, proba, ratios=(1, 5, 30))
    assert [r["ratio"] for r in rows] == [1, 5, 30]
