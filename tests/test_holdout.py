"""Held-out evaluation summarisation.

The queue-size test exists because of a real reporting error: the README's
headline claimed a "3.2x smaller review queue" using a false-positive ratio.
A reviewer opens every alert, not only the wrong ones, so queue size is tp+fp
and the true reduction was 1.95x. The headline overstated itself by 1.6x until
an audit caught it.
"""
import numpy as np
import pytest

from costs import CostModel
from holdout import clopper_pearson, summarise


def _pair(m_tp, m_fp, m_fn, r_tp, r_fp, r_fn, cost_model):
    def side(tp, fp, fn):
        return {"tp": tp, "fp": fp, "fn": fn, "tn": 100,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "cost": cost_model.total(fp, fn)}
    return {"model": side(m_tp, m_fp, m_fn), "rule": side(r_tp, r_fp, r_fn)}


@pytest.fixture
def rows():
    c = CostModel()
    return [_pair(50, 40, 5, 55, 140, 2, c), _pair(60, 50, 3, 58, 150, 1, c)]


def test_queue_size_counts_true_positives_not_just_false_alarms(rows):
    """The distinction the headline got wrong. Queue is what a human opens."""
    s = summarise(rows)
    assert s["mean_queue_size"]["model"] == pytest.approx(np.mean([90, 110]))
    assert s["mean_queue_size"]["rule"] == pytest.approx(np.mean([195, 208]))
    assert s["mean_false_positives"]["model"] == pytest.approx(45.0)
    # the two ratios must not be conflated: they differ materially
    queue_ratio = s["mean_queue_size"]["rule"] / s["mean_queue_size"]["model"]
    fp_ratio = s["mean_false_positives"]["rule"] / s["mean_false_positives"]["model"]
    assert queue_ratio < fp_ratio
    assert s["mean_queue_size"]["model"] > s["mean_false_positives"]["model"]


def test_summarise_counts_wins_by_cost(rows):
    s = summarise(rows)
    w = s["model_beats_rule"]
    assert w["wins"] + w["ties"] + w["losses"] == len(rows)


def test_cost_ratio_reports_median_alongside_the_mean(rows):
    # the ratio of means is the figure most flattered by one lopsided world
    s = summarise(rows)
    cr = s["cost_ratio"]
    for key in ("of_means", "per_pair_median", "per_pair_min", "per_pair_max"):
        assert cr[key] is not None


def test_clopper_pearson_matches_published_values():
    lo5, hi5 = clopper_pearson(5, 5)
    lo12, hi12 = clopper_pearson(12, 12)
    assert lo5 == pytest.approx(0.478, abs=0.002)
    assert lo12 == pytest.approx(0.735, abs=0.002)
    assert hi5 == 1.0 and hi12 == 1.0


def test_clopper_pearson_boundaries():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi < 1.0
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def test_a_clean_sweep_still_reports_an_honest_lower_bound():
    # the whole reason the interval is printed: 5 of 5 is compatible with a
    # coin flip and must not be presented as certainty
    lo, _ = clopper_pearson(5, 5)
    assert lo < 0.5
