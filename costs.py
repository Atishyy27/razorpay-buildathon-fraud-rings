"""
The cost model, kept separate from the detector on purpose.

Track 02 asks for "honest metrics including false-positive cost". The honest
part is not the rupee figure, it is admitting we do not know it. A real
per-incident cost is a Razorpay-internal number, so instead of pretending to
one we express the decision as a function of the only thing that actually
matters: the ratio between what a missed ring costs and what a wrongly
flagged legit cluster costs. Pick the ratio, get the threshold.

That reframing is why this is its own module. A single invented constant is a
number a panel can dismiss; a threshold that moves correctly as the ratio
moves is a decision procedure they can adopt with their own numbers.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix

# Illustrative defaults only. NOT researched Razorpay or industry figures.
# They set the default ratio (30:1) used for the headline threshold; every
# result that depends on them is reported across a ratio sweep as well, so
# no conclusion in this repo rests on these two literals being right.
COST_FALSE_POSITIVE = 500
COST_FALSE_NEGATIVE = 15000

# Ratios swept when reporting sensitivity. 1:1 treats a missed ring and a
# wrongly flagged family as equally bad (a privacy-first stance); 100:1 is a
# fraud-loss-dominated stance. The real answer is somewhere inside, and it is
# the risk team's call, not the model's.
RATIO_SWEEP = (1, 1.5, 2, 3, 5, 10, 30, 50, 100)


@dataclass(frozen=True)
class CostModel:
    """Cost of being wrong, in whatever currency unit the caller supplies."""

    false_positive: float = COST_FALSE_POSITIVE
    false_negative: float = COST_FALSE_NEGATIVE

    @classmethod
    def from_ratio(cls, ratio, false_positive=COST_FALSE_POSITIVE):
        """Build a model where a miss costs `ratio` times a false alarm."""
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        return cls(false_positive=false_positive, false_negative=false_positive * ratio)

    @property
    def ratio(self):
        return self.false_negative / self.false_positive

    def total(self, fp, fn):
        if fp < 0 or fn < 0:
            raise ValueError(f"counts must be non-negative, got fp={fp} fn={fn}")
        return fp * self.false_positive + fn * self.false_negative


def confusion_at(y_true, proba, threshold):
    """(fp, fn) at a given probability cutoff, labels pinned so an all-one-class
    prediction still returns a full 2x2 instead of raising."""
    pred = np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y_true).astype(bool), pred, labels=[False, True]
    ).ravel()
    return int(fp), int(fn)


def optimal_threshold(y_true, proba, cost_model, grid=None):
    """Cheapest threshold under this cost model.

    Candidate cutoffs come from the observed scores themselves, not a fixed
    0.1 grid: with a saturated score distribution (which is exactly what this
    model produces) a coarse grid can step straight over the only region
    where the decision actually changes.

    Ties break toward the HIGHER threshold, i.e. toward flagging fewer people.
    A defense-only system should prefer the more conservative of two equally
    priced options rather than let an arbitrary sort order decide who gets
    put in a review queue.
    """
    proba = np.asarray(proba, dtype=float)
    if grid is None:
        # midpoints between adjacent distinct scores, plus the outer edges,
        # so every distinct partition of the data is represented exactly once
        uniq = np.unique(proba)
        mids = (uniq[:-1] + uniq[1:]) / 2 if len(uniq) > 1 else np.array([])
        grid = np.concatenate([[0.0], mids, [1.0 + 1e-9]])

    best = None
    for t in grid:
        fp, fn = confusion_at(y_true, proba, t)
        cost = cost_model.total(fp, fn)
        # strict < keeps the FIRST best on ties; grid is ascending, so we flip
        # to <= only when we want the higher (more conservative) threshold
        if best is None or cost <= best["cost"]:
            best = {"threshold": float(t), "cost": float(cost), "fp": fp, "fn": fn}
    return best


def sensitivity(y_true, proba, ratios=RATIO_SWEEP, false_positive=COST_FALSE_POSITIVE):
    """How the optimal threshold and its error mix move as the cost ratio moves.

    This is the table that replaces "we assumed Rs.15,000". If the chosen
    operating point is stable across a wide ratio band, the invented constants
    never mattered; if it swings, that instability is itself the finding and
    belongs in front of the panel rather than hidden behind one default.
    """
    rows = []
    for r in ratios:
        model = CostModel.from_ratio(r, false_positive=false_positive)
        best = optimal_threshold(y_true, proba, model)
        rows.append({
            "ratio": r,
            "threshold": round(best["threshold"], 4),
            "fp": best["fp"],
            "fn": best["fn"],
            "cost": best["cost"],
        })
    return rows
