"""Guards against construction artefacts that act as labels.

This file exists because the same bug was found three separate times, each
time in a different column, and never by looking at the model:

  1. no legit account shared a card or bank account, so
     `n_cards < size OR n_banks < size` scored precision 1.00, recall 1.00
  2. rings were appended after every legit account, so `cluster_id` was
     monotonic in the label (legit 0-1400, rings 1401+), AUC 1.00
  3. account ids encoded the class in their prefix and length
     ("A00042" vs "R07-003"), AUC 1.00, shipped inside `account_ids`

None of the three was ever a model feature, which is exactly why they
survived: auditing the features you feed the model does not audit the columns
you discarded, and a discarded column can still be the answer. These tests
audit the generated data itself rather than the model.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent

# Ground truth, present for evaluation and never a model feature. Asserted
# separately in the end-to-end suite.
LABEL_COLUMNS = {"is_ring", "ring_id", "archetype"}

# A construction artefact carries no information about the class, so its AUC
# should sit near 0.5. The band is wide because these are finite samples, not
# because near-perfect separation is tolerable.
ARTEFACT_AUC_BAND = (0.30, 0.70)


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    work = tmp_path_factory.mktemp("leak")
    data = work / "data"
    subprocess.run(
        [sys.executable, "generator.py", "--seed", "11", "--n-rings", "25",
         "--n-legit", "2500", "--out", str(data)],
        cwd=REPO, check=True, capture_output=True,
    )
    subprocess.run(
        [sys.executable, "graph_features.py", "--data", str(data),
         "--out", str(data / "cluster_features.csv")],
        cwd=REPO, check=True, capture_output=True,
    )
    return pd.read_csv(data / "cluster_features.csv")


def _auc(y, values):
    """Direction-agnostic separability of one column."""
    return max(roc_auc_score(y, values), roc_auc_score(y, -np.asarray(values, dtype=float)))


def test_cluster_id_carries_no_signal(features):
    """cluster_id is an enumeration index. If rings are appended after legit
    accounts it becomes a perfect label without anyone noticing."""
    y = features["is_ring"].astype(bool)
    auc = _auc(y, features["cluster_id"])
    assert ARTEFACT_AUC_BAND[0] <= auc <= ARTEFACT_AUC_BAND[1], (
        f"cluster_id separates the classes at AUC {auc:.3f}; row order is leaking"
    )


def test_account_id_format_carries_no_signal(features):
    """Account ids must not encode the class in their prefix or their length.
    The id string ships inside `account_ids`, so any reader of the CSV sees it."""
    y = features["is_ring"].astype(bool)
    ids = features["account_ids"].str.split("|")
    mean_len = ids.apply(lambda v: sum(len(x) for x in v) / len(v))
    auc = _auc(y, mean_len)
    assert ARTEFACT_AUC_BAND[0] <= auc <= ARTEFACT_AUC_BAND[1], (
        f"account id length separates the classes at AUC {auc:.3f}"
    )
    prefixes = ids.apply(lambda v: {x[0] for x in v})
    ring_prefixes = set().union(*prefixes[y]) if y.any() else set()
    legit_prefixes = set().union(*prefixes[~y]) if (~y).any() else set()
    assert ring_prefixes == legit_prefixes, (
        f"ring ids start with {ring_prefixes}, legit with {legit_prefixes}; "
        f"the prefix is a label"
    )


def test_no_raw_column_is_a_near_perfect_separator(features):
    """No single column, feature or not, may act as an oracle.

    The bar is deliberately about PRECISION at high recall rather than AUC: the
    historical failures were all "one threshold catches nearly every ring with
    almost no false positives", which is what a perfect construction artefact
    looks like.
    """
    y = features["is_ring"].astype(bool).values
    offenders = []
    for col in features.columns:
        # pandas 3 backs string columns with Arrow, so a plain
        # `dtype == object` check misses them entirely
        if col in LABEL_COLUMNS or not pd.api.types.is_numeric_dtype(features[col]):
            continue
        if features[col].isna().any():
            continue
        values = features[col].astype(float).values
        for direction in (1, -1):
            v = direction * values
            for t in np.unique(np.quantile(v, np.linspace(0.01, 0.99, 60))):
                pred = v >= t
                tp = int((pred & y).sum())
                fp = int((pred & ~y).sum())
                fn = int((~pred & y).sum())
                if tp + fp == 0 or tp + fn == 0:
                    continue
                precision, recall = tp / (tp + fp), tp / (tp + fn)
                if precision >= 0.95 and recall >= 0.80:
                    offenders.append((col, direction, round(precision, 3), round(recall, 3)))
                    break
    assert not offenders, (
        "single columns act as near-oracles, which means the generator is "
        f"leaking the label rather than the model being good: {offenders}"
    )


def test_derived_sharing_boolean_is_not_an_oracle(features):
    """The exact boolean from the first leak, pinned forever."""
    y = features["is_ring"].astype(bool)
    pred = (features["n_cards"] < features["size"]) | (features["n_banks"] < features["size"])
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    assert precision < 0.5, (
        f"'shares a card or bank' has precision {precision:.3f}; legit groups "
        f"are not sharing identifiers the way real households do"
    )


def test_legit_groups_share_every_identifier_type(features):
    """Legit clusters must be able to share each identifier a ring can share.
    If any type is ring-exclusive, its raw count is a label."""
    legit = features[~features["is_ring"].astype(bool)]
    for col in ("n_devices", "n_cards", "n_banks"):
        sharing = (legit[col] < legit["size"]).mean()
        assert sharing > 0.05, (
            f"only {sharing:.1%} of legit clusters share by {col}; that makes "
            f"it a near-exclusive property of rings"
        )
