import pandas as pd

from explain import explain_cluster, review_note


def test_flags_low_id_sharing_ratio():
    row = pd.Series({
        "size": 5, "n_devices": 1, "n_cards": 1, "n_banks": 1,
        "id_sharing_ratio": 0.2, "chargeback_rate": 0.0,
        "burst_ratio": 1.0, "peak_hourly_velocity": 0,
    })
    assert "share only" in explain_cluster(row)


def test_no_signal_message_does_not_accuse():
    """explain.py runs over EVERY cluster, before any flag decision exists, so
    its wording must never imply the cluster was flagged. This test pins that:
    the module previously returned a sentence beginning "Flagged because",
    which put an accusation into the audit trail of accounts that passed."""
    row = pd.Series({
        "size": 2, "n_devices": 2, "n_cards": 2, "n_banks": 2,
        "id_sharing_ratio": 1.0, "chargeback_rate": 0.0,
        "burst_ratio": 1.0, "peak_hourly_velocity": 0,
    })
    text = explain_cluster(row)
    assert "flag" not in text.lower()
    assert text == "No individual signal crossed its threshold."


def test_signals_present_wording_is_neutral():
    row = pd.Series({
        "size": 5, "n_devices": 1, "n_cards": 1, "n_banks": 1,
        "id_sharing_ratio": 0.2, "chargeback_rate": 0.5,
        "burst_ratio": 1.0, "peak_hourly_velocity": 0,
    })
    assert explain_cluster(row).startswith("Signals present:")


def test_review_note_adds_the_decision_only_when_flagged():
    row = pd.Series({
        "size": 5, "n_devices": 1, "n_cards": 1, "n_banks": 1,
        "id_sharing_ratio": 0.2, "chargeback_rate": 0.5,
        "burst_ratio": 1.0, "peak_hourly_velocity": 0,
    })
    assert review_note(row, "flag_for_review").startswith("Queued for manual review.")
    assert review_note(row, "no_action").startswith("No action.")


def test_high_resale_share_is_reported_when_dominant():
    row = pd.Series({
        "size": 4, "n_devices": 1, "n_cards": 1, "n_banks": 1,
        "id_sharing_ratio": 0.8, "chargeback_rate": 0.0,
        "burst_ratio": 1.0, "peak_hourly_velocity": 0,
        "high_resale_share": 0.9,
    })
    assert "resale-friendly" in explain_cluster(row)


def test_synchrony_is_reported_when_accounts_peak_together():
    row = pd.Series({
        "size": 4, "n_devices": 1, "n_cards": 1, "n_banks": 1,
        "id_sharing_ratio": 0.8, "chargeback_rate": 0.0,
        "burst_ratio": 1.0, "peak_hourly_velocity": 0,
        "burst_synchrony": 1.0,
    })
    assert "same day" in explain_cluster(row)


def test_missing_optional_features_do_not_crash_the_explanation():
    # flagged_clusters.csv from an older run will not carry the new columns;
    # the explainer must degrade rather than raise on a stale file
    row = pd.Series({
        "size": 2, "n_devices": 2, "n_cards": 2, "n_banks": 2,
        "id_sharing_ratio": 1.0, "chargeback_rate": 0.0,
        "burst_ratio": 1.0, "peak_hourly_velocity": 0,
    })
    assert isinstance(explain_cluster(row), str)

def test_high_chargeback_rate_mentioned():
    row = pd.Series({
        "size": 5, "n_devices": 1, "n_cards": 1, "n_banks": 1,
        "id_sharing_ratio": 0.2, "chargeback_rate": 0.5,
        "burst_ratio": 1.0, "peak_hourly_velocity": 0,
    })
    assert "chargeback rate" in explain_cluster(row)
