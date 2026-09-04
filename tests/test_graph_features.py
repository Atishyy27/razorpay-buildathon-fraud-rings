import pandas as pd

from graph_features import build_attribute_graph, cluster_features


def make_accounts(rows):
    return pd.DataFrame(rows)


def test_shared_device_creates_edge():
    accounts = make_accounts([
        {"account_id": "A1", "device_id": "D1", "card_fingerprint": "C1", "bank_account_id": "B1"},
        {"account_id": "A2", "device_id": "D1", "card_fingerprint": "C2", "bank_account_id": "B2"},
        {"account_id": "A3", "device_id": "D2", "card_fingerprint": "C3", "bank_account_id": "B3"},
    ])
    g = build_attribute_graph(accounts)
    assert g.has_edge("A1", "A2")
    assert not g.has_edge("A1", "A3")
    assert not g.has_edge("A2", "A3")


def test_no_shared_attributes_no_edges():
    accounts = make_accounts([
        {"account_id": f"A{i}", "device_id": f"D{i}", "card_fingerprint": f"C{i}", "bank_account_id": f"B{i}"}
        for i in range(5)
    ])
    g = build_attribute_graph(accounts)
    assert g.number_of_edges() == 0


def test_ring_cluster_has_low_id_sharing_ratio():
    accounts = make_accounts([
        {"account_id": f"R{i}", "device_id": "D_shared", "card_fingerprint": "C_shared", "bank_account_id": "B_shared"}
        for i in range(5)
    ])
    txns = pd.DataFrame([
        {"account_id": f"R{i}", "amount": 100.0, "timestamp": "2026-01-01 00:00:00", "status": "success"}
        for i in range(5)
    ])
    feats = cluster_features(accounts["account_id"].tolist(), accounts, txns)
    assert feats["size"] == 5
    assert feats["n_devices"] == 1
    assert feats["n_cards"] == 1
    assert feats["n_banks"] == 1
    assert abs(feats["id_sharing_ratio"] - 0.2) < 1e-9  # (1+1+1)/(3*5)


def test_legit_family_has_high_id_sharing_ratio():
    accounts = make_accounts([
        {"account_id": "F1", "device_id": "D_family", "card_fingerprint": "C1", "bank_account_id": "B1"},
        {"account_id": "F2", "device_id": "D_family", "card_fingerprint": "C2", "bank_account_id": "B2"},
    ])
    txns = pd.DataFrame([
        {"account_id": "F1", "amount": 100.0, "timestamp": "2026-01-01 00:00:00", "status": "success"},
        {"account_id": "F2", "amount": 100.0, "timestamp": "2026-01-01 01:00:00", "status": "success"},
    ])
    feats = cluster_features(["F1", "F2"], accounts, txns)
    assert feats["id_sharing_ratio"] > 0.7  # (1+2+2)/(3*2) = 0.833...


def test_chargeback_rate_computed_correctly():
    accounts = make_accounts([
        {"account_id": "A1", "device_id": "D1", "card_fingerprint": "C1", "bank_account_id": "B1"},
    ])
    txns = pd.DataFrame([
        {"account_id": "A1", "amount": 100.0, "timestamp": "2026-01-01 00:00:00", "status": "chargeback"},
        {"account_id": "A1", "amount": 100.0, "timestamp": "2026-01-01 01:00:00", "status": "success"},
        {"account_id": "A1", "amount": 100.0, "timestamp": "2026-01-01 02:00:00", "status": "success"},
        {"account_id": "A1", "amount": 100.0, "timestamp": "2026-01-01 03:00:00", "status": "success"},
    ])
    feats = cluster_features(["A1"], accounts, txns)
    assert abs(feats["chargeback_rate"] - 0.25) < 1e-9


def _merchants():
    return pd.DataFrame([
        {"merchant_id": "M_RESALE", "high_resale_target": True},
        {"merchant_id": "M_PLAIN", "high_resale_target": False},
    ])


def _txn(account, merchant, amount, ts, status="success"):
    return {"account_id": account, "merchant_id": merchant, "amount": amount,
            "timestamp": ts, "status": status}


def test_high_resale_share_is_value_weighted_not_count_weighted():
    # one big resale purchase against many small ordinary ones must read as a
    # HIGH resale share; counting transactions instead of rupees would report
    # the opposite and invert the signal
    accounts = make_accounts([
        {"account_id": "A1", "device_id": "D1", "card_fingerprint": "C1", "bank_account_id": "B1"},
    ])
    txns = pd.DataFrame(
        [_txn("A1", "M_RESALE", 9000.0, "2026-01-01 00:00:00")]
        + [_txn("A1", "M_PLAIN", 100.0, f"2026-01-02 0{i}:00:00") for i in range(5)]
    )
    feats = cluster_features(["A1"], accounts, txns, merchants=_merchants())
    assert feats["high_resale_share"] > 0.9


def test_high_resale_share_is_zero_without_a_merchant_catalogue():
    accounts = make_accounts([
        {"account_id": "A1", "device_id": "D1", "card_fingerprint": "C1", "bank_account_id": "B1"},
    ])
    txns = pd.DataFrame([_txn("A1", "M_RESALE", 100.0, "2026-01-01 00:00:00")])
    feats = cluster_features(["A1"], accounts, txns, merchants=None)
    assert feats["high_resale_share"] == 0.0


def test_burst_synchrony_is_one_when_all_accounts_peak_on_the_same_day():
    accounts = make_accounts([
        {"account_id": f"R{i}", "device_id": "D1", "card_fingerprint": f"C{i}",
         "bank_account_id": "B1"} for i in range(3)
    ])
    txns = pd.DataFrame(
        [_txn(f"R{i}", "M_RESALE", 50.0, "2026-01-01 00:00:00") for i in range(3)]
        + [_txn(f"R{i}", "M_RESALE", 9000.0, f"2026-02-10 0{i}:00:00") for i in range(3)]
    )
    feats = cluster_features([f"R{i}" for i in range(3)], accounts, txns, merchants=_merchants())
    assert feats["burst_synchrony"] == 1.0


def test_burst_synchrony_is_low_when_accounts_peak_on_different_days():
    accounts = make_accounts([
        {"account_id": f"F{i}", "device_id": "D_family", "card_fingerprint": f"C{i}",
         "bank_account_id": f"B{i}"} for i in range(3)
    ])
    txns = pd.DataFrame(
        [_txn(f"F{i}", "M_PLAIN", 50.0, "2026-01-01 00:00:00") for i in range(3)]
        + [_txn(f"F{i}", "M_PLAIN", 9000.0, f"2026-0{i + 2}-10 00:00:00") for i in range(3)]
    )
    feats = cluster_features([f"F{i}" for i in range(3)], accounts, txns, merchants=_merchants())
    assert feats["burst_synchrony"] <= 1 / 3 + 1e-9


def test_formation_window_is_tight_for_a_ring_and_wide_for_a_family():
    ring = make_accounts([
        {"account_id": f"R{i}", "device_id": "D1", "card_fingerprint": "C1",
         "bank_account_id": "B1", "created_at": f"2026-01-0{i + 1} 00:00:00"} for i in range(3)
    ])
    family = make_accounts([
        {"account_id": f"F{i}", "device_id": "D2", "card_fingerprint": f"C{i}",
         "bank_account_id": f"B{i}", "created_at": f"2026-0{i + 1}-01 00:00:00"} for i in range(3)
    ])
    txns = pd.DataFrame([_txn("R0", "M_PLAIN", 10.0, "2026-01-01 00:00:00")])
    ring_feats = cluster_features(ring["account_id"].tolist(), ring, txns, merchants=_merchants())
    family_feats = cluster_features(family["account_id"].tolist(), family, txns, merchants=_merchants())
    assert ring_feats["formation_window_days"] < 3
    assert family_feats["formation_window_days"] > 50


def test_empty_transaction_cluster_does_not_crash_and_returns_neutral_values():
    accounts = make_accounts([
        {"account_id": "A1", "device_id": "D1", "card_fingerprint": "C1",
         "bank_account_id": "B1", "created_at": "2026-01-01 00:00:00"},
        {"account_id": "A2", "device_id": "D1", "card_fingerprint": "C2",
         "bank_account_id": "B2", "created_at": "2026-01-01 00:00:00"},
    ])
    empty = pd.DataFrame(columns=["account_id", "merchant_id", "amount", "timestamp", "status"])
    feats = cluster_features(["A1", "A2"], accounts, empty, merchants=_merchants())
    assert feats["n_txns"] == 0
    assert feats["high_resale_share"] == 0.0
    assert feats["burst_synchrony"] == 0.0
