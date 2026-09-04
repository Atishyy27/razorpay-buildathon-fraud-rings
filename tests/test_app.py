import pandas as pd

from app import cluster_accounts_subgraph, merge_flagged_with_actions


def test_cluster_accounts_subgraph_links_shared_device():
    accounts = pd.DataFrame([
        {"account_id": "R1", "device_id": "D1", "card_fingerprint": "C1", "bank_account_id": "B1"},
        {"account_id": "R2", "device_id": "D1", "card_fingerprint": "C2", "bank_account_id": "B1"},
    ])
    row = pd.Series({"account_ids": "R1|R2"})
    g = cluster_accounts_subgraph(row, accounts)
    assert g.has_edge("R1", "R2")
    assert set(g.nodes()) == {"R1", "R2"}


def test_merge_flagged_with_actions_sorts_by_risk_score():
    flagged = pd.DataFrame([
        {"cluster_id": 0, "explanation": "a"},
        {"cluster_id": 1, "explanation": "b"},
    ])
    actions = pd.DataFrame([
        {"cluster_id": 0, "risk_score": 0.2, "action": "no_action"},
        {"cluster_id": 1, "risk_score": 0.9, "action": "flag_for_review"},
    ])
    merged = merge_flagged_with_actions(flagged, actions)
    assert merged.iloc[0]["cluster_id"] == 1
    assert merged.iloc[0]["risk_score"] == 0.9
