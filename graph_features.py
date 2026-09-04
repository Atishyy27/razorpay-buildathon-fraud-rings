"""
Builds the shared-attribute graph from accounts.csv + transactions.csv and
extracts per-cluster features. A cluster is a connected component where
accounts are linked by sharing a device, card, or bank account. Ground-truth
is_ring/ring_id is carried through for eval only, never used as a feature.
"""
import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


def build_attribute_graph(accounts):
    g = nx.Graph()
    g.add_nodes_from(accounts["account_id"])
    for attr in ["device_id", "card_fingerprint", "bank_account_id"]:
        for _, group in accounts.groupby(attr):
            ids = group["account_id"].tolist()
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if g.has_edge(ids[i], ids[j]):
                        g[ids[i]][ids[j]]["shared"] += 1
                    else:
                        g.add_edge(ids[i], ids[j], shared=1)
    return g


def cluster_features(cluster_ids, accounts, txns, merchants=None):
    acc = accounts[accounts["account_id"].isin(cluster_ids)]
    tx = txns[txns["account_id"].isin(cluster_ids)].copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    size = len(acc)
    if size == 0:
        # unreachable from main(), which filters components to size >= 2, but
        # the function must degrade rather than raise ZeroDivisionError for any
        # other caller or an id set that matches no account row
        raise ValueError("cluster_features called with no matching accounts")
    n_devices = acc["device_id"].nunique()
    n_cards = acc["card_fingerprint"].nunique()
    n_banks = acc["bank_account_id"].nunique()

    # how tightly the accounts in this cluster were opened. A bust-out ring is
    # stood up inside a few days because the operator opens the accounts in one
    # sitting; a family accumulates accounts over months. This is a coordination
    # signal, not a volume signal, which is why no amount- or rate-threshold
    # rule can express it.
    if "created_at" in acc.columns and size > 1:
        created = pd.to_datetime(acc["created_at"])
        formation_window_days = (created.max() - created.min()).total_seconds() / 86400.0
    else:
        formation_window_days = 0.0

    if len(tx) == 0:
        peak_velocity, burst_ratio, chargeback_rate = 0, 1.0, 0.0
        high_resale_share, burst_synchrony = 0.0, 0.0
    else:
        ts_sorted = tx["timestamp"].sort_values()
        peak_velocity = 0
        for t in ts_sorted:
            window_count = ((ts_sorted >= t) & (ts_sorted < t + pd.Timedelta(hours=1))).sum()
            peak_velocity = max(peak_velocity, window_count)

        early_cutoff = tx["timestamp"].min() + pd.Timedelta(days=5)
        early_mean = tx.loc[tx["timestamp"] <= early_cutoff, "amount"].mean()
        max_amount = tx["amount"].max()
        burst_ratio = max_amount / early_mean if early_mean and early_mean > 0 else 1.0

        # over attempted transactions, matching high_resale_share's denominator
        chargeback_rate = (tx["status"] == "chargeback").mean()

        # share of ATTEMPTED rupees at resale-friendly merchants (electronics,
        # gift cards). Denominator is every transaction including failed and
        # charged-back ones, not settled volume; applied identically to both
        # classes so it does not bias the comparison, but it is "attempted
        # spend share", not "settled spend share". Bust-out fraud has to convert
        # credit into something sellable; households spread spend across
        # categories. The merchant catalogue was already in the dataset and
        # previously unused.
        if merchants is not None and "merchant_id" in tx.columns:
            resale = merchants.set_index("merchant_id")["high_resale_target"]
            flags = tx["merchant_id"].map(resale).fillna(False).astype(bool)
            total = tx["amount"].sum()
            high_resale_share = float(tx.loc[flags, "amount"].sum() / total) if total > 0 else 0.0
        else:
            high_resale_share = 0.0

        # do several accounts in this cluster spike in the SAME 24h window?
        # Family members shop independently; ring accounts are driven by one
        # operator and burst together. Measured as the largest fraction of the
        # cluster's accounts whose own biggest purchase lands in one day.
        if size > 1:
            peaks = tx.loc[tx.groupby("account_id")["amount"].idxmax(), ["account_id", "timestamp"]]
            days = peaks["timestamp"].dt.floor("D")
            burst_synchrony = float(days.value_counts().max() / size) if len(days) else 0.0
        else:
            burst_synchrony = 0.0

    return {
        "size": size,
        "n_devices": n_devices,
        "n_cards": n_cards,
        "n_banks": n_banks,
        "id_sharing_ratio": (n_devices + n_cards + n_banks) / (3 * size),
        "peak_hourly_velocity": peak_velocity,
        "burst_ratio": burst_ratio,
        "chargeback_rate": chargeback_rate,
        "high_resale_share": high_resale_share,
        "burst_synchrony": burst_synchrony,
        "formation_window_days": formation_window_days,
        "n_txns": len(tx),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--out", type=str, default="data/cluster_features.csv")
    args = parser.parse_args()

    data = Path(args.data)
    accounts = pd.read_csv(data / "accounts.csv")
    txns = pd.read_csv(data / "transactions.csv")
    merchants_path = data / "merchants.csv"
    merchants = pd.read_csv(merchants_path) if merchants_path.exists() else None

    g = build_attribute_graph(accounts)
    components = [c for c in nx.connected_components(g) if len(c) >= 2]

    rows = []
    for i, comp in enumerate(components):
        feats = cluster_features(comp, accounts, txns, merchants=merchants)
        acc_rows = accounts[accounts["account_id"].isin(comp)]
        is_ring = bool(acc_rows["is_ring"].any())
        ring_id = acc_rows["ring_id"].dropna().iloc[0] if acc_rows["ring_id"].notna().any() else None
        archetype = (
            acc_rows["archetype"].dropna().iloc[0]
            if "archetype" in acc_rows and acc_rows["archetype"].notna().any()
            else "legit"
        )
        rows.append({
            "cluster_id": i,
            "account_ids": "|".join(sorted(comp)),
            **feats,
            "is_ring": is_ring,
            "ring_id": ring_id,
            "archetype": archetype,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"clusters (size>=2): {len(out_df)}")
    print(f"of which ring: {int(out_df['is_ring'].sum())}, legit multi-account: {int((~out_df['is_ring']).sum())}")


if __name__ == "__main__":
    main()
