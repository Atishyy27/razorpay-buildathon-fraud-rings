"""
Synthetic transaction data for track 02 (AI Risk Manager): plants bust-out /
chargeback fraud rings inside an otherwise ordinary transaction log, with
ground truth for held-out precision/recall eval.

Usage: python generator.py --seed 42 --out data/
"""
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


def build_merchants(fake, n=25):
    categories = ["electronics", "gift_cards", "fashion", "grocery", "travel", "home"]
    high_resale = {"electronics", "gift_cards"}
    rows = []
    for i in range(n):
        cat = random.choice(categories)
        rows.append({
            "merchant_id": f"M{i:03d}",
            "name": fake.company(),
            "category": cat,
            "high_resale_target": cat in high_resale,
        })
    return pd.DataFrame(rows)


# Legit customers are not uniform, and pretending they are is what made the
# first two versions of this dataset trivially separable. Two ordinary
# behaviours overlap with the fraud signal and both genuinely exist in payment
# data:
#   - big_spender: an occasional large purchase (a laptop, a flight, a festival
#     order) after months of small ones. Produces exactly the burst_ratio spike
#     the rings produce.
#   - dispute_prone: a customer who actually disputes things, sometimes
#     wrongly ("friendly fraud"). Produces an elevated chargeback rate with no
#     fraud ring behind it.
# They are independent, so some accounts are both, which is the hardest and
# most realistic case.
LEGIT_BIG_SPENDER_RATE = 0.18
LEGIT_DISPUTE_PRONE_RATE = 0.12
LEGIT_DISPUTE_CB_PROB = 0.18


def make_account(account_id, device_id, card_fp, bank_id, created_at,
                 big_spender=False, dispute_prone=False):
    return {
        "account_id": account_id,
        "created_at": created_at,
        "device_id": device_id,
        "card_fingerprint": card_fp,
        "bank_account_id": bank_id,
        "is_ring": False,
        "ring_id": None,
        "big_spender": big_spender,
        "dispute_prone": dispute_prone,
    }


def _legit_profile():
    return {
        "big_spender": random.random() < LEGIT_BIG_SPENDER_RATE,
        "dispute_prone": random.random() < LEGIT_DISPUTE_PRONE_RATE,
    }


# Legit groups must be able to share the SAME identifier types rings share.
# An earlier version had legit accounts share only devices, never cards and
# never bank accounts, which meant zero of 3000 legit accounts shared either.
# The consequence was fatal and was found by adversarial review: the boolean
# "n_cards < size OR n_banks < size" scored precision 1.00 and recall 1.00 on
# every ring, one layer below the declared features. The sharing STRUCTURE was
# the label, so no amount of behavioural modelling downstream was being tested.
#
# All three of these are ordinary in Indian retail banking:
#   joint bank account   - spouses, parent and child, HUF accounts
#   supplementary card   - add-on cards issued on one primary account
#   workplace/cafe device- one shared device across unrelated customers
LEGIT_SHARED_DEVICE_RATE = 0.55
LEGIT_JOINT_BANK_RATE = 0.45
LEGIT_SUPPLEMENTARY_CARD_RATE = 0.35
LEGIT_HUB_DEVICE_RATE = 0.06
LEGIT_HOUSEHOLD_SIZES = (2, 3, 4, 5)
LEGIT_HUB_SIZES = (5, 6, 7, 8, 10, 12)


def _gen_legit_group(fake, start_index, size, created_at, kind):
    """One legit multi-account cluster.

    `household` can share a device, a joint bank account and a supplementary
    card, in any combination. `hub` is a shared device only (an office machine
    or a cafe terminal) across otherwise unrelated people, which is what
    produces large legit clusters and stops cluster SIZE from being a label.
    """
    if kind == "hub":
        # an office machine or a cafe terminal: a device link by definition
        links = {"device"}
    else:
        # A household is linked by whichever identifiers it actually shares,
        # drawn independently, with at least one guaranteed so the cluster
        # exists at all. Forcing "device" onto every household is what made
        # n_devices a near-perfect label: 100% of legit clusters had
        # n_devices == 1 against 13% of rings, so "n_devices >= 2" scored
        # precision 1.00 and recall 0.87. A couple with a joint bank account
        # but separate phones and separate cards is entirely ordinary and was
        # impossible to generate before.
        links = {name for name, rate in (
            ("device", LEGIT_SHARED_DEVICE_RATE),
            ("bank", LEGIT_JOINT_BANK_RATE),
            ("card", LEGIT_SUPPLEMENTARY_CARD_RATE),
        ) if random.random() < rate}
        if not links:
            links = {random.choice(["device", "bank", "card"])}

    shared_device = f"D{fake.uuid4()[:8]}" if "device" in links else None
    joint_bank = f"B{fake.uuid4()[:8]}" if "bank" in links else None
    supp_card = f"C{fake.uuid4()[:8]}" if "card" in links else None
    out = []
    for k in range(size):
        # signups spread over days, not one instant. An identical created_at
        # across a group would hand the model a free separator (rings form
        # inside a week) that real signup data does not contain.
        member_created = created_at + timedelta(days=random.uniform(0, 45 if kind == "hub" else 12))
        out.append(make_account(
            f"A{start_index + k:05d}",
            shared_device or f"D{fake.uuid4()[:8]}",
            supp_card or f"C{fake.uuid4()[:8]}",
            joint_bank or f"B{fake.uuid4()[:8]}",
            member_created,
            **_legit_profile(),
        ))
    return out


def gen_legit_accounts(fake, n, window_start, window_end, family_share_rate=0.05):
    accounts = []
    i = 0
    while i < n:
        created_at = fake.date_time_between(window_start, window_end)
        roll = random.random()
        if roll < family_share_rate and n - i >= 2:
            size = min(random.choice(LEGIT_HOUSEHOLD_SIZES), n - i)
            accounts.extend(_gen_legit_group(fake, i, size, created_at, "household"))
            i += size
        elif roll < family_share_rate + LEGIT_HUB_DEVICE_RATE and n - i >= 5:
            size = min(random.choice(LEGIT_HUB_SIZES), n - i)
            accounts.extend(_gen_legit_group(fake, i, size, created_at, "hub"))
            i += size
        else:
            accounts.append(make_account(
                f"A{i:05d}", f"D{fake.uuid4()[:8]}",
                f"C{fake.uuid4()[:8]}", f"B{fake.uuid4()[:8]}", created_at,
                **_legit_profile(),
            ))
            i += 1
    return accounts


# three ring archetypes, each harder to catch than the last:
#  - loud: shares device+card+bank, high chargeback rate, catchable on
#    id_sharing_ratio alone, this alone made the problem trivially easy
#  - sophisticated: shares only the cash-out bank account, distinct
#    device/card per account, lower chargeback rate, still a big-ish
#    cluster so id_sharing_ratio is a bit lower than legit family clusters
#    but not by much
#  - quiet: same sharing pattern as sophisticated, but sized 2-3 like a
#    legit family group, at that size id_sharing_ratio is mathematically
#    IDENTICAL to a legit family cluster of the same size (one shared
#    attribute out of three, divided by size, doesn't care which attribute
#    is shared). This forces the model to lean on chargeback/burst/velocity
#    alone, id_sharing_ratio genuinely cannot separate these from legit.
# burst_mean is the lognormal mean of the burst-phase purchase amount. Legit
# big spenders use 8.5, so a ring at 8.5 is trivially separable on amount and a
# ring at 7.0 is not separable on amount at all. Real bust-out operators tune
# exactly this: staying under the amount rule is the cheapest evasion there is,
# so a dataset where every ring maxes out is a dataset that flatters the model.
# `slow_drain` exists because of a measured failure, not for variety. With
# only the first three archetypes every ring bursted fast on resale merchants,
# so the single rule "peak_hourly_velocity >= 3 AND high_resale_share >= 0.5"
# reproduced the generator almost exactly and beat the model in 5 of 5
# independently generated worlds. That is a true result about that dataset and
# a false one about fraud: a population where every ring shares one mechanism
# is a population rules are guaranteed to win on. Slow-drain bust-out is a real
# documented pattern (cash out steadily under the velocity and amount rules
# rather than in one spike), and adding it is what makes the population
# heterogeneous enough that no two-condition rule covers it. See DECISIONS.md,
# "Bug 4".
RING_CONFIG = {
    "loud": {"size_range": (5, 12), "share_device_card": True,
             "cb_prob": 0.80, "burst_mean": 8.5, "pattern": "burst"},
    "sophisticated": {"size_range": (5, 12), "share_device_card": False,
                      "cb_prob": 0.45, "burst_mean": 7.6, "pattern": "burst"},
    "quiet": {"size_range": (2, 3), "share_device_card": False,
              "cb_prob": 0.25, "burst_mean": 6.9, "pattern": "burst"},
    "slow_drain": {"size_range": (4, 9), "share_device_card": False,
                   "cb_prob": 0.35, "burst_mean": 7.2, "pattern": "drain"},
}
RING_WEIGHTS = {"loud": 0.30, "sophisticated": 0.25, "quiet": 0.20, "slow_drain": 0.25}


def gen_ring_accounts(fake, ring_id, size, window_start, formation_days=5, share_device_card=True):
    if share_device_card:
        shared_devices = [f"D{fake.uuid4()[:8]}" for _ in range(random.choice([1, 2]))]
        shared_cards = [f"C{fake.uuid4()[:8]}" for _ in range(random.choice([1, 2]))]
    else:
        shared_devices = None
        shared_cards = None
    cashout_bank = f"B{fake.uuid4()[:8]}"
    ring_start = window_start + timedelta(days=random.randint(0, 30))
    accounts = []
    for j in range(size):
        created_at = ring_start + timedelta(days=random.uniform(0, formation_days))
        accounts.append({
            "account_id": f"R{ring_id:02d}-{j:03d}",
            "created_at": created_at,
            "device_id": random.choice(shared_devices) if shared_devices else f"D{fake.uuid4()[:8]}",
            "card_fingerprint": random.choice(shared_cards) if shared_cards else f"C{fake.uuid4()[:8]}",
            "bank_account_id": cashout_bank,
            "is_ring": True,
            "ring_id": ring_id,
            # rings carry the same columns as legit accounts so the two
            # populations share one schema; these flags describe legit
            # customer behaviour and are never read for ring accounts
            "big_spender": False,
            "dispute_prone": False,
        })
    return accounts, ring_start


def gen_legit_transactions(fake, account, merchants, window_end):
    txns = []
    n_txns = np.random.poisson(6) + 1
    cb_prob = LEGIT_DISPUTE_CB_PROB if account.get("dispute_prone") else 0.01
    success_prob = 1.0 - 0.04 - cb_prob
    for _ in range(n_txns):
        m = merchants.sample(1).iloc[0]
        ts = fake.date_time_between(account["created_at"], window_end)
        amount = round(np.random.lognormal(mean=6.5, sigma=0.6), 2)
        status = np.random.choice(
            ["success", "failed", "chargeback"], p=[success_prob, 0.04, cb_prob]
        )
        txns.append({
            "txn_id": f"T{fake.uuid4()[:10]}",
            "account_id": account["account_id"],
            "merchant_id": m["merchant_id"],
            "amount": amount,
            "timestamp": ts,
            "status": status,
        })
    if account.get("big_spender"):
        # one or two genuinely large purchases, drawn from the SAME amount
        # distribution the rings' burst uses. This is the point: a legit spike
        # and a fraud spike are indistinguishable on amount alone, so
        # burst_ratio stops being a free separator and the model has to use
        # the combination of signals instead of the one strongest feature.
        resale_pool = merchants[merchants["high_resale_target"]]
        for _ in range(random.randint(1, 2)):
            # a big legit purchase is usually a phone, a laptop or a gift card,
            # i.e. the SAME merchant categories a bust-out ring targets. Drawing
            # these uniformly instead made high_resale_share a near-copy of the
            # label and handed the model a signal that does not exist in
            # production data.
            m = (resale_pool if len(resale_pool) and random.random() < 0.7
                 else merchants).sample(1).iloc[0]
            ts = fake.date_time_between(account["created_at"], window_end)
            amount = round(np.random.lognormal(mean=8.5, sigma=0.5), 2)
            status = np.random.choice(
                ["success", "chargeback"], p=[1 - cb_prob, cb_prob]
            )
            txns.append({
                "txn_id": f"T{fake.uuid4()[:10]}",
                "account_id": account["account_id"],
                "merchant_id": m["merchant_id"],
                "amount": amount,
                "timestamp": ts,
                "status": status,
            })
    return txns


def gen_ring_transactions(fake, account, ring_start, merchants, formation_days=5,
                          cb_prob=0.8, burst_mean=8.5, burst_start=None,
                          pattern="burst"):
    txns = []
    # trust-building: small, ordinary-looking purchases first
    for _ in range(random.randint(3, 8)):
        m = merchants.sample(1).iloc[0]
        ts = ring_start + timedelta(days=random.uniform(0, formation_days))
        amount = round(np.random.lognormal(mean=6.0, sigma=0.4), 2)
        txns.append({
            "txn_id": f"T{fake.uuid4()[:10]}",
            "account_id": account["account_id"],
            "merchant_id": m["merchant_id"],
            "amount": amount,
            "timestamp": ts,
            "status": "success",
        })
    # burst: high-value purchases on high-resale merchants, tight window, mostly chargeback later
    pool = merchants[merchants["high_resale_target"]]
    target_merchants = pool.sample(min(3, len(pool)))
    if burst_start is None:
        burst_start = ring_start + timedelta(days=formation_days + random.uniform(1, 4))

    if pattern == "drain":
        # cash out steadily over weeks instead of in one window, and mix
        # ordinary merchants in, so neither an hourly-velocity rule nor a
        # resale-concentration rule fires. The chargebacks still arrive, which
        # is the only signal this archetype cannot hide.
        n_cashout = random.randint(5, 10)
        spread_days = random.uniform(20, 45)
    else:
        n_cashout = random.randint(2, 5)
        spread_days = 4 / 24.0

    for _ in range(n_cashout):
        if pattern == "drain" and random.random() < 0.45:
            m = merchants.sample(1).iloc[0]
        else:
            m = target_merchants.sample(1).iloc[0]
        ts = burst_start + timedelta(days=random.uniform(0, spread_days))
        amount = round(np.random.lognormal(mean=burst_mean, sigma=0.5), 2)
        status = np.random.choice(["chargeback", "success"], p=[cb_prob, 1 - cb_prob])
        txns.append({
            "txn_id": f"T{fake.uuid4()[:10]}",
            "account_id": account["account_id"],
            "merchant_id": m["merchant_id"],
            "amount": amount,
            "timestamp": ts,
            "status": status,
        })
    return txns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    # 12,000 legit accounts against 60 rings puts ring prevalence at ~4% of
    # multi-account clusters. Still far above real organized-fraud prevalence
    # (well under 1%), but two orders closer than the 58% this dataset started
    # at, and low enough that precision is measured under genuine class
    # imbalance rather than flattered by it.
    parser.add_argument("--n-legit", type=int, default=12000)
    parser.add_argument("--n-rings", type=int, default=60)
    parser.add_argument("--exclude-archetype", nargs="*", default=None,
                        help="omit these ring archetypes. Exists so the claim "
                             "'the model beats the rule' can be retested "
                             "without slow_drain, the archetype added after "
                             "watching the rule win, which is the one change "
                             "most exposed to the charge of benchmark-gaming.")
    parser.add_argument("--family-share-rate", type=float, default=0.15,
                        help="fraction of legit accounts that sit in a "
                             "device-sharing family cluster")
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    window_start = datetime(2026, 1, 1)
    window_end = datetime(2026, 4, 1)

    merchants = build_merchants(fake)

    all_accounts = gen_legit_accounts(
        fake, args.n_legit, window_start, window_end,
        family_share_rate=args.family_share_rate,
    )
    all_txns = []
    for acc in all_accounts:
        all_txns.extend(gen_legit_transactions(fake, acc, merchants, window_end))

    excluded = set(args.exclude_archetype or [])
    unknown = excluded - set(RING_WEIGHTS)
    if unknown:
        raise SystemExit(f"unknown archetype(s): {sorted(unknown)}; "
                         f"known: {sorted(RING_WEIGHTS)}")
    archetypes = [a for a in RING_WEIGHTS if a not in excluded]
    if not archetypes:
        raise SystemExit("every archetype excluded, nothing to plant")
    weights = [RING_WEIGHTS[a] for a in archetypes]
    if excluded:
        print(f"EXCLUDED archetypes: {sorted(excluded)}; planting {archetypes}")
    for ring_id in range(args.n_rings):
        archetype = random.choices(archetypes, weights=weights)[0]
        cfg = RING_CONFIG[archetype]
        size = random.randint(*cfg["size_range"])
        ring_accounts, ring_start = gen_ring_accounts(
            fake, ring_id, size, window_start, share_device_card=cfg["share_device_card"]
        )
        for acc in ring_accounts:
            acc["archetype"] = archetype
        all_accounts.extend(ring_accounts)
        # one operator, one cash-out window: the burst is a property of the
        # ring, not of each account. Drawing it per account made the ring look
        # uncoordinated and killed the synchrony signal entirely.
        ring_burst_start = ring_start + timedelta(days=5 + random.uniform(1, 4))
        for acc in ring_accounts:
            all_txns.extend(gen_ring_transactions(
                fake, acc, ring_start, merchants,
                cb_prob=cfg["cb_prob"], burst_mean=cfg["burst_mean"],
                burst_start=ring_burst_start, pattern=cfg["pattern"],
            ))

    accounts_df = pd.DataFrame(all_accounts)
    txns_df = pd.DataFrame(all_txns)

    # Two construction artefacts are scrubbed here. Both were perfect (AUC
    # 1.00) labels, both were found by adversarial review, and no amount of
    # behavioural modelling would ever have surfaced either:
    #   1. account ids encoded the class in their prefix and their length
    #      ("A00042" legit against "R07-003" ring), and that string ships
    #      verbatim inside cluster_features.account_ids.
    #   2. rings were appended after every legit account, so row order, and
    #      therefore the cluster_id graph_features assigns by enumeration
    #      order, was monotonic in the label: legit 0-1400, rings 1401-1460.
    # Reassigning ids from one shuffled pool and shuffling the rows kills both.
    # Ground truth survives only in is_ring / ring_id / archetype, which exist
    # for evaluation and are never model features.
    order = list(range(len(accounts_df)))
    random.shuffle(order)
    remap = {old_id: f"ACC{n:06d}" for old_id, n in zip(accounts_df["account_id"], order)}
    accounts_df["account_id"] = accounts_df["account_id"].map(remap)
    txns_df["account_id"] = txns_df["account_id"].map(remap)
    accounts_df = accounts_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    txns_df = txns_df.sort_values("timestamp").reset_index(drop=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merchants.to_csv(out / "merchants.csv", index=False)
    accounts_df.to_csv(out / "accounts.csv", index=False)
    txns_df.to_csv(out / "transactions.csv", index=False)

    joined = txns_df.merge(accounts_df[["account_id", "is_ring"]], on="account_id")
    cb_rate = joined.groupby("is_ring")["status"].apply(lambda s: (s == "chargeback").mean())

    print(f"accounts: {len(accounts_df)} ({int(accounts_df['is_ring'].sum())} ring, "
          f"{int((~accounts_df['is_ring']).sum())} legit)")
    print(f"transactions: {len(txns_df)}")
    print(f"rings: {args.n_rings}")
    ring_accounts_df = accounts_df[accounts_df["is_ring"]]
    print(f"ring archetype breakdown (accounts):\n{ring_accounts_df['archetype'].value_counts()}")
    print(f"chargeback rate by is_ring:\n{cb_rate}")


if __name__ == "__main__":
    main()
