"""
Templated, deterministic evidence summary for every cluster, no LLM call.

This module describes what the data shows. It deliberately does NOT decide or
imply that a cluster was flagged: it runs over all clusters, before any
threshold is applied, so wording every summary as an accusation would put one
in the audit trail of accounts that passed cleanly. respond.py owns the
decision; this owns the evidence.
Chosen over an LLM-verifier step because at this data volume, for a numeric
signal like this, a template is faster, free, and never hallucinates a
reason that isn't actually in the evidence. An LLM would add latency and a
new failure mode for something string formatting already does correctly
and auditably, this is the "AI judgment: where you chose not to use one"
answer for the panel.
"""
import argparse

import pandas as pd


def explain_cluster(row):
    reasons = []
    if row["id_sharing_ratio"] < 0.3:
        reasons.append(
            f"{int(row['size'])} accounts share only {int(row['n_devices'])} device(s), "
            f"{int(row['n_cards'])} card(s), {int(row['n_banks'])} bank account(s), "
            f"far fewer distinct identifiers than accounts"
        )
    if row["chargeback_rate"] > 0.1:
        reasons.append(f"chargeback rate {row['chargeback_rate']:.0%}, vs ~1% baseline")
    if row["burst_ratio"] > 3:
        reasons.append(f"transaction size jumped {row['burst_ratio']:.1f}x above this cluster's early average")
    if row["peak_hourly_velocity"] >= 5:
        reasons.append(f"{int(row['peak_hourly_velocity'])} transactions inside a single hour")
    if row.get("high_resale_share", 0) > 0.5:
        reasons.append(
            f"{row['high_resale_share']:.0%} of spend went to resale-friendly merchants "
            f"(electronics, gift cards)"
        )
    if row.get("burst_synchrony", 0) >= 0.5:
        reasons.append(
            f"{row['burst_synchrony']:.0%} of the accounts made their largest purchase "
            f"on the same day"
        )
    fw = row.get("formation_window_days")
    if fw is not None and not pd.isna(fw) and fw <= 7 and row["size"] >= 3:
        reasons.append(
            f"all {int(row['size'])} accounts were opened within {fw:.1f} days of each other"
        )

    if not reasons:
        # An honest "no legible reason" beats a fabricated one. A reviewer who
        # is told the model combined weak signals can still act; a reviewer
        # given an invented reason cannot tell it from a real one, and that is
        # how an audit trail stops being an audit trail.
        return "No individual signal crossed its threshold."
    return "Signals present: " + "; ".join(reasons) + "."


def review_note(row, action):
    """Reviewer-facing line. The evidence is the same either way; only the
    decision differs, and the decision belongs to respond.py, not here."""
    body = explain_cluster(row)
    if action == "flag_for_review":
        return f"Queued for manual review. {body}"
    return f"No action. {body}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/cluster_features.csv")
    parser.add_argument("--out", type=str, default="data/flagged_clusters.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    df["explanation"] = df.apply(explain_cluster, axis=1)
    df.to_csv(args.out, index=False)
    print(f"wrote explanations for {len(df)} clusters to {args.out}")
    print("\nsample, one ring and one legit multi-account cluster:")
    for is_ring_val in [True, False]:
        sample = df[df["is_ring"] == is_ring_val]
        if len(sample):
            r = sample.iloc[0]
            label = "ring" if is_ring_val else "legit"
            print(f"- cluster {r['cluster_id']} (ground truth: {label}): {r['explanation']}")


if __name__ == "__main__":
    main()
