"""
Streamlit dashboard over the pipeline's output: flagged clusters, risk
scores, explanations, and a network view of a selected cluster. Reads what
run_pipeline.py already wrote to data/, doesn't recompute anything itself.

Run: streamlit run app.py
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st


PIPELINE = [
    ["generator.py", "--seed", "42", "--n-rings", "60", "--n-legit", "12000"],
    ["graph_features.py"],
    ["detect.py"],
    ["baseline.py"],
    ["explain.py"],
    ["respond.py"],
]


def bootstrap_if_missing(data_dir="data"):
    """Generate the dataset on first run if it is not on disk.

    data/ is gitignored on purpose (a stale committed copy is one the README
    can silently disagree with), which means a fresh clone or a hosted deploy
    has nothing to read. Rather than commit generated data, the app builds it
    once and Streamlit caches the result for the session.

    It uses the CANONICAL parameters (--seed 42 --n-rings 60 --n-legit 12000),
    not a smaller world, so the figures on screen are the same figures the
    README quotes. A faster cold start with a smaller dataset would have shown
    numbers that quietly disagree with the documentation, which is the exact
    failure the gitignore on data/ exists to prevent.
    """
    needed = ["flagged_clusters.csv", "actions.json", "accounts.csv"]
    if all((Path(data_dir) / f).exists() for f in needed):
        return False
    status = st.status("First run: generating the dataset and scoring it. "
                       "About a minute, then it is cached.", expanded=True)
    for step in PIPELINE:
        status.write(f"running {step[0]}")
        result = subprocess.run(
            [sys.executable, *step], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            status.update(label=f"{step[0]} failed", state="error")
            st.code(result.stderr[-2000:])
            st.stop()
    status.update(label="Dataset ready.", state="complete")
    return True


@st.cache_data(show_spinner=False)
def load_data(data_dir="data", reports_dir="results"):
    data = Path(data_dir)
    flagged = pd.read_csv(data / "flagged_clusters.csv")
    actions_df = pd.DataFrame(json.loads((data / "actions.json").read_text()))
    eval_report = json.loads(Path(reports_dir, "eval_report.json").read_text())
    accounts = pd.read_csv(data / "accounts.csv")
    return flagged, actions_df, eval_report, accounts


def load_optional(path):
    """Reports that may not have been generated yet. The dashboard degrades to
    a note rather than crashing, so a fresh clone still renders."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def merge_flagged_with_actions(flagged, actions_df):
    merged = flagged.merge(actions_df[["cluster_id", "risk_score", "action"]], on="cluster_id")
    return merged.sort_values("risk_score", ascending=False)


def cluster_accounts_subgraph(cluster_row, accounts):
    ids = cluster_row["account_ids"].split("|")
    sub = accounts[accounts["account_id"].isin(ids)]
    g = nx.Graph()
    g.add_nodes_from(ids)
    for attr in ["device_id", "card_fingerprint", "bank_account_id"]:
        for _, group in sub.groupby(attr):
            members = group["account_id"].tolist()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    g.add_edge(members[i], members[j], attr=attr)
    return g


def render_cluster_graph(g):
    fig, ax = plt.subplots(figsize=(5, 4))
    pos = nx.spring_layout(g, seed=42)
    nx.draw(g, pos, ax=ax, with_labels=True, node_color="#f28b82",
            node_size=800, font_size=7, edge_color="#888888")
    return fig


def main():
    st.set_page_config(page_title="Fraud-ring detector", layout="wide")
    st.title("Fraud-ring detector, track 02 (AI Risk Manager)")
    st.caption("Cost figures and thresholds are illustrative, see DECISIONS.md.")

    bootstrap_if_missing()
    flagged, actions_df, eval_report, accounts = load_data()
    merged = merge_flagged_with_actions(flagged, actions_df)

    t2 = eval_report["tier2"]
    ci = t2.get("ci") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total clusters", len(merged))
    c2.metric("Queued for review", int((merged["action"] == "flag_for_review").sum()),
              help="alert rate: what a fraud-ops team has to actually staff")
    # precision and recall are shown with their bootstrap intervals, because at
    # 60 positive clusters a bare point estimate overstates what was measured
    c3.metric("Precision (out-of-fold)", f"{t2['precision']:.2f}",
              help=(f"95% CI [{ci['precision']['lo']:.2f}, {ci['precision']['hi']:.2f}]"
                    if ci.get("precision") else None))
    c4.metric("Recall (out-of-fold)", f"{t2['recall']:.2f}",
              help=(f"95% CI [{ci['recall']['lo']:.2f}, {ci['recall']['hi']:.2f}]"
                    if ci.get("recall") else None))
    c5.metric("PR-AUC", f"{t2.get('average_precision', float('nan')):.3f}",
              help="threshold-free, so it does not depend on the tuned cutoff")

    # The two results that actually carry this submission are the rule
    # comparison and the independent-world test. Showing only the model's own
    # metrics would let a reviewer leave without seeing either.
    baseline = load_optional("results/baseline_report.json")
    holdout = load_optional("results/holdout_report.json")

    st.subheader("Did the model earn its place?")
    left, right = st.columns(2)

    with left:
        st.markdown("**vs the best hand-written rule** (rule tuned in-sample "
                    "with the same features, model scored out-of-fold)")
        if baseline:
            b = baseline["best"]
            st.dataframe(pd.DataFrame([
                {"approach": f"best of {baseline['n_rules_evaluated']} rules",
                 "precision": round(b["precision"], 2), "recall": round(b["recall"], 2),
                 "false positives": b["fp"], "est. cost": f"Rs.{b['cost']:,.0f}"},
                {"approach": "Tier 2 model",
                 "precision": round(t2["precision"], 2), "recall": round(t2["recall"], 2),
                 "false positives": t2["fp"], "est. cost": f"Rs.{t2['est_cost']:,.0f}"},
            ]), width='stretch', hide_index=True)
            st.caption(f"best rule: `{b['rule']}`")
        else:
            st.info("run `python baseline.py` to generate this comparison")

    with right:
        st.markdown("**On worlds the model never saw** (train on one generated "
                    "world, test blind on another from a different seed)")
        if holdout:
            h = holdout["summary"]
            st.dataframe(pd.DataFrame([
                {"approach": who,
                 "precision": f"{h[who]['precision']['mean']:.2f} +/- {h[who]['precision']['std']:.2f}",
                 "recall": f"{h[who]['recall']['mean']:.2f} +/- {h[who]['recall']['std']:.2f}",
                 "est. cost": f"Rs.{h[who]['cost']['mean']:,.0f}"}
                for who in ("rule", "model")
            ]), width='stretch', hide_index=True)
            w = h["model_beats_rule"]
            st.caption(f"model cheaper in {w['wins']} of {h['pairs']} independent worlds "
                       f"({w['ties']} tie, {w['losses']} loss)")
        else:
            st.info("run `python holdout.py --pairs 5` to generate this test")

    st.subheader("Flagged clusters")
    display_cols = ["cluster_id", "archetype", "size", "risk_score", "action", "explanation"]
    st.dataframe(merged[display_cols], width='stretch', hide_index=True)

    st.subheader("Inspect a cluster")
    cluster_id = st.selectbox("cluster_id", merged["cluster_id"].tolist())
    row = merged[merged["cluster_id"] == cluster_id].iloc[0]
    st.write(row["explanation"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.json({
            "is_ring": bool(row["is_ring"]),
            "archetype": row["archetype"],
            "risk_score": float(row["risk_score"]),
            "action": row["action"],
            "id_sharing_ratio": float(row["id_sharing_ratio"]),
            "chargeback_rate": float(row["chargeback_rate"]),
            "burst_ratio": float(row["burst_ratio"]),
        })
    with col_b:
        g = cluster_accounts_subgraph(row, accounts)
        st.pyplot(render_cluster_graph(g))


if __name__ == "__main__":
    main()
