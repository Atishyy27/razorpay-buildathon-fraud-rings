"""
Feature ablation: what are the coordination features actually worth?

This exists because an earlier version of the README claimed that
`burst_synchrony`, `high_resale_share` and `formation_window_days` were *why*
the model beats a hand-written rule. Adversarial review pointed out that the
repo's own permutation importance did not support that, so the claim was
settled by experiment instead of by rewording.

The design is a matched pair: identical world seeds, identical model
configuration, identical rule baseline, and the only thing that changes is
which features the model may use. Anything that differs between the two arms
is therefore attributable to the features rather than to the draw.

Run: python ablation.py --pairs 6
"""
import argparse
import json
from pathlib import Path

from costs import CostModel
from detect import TIER2_FEATURES
from holdout import run_pairs

# The three signals whose value is in question. They are the coordination
# features: what the cluster buys, whether its accounts act together, and how
# fast it was stood up.
COORDINATION_FEATURES = ["burst_synchrony", "high_resale_share", "formation_window_days"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--n-rings", type=int, default=60)
    parser.add_argument("--n-legit", type=int, default=12000)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--drop", nargs="*", default=COORDINATION_FEATURES)
    parser.add_argument("--out", type=str, default="results/ablation_report.json")
    args = parser.parse_args()

    cost_model = CostModel()
    kept = [f for f in TIER2_FEATURES if f not in set(args.drop)]

    arms = {}
    for name, features in (("all_features", list(TIER2_FEATURES)),
                            ("minus_coordination", kept)):
        print(f"\n=== arm: {name} ({len(features)} features) ===")
        summary, _ = run_pairs(
            pairs=args.pairs, n_rings=args.n_rings, n_legit=args.n_legit,
            base_seed=args.base_seed, cost_model=cost_model, features=features,
        )
        arms[name] = summary

    full, abl = arms["all_features"], arms["minus_coordination"]
    report = {
        "question": ("Are burst_synchrony, high_resale_share and "
                      "formation_window_days the reason the model beats a "
                      "hand-written rule?"),
        "protocol": (f"{args.pairs} matched train/test world pairs from base seed "
                      f"{args.base_seed}. Identical seeds, identical model config, "
                      f"identical rule baseline. Only the feature set differs."),
        "dropped": sorted(args.drop),
        "arms": arms,
        "delta_from_dropping": {
            m: abl["model"][m]["mean"] - full["model"][m]["mean"]
            for m in ("precision", "recall", "cost")
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=float))

    print("\n=== ablation summary ===")
    for name, s in arms.items():
        w = s["model_beats_rule"]
        print(f"{name:>20}: precision {s['model']['precision']['mean']:.2f}  "
              f"recall {s['model']['recall']['mean']:.2f}  "
              f"cost Rs.{s['model']['cost']['mean']:,.0f}  "
              f"beats rule {w['wins']}/{s['pairs']}")
    d = report["delta_from_dropping"]
    print(f"dropping {sorted(args.drop)} costs "
          f"{-d['precision']:+.2f} precision, {-d['recall']:+.2f} recall, "
          f"Rs.{d['cost']:+,.0f} cost")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
