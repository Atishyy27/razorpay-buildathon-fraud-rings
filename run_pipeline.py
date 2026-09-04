"""
Runs the full pipeline end to end: cluster features, then detect (train+eval
both tiers), then explain, then respond. Assumes data/accounts.csv and
data/transactions.csv already exist, run generator.py first if not.
"""
import subprocess
import sys

# baseline.py runs right after detect.py on purpose: the rule comparison is
# not an appendix, it is the check that decides whether the model deserved to
# be built. Running it every time means a regression that makes the model
# worse than a rule shows up in the same output as the metrics.
STEPS = [
    [sys.executable, "graph_features.py"],
    [sys.executable, "detect.py"],
    [sys.executable, "baseline.py"],
    [sys.executable, "explain.py"],
    [sys.executable, "respond.py"],
]


def main():
    for step in STEPS:
        print(f"\n=== {' '.join(step)} ===")
        result = subprocess.run(step)
        if result.returncode != 0:
            print(f"FAILED at {' '.join(step)}, stopping.")
            sys.exit(1)


if __name__ == "__main__":
    main()
