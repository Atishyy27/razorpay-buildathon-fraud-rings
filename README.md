# Fraud-ring detector

**Razorpay AI Buildathon, track 02 (AI Risk Manager).** Detects bust-out and
chargeback fraud rings: groups of accounts that share identifiers, build trust
with ordinary purchases, then cash out and charge back. Defense-only end to
end, every action is a human review queue entry, nothing is ever auto-blocked
or auto-refunded.

**The honest headline.** Against the best of 594 hand-tuned rules, on 12
independently generated worlds the model has never seen, it is cheaper in 11 of
12 and puts **3.2x fewer clusters in the review queue** at the same catch rate
(44.6 false positives per world against 143.6, with recall 0.97 against 0.96).
Precision is the analyst's workload, and that is the number a fraud-ops team
actually feels.

Getting to numbers worth stating took **seven** rediscoveries that the problem
was too easy. One of them was a hand-written rule beating the model outright.
Three of them were columns that separated the two classes perfectly and that
the model never even used. All seven are in [DECISIONS.md](DECISIONS.md), along
with a direct answer to the obvious objection: that the data was changed until
the model won.

---

## Headline results

Canonical run: `--seed 42 --n-rings 60 --n-legit 12000`. **1,428 multi-account
clusters, 60 of them rings, 4.2% positive.** Costs use a modelled 30:1 ratio
between a missed ring and a false alarm; the rupee figures are invented and
section 6 shows how much the conclusion depends on them.

### 1. Does the model beat a hand-written rule?

This is the question that decides whether any of this was worth building. The
rules are tuned **in-sample, with access to all 9 features the model uses**, in
both threshold directions plus every two-column AND/OR pair. The model is
scored out-of-fold. The comparison is tilted toward the rule on purpose.

| | precision | recall | false positives | est. cost |
|---|---|---|---|---|
| Best of 594 hand-written rules (in-sample) | 0.30 | 1.00 | 139 | Rs.69,500 |
| Tier 1, logistic, 2 features (out-of-fold) | 0.45 | 0.87 | 64 | Rs.152,000 |
| **Tier 2, GBDT, 9 features (out-of-fold)** | **0.82** | **0.97** | **13** | **Rs.36,500** |

The rule reaches perfect recall by queueing 139 legit households for review.
Tier 2 reaches 0.97 with 13. **Tier 1 is beaten by a hand-written rule on cost**
and is kept in the repo saying so, because reporting only the tier that wins
would hide the more useful finding: two linear features are not enough here.

### 2. Does it survive data it has never seen?

`holdout.py` trains on one generated world and tests on a **separate world from
a different seed**, with its own accounts, merchants, rings and noise. The
threshold is tuned on out-of-fold training scores only. The rule is selected on
the same training world and applied to the same test world. Twelve pairs:

| | precision | recall | est. cost | false positives / world |
|---|---|---|---|---|
| Best hand-tuned rule | 0.32 +/- 0.10 | 0.96 +/- 0.05 | Rs.110,542 +/- 28,614 | 143.6 |
| **Tier 2 model** | **0.61 +/- 0.17** | **0.97 +/- 0.05** | **Rs.46,042 +/- 35,068** | **44.6** |

**Model cheaper in 11 of 12 worlds**, 95% Clopper-Pearson CI on the win rate
**[62%, 100%]**. Cost advantage 2.4x on the ratio of means, 2.8x on the
per-pair median, ranging 0.7x to 6.1x, so **in one world the rule wins
outright**. The review queue is 3.2x smaller.

Held-out precision (0.61) is well below the in-fold figure (0.82). The
independent-world number is the one to trust.

### 3. How confident are these numbers?

| metric | point | interval |
|---|---|---|
| Tier 2 precision (out-of-fold) | 0.82 | 95% CI [0.73, 0.90] |
| Tier 2 recall (out-of-fold) | 0.97 | 95% CI [0.91, 1.00] |
| Tier 2 PR-AUC (threshold-free) | 0.969 | vs 0.042 for random |
| Alert rate (share of clusters queued) | 5.0% | 71 of 1,428 |

A bootstrap CI on recall is a **tautology whenever the sample has zero false
negatives**, because the bootstrap resamples a fixed prediction vector and
never refits. An earlier version of this repo reported [1.00, 1.00] for exactly
that reason. It is a real interval here only because there are real misses.

### 4. Which rings does each tier catch?

| archetype | what it does | n | Tier 1 | Tier 2 |
|---|---|---|---|---|
| loud | shares device, card and bank; bursts hard | 24 | 1.00 | 0.96 |
| sophisticated | shares only the cash-out bank | 13 | 0.92 | 1.00 |
| quiet | shares only the bank, sized 2-3 like a household | 9 | **0.22** | **0.89** |
| slow_drain | never bursts, cashes out over 20 to 45 days | 14 | 1.00 | 1.00 |

Tier 2 now misses real rings, including one `loud` cluster, which is what an
honest 0.97 recall looks like.

### 5. Two ablations

**The coordination features earn their place, but only once the data is
clean.** `burst_synchrony`, `high_resale_share` and `formation_window_days` are
the three signals a threshold rule cannot usefully combine. Measured with
`ablation.py` on 6 matched world pairs, identical seeds and model:

| model | precision | recall | cost | beats rule |
|---|---|---|---|---|
| all 9 features | 0.65 | 0.96 | Rs.54,833 | 5 of 6 |
| minus the 3 coordination features | 0.61 | 0.93 | Rs.84,083 | 3 of 6 |

Dropping them costs 4.6 points of precision, 3.3 of recall, and Rs.29,250, and
halves the win rate. Worth knowing: on an earlier, leakier version of this
dataset the same ablation said they were worth **nothing**. They only started
paying once the construction artefacts were removed, because before that the
leaks were doing their work for them.

**Removing the archetype added at a convenient moment costs the model most of
its cost advantage, but not its queue advantage.** `slow_drain` was added right
after watching the rule beat the model 5 times out of 5, which is the change
most exposed to a charge of gaming the benchmark. Regenerating every world
without it (`holdout.py --exclude-archetype slow_drain`, 6 pairs):

| | precision | recall | cost | FPs / world |
|---|---|---|---|---|
| Best hand-tuned rule | 0.45 | 0.99 | Rs.49,333 | 73.7 |
| **Tier 2 model** | **0.74** | 0.96 | **Rs.48,667** | **27.3** |

Read that honestly: **on cost the two are a tie** (Rs.48,667 against
Rs.49,333, a 1.4% difference), though the model still wins 5 of 6 worlds on
count. The queue advantage survives intact at 2.7x. So `slow_drain` does carry
much of the cost advantage, and the concession in DECISIONS.md stands. What
does not depend on it is the review-volume result.

### 6. How much does the invented cost ratio matter?

| FN:FP ratio | false positives | misses | cost |
|---|---|---|---|
| 1:1 | 0 | 7 | Rs.3,500 |
| 3:1 | 4 | 3 | Rs.6,500 |
| 10:1 | 13 | 2 | Rs.16,500 |
| 30:1 | 13 | 2 | Rs.36,500 |
| 100:1 | 48 | 1 | Rs.74,000 |

The operating point moves a lot with the ratio, from flagging nothing at 1:1 to
48 false positives at 100:1. So the invented constants **do** matter, which is
the honest reason to state them loudly rather than bury them. What does not
depend on them at all is the review-queue comparison in section 2.

---

## How it works

```
generator.py       synthetic accounts and transactions. 4 ring archetypes
                   inside a legit population built to overlap them:
                   households sharing any of device / joint bank account /
                   supplementary card, shared office and cafe device hubs,
                   big spenders, dispute-prone customers
       |
       v
graph_features.py  shared device / card / bank-account graph, connected
                   components, 9 per-cluster features
       |
       v
detect.py          Tier 1: logistic regression, 2 features, readable
                   Tier 2: gradient-boosted trees, 9 features
                   5-fold stratified CV, cost-optimal threshold, bootstrap
                   CIs, PR-AUC, permutation importance
       |
       v
baseline.py        594 hand-written rules with the same feature access,
                   tuned in-sample. The check on whether the model earned
                   its place
       |
       v
explain.py         templated evidence summary per cluster, no LLM call
       |
       v
respond.py         queues clusters above the cost-optimal threshold for
                   HUMAN review with a per-cluster case file. Never
                   auto-blocks, never auto-refunds
```

`holdout.py` runs the independent-world test, `ablation.py` the feature
ablation. `run_pipeline.py` runs every stage. `app.py` is a Streamlit
dashboard.

Reports live in `results/`. Generated data goes to `data/`, which is
gitignored: a stale committed copy is one the README can silently disagree
with.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generator.py --seed 42 --n-rings 60         # writes data/
python run_pipeline.py                              # results/eval_report.json, baseline_report.json
python holdout.py --pairs 12                        # results/holdout_report.json   (~12 min)
python ablation.py --pairs 6                        # results/ablation_report.json  (~12 min)
python holdout.py --pairs 6 --exclude-archetype slow_drain \
    --out results/archetype_ablation_report.json    # (~6 min)
streamlit run app.py
```

Those five commands regenerate every file in `results/` that this README
quotes.

## Tests

```bash
pytest
```

**61 tests.** Unit tests per stage, an end-to-end suite that runs the real
scripts as subprocesses, and a leakage-guard suite.

`tests/test_end_to_end.py` asserts what only exists across stage boundaries:

- no action outside `{flag_for_review, no_action}` can ever be emitted
- **the queue that ships is exactly the queue that was measured** (regression
  test for a real bug, DECISIONS.md Bug 5)
- every queued cluster carries a reviewable evidence block
- ground truth is never a model feature
- no explanation accuses a cluster the pipeline did not flag

`tests/test_no_leakage.py` audits the generated **data** rather than the model,
because three separate perfect-separation leaks lived in columns the model
never used. It brute-forces every numeric column for a near-oracle threshold
and fails the build if any single column reaches precision 0.95 at recall 0.80.

## Honest limits

- **Everything is synthetic and we chose every difficulty knob.** Seven times
  we found one set too low. The independent-world test controls for overfitting
  a single draw. It cannot show the generator resembles Razorpay traffic, and
  no synthetic result can.
- **4.2% ring prevalence is still far above real organized fraud**, which sits
  well under 1%. Precision would fall further at a true base rate.
- **The graph has no hub problem because every identifier is a fresh UUID.**
  Real data grows giant connected components through shared office devices,
  aggregator nodal accounts and card BINs, and there is no hub pruning, degree
  capping or community detection here. This is the largest single gap between
  this and something deployable.
- **100% of planted fraud is graph-linkable by construction**, so reported
  recall is recall over rings that form a cluster, not over fraud in general.
- **The cost model prices Razorpay's loss, never the customer's.** A wrongly
  queued household or office cluster has no SLA, no appeals path and no
  fairness check here. Defense-only bounds the harm, it does not remove it.
- **No label-latency or retraining story.** Chargebacks mature over weeks to
  months; this pipeline assumes labels exist at train time.
- **In one of 12 held-out worlds the rule beat the model outright**, and
  without `slow_drain` the cost advantage is a statistical tie.
- **Tier 2's headline threshold is chosen on the same out-of-fold predictions
  it is scored on.** A nested check showed ~0 measured bias, but the held-out
  numbers in section 2 are the ones without the property at all.
