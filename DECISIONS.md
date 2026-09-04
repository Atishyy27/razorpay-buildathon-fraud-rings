# Decisions, and what broke

Working notes on the choices behind this build and the real failures hit while
making it. This is the source material for the application's "what broke, and
how you got out" field, and for defending the design in the panel.

The through-line: **seven separate times, this project produced a result that
looked good and was not.** Each time the fix was to make the measurement harder
rather than to make the model better. Bug 4 is the one worth reading, because
that is where the hand-written rule beat the model and the honest answer was to
accept it and find out why. Bug 5 is the one worth noting, because it was found
by an outside reviewer after everything above was already written.

A note on what is checkable. The narrative order of Bugs 1 through 4 predates
this repo's first commit, so a reader cannot verify the sequence from git
history; the final state of the code is verifiable and every number quoted is
reproducible from `results/`. Bug 5 and everything in "What adversarial review
overturned" happened after the code was in its current shape and can be read
off the diffs.

---

## Why bust-out and chargeback rings, not a generic "abuse ring"

Track 02 names fraud, returns and chargebacks. A ring that builds trust with
small ordinary purchases, then cashes out on resale-friendly categories, then
charges most of it back, is a real documented pattern (bust-out fraud), not an
invented one. It gives a concrete measurable loss signal, the chargeback,
instead of a vague "looks suspicious".

## Why a gradient-boosted tree, not a GNN

With ~60 planted rings there is not enough positive-class data for a GNN to
learn from without overfitting or heavy synthetic augmentation. A tree on
hand-engineered graph and behavioural features is the right-sized tool for
this data volume, and permutation importance on it doubles as the audit trail.
This is a deliberate judgment call, not a shortcut, and it is the answer to
"why not a GNN".

## Why a templated explanation, not an LLM call

The audit-trail explanation is generated from feature values with plain string
formatting. A template is faster, free, deterministic, and cannot hallucinate
a reason that is not in the evidence. An LLM adds latency and a new failure
mode for something formatting already does correctly and auditably.

This is the "where you chose not to use AI" answer, and there is a second
half to it that matters more: an explanation a reviewer cannot distinguish
from an invented one is worse than no explanation. When no threshold fires,
the module says exactly that rather than composing a plausible sentence.

## Why the threshold is chosen by cost, not left at 0.5

0.5 is the accuracy-optimal cutoff. On a 14%-positive problem where a missed
ring is modelled at 30x the cost of a false alarm, accuracy is the wrong
objective, and 0.5 is simply the wrong number. `costs.py` picks the cheapest
cutoff from the observed score distribution, and `respond.py` reads that same
cutoff back out of the eval report rather than hardcoding one, so the queue
that ships is the system that was measured.

Ties break toward the higher threshold, meaning toward flagging fewer people.
For a defense-only system, two equally priced options should not have their
tie broken by an arbitrary sort order when one of them puts more people in a
review queue.

---

# The seven times this looked good and was not

## Bug 1: the classifier silently predicted nothing

First full pipeline run gave Tier 2 precision 0 and recall 0, worse than the
simpler Tier 1, which should never happen for the richer tier.

Cause: `HistGradientBoostingClassifier`'s default `min_samples_leaf=20` on a
training set of a few dozen rows means it can never split, so it predicts the
majority class every time. Fixed with `min_samples_leaf=2`, tuned for this
data volume.

This is the kind of bug that reads as "the model just isn't very good" if you
never check hyperparameters against your actual data size.

## Bug 2, a design flaw not a code bug: the problem was trivially easy

After fixing Bug 1, both tiers scored precision 1.00 and recall 1.00. That is
a red flag, not a win. Rings originally shared every identifier and had an 80%
chargeback rate; nothing sat between them and legit accounts. A model cannot
fail to separate two classes that do not overlap, so "perfect" proved nothing
about the model and only that the data was too easy.

Fix: added the `sophisticated` archetype (shares only the cash-out bank,
distinct device and card per account, 45% chargeback rate), then the `quiet`
archetype (same low sharing, but sized 2-3 like a family). At size 2-3,
`id_sharing_ratio` is *mathematically identical* to a legit family cluster:
one shared attribute out of three, divided by size, and the arithmetic does
not care which attribute it is. That makes quiet rings genuinely invisible to
that feature rather than merely harder to see.

## Bug 3: a two-line rule matched the model exactly

With the harder archetypes in place, `baseline.py` was written to answer the
question that decides whether any of this was worth building: does the model
beat a rule a risk analyst would write in an afternoon?

It did not. The best rule scored precision 0.98, recall 1.00, cost Rs.500,
identical to Tier 2 on every figure, and it caught quiet rings too.

The cause was one level below Bug 2. The archetypes had been made harder, but
the *legit* population was still uniform: no legit account ever bursted, and
the baseline chargeback rate was ~1%. Any burst threshold separated the
classes cleanly, so the ML had nothing to add.

Fix, in the generator rather than the model:
- `big_spender` legit accounts (18%) make occasional large purchases drawn
  from the same amount distribution the rings use, at the same resale-friendly
  merchants. A legit spike and a fraud spike became indistinguishable on
  amount alone.
- `dispute_prone` legit accounts (12%) charge back at 18%, modelling genuine
  consumer disputes and friendly fraud.
- Family members are created over a spread of days instead of sharing one
  identical timestamp, which had been handing the model a free separator that
  real signup data does not contain.
- Class balance moved to something a risk team would recognise: 14% of
  multi-account clusters are rings, not 58%.

## Bug 4: the rule beat the model on held-out data, and the first fix made it worse

This is the important one.

`holdout.py` trains on one generated world and tests on a completely separate
one from a different seed. Cross-validation only holds out folds of a single
dataset, so every fold shares that dataset's quirks; an independent world does
not. On five such pairs:

> **model cheaper than the tuned rule in 0 of 5 worlds.**
> model Rs.43,100 mean, rule Rs.12,900 mean.

The in-fold result had not survived. First hypothesis: the threshold was being
tuned on in-sample training probabilities, which a fitted tree pushes to the
extremes, so the transferred cutoff lands in a probability region the model
never reproduces on unseen data. That is a real bug and it was fixed
(threshold now tuned on out-of-fold training scores).

**The fix made it worse: 1 of 5 became 0 of 5.** Which was the useful result,
because it eliminated the threshold as the cause.

The actual cause: every archetype shared one mechanism. All of them bursted
fast, on resale merchants, within a tight window. So the rule
`peak_hourly_velocity >= 3 AND high_resale_share >= 0.5` was not approximating
the model, it was approximating **the generator**. A rule that encodes the
data-generating process cannot be beaten by a model that has to learn it from
finite samples. The model was not underperforming; the population was
unrealistically homogeneous, and a homogeneous population is one rules are
guaranteed to win on.

Fix: added `slow_drain` (25% of rings), a real documented bust-out variant
that cashes out steadily over 20 to 45 days, mixes ordinary merchants in, and
never spikes. It stays under every velocity and amount threshold. No
two-condition rule now covers both it and the fast-burst archetypes.

Result on the same five independent world pairs:

| | precision | recall | cost |
|---|---|---|---|
| Best hand-tuned rule | 0.61 +/- 0.26 | 0.90 +/- 0.10 | Rs.113,300 |
| Tier 2 model | 0.79 +/- 0.21 | 0.99 +/- 0.01 | Rs.17,000 |

**Model cheaper in 5 of 5.**

The honest statement of what this shows: the model's advantage is *specifically*
its ability to span heterogeneous fraud mechanisms with one decision function.
On a population where every ring behaves the same way, a rule is genuinely the
better engineering choice, and this repo can demonstrate that case as well as
the other one.

## Bug 5: the queue that shipped was not the queue that was measured

Found by adversarial review after the above was already written, which is
the point of running one.

`respond.py` fitted the model on every cluster and then scored those same
clusters **in-sample**, while applying a threshold that `detect.py` had tuned on
**out-of-fold** probabilities. This is the identical mechanism as Bug 4: a
fitted tree pushes its own training rows toward the extremes, so an out-of-fold
cutoff lands somewhere else entirely on an in-sample score distribution. It had
been fixed in `holdout.py` and missed in the one file that represents what
actually ships.

Measured, on two independent seeds:

| seed | alert rate reported by the eval | alert rate actually shipped |
|---|---|---|
| 7 | 17.2% (77 of 448) | 16.5% (74 of 448) |
| 99 | 26.8% (66 of 246) | 24.8% (61 of 246) |

Not huge, but systematic and in a predictable direction, and it meant the
precision, recall and alert-rate figures shown to a reviewer described a
slightly different system than the one writing `actions.json`.

Fix: `respond.py` now scores out-of-fold with the same model config, seed and
fold count as `detect.py`, because every cluster in this batch is one the model
trained on. In production the deployed model scores clusters it has never seen,
which is what out-of-fold scoring stands in for. The old behaviour is kept
behind `--in-sample` so the defect can be reproduced. Measured and shipped
counts now match exactly (67 and 67), and
`test_shipped_queue_is_exactly_the_measured_queue` fails the build if they ever
diverge again.

## Bug 6: the sharing structure WAS the label

The worst one, and it survived every earlier round because every earlier round
looked at behaviour rather than at the graph underneath it.

An adversarial reviewer measured something nobody here had thought to measure:
how often *legit* accounts share each identifier type. The answer was that
**0 of 3000 legit accounts shared a card fingerprint, and 0 shared a bank
account.** Only devices were ever shared, and only in pairs or triples. Every
ring, meanwhile, shares a cash-out bank by construction.

The consequence was fatal. The boolean `n_cards < size OR n_banks < size`,
built from raw counts one layer below the nine declared features, scored:

> **precision 1.00, recall 1.00. 60 of 60 rings, 0 false positives.**

So the entire behavioural layer, all four archetypes, the whole
model-versus-rule question, was decorating a dataset where a one-line boolean
already separated the classes perfectly. The `quiet` archetype in particular
was never hard: `id_sharing_ratio` discards *which* attribute is shared, which
is what made quiet rings look identical to households, but the raw counts kept
the information and only rings ever shared a bank.

Two related giveaways came out of the same review. Legit clusters were capped
at size 3 while rings ran to 12, so `size >= 5` was itself close to a label
(single-feature AUC 1.00 for three of four archetypes). And ring prevalence was
14% of clusters and 11.8% of all accounts, orders of magnitude above real
organized fraud, which inflates precision directly.

Fix, all in the generator:
- Households can share a **joint bank account** (45%) and a **supplementary
  card** (35%) as well as a device. All three are ordinary in Indian retail
  banking: spouse and HUF accounts, add-on cards on one primary.
- Added **shared-device hubs** (offices, cafes) of 5 to 12 unrelated accounts,
  so legit clusters span 2 to 12 exactly as rings do and size stops being a
  label.
- Legit population raised to 12,000, dropping ring prevalence to **4.1%** of
  clusters.

### Bug 6b: the same bug, twice more, in columns nobody had audited

A second review round found the identical shape in three more places, none of
which was ever a model feature. That is the pattern worth carrying away:
**auditing the features you feed the model does not audit the columns you
discarded, and a discarded column can still be the answer.**

- **`cluster_id`, AUC 1.00.** Rings were appended after every legit account, so
  connected components enumerated in insertion order gave legit clusters ids
  0-1400 and rings 1401-1460. A pure row-order artefact that separated the
  classes perfectly.
- **`account_id`, AUC 1.00.** Legit ids were `A00042`, ring ids `R07-003`.
  Different prefix, different length, and that string ships verbatim inside
  `cluster_features.account_ids`.
- **`n_devices`, AUC 0.93, rule `n_devices >= 2` at precision 1.00 and recall
  0.87.** The Bug 6 fix made card and bank sharing probabilistic for legit
  households but left device sharing unconditional, so 100% of legit clusters
  had exactly one device while three of four ring archetypes share none. The
  original bug, relocated onto the third identifier.

Fixes: household linking is now drawn independently across all three
identifiers, with at least one guaranteed, so a couple with a joint bank
account but separate phones and separate cards is finally representable.
Account ids are reassigned from one shuffled pool (`ACC000123`) and the rows
are shuffled, so neither id format nor row order carries anything.

**All figures below are pinned to `--seed 42 --n-rings 60 --n-legit 12000`**,
1,428 clusters, 60 rings, 4.2% positive. An earlier draft quoted a leak-boolean
precision of 0.247 without naming a seed or a population size, and a later
review correctly caught that it no longer reproduced. Precision here is not a
property of the rule alone: the false-positive count depends entirely on how
many legit clusters the generator emits, so any change to prevalence or sharing
rates moves it even when nobody edits the sentence. Hence the pin.

| check | before | after |
|---|---|---|
| legit accounts sharing a card | 0.0% | 11.9% |
| legit accounts sharing a bank account | 0.0% | 13.6% |
| `n_cards < size OR n_banks < size` | precision **1.00** | precision **0.077** |
| `n_devices >= 2` | precision **1.00**, recall 0.87 | precision **0.115** |
| `cluster_id` AUC | **1.00** | 0.40 |
| account-id length AUC | **1.00** | **0.50** |

The headline numbers moved across both rounds of this fix, and not in one
direction. Tier 2 precision went 0.90, then **0.63** once prevalence became
realistic, then **0.82** once the construction artefacts were removed and the
legit population changed shape again. The held-out sweep went from 12 of 12 at
a 5.3x cost advantage to **11 of 12 at 2.4x**, with the model losing one world
outright. Held-out precision is 0.61, well below the 0.82 in-fold figure, and
the held-out number is the one that counts.

`tests/test_no_leakage.py` now pins all of this: five tests that audit the
generated data rather than the model, including one that brute-forces every
numeric column for a near-oracle threshold and fails the build if any single
column reaches precision 0.95 at recall 0.80.

---

## What adversarial review overturned

Three independent reviewers were briefed to refute this repo rather than check
it, with no knowledge of each other. What they found, and what changed:

**A claim was wrong, and the correction then flipped twice more.** The README
said the three coordination features were *why* the model beats the rule.
Review pointed out that this repo's own permutation importance contradicted it,
so it was measured rather than reworded, and the answer changed every time the
data got cleaner:

| dataset state | dropping the 3 coordination features |
|---|---|
| 14% prevalence, leaky | model wins 6 of 6 either way; features worth 16 precision points |
| 4% prevalence, still leaky | precision moves 0.01 and cost *falls*; features worth nothing |
| 4% prevalence, de-leaked | precision -4.6 pts, recall -3.3 pts, cost +Rs.29,250, wins 5 of 6 down to 3 of 6 |

The final answer is that they do earn their place, but the honest lesson is the
sequence rather than the verdict: **a feature ablation run on leaky data
measures the leak, not the feature.** While `n_devices` and the id-ordering
artefacts were still separating the classes, the coordination features had
nothing left to contribute, so they scored zero. All three results are
published because the first two were wrong for a reason worth seeing.

**The rule baseline was quietly handicapped.** "Full feature access" was false:
`size` and `n_txns`, 2 of the model's 9 features, had no rule that could
threshold them. Fixed by adding both, taking the rule family from 362 to 594
candidates. The model's win survived, but the earlier comparison was not the
fair fight it was described as.

**A reported interval was not evidence.** The bootstrap recall CI of
[1.00, 1.00] is forced by construction whenever the sample has zero false
negatives, since the bootstrap resamples a fixed prediction vector and never
refits. It is kept, with an explicit warning, and the honest recall number is
the held-out 0.99 +/- 0.01.

**The documented commands did not produce the documented results.**
`detect.py` defaulted its report into gitignored `data/`, so the "Run it"
sequence never regenerated the `results/eval_report.json` the README pointed at.
That file could only have come from an undocumented manual invocation. Reports
now default into `results/`.

**"5 of 5" was oversold.** The exact binomial lower bound on 5 wins from 5
trials is 48%, formally compatible with a coin flip. Rerun at 12 pairs, where
the lower bound is 74%, and the interval is now printed next to the tally.

Two things review checked hard and could not break: the cost-optimal threshold
logic in `costs.py` (tie-breaking, grid construction, monotonicity all verified
independently), and leakage in `holdout.py` (the threshold, the rule selection
and the model fit were each confirmed to touch training data only).

---

## Two smaller things the tests caught

**The rule baseline could not use a feature the model had.** A test asserting
feature parity between `TIER2_FEATURES` and the rule family failed on
`formation_window_days`. It needed a `<=` rule (a *short* formation window is
the suspicious one) and the rule family only generated `>=` rules, so the
baseline had been quietly handicapped. Fixed by adding an at-most rule family.
The same test did not catch `size` and `n_txns`, because it exempted them as
"structural"; adversarial review caught those, which is a fair verdict on how
much a test written by the same person who wrote the code can be trusted.

**The explainer accused clusters that were never flagged.** `explain.py` runs
over every cluster, before any threshold is applied, and every summary began
"Flagged because:". That put an accusation in the audit trail of accounts that
passed cleanly. The evidence text is now neutral ("Signals present: ..."), and
the decision wording lives in `respond.py`, which is the only stage that knows
the decision.

---

# "You changed the data until the model won." Is this p-hacking?

It is the sharpest objection to this project and it deserves a direct answer
rather than a defensive one. A reviewer put it plainly: whoever writes the
generator can make either side win, and this generator was rewritten five
times.

**The concession first.** One change is genuinely vulnerable to the charge.
`slow_drain` was added *after* watching the rule beat the model 5 times out of
5, and it was designed knowing which rule family it had to defeat. That is the
weakest link in this repo and no amount of framing removes it. It is a real,
documented bust-out pattern rather than an invented one, but it was chosen at a
moment when a particular result was wanted.

**The defence, which is checkable rather than rhetorical.** Every other change
removed a shortcut that made the problem unrealistically easy, and each is
recorded with the specific shortcut it killed:

| change | shortcut it removed | direction it moved the headline |
|---|---|---|
| `sophisticated`, `quiet` archetypes | rings shared every identifier | worse |
| legit big spenders, dispute-prone customers | no legit account ever bursted or charged back | worse |
| legit joint bank accounts, supplementary cards, device hubs | **no legit account ever shared a card or bank, so sharing structure WAS the label** | much worse |
| legit clusters sized 2-12 | legit clusters capped at 3, so `size >= 5` was a label | worse |
| ring prevalence 58% then 14% then 4% | precision flattered by an impossible base rate | much worse |
| shuffled account ids, shuffled row order | id prefix and cluster_id were perfect labels | worse |
| household device sharing made probabilistic | `n_devices >= 2` was precision 1.00 | mixed |

**The decisive point is the direction.** If the generator were being tuned to
flatter the model, the numbers would improve. They collapsed, repeatedly and on
purpose. Tier 2 precision went 0.98, then 0.90, then 0.63, and the held-out
figure is 0.61. The clean 12-of-12 sweep at a 5.3x cost advantage became 11 of
12 at 2.4x, with the model losing one world outright. Every one of those
degradations was published rather than reverted, and each earlier, better
number was available to ship and was thrown away. Someone gaming a benchmark
stops when the scoreboard reads well; this stopped when the shortcuts ran out.

The strongest evidence is that the changes kept costing us results we had
already written down. Two published conclusions were reversed by our own later
measurements, in both directions, and the reversals are still in the document.

**The concession, tested.** `slow_drain` was regenerated out of every world and
the whole comparison rerun without it (`holdout.py --exclude-archetype
slow_drain`, 6 pairs). If the model's advantage came from that archetype, it
should collapse:

| | precision | recall | cost | review queue |
|---|---|---|---|---|
| Best hand-tuned rule | 0.46 | 0.97 | Rs.72,083 | 84.2 FPs |
| Tier 2 model | 0.69 | 0.97 | Rs.39,083 | 28.2 FPs |

| | precision | recall | cost | FPs / world |
|---|---|---|---|---|
| Best hand-tuned rule | 0.45 | 0.99 | Rs.49,333 | 73.7 |
| Tier 2 model | 0.74 | 0.96 | Rs.48,667 | 27.3 |

**On cost this is a tie**: Rs.48,667 against Rs.49,333 is a 1.4% difference,
even though the model still wins 5 of 6 worlds on count. The queue advantage
survives intact at 2.7x.

So the concession is only half-rescued, and the honest reading is the
uncomfortable one: **`slow_drain` does carry most of the cost advantage.** The
result that does not depend on it is the review-volume one, which is why that
is the number this repo leads with rather than the cost ratio.

**What would settle it properly.** A dataset nobody here generated. That does
not exist in this project, which is why the honest limits section leads with
it.

**What a reader should take from this.** Not "the model beats rules at fraud
detection". The defensible claim is narrower: on a population containing
heterogeneous ring mechanisms, a model that combines partial signals holds up
better than a tuned two-condition rule, and the size of that advantage is
sensitive to how the population is built. The number to trust least is the
cost ratio. The number to trust most is the review-queue volume, because it
does not depend on the invented rupee constants at all.

---

## Known placeholders, stated plainly

`COST_FALSE_POSITIVE = 500` and `COST_FALSE_NEGATIVE = 15000` in `costs.py`
are illustrative, invented to make the cost line concrete. They are not real
Razorpay or industry figures and should not be presented as researched. This
is why every cost conclusion is reported across a ratio sweep from 1:1 to
100:1: the operating point is stable from 5:1 upward, so the specific rupee
values never drove a conclusion.

## What is still open

- **Everything is synthetic and we chose every difficulty knob.** Seven times
  we found one set too low. The independent-world test controls for overfitting
  a single draw. It cannot show the generator resembles Razorpay traffic.
- **4.2% ring prevalence is far above real organized fraud** (well under 1%).
  Precision would fall further at a true base rate.
- **The graph has no hub problem because every identifier is a fresh UUID.**
  Real data grows giant connected components through shared office devices,
  aggregator nodal accounts and card BINs. There is no hub pruning, degree
  capping, edge weighting or community detection here, and the synthetic data
  gives no signal that any is needed. Largest single gap to deployability, and
  invisible in every number above.
- **100% of planted fraud is graph-linkable by construction.** Real mule
  networks using a separate account per mule are not represented, so reported
  recall is recall over fraud that forms a cluster.
- **The cost model prices Razorpay's loss, never the customer's.** A wrongly
  queued household has no SLA, no appeals path and no fairness check.
- **No label-latency or retraining story.** Chargebacks mature over weeks to
  months; the pipeline assumes labels exist at train time.
- **The model loses one of 12 held-out worlds outright**, and without
  `slow_drain` the cost advantage is a statistical tie (Rs.48,667 against
  Rs.49,333). Only the review-queue advantage is robust across both.
- **Held-out precision (0.61 +/- 0.17) is far below in-fold (0.82)** and the
  per-world cost ratio spans 0.7x to 6.1x. No single world should be quoted.
- **Tier 1 loses to a hand-written rule** and is kept in the repo saying so.
- **The headline threshold is selected on the same out-of-fold predictions it
  is scored on.** A nested check returned identical counts, so measured bias is
  ~0, but the held-out numbers do not have the property at all.
- **`tests/test_no_leakage.py` cannot prove absence of leakage.** It brute
  forces single numeric columns and a few known booleans. A leak expressible
  only as a multi-column interaction would pass it.
