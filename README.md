# attribution-calibration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24064146.svg)](https://doi.org/10.5281/zenodo.24064146)

**Error rates for directed claims about multi-agent logs.** Bring your own detector and your own
event log; get back the three numbers any claim of the form "agent A caused / influenced / is
responsible for B" should carry, and never does.

Author: Manu Nicholas Jacob, Independent Researcher (manunicholasjacob@gmail.com).

## Why this exists

Attribution and influence detection are the same statistical problem. Both take a log of who did
what when and emit a directed claim. Both are almost always published with no error rate attached.

When one such pipeline was finally given a valid surrogate, its family-wise error on real
turn-ordered agent chat came out at **0.90 against a nominal 0.05**. On nine logs in ten containing
no influence at all, it certified at least one influence edge. The cause was not the estimator. It
was the surrogate: the conventional circular-shift recipe does not hold the actor sequence fixed, so
on a log where speakers take turns its replicates are not replicates of that log.

That failure is not specific to transfer entropy, or to influence, or to that one estimator. Any
method that reads a multi-agent log and emits directed edges can be audited the same way, and almost
none are. This package is that audit, made generic and detector-agnostic.

The 0.90 figure, and the measurements behind it, are in
**[swarm-influence-audit](https://github.com/manunicholasjacob/swarm-influence-audit)**, with a
verifier that recomputes every number in that report from the committed analysis output. The
generated benchmark it was validated against is
**[swarm-influence-bench](https://github.com/manunicholasjacob/swarm-influence-bench)**.

## Prior work, and what is and is not original here

**The idea of holding the actor sequence fixed and permuting only the payloads is not mine.** It is
the local permutation scheme of:

> Runge, J. (2018). Conditional Independence Testing Based on a Nearest-Neighbor Estimator of
> Conditional Mutual Information. *AISTATS*, PMLR 84, 938-947.

adapted to continuous-time event data by:

> Shorten, D. P., Spinney, R. E., and Lizier, J. T. (2021). Estimating transfer entropy in
> continuous time between neural spike trains or other event-based data. *PLOS Computational
> Biology* 17(4), e1008054. doi:10.1371/journal.pcbi.1008054

Runge proposed it; Shorten and colleagues established that the conventional source-time-shift
surrogate fails to represent the null of conditional independence for event-based processes, and
adapted Runge's scheme to that setting. `permute` and `blockpermute` below are that idea applied to
turn-ordered multi-agent logs. The `JIDT` toolkit has implemented the spiking estimator and its
permutation surrogate since 2022, contributed by Shorten himself.

What this package adds is narrower and worth stating plainly: packaging the comparison as a
detector-agnostic harness, the four-way surrogate contrast that lets an invalid null be *measured*
rather than asserted, the explicit floor status, and `audit_null()`, which demonstrates on your own
log why `shift` fails instead of asking you to take it on faith.

## Install

```bash
pip install -e ".[test]"
python -m pytest tests -q
python demo.py
```

`pytest` is an optional extra, so `pip install -e .` alone will not let the test line above run.

Python 3.10 or newer. `numpy` is the only dependency. scikit-learn is deliberately **not** used:
the one metric that would have needed it, the adjusted Rand index, is a dozen lines of numpy in
`attrcal/symbols.py`, and a harness whose argument is that other people's error rates are
unverified should be auditable from short files.

## Use

```python
from attrcal import calibrate, PlantSpec

events = [("agent_a", 0.0, "budget"), ("agent_b", 12.0, "budget"), ...]   # (actor, time, payload)

report = calibrate(
    events,
    my_detector,              # Callable[[events], set[(src, dst)]]
    null="permute",           # which surrogate builds the replicates
    n_null=200,               # how many replicates
    planted=PlantSpec(),      # the spike-in positive control
    seed=0,
)

report.familywise_error       # share of null replicates certifying >= 1 edge
report.detection_floor        # {"floor": ..., "status": "bracketed", "note": ..., ...}
report.power_curve            # {copy_fraction: Rate(point, lo, hi, k, n)}
print(report.summary())       # a printable block for a paper
```

Your detector receives an event list and must return a **set** of `(source, target)` tuples. A list
is rejected with an explanation: a list can carry the same edge twice, which changes every count
this harness reports, and a generator reads as an empty result after its first use.

### The three numbers

| what | where | what it means |
|---|---|---|
| family-wise error | `report.familywise_error` plus `..._interval` | the share of structure-free replicates of **your own log** on which the detector certified at least one edge |
| power | `report.power_curve`, `report.volume_curve` | recall of one planted copycat edge, at each swept copy fraction and at each swept plant volume |
| detection floor | `report.detection_floor`, `report.detection_floor_volume` | the smallest swept value at which power reaches the threshold, with a status saying whether that number means anything |

Every rate is a `Rate`: a point estimate, a 95% Wilson interval, and the denominator it came from.
Wilson rather than the normal approximation because every interesting rate here sits near 0 or near
1, which is exactly where a Wald interval leaves the unit square and starts reporting negative
error rates.

## Nulls built from your own log

This is the part that caught the original failure, and it is the reason the package is useful on a
log nobody has a generator for. Rather than comparing the detector against a simulation, the
comparison is built out of the log the claim is about: **hold the actor sequence exactly as it is,
including every turn, every volume share and every idle stretch, and only change which payload sits
at which turn.** What comes back cannot contain directed structure, because nothing about who spoke
when has anything to do with what was said any more.

Four surrogates are offered. They differ only in which nuisance structure they keep, which is why a
published error rate that does not name its null is not a rate.

| `null=` | keeps | what it answers | actor sequence |
|---|---|---|---|
| `permute` | volumes, global payload marginal | is there any structure at all? **the calibration null, and the default** | fixed |
| `blockpermute` | the above plus the channel's slow drift | is the detector confounded by a shared subject? **the confounding null** | fixed |
| `rotate` | the above plus each actor's own payload marginal and autocorrelation | is the detector just rediscovering that two actors habitually talk about the same thing? **the strict null** | fixed |
| `shift` | the conventional circular shift | nothing trustworthy. **INVALID on turn-ordered logs**, included to be measured against | **moves** |

`shift` is implemented faithfully rather than charitably: each actor's emissions are slid along the
turn index by their own random offset, which is the recipe as it is normally written. Two
consequences are measurable in a single replicate. Turns collide and turns empty, so actors end up
adjacent that the protocol never allowed to be adjacent, and a lag-1 statistic suddenly has data for
pairs the real log gave it none for. And because all of an actor's events move by one offset, each
pair of actors is compared at its own fixed arbitrary offset, so pairs whose offsets land close keep
the log's own payload autocorrelation and read as copying.

`attrcal.null_index()` refuses to express `shift` as an index map, and the refusal is the
documentation: a null that can be written as a remapping of the payload column cannot change who
spoke when, and one that cannot be written that way does.

### Seeing it in two lines

```python
from attrcal import compare_nulls
print(compare_nulls(events, my_detector, nulls=("permute", "rotate", "shift"), n_null=200))
```

On the demo's synthetic log, which has turn-taking and a drifting channel subject, the same
uncalibrated detector comes out at a family-wise error of 0.000 under `permute` and 1.000 under
`shift`. At most one of those is that detector's real error rate.

### Auditing the surrogate itself

```python
from attrcal import audit_null
audit_null(events, "shift")
```

A null model is a claim, and the claim is checkable. `audit_null` measures one replicate the same
way the log was measured and reports what moved: the effective number of adjacent actor pairs, the
rate of tied timestamps, the turn-taking index, the payload repeat rate and the payload purity lift,
log against replicate. On the demo log the effective number of adjacent actor pairs goes from 6.8 to
27.0 out of 30 possible under `shift`, and does not move at all under the three that hold the actor
sequence fixed. `calibrate` runs this automatically and prints it under `NULL AUDIT`.

## The spike-in positive control

A family-wise error rate alone is satisfied by a detector that certifies nothing, so it cannot be
the only number reported. `PlantSpec` inserts a synthetic actor into the real log that copies a
chosen leader's payload a chosen fraction of the time at a chosen lag, at a chosen volume, and the
harness asks whether the detector returns that one edge.

```python
PlantSpec(
    strengths=(0.05, 0.1, 0.2, 0.4, 0.8),   # the copy fraction swept for the floor
    volume=200,                             # plant messages, held fixed during that sweep
    volumes=(25, 50, 100, 200, 400),        # optional second sweep, at `strength`
    strength=0.4,
    leader=None,                            # None means the busiest actor
    lag=1,
    n_seeds=10,
)
```

Two honest limits, both reported rather than argued.

**The plant is the easiest possible positive.** It copies a payload verbatim at a fixed lag with no
paraphrase, no jitter and no competing obligations, so a detection floor measured against it is a
**lower bound** on the strength of real influence a detector needs. Real agents copy ideas, not
strings.

**The requested copy fraction is a label, not a measurement.** Inserting the plant lengthens the
turn sequence, so at lags above 1 another turn can land in between and push a copy off its intended
lag; and a noise message drawn from the log's own payload distribution can land after the leader and
match it by chance. `plant_copycat` therefore measures the fraction actually realised in the stream
it produces, `calibrate` averages it per cell into `report.realised_strength`, and the summary prints
it beside every request. **Index your conclusions by the realised value.**

By default the leader is the busiest actor, which supplies the most opportunities and so is the
right default for a floor meant to be read as a lower bound. It is the wrong default if the question
is whether a quiet actor can be attributed from at all; pass `leader` explicitly for that. If more
copies are requested than the leader has turns, the plant is capped, and the cap is reported in
`report.notes` rather than silently lowering the realised fraction.

## The detection floor is a status, not a number

```python
report.detection_floor
# {"floor": 0.1, "status": "bracketed", "note": None, "threshold": 0.8, "monotone": True,
#  "n_seeds": 10}
```

| status | meaning |
|---|---|
| `no_data` | nothing was swept, so there is no floor |
| `never_reached` | power stayed below the threshold at every value swept |
| `below_grid` | power was already above the threshold at the smallest value swept, so this is an **upper bound** on the true floor |
| `bracketed` | power crossed the threshold and stayed above it. The only status that is a floor |
| `non_monotone` | power crossed the threshold and then fell back below it at a larger value, so this is **not** a floor |

Returning a bare number conflated all five, and `non_monotone` is the dangerous one: the first
crossing of a ragged curve reads exactly like a floor. The `note` field warns when the seed count
cannot support the threshold, because with *n* seeds a recall estimate can only take multiples of
1/*n*, so a floor defined at 80% power from three seeds is a property of the grid and not of the
detector. `calibrate` also raises a `UserWarning` for that case rather than only recording it.

## The symbolisation check, and what it does not cover

Most detectors do not work on text. They work on a discrete symbol per turn, obtained by clustering,
topic modelling, embedding and thresholding, or an LLM label. That front end is **inside** the
detector, so this harness cannot see it, and it cannot tell a detector whose statistics found nothing
from one whose symboliser failed. Those two produce the same empty edge set and support opposite
conclusions.

If you can export the symbols your detector assigned and you have ground-truth labels for the same
turns, pass both:

```python
report = calibrate(events, my_detector, symbols=my_symbols, labels=true_labels)
report.symbolisation   # {"adjusted_rand_index": 0.44, "n_symbols": 8, "n_true_classes": 8, ...}
```

The adjusted Rand index is used because cluster identities are arbitrary, so a symboliser that
recovers the partition perfectly and names it differently must still score 1.0, and because an
unadjusted index rises towards 1 as clusters multiply, which would reward a symboliser for being
indecisive. **Scope:** this compares two partitions you supplied. It does not run a symboliser and
it says nothing about a detector whose symbols you cannot export. If you publish a directed graph
over open text, report your symboliser's agreement with something.

## Log diagnostics

`describe_log(events)`, which `calibrate` runs automatically, reports the measurements that decide
which surrogate is safe on your log rather than leaving it to judgement.

- **`turn_taking_index`**: how often an actor follows itself, divided by how often it would under
  independent draws from the volume shares. At 1.0 the speaker sequence carries no order
  information. Well below 1.0 is a turn-taking protocol or a round-robin scheduler, and is the
  regime in which the conventional shift surrogate was measured at 0.90.
- **`lag1_pairs_effective`** against **`lag1_pairs_possible`**: the effective number of ordered actor
  pairs that a lag-1 statistic really has data for, as an inverse Simpson index over the transition
  counts. A plain count of pairs seen at least once saturates on any long log, because one stray
  transition puts a pair on the list; weighting by how often each pair actually occurs does not.
- **`payload_repeat_rate`**: how often consecutive turns carry the same payload. The nulls can only
  disagree when this sits above chance, so reporting it turns a surprise into a prediction.
- **`payload_purity_lift`**: how much better a turn's payload is guessed from its actor than from
  nothing. Well above 1.0 means actors have characteristic subjects, which `rotate` and `shift`
  preserve and `permute` destroys.
- **`tied_timestamp_rate`**: a real log usually has none, which is what makes it able to detect a
  surrogate that invents simultaneous speech.

## What this does not do

**It does not control for a common cause acting at the same lag as the claimed influence.** No
reshuffle of the payload column can separate those, because at that lag they are the same pattern.
`blockpermute` catches a *slower* shared drift, which is the common case, but a confounder that acts
at exactly lag 1 needs a detector that conditions on the channel's own state. This harness will not
tell you that you need one; it will only tell you that your error rate is not what you said.

**It does not make your detector correct.** A detector can pass `permute` and fail `blockpermute`, or
pass both and still be measuring something you did not mean. The harness bounds the false positive
rate against a stated null and nothing more.

**It does not evaluate precision against a known graph.** For that you need labels, which means a
generator, which is a different piece of work. The spike-in gives recall of one edge you planted,
not precision against a full ground-truth graph.

**It does not symbolise text.** See above.

## Reference detectors

`attrcal.detectors` ships four, provided so the harness can be demonstrated and tested against
behaviour that is known in advance. **None is a recommendation.**

- `all_pairs` certifies every ordered pair. Perfect recall of anything, family-wise error 1.0.
- `no_edges` certifies nothing. Perfect family-wise error, no power. The failure mode that a false
  positive rate alone cannot catch.
- `CopyRateDetector` is a lag-1 payload copy rate against a threshold somebody chose, with no null
  model. The shape of most published attribution claims.
- `SurrogateCopyDetector` is the same statistic, standardised so pairs with different opportunity
  counts are comparable, with a surrogate test whose null distribution is the **maximum** over all
  pairs tested. A directed claim about a swarm is a claim about one edge drawn from the whole family,
  so the family-wise rate is the one that matters: testing each pair at 0.05 on a twelve-actor log
  tests 132 pairs. It records `last_tested`, because an error rate cannot be read without the family
  size and because a turn-taking protocol means most pairs are never adjacent and so are never
  tested at all.

`attrcal.synth_log` builds a small synthetic log for the demo and the tests. It is **not** a
benchmark: a benchmark needs exact labels, separated null families and measured realism limits, and
that is separate work. What `synth_log` provides is the two properties the surrogate comparison
depends on, turn-taking and payload drift, with a knob for each.

## Independence

This package shares no code with any other project. It imports `numpy` and the standard library and
nothing else, which `tests/test_packaging.py` asserts by walking each file's syntax tree. There are
no absolute paths, no import-path manipulation and no non-ASCII characters anywhere, each also
asserted mechanically, so a clean clone runs on Linux unchanged.

**Code independence is not intellectual independence**, and the two were easy to confuse here. The
statistical idea the package is built on belongs to Runge (2018) and to Shorten, Spinney and Lizier
(2021), as set out under Prior work above.

## Tests

```bash
python -m pytest tests -q
```

Every test asserts a behaviour. The load-bearing ones are: a detector that certifies everything
scores as badly as possible on nulls while having perfect recall; a detector that certifies nothing
has perfect family-wise error and no power; all five floor statuses are reachable; `permute` and
`shift` disagree on turn-ordered input, and **agree** on a log without turn-taking, which is the
negative control that keeps the first assertion from being a property of the harness; and a fixed
seed reproduces a report exactly.

## Provenance

Version 0.1.0, dated 2026-11-06 (see [CITATION.cff](CITATION.cff)). This is general-purpose
infrastructure: it is detector-agnostic and log-agnostic, and it was written to be reused rather
than for one study. It shares no code with any other project of the author's or anyone else's, and
the statistical method it implements is credited under Prior work above.

## Licence

MIT, see [LICENSE](LICENSE).
