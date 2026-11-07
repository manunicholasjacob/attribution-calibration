"""Run the harness on a synthetic log, against a bad detector and a good one.

Usage:  python demo.py

The log is built with the two properties that matter: speakers take turns, and the channel's
subject drifts so that consecutive turns often carry the same payload. Both are true of real agent
chat, and together they are the regime in which the conventional circular-shift surrogate was found
to fail. Nothing is planted in the log itself, so every edge any detector certifies on a replicate
of it is a false positive by construction.

Four blocks are printed.

  1. A deliberately bad detector: a lag-1 copy rate with a threshold somebody chose and no null
     model at all. This is the shape of most published attribution claims.
  2. A deliberately good detector: the same statistic, standardised, with a permutation surrogate
     test and a family-wise correction over every pair tested.
  3. The bad detector audited under three surrogates at once. This is the five-minute version of
     the package's argument, and the result is not the one the first block suggests.
  4. The good detector under the calibration null, the confounding null and the invalid one.
"""
from __future__ import annotations

import warnings

from attrcal import PlantSpec, calibrate, compare_nulls, detectors, synth_log

N_EVENTS = 2500
N_NULL = 200
N_NULL_SLOW = 60
N_SEEDS = 10


def header(text: str) -> None:
    print()
    print("#" * 88)
    print(f"# {text}")
    print("#" * 88)
    print()


def main() -> int:
    built = synth_log(n_events=N_EVENTS, n_actors=6, n_payloads=8, turn_taking=0.9, drift=0.05,
                      common_cause=0.3, seed=3)
    events = built["events"]
    print(f"synthetic log: {len(events)} turns, {built['config']['n_actors']} actors, "
          f"{built['config']['n_payloads']} payload symbols, no planted influence")
    print(f"first three turns: {events[:3]}")

    spec = PlantSpec(strengths=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8), volume=200,
                     volumes=(25, 50, 100, 200, 400), strength=0.4, n_seeds=N_SEEDS)
    bad = detectors.CopyRateDetector(lag=1, threshold=0.4)
    good = detectors.SurrogateCopyDetector(lag=1, n_surrogates=120, null="permute", alpha=0.05,
                                           seed=0)

    header("1. A DELIBERATELY BAD DETECTOR: a copy rate, a round-number threshold, no null model")
    print(calibrate(events, bad, null="permute", n_null=N_NULL, planted=spec, seed=0).summary())

    header("2. A DELIBERATELY GOOD DETECTOR: the same statistic with a surrogate test")
    print(calibrate(events, good, null="permute", n_null=N_NULL, planted=spec, seed=0).summary())

    with warnings.catch_warnings():
        # compare_nulls states the shift surrogate's problem in its own output, so the warning
        # would only repeat it.
        warnings.simplefilter("ignore")

        header("3. THE BAD DETECTOR UNDER THREE SURROGATES: its error rate depends on the null")
        print(compare_nulls(events, bad, nulls=("permute", "rotate", "shift"), n_null=N_NULL,
                            seed=0).summary())

        header("4. THE GOOD DETECTOR UNDER THE SAME TREATMENT")
        print(compare_nulls(events, good, nulls=("permute", "blockpermute", "shift"),
                            n_null=N_NULL_SLOW, seed=0).summary())

    header("WHAT TO TAKE FROM THIS")
    print("""Block 1 is the trap. Under the permutation null the bad detector looks excellent: it
certifies nothing at all on 200 structure-free replicates and finds a planted copycat down to a
copy fraction of 0.05, which beats the good detector on both numbers. Its threshold of 0.4 simply
happens to sit above the rate at which two payloads drawn from this log's eight-symbol alphabet
agree by chance. Nothing about the detector earned that. Change the alphabet and it moves.

Block 3 is what the threshold is actually worth. The same detector on the same log goes from 0.000
under the permutation null to 1.000 under rotation, which keeps nothing but each actor's own
habitual subjects, and to 1.000 under the conventional circular shift. At most one of those three
is that detector's real error rate, and a paper quoting any one of them without naming the null is
not quoting a rate.

Block 2 is what a stated error rate looks like. 0.055 over 200 replicates, an interval covering the
nominal 0.05, and a detection floor of 0.10 that is bracketed rather than guessed. It is less
powerful than the bad detector and that is the trade being made visible, which is the point of
reporting both: a floor means nothing next to an unstated error rate, and an error rate means
nothing next to an unstated floor.

Block 4 separates the two ways a detector can be wrong. The good detector holds near nominal under
the permutation null and under the block permutation, which keeps the channel's slow drift, so on
this log it is not merely rediscovering a shared subject. It still reads as broken under the
circular shift, and the NULL AUDIT section in blocks 1 and 2 says why: the shift surrogate does not
hold the actor sequence fixed, so its replicates carry adjacencies this log's turn-taking never
allowed and an error rate measured against them belongs to a conversation that did not happen.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
