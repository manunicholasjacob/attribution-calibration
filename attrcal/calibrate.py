"""The entry point: hand over a log and a detector, get back the three numbers.

The design constraint that shapes this file is that the detector is opaque. The harness never sees
the statistic, the threshold, the symboliser or the internal null; it sees a callable that turns an
event log into a set of directed edges. Everything measured here is therefore measured by changing
the input and watching the output, which is the only thing that works on a method somebody else
wrote and the only thing that works on a method whose author did not plan for it to be audited.

Three measurements come out of that:

  family-wise error  run the detector on replicates of the user's own log that cannot contain any
                     directed structure, and count how often it certifies at least one edge
  power              insert an actor that really does copy somebody, and count how often that one
                     edge comes back
  detection floor    the smallest copy fraction at which power reaches a threshold, with a status
                     that says whether that number means anything

Seeding is done with a `SeedSequence` and separate spawned streams for the nulls, the strength
sweep and the volume sweep. The point is that adding a volume sweep must not change the null
replicates, so two runs at the same seed that differ only in what was swept still agree on the
numbers they share.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Sequence

import numpy as np

from .events import check_edges, describe_log, detector_name, normalise_events
from .nulls import NULLS, make_null, preserves_actor_sequence
from .plant import PlantSpec, choose_leader, plant_copycat
from .report import NullComparison, Report
from .stats import MIN_SEEDS_FOR_POWER, Rate, detection_floor
from .symbols import symbolisation_check


def _as_spec(planted: Any) -> PlantSpec | None:
    if planted is None or planted is False:
        return None
    if planted is True:
        return PlantSpec()
    if isinstance(planted, PlantSpec):
        return planted
    if isinstance(planted, dict):
        return PlantSpec(**planted)
    raise TypeError(f"planted must be a PlantSpec, a dict of its fields, True or None, "
                    f"got {type(planted).__name__}")


def _run_nulls(events, detector, kind, n_null, rng, block_size, progress, who):
    counts: list[int] = []
    for r in range(n_null):
        replicate = make_null(events, kind, rng, block_size=block_size)
        found = check_edges(detector(replicate), who)
        counts.append(len(found))
        if progress is not None:
            progress("null", r + 1, n_null, len(found))
    return counts


def audit_null(events: Sequence[Any], kind: str, seed: int = 0,
               block_size: int | None = None) -> dict:
    """Check what a surrogate did to the log, rather than trusting that it removed only structure.

    A null model is a claim: that its replicates could have arisen from this log with no directed
    influence in it. That claim is checkable, cheaply, by measuring the replicate the same way the
    log was measured and looking at what moved. A surrogate that puts actors next to each other who
    were never next to each other has not removed structure, it has replaced the conversation, and
    the lag-1 statistics it then supports are about pairs the real log had no data for.

    One replicate is enough, because the properties compared are aggregates over the whole log and
    what is being looked for is a structural change rather than a small shift.
    """
    ev = normalise_events(events)
    real = describe_log(ev)
    rep = describe_log(make_null(ev, kind, np.random.default_rng(seed), block_size=block_size))
    fixed = preserves_actor_sequence(kind)
    before, after = real["lag1_pairs_effective"], rep["lag1_pairs_effective"]
    widened = after > 1.5 * before if before else False
    if fixed:
        verdict = ("the actor sequence is untouched, so this replicate is a surrogate of this log "
                   "and the error rate above belongs to this log")
    elif widened:
        verdict = (f"the actor sequence moved: the effective number of adjacent actor pairs went "
                   f"from {before:.1f} to {after:.1f} out of {real['lag1_pairs_possible']}. A "
                   "lag-1 statistic now has data for pairs the real conversation barely put next "
                   "to each other, so this is not a null for this log and the error rate above is "
                   "not this detector's")
    else:
        verdict = ("the actor sequence moved, so this replicate is not a surrogate of this log, "
                   "even though its adjacency structure happens to be about as narrow as the real "
                   "log's")
    return {
        "null": kind,
        "preserves_actor_sequence": fixed,
        "lag1_pairs_effective": {"log": before, "replicate": after,
                                 "possible": real["lag1_pairs_possible"]},
        "tied_timestamp_rate": {"log": real["tied_timestamp_rate"],
                                "replicate": rep["tied_timestamp_rate"]},
        "turn_taking_index": {"log": real["turn_taking_index"],
                              "replicate": rep["turn_taking_index"]},
        "payload_repeat_rate": {"log": real["payload_repeat_rate"],
                                "replicate": rep["payload_repeat_rate"]},
        "payload_purity_lift": {"log": real["payload_purity_lift"],
                                "replicate": rep["payload_purity_lift"]},
        "widened_adjacency": widened,
        "verdict": verdict,
    }


def _run_cells(events, detector, spec, values, vary, rng, progress, who, notes):
    """Recall and realised strength for one sweep, where `vary` names the field being swept."""
    curve: dict = {}
    realised: dict = {}
    cells: list[dict] = []
    leader = spec.leader if spec.leader is not None else choose_leader(events)
    for value in values:
        strength = float(value) if vary == "strength" else float(spec.strength)
        volume = int(value) if vary == "volume" else int(spec.volume)
        hits = 0
        realised_values: list[float] = []
        for s in range(spec.n_seeds):
            built = plant_copycat(events, rng, leader=leader, strength=strength, volume=volume,
                                  lag=spec.lag, name=spec.name)
            if built["capped"] and built["cap_note"] not in notes:
                notes.append(built["cap_note"])
            found = check_edges(detector(built["events"]), who)
            hits += built["edge"] in found
            realised_values.append(built["strength_realised"])
            if progress is not None:
                progress(f"power:{vary}", s + 1, spec.n_seeds, value)
        curve[value] = Rate.of(hits, spec.n_seeds)
        realised[value] = round(float(np.mean(realised_values)), 4)
        cells.append({vary: value, "recall": curve[value].as_dict(),
                      "strength_realised": realised[value], "leader": leader})
    return curve, realised, cells, leader


def calibrate(events: Sequence[Any], detector: Callable[[Sequence[Any]], set], *,
              null: str = "permute", n_null: int = 200, planted: Any = None,
              alpha: float = 0.05, seed: int = 0, floor_threshold: float = 0.8,
              symbols: Sequence[Any] | None = None, labels: Sequence[Any] | None = None,
              block_size: int | None = None, name: str | None = None,
              progress: Callable[..., None] | None = None) -> Report:
    """Calibrate one detector on one log and return a `Report`.

    events
        `[(actor, timestamp, payload), ...]`. Sorted stably by timestamp, so ties keep the caller's
        order, which in a coarse-timestamped log is the turn order.
    detector
        `Callable[[events], set[(src, dst)]]`. Must return a set; a list or a generator is rejected
        with an explanation, because both silently change the counts this harness reports.
    null
        Which surrogate builds the replicates: `permute` (calibration), `blockpermute`
        (confounding) or `shift` (the conventional one, included to be measured against).
    n_null
        Number of null replicates. This bounds the resolution of the result: with n replicates the
        smallest non-zero error rate observable is 1/n, so 20 replicates cannot distinguish a
        detector at 0.05 from one at 0.00.
    planted
        A `PlantSpec`, a dict of its fields, or `True` for the defaults. Omit it and the report
        carries no power curve, which is stated rather than left blank.
    symbols, labels
        Optional, and only meaningful together: the symbols the detector assigned to each turn and
        ground-truth labels for the same turns, in the same order. Scored with the adjusted Rand
        index.
    """
    if null not in NULLS:
        raise ValueError(f"unknown null {null!r}; expected one of {NULLS}")
    if n_null < 0:
        raise ValueError("n_null cannot be negative")
    ev = normalise_events(events)
    log = describe_log(ev)
    who = name or detector_name(detector)
    spec = _as_spec(planted)
    notes: list[str] = []

    root = np.random.SeedSequence(seed)
    null_ss, strength_ss, volume_ss = root.spawn(3)

    if null == "shift":
        notes.append("the shift surrogate was chosen. It slides each actor's emissions along the "
                     "turn index, so its replicates do not have this log's actor sequence and any "
                     "error rate from it describes a conversation the user does not have.")
        warnings.warn("null='shift' is the conventional circular-shift surrogate. It moves the "
                      "actor sequence, so its replicates are not surrogates of this log, and on "
                      "turn-ordered agent chat it was measured at a family-wise error of 0.90 "
                      "against a nominal 0.05. Use it for comparison, not for a headline.",
                      stacklevel=2)
    if n_null == 0:
        notes.append("no null replicates were requested, so there is no measured error rate.")
    elif n_null < 20:
        notes.append(f"{n_null} null replicates resolve an error rate only to 1/{n_null}; the "
                     "interval printed above is correspondingly wide.")

    counts = _run_nulls(ev, detector, null, n_null, np.random.default_rng(null_ss), block_size,
                        progress, who)
    fired = sum(1 for c in counts if c > 0)
    fwer = Rate.of(fired, n_null)
    null_check = audit_null(ev, null, seed=seed, block_size=block_size) if n_null else {}

    power: dict = {}
    realised: dict = {}
    volume_curve: dict = {}
    planted_out: dict | None = None
    floor_v: dict = {}
    if spec is not None:
        if spec.name in {a for a, _, _ in ev}:
            raise ValueError(f"the plant's name {spec.name!r} is already an actor in this log")
        if spec.n_seeds < MIN_SEEDS_FOR_POWER:
            warnings.warn(
                f"planted sweep uses {spec.n_seeds} seed(s) per cell. With n seeds a recall "
                f"estimate can only take multiples of 1/n, so a {floor_threshold:.0%} power "
                "threshold is not estimable and the reported floor is a property of the grid.",
                stacklevel=2)
        power, realised, strength_cells, leader = _run_cells(
            ev, detector, spec, list(spec.strengths), "strength",
            np.random.default_rng(strength_ss), progress, who, notes)
        volume_cells: list[dict] = []
        if spec.volumes:
            volume_curve, _vr, volume_cells, _l = _run_cells(
                ev, detector, spec, list(spec.volumes), "volume",
                np.random.default_rng(volume_ss), progress, who, notes)
            floor_v = detection_floor(volume_curve, floor_threshold, spec.n_seeds)
        planted_out = {**spec.as_dict(), "leader": leader,
                       "strength_cells": strength_cells, "volume_cells": volume_cells}

    floor = detection_floor(power, floor_threshold, spec.n_seeds if spec else None)

    sym = None
    if symbols is not None or labels is not None:
        if symbols is None or labels is None:
            raise ValueError("symbols and labels are only meaningful together: one partition "
                             "cannot be scored against nothing")
        sym = symbolisation_check(symbols, labels)

    return Report(
        familywise_error=(float("nan") if n_null == 0 else fwer.point),
        familywise_error_interval=(fwer.lo, fwer.hi),
        familywise_error_n=n_null,
        familywise_error_k=fired,
        mean_edges_per_null=round(float(np.mean(counts)), 4) if counts else 0.0,
        max_edges_on_a_null=max(counts) if counts else 0,
        detection_floor=floor,
        power_curve=power,
        volume_curve=volume_curve,
        detection_floor_volume=floor_v,
        realised_strength=realised,
        symbolisation=sym,
        log=log,
        null=null,
        n_null=n_null,
        alpha=alpha,
        seed=seed,
        detector=who,
        planted=planted_out,
        null_audit=null_check,
        notes=notes,
    )


def compare_nulls(events: Sequence[Any], detector: Callable[[Sequence[Any]], set], *,
                  nulls: Sequence[str] = ("permute", "shift"), n_null: int = 200,
                  seed: int = 0, block_size: int | None = None,
                  name: str | None = None,
                  progress: Callable[..., None] | None = None) -> NullComparison:
    """Family-wise error for one detector on one log under several surrogates.

    This exists because the single most useful thing a user can learn in five minutes is that their
    error rate depends on which surrogate they chose. One number cannot show that. Each surrogate
    gets its own spawned stream from the same root seed, so the comparison is reproducible and no
    surrogate is advantaged by having been run first.
    """
    for kind in nulls:
        if kind not in NULLS:
            raise ValueError(f"unknown null {kind!r}; expected from {NULLS}")
    if len(set(nulls)) < 2:
        raise ValueError("comparing a surrogate against itself measures nothing; pass at least "
                         "two distinct nulls")
    ev = normalise_events(events)
    who = name or detector_name(detector)
    streams = np.random.SeedSequence(seed).spawn(len(nulls))
    rates: dict = {}
    with warnings.catch_warnings():
        # The shift warning is the finding here rather than a surprise, and it is stated in the
        # comparison's own summary, so it is not also raised once per call.
        warnings.simplefilter("ignore")
        for kind, stream in zip(nulls, streams):
            counts = _run_nulls(ev, detector, kind, n_null, np.random.default_rng(stream),
                                block_size, progress, who)
            rates[kind] = Rate.of(sum(1 for c in counts if c > 0), n_null)
    return NullComparison(rates=rates, log=describe_log(ev), detector=who, n_null=n_null,
                          seed=seed,
                          moves_actor_sequence={k: not preserves_actor_sequence(k) for k in nulls})
