"""The event contract, and the diagnostics that decide whether a surrogate is safe on your log.

An event is `(actor, timestamp, payload)`. That is the whole input contract, because it is the
intersection of what every multi-agent log actually has: somebody emitted something at some time.
A harness that demanded more would exclude the logs it is meant to audit.

`describe_log` exists because the failure this package was built around is a property of the log and
not of the detector. On a log where speakers take turns, the conventional circular-shift surrogate
manufactures adjacencies the protocol forbade, and the user has no way to see that from the
detector's own output. These diagnostics are reported so the decision is made on a measurement.

The sharpest of them is `lag1_pairs_effective` against `lag1_pairs_possible`. Where a turn-taking
protocol is in force, almost all the adjacency mass sits on a handful of ordered pairs, and those
are the only pairs a lag-1 statistic really has data for. A surrogate that comes back with the mass
spread evenly over every pair has not removed structure from the log, it has removed the protocol,
and no error rate measured against it is about the log the user has.
"""
from __future__ import annotations

import collections
import collections.abc
from typing import Any, Callable, Iterable, Sequence

Event = tuple[Any, float, Any]
_SEQ = collections.abc.Sequence


def normalise_events(events: Iterable[Any]) -> list[Event]:
    """Validate and sort an event log into the canonical form this package works on.

    Sorting is stable, which matters more than it looks. Many agent logs carry a coarse timestamp
    with ties inside a turn, and in those logs the list order is the turn order and the timestamp
    is not. A stable sort keeps the caller's ordering for tied timestamps instead of silently
    reshuffling the turn sequence that every lagged statistic depends on.
    """
    out: list[Event] = []
    for i, ev in enumerate(events):
        if isinstance(ev, (str, bytes)) or not isinstance(ev, _SEQ):
            raise TypeError(f"event {i} is not a sequence: {ev!r}")
        if len(ev) != 3:
            raise ValueError(
                f"event {i} has {len(ev)} fields, expected 3 as (actor, timestamp, payload). "
                "If your log carries extra columns, drop them or fold them into the payload; the "
                "harness must not see a ground-truth label it could leak to a detector.")
        actor, ts, payload = ev
        try:
            hash(actor)
        except TypeError as exc:
            raise TypeError(f"event {i}: actor {actor!r} is not hashable and cannot name an "
                            f"edge endpoint") from exc
        try:
            ts = float(ts)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"event {i}: timestamp {ts!r} is not a number") from exc
        if ts != ts:
            raise ValueError(f"event {i}: timestamp is NaN, so the turn order is undefined")
        out.append((actor, ts, payload))
    if not out:
        raise ValueError("the event log is empty; there is nothing to calibrate against")
    out.sort(key=lambda e: e[1])
    return out


def actor_list(events: Sequence[Event]) -> list:
    """Distinct actors in order of first appearance, so reports are stable across runs."""
    seen: dict = {}
    for a, _, _ in events:
        seen.setdefault(a, None)
    return list(seen)


def check_edges(found: Any, who: str = "detector") -> frozenset:
    """Validate a detector's return value against the contract, and explain any rejection.

    A `set` is required rather than any iterable, and this is not pedantry. Scoring is set
    intersection, so a list that happens to contain the same edge twice inflates the denominator of
    precision and the numerator of "edges certified" without changing the claim being made. A
    generator is worse: it is consumed by the first use and reads as an empty result afterwards.
    Both used to pass silently.
    """
    if isinstance(found, (set, frozenset)):
        pairs = found
    else:
        raise TypeError(
            f"{who} returned {type(found).__name__}, expected a set of (source, target) tuples. "
            "A list can carry the same edge twice, which changes every count this harness "
            "reports; a generator is consumed on first use and then reads as no edges at all.")
    out = []
    for e in pairs:
        if isinstance(e, (str, bytes)) or not isinstance(e, _SEQ) or len(e) != 2:
            raise ValueError(f"{who} returned {e!r}, expected a 2-tuple (source, target)")
        src, dst = e
        if src == dst:
            raise ValueError(
                f"{who} certified the self-edge {e!r}. No lagged pairwise statistic tests an "
                "actor against itself, so a self-edge cannot be confirmed or refuted here.")
        out.append((src, dst))
    return frozenset(out)


def describe_log(events: Sequence[Event]) -> dict:
    """Summary statistics of a log, including the two that decide which surrogate is safe.

    `turn_taking_index` is the observed rate at which the same actor speaks twice in a row divided
    by the rate expected if speakers were drawn independently from their own volume shares. At 1.0
    the speaker sequence carries no order information and a shift surrogate has little to preserve.
    Well below 1.0 means speakers avoid following themselves, which is what a turn-taking protocol
    or a round-robin scheduler produces, and it is the regime in which the circular-shift surrogate
    was measured to certify at least one edge on 90% of structure-free replicates.

    `payload_repeat_rate` is the rate at which consecutive turns carry the same payload. A shifting
    or rotating surrogate preserves it by construction while a full permutation destroys it, so two
    nulls can only disagree when this sits well above chance. Reporting it turns "the nulls
    disagree" from a surprise into a prediction.

    `payload_purity_lift` is how much better one can guess a turn's payload from knowing its actor
    than from knowing nothing. At 1.0 actors are interchangeable; well above 1.0 the actors have
    characteristic subjects, which is true of essentially every real agent team, and it is the
    structure that a shift or rotation surrogate keeps and a permutation destroys. A detector that
    is really keying on "these two actors habitually talk about the same thing" will look clean
    under one of those nulls and dirty under the other, and this is the number that predicts it.
    """
    n = len(events)
    counts = collections.Counter(a for a, _, _ in events)
    repeats = sum(1 for i in range(1, n) if events[i][0] == events[i - 1][0])
    shares = [c / n for c in counts.values()]
    expected_repeat = sum(s * s for s in shares)
    observed_repeat = repeats / (n - 1) if n > 1 else 0.0
    payload_repeat = (sum(1 for i in range(1, n) if events[i][2] == events[i - 1][2]) / (n - 1)
                      if n > 1 else 0.0)
    transitions: collections.Counter = collections.Counter(
        (events[i - 1][0], events[i][0]) for i in range(1, n) if events[i - 1][0] != events[i][0])
    total_tr = sum(transitions.values())
    # Inverse Simpson index: the effective number of ordered actor pairs that a lag-1 statistic
    # really has data for. A plain count of pairs seen at least once saturates on any long log,
    # because one stray transition is enough to put a pair on the list, and then it cannot
    # distinguish a strict protocol from an open free-for-all. Weighting by how often each pair
    # actually occurs does distinguish them.
    diversity = (1.0 / sum((c / total_tr) ** 2 for c in transitions.values())
                 if total_tr else 0.0)
    tied = (sum(1 for i in range(1, n) if events[i][1] == events[i - 1][1]) / (n - 1)
            if n > 1 else 0.0)
    n_actors = len(counts)
    try:
        n_payloads: int | None = len({p for _, _, p in events})
        purity, purity_flat = _payload_purity(events, counts, n)
    except TypeError:
        # Payloads need not be hashable. They are only reordered and compared for equality in the
        # rest of the package, so an unhashable payload loses these two diagnostics and nothing else.
        n_payloads = None
        purity = purity_flat = None
    times = [t for _, t, _ in events]
    return {
        "n_events": n,
        "n_actors": n_actors,
        "n_distinct_payloads": n_payloads,
        "lag1_pairs_observed": len(transitions),
        "lag1_pairs_effective": round(diversity, 4),
        "lag1_pairs_possible": n_actors * (n_actors - 1),
        "tied_timestamp_rate": round(tied, 4),
        "busiest_actor": counts.most_common(1)[0][0] if counts else None,
        "busiest_actor_share": round(counts.most_common(1)[0][1] / n, 4) if counts else None,
        "time_span": round(times[-1] - times[0], 4) if n > 1 else 0.0,
        "repeat_speaker_rate": round(observed_repeat, 4),
        "repeat_speaker_rate_if_independent": round(expected_repeat, 4),
        "turn_taking_index": (round(observed_repeat / expected_repeat, 4)
                              if expected_repeat > 0 else None),
        "payload_repeat_rate": round(payload_repeat, 4),
        "payload_purity": None if purity is None else round(purity, 4),
        "payload_purity_if_actors_interchangeable": (None if purity_flat is None
                                                     else round(purity_flat, 4)),
        "payload_purity_lift": (None if not purity_flat else round(purity / purity_flat, 4)),
    }


def _payload_purity(events: Sequence[Event], counts: collections.Counter,
                    n: int) -> tuple[float, float]:
    """How predictable a payload is from its actor, against how predictable it is from nothing.

    Deliberately the crudest possible measure, the accuracy of guessing each turn's payload with its
    actor's modal payload, rather than a mutual information. A lift is readable without knowing what
    units it is in, and this diagnostic exists to be read by somebody deciding which null to trust.
    """
    per_actor: dict = collections.defaultdict(collections.Counter)
    overall: collections.Counter = collections.Counter()
    for a, _, p in events:
        per_actor[a][p] += 1
        overall[p] += 1
    purity = sum(c.most_common(1)[0][1] for c in per_actor.values()) / n
    flat = overall.most_common(1)[0][1] / n
    return purity, flat


def detector_name(detector: Callable) -> str:
    """A readable name for whatever callable the user handed us, for the report header."""
    for attr in ("name", "__name__"):
        value = getattr(detector, attr, None)
        if isinstance(value, str) and value:
            return value
    cls = getattr(detector, "__class__", None)
    if cls is not None and getattr(cls, "__name__", None):
        return cls.__name__
    return repr(detector)
