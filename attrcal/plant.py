"""A positive control: insert an actor that really does copy somebody, then see if it is found.

A family-wise error rate alone is satisfied by a detector that certifies nothing, so it cannot be
the only number reported. The cheapest honest complement is a spike-in. A synthetic actor is added
to the real log, it copies a chosen leader's payload a known fraction of the time at a known lag,
and the harness asks whether the detector returns that one edge. Sweeping the fraction and the
synthetic actor's volume traces where the detector stops working, on the user's own log rather than
on a simulation of it.

Two properties of the spike-in are worth being blunt about, because both limit what the resulting
power curve means.

The plant is the easiest possible positive. It copies a payload verbatim at a fixed lag with no
paraphrase, no delay jitter and no competing obligations, so a detection floor measured against it
is a LOWER bound on the strength of real influence a detector needs. Real agents copy ideas, not
strings, and nothing here measures that.

Inserting the plant lengthens the turn sequence. At lag 1 the plant's message sits immediately
after its leader's and the requested lag is realised exactly. At larger lags, or when several plant
messages are scheduled close together, other turns can land in between and push a copy past its
intended lag. The requested fraction is therefore a label and not a measurement, so
`plant_copycat` measures the fraction actually realised in the stream it produces and reports it
next to the request. Index any conclusion by the realised value.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .events import Event

DEFAULT_NAME = "attrcal_plant"


@dataclass(frozen=True)
class PlantSpec:
    """How to build the positive control and what to sweep.

    `strengths` is the copy fraction swept for the detection floor, at the fixed `volume`.
    `volumes` is an optional second sweep of the plant's message count at the fixed `strength`; it
    answers a different question, which is how much a quiet actor has to say before any method can
    attribute anything to it, and that is usually the binding constraint in a real log.

    `n_seeds` defaults to 10 because with n replicates a recall estimate can only take multiples of
    1/n, and below ten a floor defined at 80% power is a property of the grid rather than of the
    detector. The harness warns rather than refuses, because a user exploring with a slow detector
    has a legitimate reason to run three.
    """

    strengths: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.8)
    volume: int = 200
    volumes: Sequence[int] = ()
    strength: float = 0.4
    leader: Any = None
    lag: int = 1
    name: str = DEFAULT_NAME
    n_seeds: int = 10

    def __post_init__(self) -> None:
        for c in self.strengths:
            if not 0.0 <= float(c) <= 1.0:
                raise ValueError(f"strength {c} is not a fraction in [0, 1]")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError(f"strength {self.strength} is not a fraction in [0, 1]")
        if int(self.volume) < 1:
            raise ValueError("volume must be at least one message")
        for v in self.volumes:
            if int(v) < 1:
                raise ValueError(f"volume {v} must be at least one message")
        if int(self.lag) < 1:
            raise ValueError("lag must be at least one turn; a lag of 0 is simultaneity, "
                             "not influence")
        if int(self.n_seeds) < 1:
            raise ValueError("n_seeds must be at least 1")

    def as_dict(self) -> dict:
        return {"strengths": list(self.strengths), "volume": int(self.volume),
                "volumes": list(self.volumes), "strength": float(self.strength),
                "leader": self.leader, "lag": int(self.lag), "name": self.name,
                "n_seeds": int(self.n_seeds)}


def choose_leader(events: Sequence[Event]) -> Any:
    """Default leader for a spike-in: the busiest actor.

    The busiest actor is the easiest leader to attribute from, because it supplies the most
    opportunities for a copy to land. That makes it the right default for a floor that is meant to
    be read as a lower bound, and the wrong default if the question is whether a quiet actor can be
    attributed from at all. Pass `leader` explicitly for the second question.
    """
    counts = collections.Counter(a for a, _, _ in events)
    return counts.most_common(1)[0][0]


def plant_copycat(events: Sequence[Event], rng: np.random.Generator, *, leader: Any,
                  strength: float, volume: int, lag: int = 1,
                  name: str = DEFAULT_NAME) -> dict:
    """Insert a synthetic copycat of `leader` and return the spiked log plus what it really did.

    The plant emits `volume` messages. A fraction `strength` of them are copies: each is placed so
    that it falls `lag` turns after one of the leader's own messages and carries that message's
    payload. The rest are placed at uniformly random points in the log and carry payloads drawn
    from the log's own payload distribution, so the plant's marginal looks like everybody else's
    and a detector cannot find it by noticing an actor that says unusual things.

    Timestamps for inserted messages are interpolated strictly between their host turn and the next
    one. When the host and its successor share a timestamp, which coarse logs do constantly, the
    insert takes the host's timestamp and relies on the stable sort in `normalise_events` to keep
    it immediately after its host.
    """
    n = len(events)
    if n < 2:
        raise ValueError("cannot plant into a log of fewer than two events")
    actors_present = {a for a, _, _ in events}
    if name in actors_present:
        raise ValueError(f"the plant's name {name!r} is already an actor in the log; choose "
                         "another, otherwise the planted edge is not distinguishable from "
                         "whatever that actor really did")
    if leader not in actors_present:
        raise ValueError(f"leader {leader!r} does not appear in the log")
    leader_pos = [i for i, (a, _, _) in enumerate(events) if a == leader]
    volume = int(volume)
    lag = int(lag)
    n_copy_requested = int(round(float(strength) * volume))
    capped = n_copy_requested > len(leader_pos)
    n_copy = min(n_copy_requested, len(leader_pos))
    n_noise = volume - n_copy

    payloads = [e[2] for e in events]
    inserts: list[tuple[int, Any]] = []
    if n_copy:
        # Without replacement: two copies hosted on the same leader message cannot both land at the
        # requested lag, so sampling with replacement would quietly lower the realised fraction.
        picked = rng.choice(len(leader_pos), size=n_copy, replace=False)
        for j in picked:
            src = leader_pos[int(j)]
            host = min(n - 1, src + lag - 1)
            inserts.append((host, payloads[src]))
    if n_noise:
        hosts = rng.integers(0, n, size=n_noise)
        draws = rng.integers(0, n, size=n_noise)
        for h, d in zip(hosts, draws):
            inserts.append((int(h), payloads[int(d)]))

    by_host: dict[int, list[Any]] = collections.defaultdict(list)
    for host, payload in inserts:
        by_host[host].append(payload)

    out: list[Event] = []
    for i, (a, t, p) in enumerate(events):
        out.append((a, float(t), p))
        extra = by_host.get(i)
        if not extra:
            continue
        nxt = float(events[i + 1][1]) if i + 1 < n else float(t) + 1.0
        span = max(nxt - float(t), 0.0)
        for k, payload in enumerate(extra):
            frac = (k + 1) / (len(extra) + 1)
            out.append((name, float(t) + span * frac, payload))

    realised = realised_copy_fraction(out, leader=leader, name=name, lag=lag)
    return {
        "events": out,
        "edge": (leader, name),
        "leader": leader,
        "name": name,
        "lag": lag,
        "strength_requested": float(strength),
        "strength_realised": realised,
        "volume": volume,
        "n_copies_requested": n_copy_requested,
        "n_copies_placed": n_copy,
        "capped": capped,
        "cap_note": (f"requested {n_copy_requested} copies but {leader!r} only speaks "
                     f"{len(leader_pos)} times, so only {n_copy} were placed" if capped else None),
    }


def realised_copy_fraction(events: Sequence[Event], *, leader: Any, name: Any, lag: int) -> float:
    """The fraction of the plant's messages that really sit `lag` turns after the leader's, copying it.

    This is the number to index a power curve by. The requested fraction is what was asked for; this
    is what the detector was actually shown, and the two come apart whenever an insert gets pushed
    off its intended lag by another turn landing in between.
    """
    acts = [e[0] for e in events]
    pays = [e[2] for e in events]
    plant_idx = [i for i, a in enumerate(acts) if a == name]
    if not plant_idx:
        return 0.0
    hits = sum(1 for i in plant_idx
               if i - lag >= 0 and acts[i - lag] == leader and pays[i] == pays[i - lag])
    return round(hits / len(plant_idx), 4)
