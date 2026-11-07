"""A small synthetic log, for demonstrating and testing the harness. Not a benchmark.

This is scoped deliberately. A benchmark needs a generator whose labels are exact, whose nulls are
separated into calibration and confounding families, and whose realism limits are measured; that is
a separate piece of work and it is not this file. What is needed here is a log with two properties
the harness depends on, so that the tests assert against something whose shape is known:

  turn-taking     speakers do not follow themselves, so the actor sequence carries order
                  information that a circular-shift surrogate preserves
  payload drift   consecutive turns often share a payload, so there is autocorrelation for a
                  shift surrogate to keep and a permutation to destroy

Those two together are the regime in which the conventional surrogate was measured to fail, and a
log without them cannot show the difference between the nulls. `turn_taking` and `drift` are the
knobs; set `turn_taking=0.0` for independently drawn speakers, which is the regime in which the
conventional surrogate behaves acceptably.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .events import Event


def synth_log(n_events: int = 3000, n_actors: int = 6, n_payloads: int = 8,
              turn_taking: float = 0.85, drift: float = 0.08, common_cause: float = 0.6,
              mean_gap: float = 20.0, seed: int = 0,
              copy_edges: Sequence[tuple[int, int, float]] = ()) -> dict:
    """One synthetic log with known structure, returned as events plus the planted edge set.

    `turn_taking` is the probability that the next speaker is the next actor in a fixed rotation
    rather than a uniform draw. At 1.0 the sequence is strict round robin; at 0.0 speakers are
    independent. `drift` is the per-turn probability that the channel changes subject, and
    `common_cause` is the probability that a speaker addresses the channel's current subject rather
    than its own habitual one, so raising it makes every pair of actors genuinely dependent through
    the channel without any actor influencing another.

    `copy_edges` is an iterable of `(leader, follower, probability)`: when the leader speaks, the
    follower takes the next turn with that probability and repeats the leader's payload. The
    returned `ground_truth` names those edges. Note that a follower taking the turn displaces
    whoever would otherwise have spoken, so a planted edge changes the sequence for everybody and
    the label set does not say so; precision measured against `ground_truth` is a lower bound.
    """
    if not 0.0 <= turn_taking <= 1.0:
        raise ValueError(f"turn_taking must be a probability, got {turn_taking}")
    if not 0.0 <= common_cause <= 1.0:
        raise ValueError(f"common_cause must be a probability, got {common_cause}")
    if n_actors < 2:
        raise ValueError("a log needs at least two actors")
    rng = np.random.default_rng(seed)
    edges = [(int(a), int(b), float(c)) for a, b, c in copy_edges]
    for a, b, c in edges:
        if not (0 <= a < n_actors and 0 <= b < n_actors):
            raise ValueError(f"edge ({a},{b}) is outside 0..{n_actors - 1}")
        if a == b:
            raise ValueError(f"self-edge ({a},{b}): no pairwise statistic tests an actor "
                             "against itself")
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"probability {c} for edge ({a},{b}) is not in [0, 1]")
    followers: dict[int, list[tuple[int, float]]] = {}
    for a, b, c in edges:
        followers.setdefault(a, []).append((b, c))

    home = rng.integers(0, n_payloads, size=n_actors)
    subject = int(rng.integers(n_payloads))
    acts: list[int] = []
    pays: list[int] = []
    times: list[float] = []
    t = 0.0
    queued: tuple[int, int] | None = None
    last = int(rng.integers(n_actors))
    for _ in range(n_events):
        t += float(rng.exponential(mean_gap))
        if rng.random() < drift:
            subject = int(rng.integers(n_payloads))
        if queued is not None:
            speaker, payload = queued
            queued = None
        else:
            if rng.random() < turn_taking:
                speaker = (last + 1) % n_actors
            else:
                speaker = int(rng.integers(n_actors))
            payload = (subject if rng.random() < common_cause else int(home[speaker]))
        acts.append(speaker)
        pays.append(payload)
        times.append(t)
        last = speaker
        for follower, prob in followers.get(speaker, ()):
            if queued is None and rng.random() < prob:
                queued = (follower, payload)

    events: list[Event] = [(f"actor{a:02d}", float(tt), int(p))
                           for a, tt, p in zip(acts, times, pays)]
    truth = {(f"actor{a:02d}", f"actor{b:02d}") for a, b, c in edges if c > 0}
    return {"events": events, "ground_truth": truth,
            "config": {"n_events": n_events, "n_actors": n_actors, "n_payloads": n_payloads,
                       "turn_taking": turn_taking, "drift": drift,
                       "common_cause": common_cause, "seed": seed,
                       "copy_edges": [[a, b, c] for a, b, c in edges]}}
