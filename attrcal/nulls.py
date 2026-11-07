"""Surrogate logs built from the user's own log, which is the whole point of this package.

Every attribution claim is a comparison against something. The usual something is a simulation, a
hand-built toy, or nothing at all, and all three let a detector's error rate be whatever the author
hopes it is. The alternative implemented here is to build the comparison out of the log the claim is
being made about: hold the actor sequence exactly as it is, including every turn, every volume share
and every idle stretch, and only change which payload sits at which turn. What comes out contains no
directed structure by construction, because nothing about who spoke when now has anything to do with
what was said. Running a detector over many such replicates measures the rate at which it certifies
an edge where none can exist, on the user's own data rather than on a stand-in for it.

The four nulls differ in which nuisance structure they keep, and that is the only thing that
separates them. A detector's error rate is therefore a property of the detector AND the null, and a
published rate that does not name its null is not a rate.

  permute       reshuffle payloads over the whole log. Keeps volumes and the global payload
                marginal, destroys everything else, including each actor's characteristic payloads
                and the channel's drift. This is the CALIBRATION null and the default: it asks what
                the detector does when there is no structure of any kind left to find.

  blockpermute  reshuffle payloads only within contiguous blocks of turns. Slow drift in what the
                channel is talking about survives, so every pair of actors stays genuinely dependent
                through the channel while no actor influences another. This is a CONFOUNDING null,
                and a detector can pass permute and fail this one.

  rotate        rotate each actor's own payload subsequence circularly among that actor's own turns.
                Keeps each actor's payload marginal and each actor's own autocorrelation exactly, so
                an apparent edge that is really just two actors with overlapping habitual subjects
                survives into the surrogate and stops being certified. The strictest of the three
                that hold the actor sequence fixed.

  shift         the conventional circular-shift surrogate, implemented faithfully: each actor's
                emissions are slid along the turn index by their own random offset. INVALID on
                turn-ordered logs, and offered here to be measured rather than used. It does not
                hold the actor sequence fixed, which has two consequences, both measurable in the
                replicate it returns. Turns collide and turns empty, so actors end up adjacent that
                the protocol never allowed to be adjacent, and a lag-1 statistic suddenly has data
                for pairs the real log gave it none for. And because every one of an actor's events
                moves by the same offset, each pair of actors ends up compared at its own fixed
                arbitrary offset, so the pairs whose offsets happen to land close to each other keep
                the log's payload autocorrelation intact and look like copies. This is the recipe
                that came out at a family-wise error of 0.90 against a nominal 0.05 on real agent
                chat, and `null_index` refuses to express it as a payload remapping precisely
                because it is not one.

What none of these control for is a common cause acting at the same lag as the influence being
claimed. No reshuffle of the payload column can separate those, because at that lag they are the
same pattern. Separating them needs a detector that conditions on the channel's own state, and this
harness will not tell anyone they need one; it will only tell them that their error rate is not what
they said it was.
"""
from __future__ import annotations

import collections
from typing import Any, Sequence

import numpy as np

from .events import Event

#: Surrogate kinds, in the order they should be preferred. `shift` is last because it is the one
#: known to be invalid on the logs this package exists for.
NULLS = ("permute", "blockpermute", "rotate", "shift")

#: The subset expressible as a remapping of the payload column, which is also the subset that holds
#: the actor sequence fixed and is therefore the subset that is a surrogate of the caller's log.
INDEX_NULLS = ("permute", "blockpermute", "rotate")

NULL_NOTES = {
    "permute": "payloads permuted over the whole log, actor sequence held fixed "
               "(calibration null)",
    "blockpermute": "payloads permuted within blocks of turns, so channel drift survives "
                    "(confounding null)",
    "rotate": "each actor's payloads rotated among its own turns, so actor-characteristic "
              "payloads survive (strict null)",
    "shift": "each actor's emissions slid along the turn index: the conventional circular shift, "
             "which moves the actor sequence and is INVALID on turn-ordered logs",
}

#: Default number of blocks the log is cut into for `blockpermute`. Twenty is a compromise: enough
#: blocks that each is short relative to the drift timescale of a typical chat log, and few enough
#: that each still holds plenty of turns to permute.
DEFAULT_N_BLOCKS = 20


def _groups(actors: Sequence[Any]) -> dict[Any, list[int]]:
    out: dict[Any, list[int]] = collections.defaultdict(list)
    for i, a in enumerate(actors):
        out[a].append(i)
    return out


def _permute_index(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n)


def _blockpermute_index(n: int, rng: np.random.Generator, block_size: int) -> np.ndarray:
    if block_size < 2:
        raise ValueError(f"block_size must be at least 2, got {block_size}")
    idx = np.arange(n)
    # A trailing block shorter than two turns cannot be permuted, so it is folded into the previous
    # block rather than left in place. Left in place it would be a stretch of the surrogate that is
    # identical to the real log, which is a small but systematic leak of real structure.
    starts = list(range(0, n, block_size))
    if len(starts) > 1 and n - starts[-1] < 2:
        starts.pop()
    for b, start in enumerate(starts):
        stop = starts[b + 1] if b + 1 < len(starts) else n
        if stop - start < 2:
            continue
        idx[start:stop] = start + rng.permutation(stop - start)
    return idx


def _rotate_index(actors: Sequence[Any], rng: np.random.Generator) -> np.ndarray:
    n = len(actors)
    idx = np.arange(n)
    for pos in _groups(actors).values():
        m = len(pos)
        if m < 2:
            continue
        # A rotation of zero would hand back the real log, so offsets are drawn from 1..m-1. Note
        # that an actor with m turns has only m-1 non-trivial rotations available, which caps how
        # many genuinely distinct replicates this null can build for a quiet actor. That cap is one
        # reason rotation-style surrogates behave badly on sparse series, and it is why the number
        # of replicates requested is not the same thing as the number obtained.
        k = int(rng.integers(1, m))
        arr = np.asarray(pos)
        idx[arr] = np.roll(arr, -k)
    return idx


def null_index(kind: str, actors: Sequence[Any], rng: np.random.Generator,
               block_size: int | None = None) -> np.ndarray:
    """The index map for one replicate: surrogate payload at turn i is real payload at turn idx[i].

    Returning an index map rather than a rebuilt log lets a detector apply the same surrogate to its
    own coded arrays without paying to rebuild a list of tuples, and guarantees that a detector's
    internal null and the harness's audit null are literally the same operation.

    `shift` is rejected here, and the rejection is the documentation. A null that can be written as a
    remapping of the payload column cannot change who spoke when; one that cannot be written that way
    does change it, and has therefore stopped being a surrogate of the caller's log.
    """
    if kind == "shift":
        raise ValueError(
            "the 'shift' surrogate cannot be expressed as a payload remapping, because it slides "
            "actors' emissions along the turn index and so changes who spoke when. That is exactly "
            "why it is invalid on a turn-ordered log. Use make_null() if you want to measure it.")
    if kind not in INDEX_NULLS:
        raise ValueError(f"unknown null {kind!r}; expected one of {INDEX_NULLS}")
    n = len(actors)
    if n < 2:
        raise ValueError("a surrogate needs at least two events")
    if kind == "permute":
        return _permute_index(n, rng)
    if kind == "blockpermute":
        size = block_size if block_size is not None else max(2, n // DEFAULT_N_BLOCKS)
        return _blockpermute_index(n, rng, size)
    return _rotate_index(actors, rng)


def _shift_events(events: Sequence[Event], rng: np.random.Generator) -> list[Event]:
    """The conventional circular shift, applied to the turn index, one offset per actor.

    This is the recipe as it is usually written: represent the log as one series per actor over a
    shared time grid, slide each series by its own random offset, and read the result back. Each
    actor keeps its volume and its own payload sequence in cyclic order, which is what makes the
    recipe look conservative.

    What it also does is move the actor sequence, and on a turn-ordered log that is fatal twice over.
    Two actors shifted onto the same turn now speak simultaneously, which the real protocol never
    allowed, and a turn nobody is shifted onto simply disappears; the adjacencies that a lag-1
    statistic reads off the replicate therefore include pairs the real conversation never put next to
    each other. Worse, because an actor's whole series moves by one offset, each pair of actors is
    compared at a fixed arbitrary offset for the whole replicate, so the pairs whose offsets land
    near each other keep the log's own payload autocorrelation and read as copying. The replicate has
    the right actors, the right volumes and the right payloads, and the wrong conversation. Any error
    rate measured against it is an error rate for a log the user does not have.
    """
    n = len(events)
    times = [float(e[1]) for e in events]
    offsets = {a: int(rng.integers(1, n)) for a in _groups([e[0] for e in events])}
    moved = [((i + offsets[a]) % n, a, p) for i, (a, _t, p) in enumerate(events)]
    # Stable sort on destination turn only, so actors colliding on one turn keep a deterministic
    # order instead of being reordered by their payloads.
    moved.sort(key=lambda r: r[0])
    return [(a, times[j], p) for j, a, p in moved]


def make_null(events: Sequence[Event], kind: str, rng: np.random.Generator,
              block_size: int | None = None) -> list[Event]:
    """One surrogate event log of the same length, carrying the same multiset of payloads.

    For the three index nulls the actor and timestamp columns are returned untouched, so anything a
    detector measures about volumes, timing or the payload marginal is unchanged and only the
    association between the columns is gone. For `shift` the actor column moves, which is its defect
    and is documented above.
    """
    if kind not in NULLS:
        raise ValueError(f"unknown null {kind!r}; expected one of {NULLS}")
    if len(events) < 2:
        raise ValueError("a surrogate needs at least two events")
    if kind == "shift":
        return _shift_events(events, rng)
    payloads = [e[2] for e in events]
    idx = null_index(kind, [e[0] for e in events], rng, block_size=block_size)
    return [(a, t, payloads[int(j)]) for (a, t, _), j in zip(events, idx)]


def preserves_actor_sequence(kind: str) -> bool:
    """Whether a null leaves who-spoke-when exactly as it was.

    The single most useful property to be able to ask about a surrogate, and the one the
    conventional recipe fails. A null that answers False is not a surrogate of the caller's log, and
    the harness says so in its report rather than leaving the reader to work it out.
    """
    if kind not in NULLS:
        raise ValueError(f"unknown null {kind!r}; expected one of {NULLS}")
    return kind in INDEX_NULLS
