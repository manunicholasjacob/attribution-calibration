"""Reference detectors, provided so the harness can be demonstrated and tested on itself.

These are not the point of the package and they are not a recommendation. A user brings their own
detector. What these exist for is to give the harness something whose behaviour is known in
advance, so that a test can assert that the harness reports the right thing: a detector that
certifies every pair must come out with a family-wise error of 1, a detector that certifies nothing
must come out with no power and a family-wise error of 0, and a detector that does a defensible
thing must come out somewhere sensible in between.

`CopyRateDetector` is the shape of method this package was written about: a statistic, a threshold
that sounds reasonable, and no null model at all. `SurrogateCopyDetector` is the same statistic with
a surrogate test and a family-wise correction wrapped round it. Running both through `calibrate` on
the same log is the shortest demonstration of what the harness is for.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .events import Event, normalise_events
from .nulls import INDEX_NULLS, NULLS, make_null, null_index


def all_pairs(events: Sequence[Any]) -> set:
    """Certify every ordered pair of distinct actors. Deliberately the worst possible detector.

    It has perfect recall of anything, which is exactly why recall on its own is not evidence. Used
    in tests as the upper bound on how badly a detector can score on nulls.
    """
    ev = normalise_events(events)
    actors = sorted({a for a, _, _ in ev}, key=repr)
    return {(a, b) for a in actors for b in actors if a != b}


def no_edges(events: Sequence[Any]) -> set:
    """Certify nothing, ever. Perfect family-wise error, zero power.

    The reason this is in the package rather than only in the tests: a detector that stays silent
    is indistinguishable from a well calibrated one if the only number reported is a false positive
    rate, and the easiest way to see that is to run this one through the harness and read its
    report.
    """
    normalise_events(events)
    return set()


def _codes(values: Sequence[Any]) -> tuple[np.ndarray, list]:
    seen: dict = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        code = seen.get(v)
        if code is None:
            code = len(seen)
            seen[v] = code
        out[i] = code
    return out, list(seen)


def _opportunities(actor_codes: np.ndarray, n_actors: int, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """For each ordered pair, how many turns offer that pair the chance to show a lag-`lag` copy.

    The opportunity count depends only on the actor sequence, which no surrogate in this package
    touches, so it is computed once and reused across every replicate. That is what makes the
    surrogate test affordable.
    """
    key = actor_codes[:-lag] * n_actors + actor_codes[lag:]
    opps = np.bincount(key, minlength=n_actors * n_actors).astype(float)
    return key, opps


class CopyRateDetector:
    """Fires when a pair's lag-`lag` payload copy rate clears a threshold. No null model at all.

    This is deliberately uncalibrated, and it is uncalibrated in the specific way that published
    attribution claims usually are: the statistic is reasonable, the threshold is a round number
    somebody chose, and there is nothing to compare against, so the method has no stated error rate
    and no way to acquire one.

    `threshold` is an absolute copy rate, which is how such a rule is normally written down: "we
    record an influence edge where the follower repeats the leader's subject more than 40% of the
    time". Pass `ratio` instead to threshold against the log's own overall copy rate, which is a
    slightly better rule and still has no error rate.

    What makes this worth shipping rather than only writing about: it will look clean under a
    permutation null and certify most of the graph under a shift or rotation null, on the same log,
    because what it is really measuring is that two actors habitually talk about the same thing. A
    user who runs it through `compare_nulls` sees that in two lines.
    """

    name = "CopyRateDetector"

    def __init__(self, lag: int = 1, min_opportunities: int = 20, threshold: float = 0.4,
                 ratio: float | None = None) -> None:
        if lag < 1:
            raise ValueError("lag must be at least one turn")
        if ratio is not None and ratio <= 0:
            raise ValueError("ratio must be positive")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be a copy rate in (0, 1]")
        self.lag = int(lag)
        self.min_opportunities = int(min_opportunities)
        self.threshold = float(threshold)
        self.ratio = None if ratio is None else float(ratio)

    def __call__(self, events: Sequence[Any]) -> set:
        ev = normalise_events(events)
        lag = self.lag
        if len(ev) <= lag + 1:
            return set()
        acts, actor_names = _codes([e[0] for e in ev])
        pays, _ = _codes([e[2] for e in ev])
        n_actors = len(actor_names)
        key, opps = _opportunities(acts, n_actors, lag)
        match = (pays[lag:] == pays[:-lag]).astype(float)
        hits = np.bincount(key, weights=match, minlength=n_actors * n_actors)
        cut = (self.ratio * float(match.mean())) if self.ratio is not None else self.threshold
        out = set()
        for k in np.nonzero(opps >= self.min_opportunities)[0]:
            s, d = divmod(int(k), n_actors)
            if s == d:
                continue
            if hits[k] / opps[k] > cut:
                out.add((actor_names[s], actor_names[d]))
        return out


class SurrogateCopyDetector:
    """The same copy statistic, standardised, with a max-statistic surrogate test over all pairs.

    Three choices here are worth the words, because each one is a mistake that a detector of this
    shape commonly makes.

    The statistic is standardised rather than used as a raw rate. Under the null the number of
    copies for a pair is roughly binomial in that pair's opportunity count, so a raw rate from a
    pair with thirty opportunities is far noisier than one from a pair with three thousand. A
    maximum taken over raw rates is therefore dominated by the quietest pair tested, which makes
    the test conservative for exactly the pairs a reader cares about.

    The null distribution is the maximum over all tested pairs, not the per-pair distribution. A
    directed claim about a swarm is a claim about one edge selected out of the whole family, so the
    error rate that matters is the family-wise one. Testing each pair at 0.05 on a twelve-actor log
    tests 132 pairs and certifies something on most structure-free replicates.

    Which surrogate it uses is a constructor argument rather than a fixed choice, so the same
    detector can be audited against a different surrogate than the one it trusts. That is the
    comparison the harness exists to make: passing `null="shift"` here and auditing with
    `null="permute"` in `calibrate` measures what the conventional surrogate costs.
    """

    name = "SurrogateCopyDetector"

    def __init__(self, lag: int = 1, min_opportunities: int = 20, n_surrogates: int = 200,
                 alpha: float = 0.05, null: str = "permute", seed: int = 0,
                 block_size: int | None = None) -> None:
        if lag < 1:
            raise ValueError("lag must be at least one turn")
        if n_surrogates < 1:
            raise ValueError("n_surrogates must be at least 1")
        if null not in NULLS:
            raise ValueError(f"unknown null {null!r}; expected one of {NULLS}")
        self.lag = int(lag)
        self.min_opportunities = int(min_opportunities)
        self.n_surrogates = int(n_surrogates)
        self.alpha = float(alpha)
        self.null = str(null)
        self.seed = int(seed)
        self.block_size = block_size
        #: Diagnostics from the most recent call. The family size is needed to read any
        #: family-wise error rate, and the number of pairs dropped for want of opportunities is
        #: needed to know whether the edge a user cares about was tested at all.
        self.last_tested: dict = {}

    def __call__(self, events: Sequence[Any]) -> set:
        ev = normalise_events(events)
        lag = self.lag
        self.last_tested = {"n_actors": 0, "family_size": 0, "tested": 0, "dropped": 0}
        if len(ev) <= lag + 1:
            return set()
        acts, actor_names = _codes([e[0] for e in ev])
        pays, _ = _codes([e[2] for e in ev])
        n_actors = len(actor_names)
        key, opps = _opportunities(acts, n_actors, lag)
        eligible = opps >= self.min_opportunities
        for s in range(n_actors):
            eligible[s * n_actors + s] = False
        tested = np.nonzero(eligible)[0]
        family = n_actors * (n_actors - 1)
        self.last_tested = {"n_actors": n_actors, "family_size": family,
                            "tested": int(tested.size), "dropped": int(family - tested.size)}
        if tested.size == 0:
            return set()
        opps_t = opps[tested]

        def z_scores(payload_codes: np.ndarray) -> np.ndarray:
            match = (payload_codes[lag:] == payload_codes[:-lag]).astype(float)
            q = float(match.mean())
            hits = np.bincount(key, weights=match, minlength=n_actors * n_actors)[tested]
            var = opps_t * q * (1.0 - q)
            return np.where(var > 0.0, (hits - opps_t * q) / np.sqrt(np.maximum(var, 1e-12)), 0.0)

        observed = z_scores(pays)
        rng = np.random.default_rng(self.seed)
        null_max = np.empty(self.n_surrogates)
        actor_seq = acts.tolist()
        for r in range(self.n_surrogates):
            if self.null in INDEX_NULLS:
                idx = null_index(self.null, actor_seq, rng, block_size=self.block_size)
                null_max[r] = z_scores(pays[idx]).max()
            else:
                # The shift surrogate moves the actor sequence, so the opportunity counts and the
                # whole pair indexing change per replicate and none of the precomputation above
                # applies. It is handled by rebuilding the replicate and recomputing from scratch,
                # which is slower and is the honest cost of a null that is not a surrogate of the
                # log it came from.
                rep = make_null(ev, self.null, rng, block_size=self.block_size)
                null_max[r] = self._recomputed_max(rep)
        # The +1 in both numerator and denominator is the standard correction for a Monte Carlo
        # p-value. Without it a statistic larger than every surrogate gets p = 0, which claims more
        # resolution than the number of surrogates can support and is strictly anti-conservative.
        out = set()
        for i, k in enumerate(tested):
            p = (1.0 + float((null_max >= observed[i]).sum())) / (self.n_surrogates + 1.0)
            if p < self.alpha:
                s, d = divmod(int(k), n_actors)
                out.add((actor_names[s], actor_names[d]))
        return out

    def _recomputed_max(self, events: Sequence[Event]) -> float:
        """The maximum standardised statistic over a replicate whose actor sequence has moved."""
        lag = self.lag
        acts, names = _codes([e[0] for e in events])
        pays, _ = _codes([e[2] for e in events])
        n_actors = len(names)
        key, opps = _opportunities(acts, n_actors, lag)
        eligible = opps >= self.min_opportunities
        for s in range(n_actors):
            eligible[s * n_actors + s] = False
        tested = np.nonzero(eligible)[0]
        if tested.size == 0:
            return 0.0
        match = (pays[lag:] == pays[:-lag]).astype(float)
        q = float(match.mean())
        hits = np.bincount(key, weights=match, minlength=n_actors * n_actors)[tested]
        o = opps[tested]
        var = o * q * (1.0 - q)
        z = np.where(var > 0.0, (hits - o * q) / np.sqrt(np.maximum(var, 1e-12)), 0.0)
        return float(z.max())
