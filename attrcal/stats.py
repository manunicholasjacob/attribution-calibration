"""Rates that carry intervals, and a detection floor that admits what it does not know.

Two decisions here are deliberate, and both came from watching bare numbers mislead readers.

First, no rate leaves this package without an interval. A family-wise error of 0.00 measured over
20 replicates and one measured over 2000 are very different claims and have the same point
estimate, so a reader given only the point estimate cannot tell them apart. The Wilson interval is
used rather than the normal approximation because every interesting rate in this harness sits near
0 or near 1, and that is exactly where the normal interval leaves the unit square and starts
reporting negative error rates.

Second, a detection floor is a status and not a number. "Power never reached the threshold",
"power was already above it at the smallest value swept", "there was no sweep at all" and "power
crossed the threshold and then fell back below it" are four different findings. Returning None for
the first three and a first crossing for the fourth hides all four, and the fourth is the dangerous
one: it reads as a floor while the curve above it says the effect is not reliably detected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

# Two-sided 95% normal quantile. Named rather than inlined so a caller changing the confidence
# level has one place to look.
Z95 = 1.959963984540054

# Below this many replicates per cell, a recall threshold such as 0.8 is not estimable: with n
# seeds the only reachable recall values are multiples of 1/n, so a reported floor is an artifact
# of the grid rather than a measurement.
MIN_SEEDS_FOR_POWER = 10


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    Chosen over the Wald interval because it stays inside [0, 1] and keeps useful width at k = 0
    and k = n, which is where a well behaved detector's family-wise error and a saturated power
    curve both live. An empty sample returns the whole unit interval, which is the honest answer.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Rate:
    """A proportion together with the sample it came from and its interval.

    The sample size travels with the estimate on purpose. Every rate this package prints is a
    `Rate`, so there is no code path that can emit a proportion with its denominator dropped.
    """

    k: int
    n: int
    point: float
    lo: float
    hi: float
    z: float = Z95

    @classmethod
    def of(cls, k: int, n: int, z: float = Z95) -> "Rate":
        k, n = int(k), int(n)
        if k < 0 or n < 0 or k > n:
            raise ValueError(f"Rate.of needs 0 <= k <= n, got k={k}, n={n}")
        lo, hi = wilson(k, n, z)
        return cls(k=k, n=n, point=(k / n if n else float("nan")), lo=lo, hi=hi, z=z)

    def covers(self, value: float) -> bool:
        """Whether the interval contains a reference value, for example a nominal 0.05."""
        return self.lo <= value <= self.hi

    def __float__(self) -> float:
        return float(self.point)

    def __str__(self) -> str:
        if self.n == 0:
            return "no data"
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] ({self.k}/{self.n})"

    def as_dict(self) -> dict:
        return {"point": None if self.n == 0 else round(self.point, 6),
                "lo": round(self.lo, 6), "hi": round(self.hi, 6), "k": self.k, "n": self.n}


def detection_floor(curve: Mapping[Any, Any], threshold: float = 0.8,
                    n_seeds: int | None = None) -> dict:
    """The smallest swept value at which power reaches `threshold`, with an explicit status.

    `curve` maps a swept value to either a float recall or a `Rate`. Values are compared on their
    point estimates; the interval is not used to decide the crossing, because a floor defined by an
    interval edge moves when the replicate count changes and so is not a property of the detector.

    Status is one of:

      no_data        the curve is empty, so nothing was swept
      never_reached  power stayed below the threshold at every value swept
      below_grid     power was already at or above the threshold at the smallest value swept, so
                     the true floor is somewhere below the grid and this is an upper bound
      bracketed      power crossed the threshold and stayed at or above it for every larger value
      non_monotone   power crossed the threshold and then fell back below it at some larger value

    `non_monotone` exists because the first crossing of a ragged curve reads exactly like a floor
    and is not one. With few replicates per cell it is usually noise, which is why the seed count
    is folded into the note rather than left for the reader to look up.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    note = None
    if n_seeds is not None and n_seeds < MIN_SEEDS_FOR_POWER:
        note = (f"coarse: {n_seeds} seed(s) per cell resolve recall only to 1/{max(1, n_seeds)}, "
                f"so a {threshold:.0%} threshold is not estimable from this grid")
    base = {"floor": None, "status": "no_data", "note": note, "threshold": threshold,
            "monotone": None, "n_seeds": n_seeds}
    if not curve:
        return base
    items = [(k, float(v)) for k, v in sorted(curve.items())]
    above = [k for k, v in items if v >= threshold]
    if not above:
        return {**base, "status": "never_reached"}
    first = above[0]
    tail = [v for k, v in items if k >= first]
    monotone = all(v >= threshold for v in tail)
    if items[0][1] >= threshold:
        return {**base, "floor": items[0][0], "status": "below_grid", "monotone": monotone}
    return {**base, "floor": first, "status": "bracketed" if monotone else "non_monotone",
            "monotone": monotone}
