"""The three numbers a directed claim should carry, and a printable block that states them.

The format of `summary()` is part of the deliverable rather than decoration. A user who runs this
harness and then writes "agent A influenced agent B" has to put something next to that sentence,
and if the harness does not hand them a paste-ready block they will paste the point estimates and
drop the intervals, the sample sizes and the status of the floor. So every line that carries a rate
carries its interval and its denominator, the floor is printed with its status word, and the
caveats that apply to the numbers above them are printed underneath rather than left in the
documentation.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

from .nulls import NULL_NOTES
from .stats import Rate

#: Below this, a log's speaker sequence carries enough order information that the circular-shift
#: surrogate has something real to preserve. The value is a judgement call and is printed with the
#: measurement so a reader can disagree with it.
TURN_TAKING_FLAG = 0.5

_WIDTH = 86


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=_WIDTH, initial_indent=indent, subsequent_indent=indent)


@dataclass
class Report:
    """Everything `calibrate` measured, with the point estimates the brief asks for on top.

    `familywise_error` is a plain float because that is what a caller wants to compare against
    0.05. Its interval and denominator are beside it rather than inside it, so there is no way to
    read the rate without them being one attribute away.
    """

    familywise_error: float
    familywise_error_interval: tuple[float, float]
    familywise_error_n: int
    familywise_error_k: int
    mean_edges_per_null: float
    max_edges_on_a_null: int
    detection_floor: dict
    power_curve: dict
    volume_curve: dict = field(default_factory=dict)
    detection_floor_volume: dict = field(default_factory=dict)
    realised_strength: dict = field(default_factory=dict)
    symbolisation: dict | None = None
    log: dict = field(default_factory=dict)
    null_audit: dict = field(default_factory=dict)
    null: str = "permute"
    n_null: int = 0
    alpha: float = 0.05
    seed: int = 0
    detector: str = "detector"
    planted: dict | None = None
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------------ derived readings

    @property
    def familywise_error_rate(self) -> Rate:
        """The family-wise error as a `Rate`, for code that wants the interval to travel with it.

        The success count is carried as its own field rather than recovered from the point estimate.
        With no replicates the point estimate is NaN, and reconstructing the count from it raised
        rather than reporting that there was no measurement.
        """
        lo, hi = self.familywise_error_interval
        return Rate(k=self.familywise_error_k, n=self.familywise_error_n,
                    point=self.familywise_error, lo=lo, hi=hi)

    def calibrated(self, nominal: float | None = None) -> bool:
        """Whether the family-wise error interval covers the nominal rate.

        Deliberately not "whether the point estimate is below nominal". With 200 replicates an
        observed 0.03 and an observed 0.09 are both consistent with a nominal 0.05, and a harness
        that called the first calibrated and the second broken would be reporting noise.
        """
        target = self.alpha if nominal is None else nominal
        return self.familywise_error_rate.covers(target)

    # ------------------------------------------------------------------ output

    def summary(self) -> str:
        """A short, honest block suitable for pasting into a paper or an appendix."""
        lines: list[str] = []
        add = lines.append
        add("attrcal calibration report")
        add("=" * 26)
        log = self.log
        # The payload count is None when payloads are unhashable, which is allowed: they are only
        # reordered and compared for equality. Printing the word is clearer than printing None.
        payloads = log.get("n_distinct_payloads")
        add(f"log           : {log.get('n_events')} events, {log.get('n_actors')} actors, "
            f"{'unknown' if payloads is None else payloads} distinct payloads")
        add(f"detector      : {self.detector}")
        add(f"surrogate     : {self.null} ({NULL_NOTES.get(self.null, 'unknown')})")
        add(f"replicates    : {self.n_null} null, seed {self.seed}")
        add("")

        add("1. FAMILY-WISE ERROR  (share of structure-free replicates certifying >= 1 edge)")
        add(f"   {self.familywise_error_rate}   nominal target {self.alpha:.3f}")
        if self.familywise_error_n == 0:
            add(_wrap("no null replicates were run, so there is no error rate here. Any edge this "
                      "detector certifies is uncalibrated.", "   "))
        elif self.calibrated():
            add(_wrap(f"the interval covers {self.alpha:.3f}, so this run does not show the "
                      "detector to be miscalibrated on this log. It is not a proof that it is "
                      "calibrated: a wider sample would narrow the interval.", "   "))
        elif self.familywise_error_interval[0] > self.alpha:
            add(_wrap(f"the interval lies ABOVE {self.alpha:.3f}. On this log the detector's real "
                      "error rate exceeds the one it claims, and every edge it certifies inherits "
                      "that.", "   "))
        else:
            # Below nominal is a different finding from above nominal, and conflating the two is
            # how a conservative detector gets reported as a broken one. Below nominal costs power
            # rather than validity, and on a detector with no null model of its own it is luck
            # rather than design, which is the part worth saying.
            add(_wrap(f"the interval lies BELOW {self.alpha:.3f}, so on this log the detector is "
                      "conservative rather than miscalibrated. That costs power, not validity. If "
                      "the detector has no null model of its own, this is a property of this log "
                      "and this surrogate rather than of the detector, so check it against a "
                      "stricter null before relying on it.", "   "))
        add(f"   mean edges per null replicate: {self.mean_edges_per_null:.3f} "
            f"(worst replicate: {self.max_edges_on_a_null})")
        add("")

        add("2. POWER  (recall of one planted copycat edge, by copy fraction)")
        if not self.power_curve:
            add(_wrap("no spike-in was run, so there is no power curve. A family-wise error rate "
                      "on its own is satisfied by a detector that certifies nothing, so this "
                      "report is half a result.", "   "))
        else:
            for value in sorted(self.power_curve):
                realised = self.realised_strength.get(value)
                tail = f"   realised {realised:.3f}" if realised is not None else ""
                add(f"   requested {float(value):<6.3f} recall {self.power_curve[value]}{tail}")
            add(_wrap("the plant copies a payload verbatim at a fixed lag, which is the easiest "
                      "positive there is, so these recalls are an upper bound and the floor below "
                      "is a lower bound on what real influence needs.", "   "))
        add("")

        if self.volume_curve:
            add("3. POWER  (recall by the planted actor's message volume)")
            for value in sorted(self.volume_curve):
                add(f"   volume {int(value):<8d} recall {self.volume_curve[value]}")
            add(f"   detection floor (volume): {_floor_line(self.detection_floor_volume)}")
            add("")

        add("DETECTION FLOOR")
        add(f"   {_floor_line(self.detection_floor)}")
        if self.detection_floor.get("note"):
            add(_wrap(str(self.detection_floor["note"]), "   "))
        add("")

        if self.symbolisation is not None:
            s = self.symbolisation
            add("SYMBOLISATION")
            add(f"   adjusted Rand index {s['adjusted_rand_index']:.3f} over {s['n_items']} turns, "
                f"{s['n_symbols']} symbols against {s['n_true_classes']} true classes")
            if s.get("note"):
                add(_wrap(str(s["note"]), "   "))
            add(_wrap("scope: this compares symbols you supplied against labels you supplied. It "
                      "says nothing about what your detector does internally, and the harness "
                      "cannot see that.", "   "))
            add("")

        if log.get("turn_taking_index") is not None:
            tti = log["turn_taking_index"]
            add("LOG DIAGNOSTICS")
            add(f"   turn-taking index {tti:.3f} "
                f"(1.0 means speakers are drawn independently of order)")
            add(f"   consecutive turns repeat a payload {log.get('payload_repeat_rate', 0):.3f} "
                "of the time")
            if log.get("payload_purity_lift") is not None:
                add(f"   payload is {log['payload_purity_lift']:.2f}x more predictable from the "
                    "actor than from nothing")
            if tti < TURN_TAKING_FLAG:
                add(_wrap("speakers avoid following themselves, so the speaker sequence carries "
                          "order information. This is the regime in which the conventional "
                          "circular-shift surrogate was measured at a family-wise error of 0.90 "
                          "against a nominal 0.05. Do not report a shift-based error rate on this "
                          "log.", "   "))
            add("")

        if self.null_audit:
            na = self.null_audit
            add("NULL AUDIT  (what the surrogate did to the log)")
            pairs = na.get("lag1_pairs_effective", {})
            add(f"   {'adjacent pairs (effective)':<26s} log {pairs.get('log', 0):>7.2f}   "
                f"replicate {pairs.get('replicate', 0):>7.2f}   "
                f"(of {pairs.get('possible')} possible)")
            for key, label in (("tied_timestamp_rate", "tied timestamps"),
                               ("turn_taking_index", "turn-taking index"),
                               ("payload_repeat_rate", "payload repeat rate"),
                               ("payload_purity_lift", "payload purity lift")):
                pair = na.get(key, {})
                shown = {k: ("n/a" if v is None else f"{v:.3f}") for k, v in pair.items()}
                add(f"   {label:<26s} log {shown.get('log', 'n/a'):>7s}   "
                    f"replicate {shown.get('replicate', 'n/a'):>7s}")
            add(_wrap(str(na.get("verdict", "")), "   "))
            add("")

        for note in self.notes:
            add(_wrap(f"note: {note}", "   "))
        if self.notes:
            add("")
        add("Every interval is a 95% Wilson interval. Rates are printed as "
            "point [lo, hi] (successes/trials).")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        """A JSON-serialisable form, with every rate expanded to point, interval and denominator."""
        return {
            "detector": self.detector,
            "null": self.null,
            "n_null": self.n_null,
            "alpha": self.alpha,
            "seed": self.seed,
            "log": self.log,
            "familywise_error": self.familywise_error_rate.as_dict(),
            "mean_edges_per_null": self.mean_edges_per_null,
            "max_edges_on_a_null": self.max_edges_on_a_null,
            "power_curve": {str(k): v.as_dict() for k, v in self.power_curve.items()},
            "volume_curve": {str(k): v.as_dict() for k, v in self.volume_curve.items()},
            "realised_strength": {str(k): v for k, v in self.realised_strength.items()},
            "detection_floor": self.detection_floor,
            "detection_floor_volume": self.detection_floor_volume,
            "symbolisation": self.symbolisation,
            "planted": self.planted,
            "null_audit": self.null_audit,
            "notes": list(self.notes),
        }


def _floor_line(floor: dict) -> str:
    if not floor:
        return "not computed"
    status = floor.get("status", "no_data")
    value = floor.get("floor")
    threshold = floor.get("threshold", 0.8)
    meanings = {
        "no_data": "nothing was swept, so there is no floor to report",
        "never_reached": f"power never reached {threshold:.0%} anywhere on the grid",
        "below_grid": "already above the threshold at the smallest value swept, so this is an "
                      "upper bound on the true floor",
        "bracketed": "power crossed the threshold here and stayed above it",
        "non_monotone": "power crossed the threshold here and then fell back below it, so this is "
                        "not a floor",
    }
    shown = "none" if value is None else f"{value}"
    return f"{shown}  status={status}  ({meanings.get(status, 'unknown status')})"


@dataclass
class NullComparison:
    """Family-wise error for the same detector and log under two or more surrogates.

    This has its own type because the comparison is the finding. A single number from a single
    surrogate cannot show that the surrogate is the problem; two numbers from the same detector on
    the same log can, and the only thing that differs between them is the null.
    """

    rates: dict
    log: dict
    detector: str
    n_null: int
    seed: int
    moves_actor_sequence: dict = field(default_factory=dict)

    @property
    def disagree(self) -> bool:
        """Whether any two surrogates' intervals fail to overlap.

        Non-overlapping Wilson intervals are used rather than a difference in point estimates
        because two surrogates will always differ a little by chance, and the claim worth making is
        that they cannot both be right about the same detector on the same log.
        """
        items = list(self.rates.values())
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if a.hi < b.lo or b.hi < a.lo:
                    return True
        return False

    def summary(self) -> str:
        lines = ["attrcal null comparison", "=" * 23,
                 f"log           : {self.log.get('n_events')} events, "
                 f"{self.log.get('n_actors')} actors",
                 f"detector      : {self.detector}",
                 f"replicates    : {self.n_null} per surrogate, seed {self.seed}", ""]
        for kind, rate in self.rates.items():
            flag = "  [moves the actor sequence]" if self.moves_actor_sequence.get(kind) else ""
            lines.append(f"   {kind:<14s} family-wise error {rate}{flag}")
        lines.append("")
        if self.disagree:
            lines.append(_wrap("these surrogates do not agree, and they are measuring the same "
                               "detector on the same log, so at most one of them is reporting "
                               "that detector's real error rate.", "   "))
            if any(self.moves_actor_sequence.values()):
                lines.append("")
                lines.append(_wrap("a surrogate marked above as moving the actor sequence is not "
                                   "a surrogate of this log. It slides actors onto each other's "
                                   "turns, so the replicate carries adjacencies the turn-taking "
                                   "here never allowed, and it compares each pair at its own fixed "
                                   "offset, so pairs whose offsets land close keep this log's own "
                                   "payload autocorrelation and read as copying. Believe the nulls "
                                   "that hold the actor sequence fixed, and run audit_null to see "
                                   "the damage.", "   "))
            else:
                lines.append("")
                lines.append(_wrap("all of these hold the actor sequence fixed, so the "
                                   "disagreement is about which nuisance structure each one "
                                   "keeps. The permutation null keeps the least and answers "
                                   "'is there any structure at all'; the stricter nulls keep each "
                                   "actor's habitual subjects or the channel's drift, and a "
                                   "detector that fires only under those is detecting those.",
                                   "   "))
        else:
            lines.append(_wrap("the surrogates agree on this log. That is the expected outcome "
                               "when the speaker sequence carries little order information and "
                               "actors have no characteristic subjects, and it is not evidence "
                               "that any of them is safe on a different log.", "   "))
        tti = self.log.get("turn_taking_index")
        if tti is not None:
            lines.append("")
            lines.append(f"   turn-taking index {tti:.3f}, consecutive payload repeat rate "
                         f"{self.log.get('payload_repeat_rate', 0):.3f}, payload purity lift "
                         f"{self.log.get('payload_purity_lift')}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        return {"detector": self.detector, "n_null": self.n_null, "seed": self.seed,
                "log": self.log, "disagree": self.disagree,
                "moves_actor_sequence": dict(self.moves_actor_sequence),
                "rates": {k: v.as_dict() for k, v in self.rates.items()}}
