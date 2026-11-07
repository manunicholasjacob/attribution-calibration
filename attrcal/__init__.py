"""attrcal: error rates for directed attribution claims about multi-agent logs.

An attribution claim and an influence claim are the same statistical object. Both take a log of who
did what when and emit a directed statement, "A caused B" or "A influenced B", and both are usually
published with no error rate at all. When one of these pipelines was finally given a valid
surrogate, its family-wise error on real turn-ordered agent chat came out at 0.90 against a nominal
0.05: on nine logs in ten with no influence in them, it certified at least one influence edge.

This package is the generic form of that check. A user brings their own detector and their own log
and gets back the three numbers any directed claim should carry: how often the detector certifies
an edge on replicates of that same log that cannot contain one, how strong an influence has to be
before the detector finds it, and whether that floor means anything.

    from attrcal import calibrate, PlantSpec

    report = calibrate(events, my_detector, null="permute", n_null=200,
                       planted=PlantSpec(strengths=(0.05, 0.1, 0.2, 0.4, 0.8)))
    print(report.summary())

Author: Manu Nicholas Jacob, Independent Researcher (manunicholasjacob@gmail.com).
"""
from __future__ import annotations

from . import detectors
from .calibrate import audit_null, calibrate, compare_nulls
from .events import Event, check_edges, describe_log, normalise_events
from .nulls import (INDEX_NULLS, NULL_NOTES, NULLS, make_null, null_index,
                    preserves_actor_sequence)
from .plant import PlantSpec, choose_leader, plant_copycat, realised_copy_fraction
from .report import NullComparison, Report
from .stats import MIN_SEEDS_FOR_POWER, Rate, detection_floor, wilson
from .symbols import adjusted_rand_index, symbolisation_check
from .synth import synth_log

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "calibrate",
    "compare_nulls",
    "audit_null",
    "Report",
    "NullComparison",
    "PlantSpec",
    "Rate",
    "Event",
    "NULLS",
    "INDEX_NULLS",
    "NULL_NOTES",
    "make_null",
    "null_index",
    "preserves_actor_sequence",
    "plant_copycat",
    "realised_copy_fraction",
    "choose_leader",
    "detection_floor",
    "wilson",
    "MIN_SEEDS_FOR_POWER",
    "adjusted_rand_index",
    "symbolisation_check",
    "normalise_events",
    "describe_log",
    "check_edges",
    "synth_log",
    "detectors",
]
