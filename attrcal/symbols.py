"""An optional check on the step this harness cannot see: turning payloads into symbols.

Most attribution and influence detectors do not work on text. They work on a discrete symbol per
turn, and they get it by clustering, topic modelling, embedding and thresholding, or an LLM label.
That front end is inside the detector, so the harness only ever sees the callable's final edge set
and has no way to tell a detector whose statistics failed from one whose symboliser failed. The two
produce the same output and support opposite conclusions: the first says there is no influence, the
second says the measurement never happened.

The check offered here is narrow on purpose. If the user can supply both the symbols their detector
assigned and a set of ground-truth labels for the same turns, the adjusted Rand index between them
is a number that can be put in a paper next to the edge set. It is chosen over accuracy because
cluster identities are arbitrary: a symboliser that recovers the partition perfectly but names the
clusters differently scores 1.0 here and near 0 on accuracy. It is adjusted for chance because an
unadjusted Rand index rises towards 1 as the number of clusters grows, which rewards a symboliser
for being indecisive.

What this is not: it is not a measurement of a detector whose symbols the user cannot export, and
it does not run a symboliser. It scores two partitions the user hands over. If there are no
ground-truth labels for any part of the log, there is nothing to score and this check does not
apply; say so in the paper rather than leaving the symboliser unreported.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _codes(values: Sequence[Any]) -> np.ndarray:
    """Integer codes in order of first appearance, so the contingency table is deterministic."""
    seen: dict = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        code = seen.get(v)
        if code is None:
            code = len(seen)
            seen[v] = code
        out[i] = code
    return out


def _pairs(x: np.ndarray) -> np.ndarray:
    return x * (x - 1.0) / 2.0


def adjusted_rand_index(labels_true: Sequence[Any], labels_pred: Sequence[Any]) -> float:
    """Adjusted Rand index between two partitions of the same items.

    Implemented here in about a dozen lines of numpy rather than pulled in from scikit-learn. The
    reason is not purity: it is that a calibration harness whose whole argument is that other
    people's error rates are unverified should be installable and auditable from one short file,
    and a single metric is not worth a dependency that brings its own build chain.

    Two partitions that each put every item in one cluster agree perfectly and have no variance to
    adjust for; by convention that returns 1.0 rather than a division by zero.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError(f"partitions have different lengths: {len(labels_true)} and "
                         f"{len(labels_pred)}")
    n = len(labels_true)
    if n < 2:
        raise ValueError("the adjusted Rand index needs at least two items")
    t = _codes(labels_true)
    p = _codes(labels_pred)
    n_t, n_p = int(t.max()) + 1, int(p.max()) + 1
    table = np.bincount(t * n_p + p, minlength=n_t * n_p).reshape(n_t, n_p).astype(float)
    sum_cells = _pairs(table).sum()
    sum_rows = _pairs(table.sum(axis=1)).sum()
    sum_cols = _pairs(table.sum(axis=0)).sum()
    total = _pairs(np.array(float(n)))
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    if maximum == expected:
        return 1.0
    return float((sum_cells - expected) / (maximum - expected))


def symbolisation_check(symbols: Sequence[Any], labels: Sequence[Any]) -> dict:
    """Score a detector's symbols against ground-truth labels for the same turns.

    Both sequences must cover the same turns in the same order. The cluster and class counts are
    reported alongside the index because an ARI near zero means something different when the
    symboliser produced one cluster than when it produced four hundred: the first failed to
    separate anything and the second failed to merge anything, and the fix is not the same.
    """
    ari = adjusted_rand_index(labels, symbols)
    n_sym = len(set(symbols))
    n_lab = len(set(labels))
    if n_sym == 1:
        note = ("the symboliser produced a single symbol, so every turn looks alike and no lagged "
                "statistic can carry information")
    elif n_sym > 10 * max(1, n_lab):
        note = (f"{n_sym} symbols against {n_lab} true classes: the symboliser is splitting, which "
                "starves every cell of counts")
    elif ari < 0.2:
        note = ("agreement is near chance, so an empty edge set from this detector is evidence "
                "about its front end and not about the log")
    else:
        note = None
    return {"adjusted_rand_index": round(ari, 4), "n_items": len(symbols),
            "n_symbols": n_sym, "n_true_classes": n_lab, "note": note}
