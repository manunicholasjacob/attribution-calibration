"""Tests for the optional symbolisation check.

The index is implemented here rather than imported, so it has to be shown to behave like the thing
it claims to be: invariant to how clusters are named, chance-corrected, and 1.0 only on exact
agreement.
"""
import pytest

from attrcal import adjusted_rand_index, symbolisation_check


def test_identical_partitions_score_one():
    labels = [0, 0, 1, 1, 2, 2]
    assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)


def test_relabelling_the_clusters_does_not_change_the_score():
    """The reason this index is used instead of accuracy: cluster identities are arbitrary, so a
    symboliser that recovers the partition perfectly and names it differently must still score 1."""
    truth = ["a", "a", "b", "b", "c", "c"]
    renamed = [7, 7, 99, 99, 0, 0]
    assert adjusted_rand_index(truth, renamed) == pytest.approx(1.0)


def _shuffled(values, seed=0):
    """A genuinely independent partition of the same items, built without numpy.

    An arithmetic recoding such as (7i + 3) mod 4 looks scrambled and is in fact a bijection on the
    residues, so it agrees with the truth perfectly and scores 1.0. That mistake is easy to make in
    a test and would have let a broken index pass, so the shuffle here is a real one.
    """
    import random
    out = list(values)
    random.Random(seed).shuffle(out)
    return out


def test_a_random_partition_scores_near_zero_rather_than_near_a_half():
    """Chance correction is the point. An unadjusted Rand index on these would sit well above 0."""
    truth = [i % 4 for i in range(2000)]
    scrambled = _shuffled(truth, seed=1)
    assert abs(adjusted_rand_index(truth, scrambled)) < 0.05


def test_an_arithmetic_recoding_of_the_labels_is_not_a_random_partition():
    """Pinned down because it is the trap the test above avoids: this recoding is a relabelling."""
    truth = [i % 4 for i in range(400)]
    recoded = [(i * 7 + 3) % 4 for i in range(400)]
    assert adjusted_rand_index(truth, recoded) == pytest.approx(1.0)


def test_a_symboliser_that_splits_every_item_scores_near_zero_not_near_one():
    """An unadjusted index rewards indecision, which would let a useless symboliser look good."""
    truth = [i % 4 for i in range(200)]
    all_singletons = list(range(200))
    assert adjusted_rand_index(truth, all_singletons) < 0.05


def test_a_symboliser_that_merges_everything_scores_near_zero():
    truth = [i % 4 for i in range(200)]
    one_cluster = [0] * 200
    assert adjusted_rand_index(truth, one_cluster) == pytest.approx(0.0, abs=1e-9)


def test_partial_agreement_lands_between_the_extremes():
    truth = [0, 0, 0, 0, 1, 1, 1, 1]
    partial = [0, 0, 0, 1, 1, 1, 1, 1]
    score = adjusted_rand_index(truth, partial)
    assert 0.2 < score < 0.9


def test_two_single_cluster_partitions_agree_perfectly_by_convention():
    """Both partitions are trivial, so there is no variance to adjust for and no division to do."""
    assert adjusted_rand_index([0, 0, 0], [5, 5, 5]) == pytest.approx(1.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="different lengths"):
        adjusted_rand_index([0, 1], [0, 1, 2])


def test_a_single_item_is_rejected_because_the_index_is_about_pairs():
    with pytest.raises(ValueError, match="at least two"):
        adjusted_rand_index([0], [0])


def test_symbolisation_check_reports_both_cluster_counts_alongside_the_index():
    """An index near zero means something different when the symboliser produced one cluster than
    when it produced four hundred, and the fix is not the same."""
    out = symbolisation_check(symbols=[0, 0, 1, 1], labels=["x", "x", "y", "y"])
    assert out["adjusted_rand_index"] == pytest.approx(1.0)
    assert out["n_symbols"] == 2 and out["n_true_classes"] == 2 and out["n_items"] == 4


def test_symbolisation_check_flags_a_collapsed_symboliser():
    out = symbolisation_check(symbols=[0] * 40, labels=[i % 4 for i in range(40)])
    assert out["n_symbols"] == 1
    assert out["note"] and "single symbol" in out["note"]


def test_symbolisation_check_flags_a_splitting_symboliser():
    out = symbolisation_check(symbols=list(range(60)), labels=[i % 2 for i in range(60)])
    assert out["note"] and "splitting" in out["note"]


def test_symbolisation_check_flags_near_chance_agreement_as_being_about_the_front_end():
    labels = [i % 4 for i in range(2000)]
    out = symbolisation_check(symbols=_shuffled(labels, seed=2), labels=labels)
    assert out["adjusted_rand_index"] < 0.2
    assert out["note"] and "front end" in out["note"]


def test_a_good_symboliser_gets_no_note():
    out = symbolisation_check(symbols=[0, 0, 0, 1, 1, 1], labels=[0, 0, 0, 1, 1, 1])
    assert out["note"] is None
