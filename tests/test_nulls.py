"""Tests for the surrogates, which are the part of this package that has to be right.

The assertions fall into two groups. The first group is what every null must preserve, because a
null that changes the wrong thing is measuring a different log. The second group is the documented
difference between the nulls, including the one that makes the conventional recipe invalid: it does
not hold the actor sequence fixed, and the damage is measurable in a single replicate.
"""
import collections

import numpy as np
import pytest

from attrcal import (INDEX_NULLS, NULLS, audit_null, describe_log, make_null, null_index,
                     preserves_actor_sequence, synth_log)


def turn_ordered(n=2000, seed=3):
    return synth_log(n_events=n, n_actors=6, n_payloads=8, turn_taking=0.9, drift=0.05,
                     common_cause=0.3, seed=seed)["events"]


@pytest.mark.parametrize("kind", NULLS)
def test_every_null_preserves_the_length_and_the_payload_multiset(kind):
    """A surrogate that changes the payload marginal changes what a detector measures about it."""
    ev = turn_ordered(800)
    rep = make_null(ev, kind, np.random.default_rng(0))
    assert len(rep) == len(ev)
    assert collections.Counter(e[2] for e in rep) == collections.Counter(e[2] for e in ev)


@pytest.mark.parametrize("kind", NULLS)
def test_every_null_preserves_each_actors_volume(kind):
    ev = turn_ordered(800)
    rep = make_null(ev, kind, np.random.default_rng(0))
    assert (collections.Counter(e[0] for e in rep)
            == collections.Counter(e[0] for e in ev))


@pytest.mark.parametrize("kind", INDEX_NULLS)
def test_the_index_nulls_hold_the_actor_and_timestamp_columns_exactly_fixed(kind):
    """The property that makes a replicate a surrogate of the caller's own log."""
    ev = turn_ordered(800)
    rep = make_null(ev, kind, np.random.default_rng(0))
    assert [e[0] for e in rep] == [e[0] for e in ev]
    assert [e[1] for e in rep] == [e[1] for e in ev]
    assert preserves_actor_sequence(kind)


def test_the_shift_null_does_not_hold_the_actor_sequence_fixed_and_says_so():
    ev = turn_ordered(800)
    rep = make_null(ev, "shift", np.random.default_rng(0))
    assert [e[0] for e in rep] != [e[0] for e in ev]
    assert not preserves_actor_sequence("shift")


def test_null_index_refuses_the_shift_null_because_it_is_not_a_payload_remapping():
    """The API encodes the distinction: a null that can be written as an index map cannot move
    who spoke when, and one that cannot be written that way does."""
    with pytest.raises(ValueError, match="not a surrogate|cannot be expressed"):
        null_index("shift", ["a", "b", "a", "b"], np.random.default_rng(0))


def test_permute_drops_the_payload_repeat_rate_to_chance_and_rotate_stays_above_it():
    """The documented difference between them, and the reason their error rates can disagree.

    Chance here is the probability that two payloads drawn independently from the log's own
    marginal agree, which is what a full permutation leaves behind. It is well above zero on a log
    with few payload symbols, so the right assertion is that permutation lands on chance rather
    than that it lands near zero.
    """
    ev = turn_ordered(2000)
    counts = collections.Counter(e[2] for e in ev)
    chance = sum((c / len(ev)) ** 2 for c in counts.values())
    real = describe_log(ev)["payload_repeat_rate"]
    permuted = describe_log(make_null(ev, "permute", np.random.default_rng(1)))["payload_repeat_rate"]
    rotated = describe_log(make_null(ev, "rotate", np.random.default_rng(1)))["payload_repeat_rate"]
    assert real > chance + 0.05, "the log must have payload autocorrelation to tell the nulls apart"
    assert permuted == pytest.approx(chance, abs=0.04)
    assert rotated > permuted + 0.02


def test_rotate_keeps_each_actors_own_payload_distribution_exactly():
    """What makes rotation the strict null: an actor's habitual subjects survive into the replicate,
    so a detector cannot be credited for rediscovering them."""
    ev = turn_ordered(1500)
    rep = make_null(ev, "rotate", np.random.default_rng(2))
    for who in {e[0] for e in ev}:
        before = collections.Counter(e[2] for e in ev if e[0] == who)
        after = collections.Counter(e[2] for e in rep if e[0] == who)
        assert before == after


def test_permute_does_not_keep_each_actors_own_payload_distribution():
    ev = turn_ordered(1500)
    rep = make_null(ev, "permute", np.random.default_rng(2))
    changed = [who for who in {e[0] for e in ev}
               if collections.Counter(e[2] for e in ev if e[0] == who)
               != collections.Counter(e[2] for e in rep if e[0] == who)]
    assert changed, "a full permutation must move at least one actor's payload distribution"


def test_blockpermute_keeps_more_of_the_channel_drift_than_a_full_permutation():
    """It is the confounding null: slow drift survives, so every pair stays dependent through the
    channel while no actor influences another."""
    ev = synth_log(n_events=3000, n_actors=6, turn_taking=0.9, drift=0.01, common_cause=1.0,
                   seed=7)["events"]
    rng = np.random.default_rng(4)
    blocked = describe_log(make_null(ev, "blockpermute", rng, block_size=40))["payload_repeat_rate"]
    permuted = describe_log(make_null(ev, "permute", rng))["payload_repeat_rate"]
    assert blocked > permuted * 1.5


def test_blockpermute_block_size_controls_how_much_drift_survives():
    ev = synth_log(n_events=3000, turn_taking=0.9, drift=0.01, common_cause=1.0, seed=8)["events"]
    rng = np.random.default_rng(5)
    tight = describe_log(make_null(ev, "blockpermute", rng, block_size=5))["payload_repeat_rate"]
    loose = describe_log(make_null(ev, "blockpermute", rng, block_size=400))["payload_repeat_rate"]
    assert tight > loose


def test_audit_null_detects_that_the_shift_surrogate_widened_the_adjacency_structure():
    """The mechanical reason the conventional surrogate is invalid here, measured rather than
    argued: it manufactures adjacencies between actor pairs the protocol kept apart."""
    ev = turn_ordered(2000)
    shifted = audit_null(ev, "shift", seed=1)
    permuted = audit_null(ev, "permute", seed=1)
    assert shifted["widened_adjacency"] is True
    assert permuted["widened_adjacency"] is False
    before = shifted["lag1_pairs_effective"]["log"]
    after = shifted["lag1_pairs_effective"]["replicate"]
    assert after > 3 * before
    assert permuted["lag1_pairs_effective"]["replicate"] == before


def test_audit_null_detects_that_the_shift_surrogate_makes_actors_speak_simultaneously():
    ev = turn_ordered(1500)
    shifted = audit_null(ev, "shift", seed=2)
    assert shifted["tied_timestamp_rate"]["log"] == 0.0
    assert shifted["tied_timestamp_rate"]["replicate"] > 0.1


@pytest.mark.parametrize("kind", NULLS)
def test_a_null_is_reproducible_from_a_fixed_seed(kind):
    ev = turn_ordered(600)
    a = make_null(ev, kind, np.random.default_rng(11))
    b = make_null(ev, kind, np.random.default_rng(11))
    assert a == b


@pytest.mark.parametrize("kind", NULLS)
def test_consecutive_replicates_from_one_generator_differ(kind):
    """Drawing n replicates from one generator must give n different replicates, otherwise the
    error rate is estimated from one sample repeated."""
    ev = turn_ordered(600)
    rng = np.random.default_rng(12)
    first = make_null(ev, kind, rng)
    second = make_null(ev, kind, rng)
    assert first != second


def test_the_rotation_null_never_draws_an_offset_of_zero():
    """A rotation offset of zero would hand back that actor's real payloads, which is not a null.

    Checked over twenty draws rather than one, because the bug it guards against is an off-by-one
    in the offset range and that shows up as a rare identity rather than a constant one.
    """
    ev = turn_ordered(600)
    for seed in range(20):
        idx = null_index("rotate", [e[0] for e in ev], np.random.default_rng(seed))
        assert not np.array_equal(idx, np.arange(len(ev)))


def test_unknown_null_is_rejected_by_name():
    ev = turn_ordered(200)
    with pytest.raises(ValueError, match="unknown null"):
        make_null(ev, "jackknife", np.random.default_rng(0))


def test_a_two_event_log_is_the_smallest_a_surrogate_can_be_built_from():
    with pytest.raises(ValueError, match="at least two"):
        make_null([("a", 0.0, 1)], "permute", np.random.default_rng(0))


def test_block_size_below_two_is_rejected_because_a_block_of_one_cannot_be_permuted():
    ev = turn_ordered(200)
    with pytest.raises(ValueError, match="at least 2"):
        make_null(ev, "blockpermute", np.random.default_rng(0), block_size=1)
