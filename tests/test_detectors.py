"""Tests for the reference detectors and the synthetic log they are demonstrated on.

These exist because the end-to-end tests use them as instruments. An instrument whose behaviour is
not pinned down cannot be used to check anything, so the known-good detector is asserted to find a
real planted edge and to stay quiet when there is nothing to find.
"""
import collections

import pytest

from attrcal import detectors, synth_log
from attrcal.events import check_edges


def test_all_pairs_returns_every_ordered_pair_and_no_self_edge():
    ev = synth_log(n_events=300, n_actors=5, seed=1)["events"]
    found = check_edges(detectors.all_pairs(ev))
    assert len(found) == 5 * 4
    assert not any(s == d for s, d in found)


def test_no_edges_returns_an_empty_set_and_still_validates_its_input():
    ev = synth_log(n_events=200, seed=1)["events"]
    assert detectors.no_edges(ev) == set()
    with pytest.raises(ValueError):
        detectors.no_edges([])


def test_both_reference_detectors_satisfy_the_contract():
    ev = synth_log(n_events=300, seed=1)["events"]
    for detector in (detectors.all_pairs, detectors.no_edges,
                     detectors.CopyRateDetector(),
                     detectors.SurrogateCopyDetector(n_surrogates=10)):
        assert isinstance(detector(ev), set)


def test_the_surrogate_detector_finds_a_strong_planted_copy_edge():
    """The known-good instrument has to actually work, otherwise every power number measured with
    it is a measurement of a broken detector."""
    built = synth_log(n_events=2500, n_actors=6, turn_taking=0.8, drift=0.05, common_cause=0.2,
                      copy_edges=((0, 3, 0.8),), seed=2)
    found = detectors.SurrogateCopyDetector(n_surrogates=100, seed=0)(built["events"])
    assert built["ground_truth"] <= found


def test_the_surrogate_detector_stays_quiet_on_a_log_with_no_structure_left():
    """Its own internal permutation null should hold on data that has been permuted already."""
    import numpy as np
    from attrcal import make_null
    ev = synth_log(n_events=2000, n_actors=6, turn_taking=0.9, common_cause=0.3, seed=4)["events"]
    detector = detectors.SurrogateCopyDetector(n_surrogates=120, null="permute", seed=0)
    fired = sum(bool(detector(make_null(ev, "permute", np.random.default_rng(s))))
                for s in range(20))
    assert fired <= 3, f"fired on {fired} of 20 structure-free replicates"


def test_the_surrogate_detector_records_how_many_pairs_it_actually_tested():
    """A family-wise error rate cannot be read without the family size, and a turn-taking protocol
    means most pairs are never adjacent and so are never tested at all."""
    ev = synth_log(n_events=2000, n_actors=6, turn_taking=0.95, seed=5)["events"]
    detector = detectors.SurrogateCopyDetector(n_surrogates=20, seed=0)
    detector(ev)
    assert detector.last_tested["family_size"] == 30
    assert detector.last_tested["tested"] < 30
    assert detector.last_tested["dropped"] > 0


def test_the_copy_rate_detector_fires_more_as_its_threshold_falls():
    ev = synth_log(n_events=1500, n_actors=6, turn_taking=0.9, common_cause=0.3, seed=6)["events"]
    loose = detectors.CopyRateDetector(threshold=0.2)(ev)
    tight = detectors.CopyRateDetector(threshold=0.9)(ev)
    assert len(loose) > len(tight)


def test_the_copy_rate_detector_can_threshold_against_the_logs_own_rate_instead():
    ev = synth_log(n_events=1500, n_actors=6, turn_taking=0.9, common_cause=0.3, seed=6)["events"]
    absolute = detectors.CopyRateDetector(threshold=0.4)(ev)
    relative = detectors.CopyRateDetector(ratio=3.0)(ev)
    assert isinstance(absolute, set) and isinstance(relative, set)
    assert len(relative) < len(absolute)


def test_a_detector_with_a_monte_carlo_p_value_never_claims_more_resolution_than_it_has():
    """With n surrogates the smallest reportable p-value is 1/(n+1), so a detector run at alpha
    0.05 with fewer than nineteen surrogates can never certify anything. That is a property worth
    pinning down: the alternative is a p-value of exactly zero, which is anti-conservative."""
    built = synth_log(n_events=2000, copy_edges=((0, 3, 0.9),), turn_taking=0.8, seed=7)
    weak = detectors.SurrogateCopyDetector(n_surrogates=5, alpha=0.05, seed=0)
    assert weak(built["events"]) == set()


@pytest.mark.parametrize("bad", [{"lag": 0}, {"n_surrogates": 0}, {"null": "jackknife"}])
def test_the_surrogate_detector_rejects_impossible_configuration(bad):
    with pytest.raises(ValueError):
        detectors.SurrogateCopyDetector(**bad)


@pytest.mark.parametrize("bad", [{"lag": 0}, {"threshold": 0.0}, {"threshold": 1.5},
                                 {"ratio": -1.0}])
def test_the_copy_rate_detector_rejects_impossible_configuration(bad):
    with pytest.raises(ValueError):
        detectors.CopyRateDetector(**bad)


# --------------------------------------------------------------------- the synthetic log


def test_synth_log_is_reproducible_from_a_fixed_seed():
    a = synth_log(n_events=500, copy_edges=((0, 2, 0.5),), seed=9)
    b = synth_log(n_events=500, copy_edges=((0, 2, 0.5),), seed=9)
    assert a["events"] == b["events"]


def test_different_seeds_give_different_logs():
    a = synth_log(n_events=500, seed=9)["events"]
    b = synth_log(n_events=500, seed=10)["events"]
    assert a != b


def test_turn_taking_produces_a_near_cyclic_speaker_sequence():
    ev = synth_log(n_events=1200, n_actors=5, turn_taking=1.0, seed=11)["events"]
    actors = [e[0] for e in ev]
    advances = sum(1 for i in range(1, len(actors))
                   if int(actors[i][-2:]) == (int(actors[i - 1][-2:]) + 1) % 5)
    assert advances / (len(actors) - 1) > 0.95


def test_the_planted_edge_shows_up_as_a_lag_one_excess_and_not_merely_a_busy_follower():
    built = synth_log(n_events=3000, n_actors=6, turn_taking=0.5, common_cause=0.2,
                      copy_edges=((0, 4, 0.7),), seed=12)
    ev = built["events"]
    actors = [e[0] for e in ev]
    pays = [e[2] for e in ev]
    base = actors.count("actor04") / len(actors)
    opps = hits = 0
    for i in range(1, len(ev)):
        if actors[i - 1] == "actor00":
            opps += 1
            hits += actors[i] == "actor04" and pays[i] == pays[i - 1]
    assert (hits / opps) / base > 2.0


def test_ground_truth_names_exactly_the_requested_edges():
    built = synth_log(n_events=400, copy_edges=((0, 2, 0.5), (1, 3, 0.4)), seed=13)
    assert built["ground_truth"] == {("actor00", "actor02"), ("actor01", "actor03")}


def test_a_zero_probability_edge_is_not_labelled():
    assert synth_log(n_events=400, copy_edges=((0, 2, 0.0),), seed=14)["ground_truth"] == set()


@pytest.mark.parametrize("bad,msg", [
    (dict(copy_edges=((0, 0, 0.5),)), "self-edge"),
    (dict(copy_edges=((0, 99, 0.5),)), "outside"),
    (dict(copy_edges=((0, 2, 1.7),)), r"\[0, 1\]"),
    (dict(turn_taking=1.5), "turn_taking"),
    (dict(common_cause=-0.1), "common_cause"),
    (dict(n_actors=1), "at least two"),
])
def test_synth_log_rejects_impossible_configuration(bad, msg):
    with pytest.raises(ValueError, match=msg):
        synth_log(n_events=200, seed=15, **bad)


def test_every_actor_speaks_at_least_once_on_a_long_log():
    built = synth_log(n_events=2000, n_actors=6, seed=16)
    assert len(collections.Counter(e[0] for e in built["events"])) == 6
