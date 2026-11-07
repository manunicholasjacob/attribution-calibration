"""Tests for the spike-in positive control.

The important ones here are about the gap between what was requested and what the detector was
actually shown. A planted copy fraction is a label; the stream is the measurement, and they come
apart for reasons the user has to be told about rather than left to discover.
"""
import collections

import numpy as np
import pytest

from attrcal import PlantSpec, choose_leader, plant_copycat, realised_copy_fraction, synth_log


def log(n=1500, seed=3):
    return synth_log(n_events=n, n_actors=6, n_payloads=8, turn_taking=0.9, drift=0.05,
                     common_cause=0.3, seed=seed)["events"]


def test_the_plant_is_added_as_a_new_actor_with_the_requested_volume():
    ev = log()
    out = plant_copycat(ev, np.random.default_rng(0), leader="actor00", strength=0.4, volume=150)
    counts = collections.Counter(e[0] for e in out["events"])
    assert counts[out["name"]] == 150
    assert len(out["events"]) == len(ev) + 150


def test_the_plants_edge_names_the_leader_and_the_plant():
    out = plant_copycat(log(), np.random.default_rng(0), leader="actor02", strength=0.4,
                        volume=100, name="spike")
    assert out["edge"] == ("actor02", "spike")


def test_the_original_log_is_not_mutated():
    ev = log(600)
    before = list(ev)
    plant_copycat(ev, np.random.default_rng(0), leader="actor00", strength=0.5, volume=50)
    assert ev == before


def test_the_spiked_log_stays_sorted_by_timestamp():
    out = plant_copycat(log(600), np.random.default_rng(0), leader="actor00", strength=0.5,
                        volume=80)
    times = [e[1] for e in out["events"]]
    assert times == sorted(times)


def test_realised_copy_fraction_tracks_the_request_at_lag_one():
    """At lag 1 the plant's message sits immediately after its host, so the request is realised."""
    for requested in (0.2, 0.5, 0.9):
        out = plant_copycat(log(2000), np.random.default_rng(1), leader="actor00",
                            strength=requested, volume=200, lag=1)
        assert out["strength_realised"] >= requested
        assert out["strength_realised"] < requested + 0.15


def test_realised_copy_fraction_exceeds_the_request_because_noise_can_coincide():
    """A noise message drawn from the log's own payload distribution can land right after the
    leader and match it by chance, and a reader indexing a power curve has to know that the
    realised value includes those."""
    out = plant_copycat(log(2000), np.random.default_rng(2), leader="actor00", strength=0.0,
                        volume=300, lag=1)
    assert out["n_copies_placed"] == 0
    assert out["strength_realised"] > 0.0


def test_a_larger_lag_loses_some_of_the_requested_fraction():
    """Inserting the plant lengthens the turn sequence, so at lag greater than one another turn can
    land in between and push a copy off its intended lag."""
    near = plant_copycat(log(2000), np.random.default_rng(3), leader="actor00", strength=0.8,
                         volume=200, lag=1)
    far = plant_copycat(log(2000), np.random.default_rng(3), leader="actor00", strength=0.8,
                        volume=200, lag=4)
    assert far["strength_realised"] < near["strength_realised"]


def test_requesting_more_copies_than_the_leader_has_turns_is_capped_and_reported():
    """Silently placing fewer copies than requested would make every power curve above that point
    a measurement of something else."""
    ev = log(400)
    quiet = collections.Counter(e[0] for e in ev).most_common()[-1][0]
    n_turns = sum(1 for e in ev if e[0] == quiet)
    out = plant_copycat(ev, np.random.default_rng(4), leader=quiet, strength=1.0,
                        volume=n_turns * 3)
    assert out["capped"] is True
    assert out["n_copies_placed"] == n_turns
    assert out["cap_note"] and "only" in out["cap_note"]


def test_copies_are_hosted_on_distinct_leader_turns():
    """Two copies on one host cannot both land at the requested lag, so hosts are sampled without
    replacement."""
    out = plant_copycat(log(1500), np.random.default_rng(5), leader="actor00", strength=1.0,
                        volume=120, lag=1)
    assert out["n_copies_placed"] == 120
    assert out["strength_realised"] > 0.9


def test_the_plants_name_cannot_collide_with_a_real_actor():
    with pytest.raises(ValueError, match="already an actor"):
        plant_copycat(log(400), np.random.default_rng(0), leader="actor00", strength=0.4,
                      volume=20, name="actor01")


def test_an_absent_leader_is_rejected():
    with pytest.raises(ValueError, match="does not appear"):
        plant_copycat(log(400), np.random.default_rng(0), leader="nobody", strength=0.4, volume=20)


def test_planting_is_reproducible_from_a_fixed_seed():
    a = plant_copycat(log(600), np.random.default_rng(9), leader="actor00", strength=0.4,
                      volume=50)
    b = plant_copycat(log(600), np.random.default_rng(9), leader="actor00", strength=0.4,
                      volume=50)
    assert a["events"] == b["events"]
    assert a["strength_realised"] == b["strength_realised"]


def test_tied_timestamps_still_place_the_plant_immediately_after_its_host():
    """Coarse logs tie timestamps constantly, and the plant has to survive that: the interpolated
    timestamp collapses onto the host's and the stable sort keeps the order."""
    ev = [("a", 0.0, 1), ("b", 0.0, 2), ("a", 0.0, 3), ("b", 0.0, 4)] * 30
    out = plant_copycat(ev, np.random.default_rng(0), leader="a", strength=1.0, volume=30, lag=1)
    assert realised_copy_fraction(out["events"], leader="a", name=out["name"], lag=1) == 1.0


def test_choose_leader_picks_the_busiest_actor():
    ev = log(1200)
    counts = collections.Counter(e[0] for e in ev)
    assert choose_leader(ev) == counts.most_common(1)[0][0]


def test_plantspec_rejects_a_strength_outside_the_unit_interval():
    with pytest.raises(ValueError, match="fraction"):
        PlantSpec(strengths=(0.5, 1.5))


def test_plantspec_rejects_a_lag_of_zero_because_simultaneity_is_not_influence():
    with pytest.raises(ValueError, match="lag"):
        PlantSpec(lag=0)


def test_plantspec_rejects_a_volume_below_one():
    with pytest.raises(ValueError, match="volume"):
        PlantSpec(volume=0)
    with pytest.raises(ValueError, match="volume"):
        PlantSpec(volumes=(100, 0))


def test_plantspec_defaults_sweep_a_usable_range_with_enough_seeds_to_estimate_a_floor():
    spec = PlantSpec()
    assert min(spec.strengths) <= 0.05 and max(spec.strengths) >= 0.8
    assert spec.n_seeds >= 10
