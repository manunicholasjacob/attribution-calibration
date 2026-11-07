"""Tests for the interval arithmetic and the detection floor.

Each of these asserts a behaviour that a bare point estimate or a bare floor number would have
hidden, which is the reason both types exist.
"""
import math

import pytest

from attrcal.stats import MIN_SEEDS_FOR_POWER, Rate, detection_floor, wilson


def test_wilson_interval_stays_inside_the_unit_interval_at_the_extremes():
    """The reason Wilson is used rather than the normal approximation.

    At zero successes the Wald interval is [0, 0], which claims the rate is known exactly, and at
    small counts it goes negative. Both are wrong in the direction that flatters a detector.
    """
    lo, hi = wilson(0, 40)
    assert lo == 0.0 and 0.0 < hi < 0.2
    lo, hi = wilson(40, 40)
    assert hi == 1.0 and 0.8 < lo < 1.0


def test_wilson_interval_narrows_as_the_sample_grows():
    narrow = wilson(5, 1000)
    wide = wilson(1, 20)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_covers_the_true_rate_for_a_textbook_case():
    lo, hi = wilson(9, 200)
    assert lo < 0.045 < hi
    assert 0.02 < lo and hi < 0.09


def test_empty_sample_returns_the_whole_unit_interval_rather_than_a_point():
    assert wilson(0, 0) == (0.0, 1.0)
    r = Rate.of(0, 0)
    assert math.isnan(r.point)
    assert str(r) == "no data"


def test_rate_carries_its_denominator_into_its_string_form():
    r = Rate.of(9, 200)
    assert "9/200" in str(r)
    assert "0.045" in str(r)


def test_rate_covers_distinguishes_a_wide_sample_from_a_narrow_one():
    """Two detectors with the same point estimate and different sample sizes are different claims."""
    assert Rate.of(0, 20).covers(0.05)
    assert not Rate.of(0, 2000).covers(0.05)


def test_rate_rejects_impossible_counts():
    with pytest.raises(ValueError):
        Rate.of(5, 2)
    with pytest.raises(ValueError):
        Rate.of(-1, 10)


def test_rate_is_floatable_so_curves_can_be_compared_numerically():
    assert float(Rate.of(8, 10)) == pytest.approx(0.8)


# --------------------------------------------------------------------------- detection floor
# Every status the floor can return is reachable, which is the whole reason it returns a status.


def test_floor_status_no_data_when_nothing_was_swept():
    out = detection_floor({})
    assert out["status"] == "no_data" and out["floor"] is None


def test_floor_status_never_reached_when_power_stays_low():
    out = detection_floor({0.1: 0.1, 0.2: 0.3, 0.4: 0.5}, threshold=0.8, n_seeds=20)
    assert out["status"] == "never_reached" and out["floor"] is None


def test_floor_status_below_grid_when_the_smallest_value_already_passes():
    out = detection_floor({0.1: 0.9, 0.2: 1.0}, threshold=0.8, n_seeds=20)
    assert out["status"] == "below_grid" and out["floor"] == 0.1


def test_floor_status_bracketed_when_power_crosses_and_holds():
    out = detection_floor({0.05: 0.1, 0.1: 0.5, 0.2: 0.9, 0.4: 1.0}, threshold=0.8, n_seeds=20)
    assert out["status"] == "bracketed" and out["floor"] == 0.2 and out["monotone"] is True


def test_floor_status_non_monotone_when_power_crosses_then_falls_back():
    """A ragged curve's first crossing reads exactly like a floor and is not one."""
    out = detection_floor({0.05: 0.1, 0.1: 0.9, 0.2: 0.4, 0.4: 1.0}, threshold=0.8, n_seeds=20)
    assert out["status"] == "non_monotone" and out["floor"] == 0.1 and out["monotone"] is False


def test_all_five_floor_statuses_are_reachable():
    seen = {
        detection_floor({})["status"],
        detection_floor({0.1: 0.0})["status"],
        detection_floor({0.1: 1.0})["status"],
        detection_floor({0.1: 0.0, 0.2: 1.0})["status"],
        detection_floor({0.1: 0.0, 0.2: 1.0, 0.3: 0.0})["status"],
    }
    assert seen == {"no_data", "never_reached", "below_grid", "bracketed", "non_monotone"}


def test_floor_warns_in_its_note_when_the_seed_count_cannot_resolve_the_threshold():
    """With n seeds recall only takes multiples of 1/n, so a floor from three seeds is grid noise."""
    coarse = detection_floor({0.1: 0.0, 0.2: 1.0}, threshold=0.8, n_seeds=3)
    assert coarse["note"] is not None and "coarse" in coarse["note"]
    fine = detection_floor({0.1: 0.0, 0.2: 1.0}, threshold=0.8, n_seeds=MIN_SEEDS_FOR_POWER)
    assert fine["note"] is None


def test_floor_accepts_rate_objects_and_compares_on_point_estimates():
    out = detection_floor({0.1: Rate.of(2, 10), 0.2: Rate.of(9, 10)}, threshold=0.8, n_seeds=10)
    assert out["status"] == "bracketed" and out["floor"] == 0.2


def test_floor_rejects_a_threshold_outside_the_unit_interval():
    with pytest.raises(ValueError):
        detection_floor({0.1: 0.5}, threshold=1.5)
