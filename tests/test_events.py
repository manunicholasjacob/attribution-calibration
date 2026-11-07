"""Tests for the event contract and the log diagnostics.

The contract tests are here because every one of these inputs used to be accepted and then produced
a wrong number somewhere downstream rather than an error at the boundary.
"""
import pytest

from attrcal import describe_log, normalise_events, synth_log
from attrcal.events import check_edges, detector_name


def test_events_are_sorted_by_timestamp():
    out = normalise_events([("b", 3.0, "y"), ("a", 1.0, "x"), ("c", 2.0, "z")])
    assert [e[0] for e in out] == ["a", "c", "b"]


def test_tied_timestamps_keep_the_callers_order():
    """In a coarsely timestamped log the list order is the turn order and the timestamp is not.

    An unstable sort would reshuffle the turn sequence that every lagged statistic depends on, and
    it would do so differently on different machines.
    """
    given = [("a", 5.0, 1), ("b", 5.0, 2), ("c", 5.0, 3), ("d", 5.0, 4)]
    assert [e[0] for e in normalise_events(given)] == ["a", "b", "c", "d"]


def test_timestamps_are_coerced_to_float_so_integer_turn_indices_work():
    out = normalise_events([("a", 0, "x"), ("b", 1, "y")])
    assert out[0][1] == 0.0 and isinstance(out[0][1], float)


def test_empty_log_is_rejected_rather_than_producing_an_empty_report():
    with pytest.raises(ValueError, match="empty"):
        normalise_events([])


def test_wrong_arity_is_rejected_and_the_message_says_why_extra_columns_are_dangerous():
    with pytest.raises(ValueError, match="expected 3"):
        normalise_events([("a", 1.0, "x", "topic3")])


def test_non_numeric_timestamp_is_rejected():
    with pytest.raises(TypeError, match="not a number"):
        normalise_events([("a", "noon", "x"), ("b", 1.0, "y")])


def test_nan_timestamp_is_rejected_because_it_leaves_the_turn_order_undefined():
    with pytest.raises(ValueError, match="NaN"):
        normalise_events([("a", float("nan"), "x"), ("b", 1.0, "y")])


def test_unhashable_actor_is_rejected_because_it_cannot_name_an_edge():
    with pytest.raises(TypeError, match="hashable"):
        normalise_events([(["a"], 1.0, "x"), ("b", 2.0, "y")])


def test_unhashable_payload_is_accepted_and_only_costs_the_optional_diagnostics():
    """Payloads are reordered and compared for equality, never hashed, so a dict payload is fine."""
    log = describe_log(normalise_events([("a", 1.0, {"t": 1}), ("b", 2.0, {"t": 2})]))
    assert log["n_events"] == 2
    assert log["n_distinct_payloads"] is None
    assert log["payload_purity_lift"] is None


# --------------------------------------------------------------------------- detector contract


def test_a_list_of_edges_is_rejected_with_an_explanation():
    with pytest.raises(TypeError, match="set of"):
        check_edges([("a", "b")])


def test_a_generator_of_edges_is_rejected():
    with pytest.raises(TypeError):
        check_edges((e for e in [("a", "b")]))


def test_a_self_edge_is_rejected_because_no_pairwise_statistic_can_test_it():
    with pytest.raises(ValueError, match="self-edge"):
        check_edges({("a", "a")})


def test_a_malformed_edge_is_rejected():
    with pytest.raises(ValueError, match="2-tuple"):
        check_edges({("a", "b", "c")})


def test_a_well_formed_edge_set_passes_through_unchanged():
    assert check_edges({("a", "b"), ("b", "c")}) == frozenset({("a", "b"), ("b", "c")})


def test_empty_edge_set_is_valid_because_certifying_nothing_is_an_answer():
    assert check_edges(set()) == frozenset()


def test_detector_name_prefers_an_explicit_name_then_falls_back_to_the_class():
    class Anon:
        def __call__(self, events):
            return set()

    class Named:
        name = "my-detector"

        def __call__(self, events):
            return set()

    def plain(events):
        return set()

    assert detector_name(Named()) == "my-detector"
    assert detector_name(Anon()) == "Anon"
    assert detector_name(plain) == "plain"


# --------------------------------------------------------------------------- log diagnostics


def test_turn_taking_index_separates_a_protocol_from_independent_speakers():
    """The diagnostic that decides whether the conventional surrogate is safe on a log."""
    ordered = describe_log(synth_log(n_events=1500, turn_taking=0.95, seed=1)["events"])
    free = describe_log(synth_log(n_events=1500, turn_taking=0.0, seed=1)["events"])
    assert ordered["turn_taking_index"] < 0.25
    assert free["turn_taking_index"] > 0.75


def test_effective_adjacent_pair_count_is_small_under_a_turn_taking_protocol():
    """A plain count of pairs seen once saturates on a long log; the effective count does not."""
    log = describe_log(synth_log(n_events=2000, n_actors=6, turn_taking=0.9, seed=2)["events"])
    assert log["lag1_pairs_possible"] == 30
    assert log["lag1_pairs_observed"] == 30, "every pair occurs at least once on a long log"
    assert log["lag1_pairs_effective"] < 10, "but almost all the mass sits on a handful"


def test_payload_purity_lift_rises_when_actors_have_characteristic_subjects():
    characteristic = describe_log(
        synth_log(n_events=1500, common_cause=0.0, seed=3)["events"])
    interchangeable = describe_log(
        synth_log(n_events=1500, common_cause=1.0, seed=3)["events"])
    assert characteristic["payload_purity_lift"] > interchangeable["payload_purity_lift"]
    assert interchangeable["payload_purity_lift"] == pytest.approx(1.0, abs=0.15)


def test_payload_repeat_rate_rises_with_slower_drift():
    slow = describe_log(synth_log(n_events=1500, drift=0.01, common_cause=1.0, seed=4)["events"])
    fast = describe_log(synth_log(n_events=1500, drift=0.9, common_cause=1.0, seed=4)["events"])
    assert slow["payload_repeat_rate"] > fast["payload_repeat_rate"]


def test_a_real_log_has_no_tied_timestamps_so_the_measure_can_detect_a_surrogate_that_adds_them():
    log = describe_log(synth_log(n_events=800, seed=5)["events"])
    assert log["tied_timestamp_rate"] == 0.0
