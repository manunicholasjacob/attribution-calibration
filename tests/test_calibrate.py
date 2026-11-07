"""End-to-end tests of the harness, against detectors whose behaviour is known in advance.

If the harness is right, then a detector that certifies every pair must come out with a family-wise
error of 1, a detector that certifies nothing must come out with no power and an error of 0, and a
detector that does a defensible thing must come out somewhere sensible in between. Those three are
the load-bearing assertions in this file.
"""
import warnings

import pytest

from attrcal import PlantSpec, calibrate, compare_nulls, detectors, synth_log


def turn_ordered(n=1500, seed=3):
    """A log with the two properties the surrogate comparison depends on: speakers take turns and
    the channel's subject drifts, so there is autocorrelation for one null to keep and another to
    destroy."""
    return synth_log(n_events=n, n_actors=6, n_payloads=8, turn_taking=0.9, drift=0.05,
                     common_cause=0.3, seed=seed)["events"]


def flat(text):
    """Collapse the printable block's line wrapping, so a phrase can be asserted without the
    assertion depending on where textwrap happened to break the line."""
    return " ".join(text.split())


# --------------------------------------------------------------------- the two extreme detectors


def test_a_detector_that_certifies_everything_scores_as_badly_as_possible_on_nulls():
    """Recall is not evidence. This detector has perfect recall of anything and is worthless, and
    the report has to say so in the family-wise error line."""
    report = calibrate(turn_ordered(600), detectors.all_pairs, n_null=40,
                       planted=PlantSpec(strengths=(0.1, 0.8), volume=100, n_seeds=10), seed=0)
    assert report.familywise_error == 1.0
    assert report.familywise_error_interval[0] > 0.9
    assert not report.calibrated()
    assert report.mean_edges_per_null > 1
    assert all(r.point == 1.0 for r in report.power_curve.values())
    assert report.detection_floor["status"] == "below_grid"


def test_a_detector_that_certifies_nothing_has_perfect_error_and_no_power():
    """The failure mode that a false positive rate alone cannot catch: silence passes it."""
    report = calibrate(turn_ordered(600), detectors.no_edges, n_null=40,
                       planted=PlantSpec(strengths=(0.1, 0.8), volume=100, n_seeds=10), seed=0)
    assert report.familywise_error == 0.0
    assert report.mean_edges_per_null == 0.0
    assert report.max_edges_on_a_null == 0
    assert all(r.point == 0.0 for r in report.power_curve.values())
    assert report.detection_floor["status"] == "never_reached"
    assert report.detection_floor["floor"] is None


def test_the_two_extreme_detectors_are_distinguished_only_by_reading_both_numbers():
    """Either number alone ranks one of these two detectors first, and both are useless."""
    ev = turn_ordered(600)
    spec = PlantSpec(strengths=(0.4,), volume=100, n_seeds=10)
    everything = calibrate(ev, detectors.all_pairs, n_null=30, planted=spec, seed=0)
    nothing = calibrate(ev, detectors.no_edges, n_null=30, planted=spec, seed=0)
    assert everything.familywise_error > nothing.familywise_error
    assert float(everything.power_curve[0.4]) > float(nothing.power_curve[0.4])


# --------------------------------------------------------------------- a defensible detector


def test_a_surrogate_tested_detector_is_near_nominal_and_has_real_power():
    detector = detectors.SurrogateCopyDetector(n_surrogates=80, null="permute", seed=0)
    report = calibrate(turn_ordered(2000), detector, null="permute", n_null=60,
                       planted=PlantSpec(strengths=(0.05, 0.1, 0.2, 0.4, 0.8), volume=200,
                                         n_seeds=10),
                       seed=0)
    assert report.calibrated(), f"family-wise error {report.familywise_error_rate}"
    assert report.familywise_error < 0.2
    assert float(report.power_curve[0.8]) == 1.0
    assert report.detection_floor["status"] in {"bracketed", "below_grid"}
    assert report.detection_floor["floor"] is not None


def test_power_rises_with_the_planted_copy_fraction():
    detector = detectors.SurrogateCopyDetector(n_surrogates=60, null="permute", seed=0)
    report = calibrate(turn_ordered(2000), detector, n_null=0,
                       planted=PlantSpec(strengths=(0.02, 0.4), volume=200, n_seeds=10), seed=0)
    assert float(report.power_curve[0.02]) < float(report.power_curve[0.4])


def test_the_realised_copy_fraction_is_reported_next_to_every_requested_one():
    report = calibrate(turn_ordered(1200), detectors.no_edges, n_null=0,
                       planted=PlantSpec(strengths=(0.2, 0.8), volume=150, n_seeds=10), seed=0)
    assert set(report.realised_strength) == {0.2, 0.8}
    assert report.realised_strength[0.8] > report.realised_strength[0.2]


# --------------------------------------------------------------------- the surrogate comparison


def test_permute_and_shift_disagree_on_a_turn_ordered_log():
    """The finding this package exists for. The same detector on the same log gets a family-wise
    error near 0 under the permutation null and near 1 under the conventional circular shift, so at
    most one of them is that detector's real error rate."""
    ev = turn_ordered(2000)
    out = compare_nulls(ev, detectors.CopyRateDetector(threshold=0.4),
                        nulls=("permute", "shift"), n_null=60, seed=0)
    assert out.disagree
    assert out.rates["permute"].hi < 0.2
    assert out.rates["shift"].lo > 0.8
    assert out.moves_actor_sequence == {"permute": False, "shift": True}
    assert "not a surrogate of this log" in flat(out.summary())


def test_the_disagreement_survives_a_detector_that_does_control_its_own_error_rate():
    """It is not only naive detectors. A detector with its own surrogate test and a family-wise
    correction still reads as broken when it is audited against shift replicates, because those
    replicates are not structure-free."""
    ev = turn_ordered(1500)
    detector = detectors.SurrogateCopyDetector(n_surrogates=60, null="permute", seed=0)
    out = compare_nulls(ev, detector, nulls=("permute", "shift"), n_null=30, seed=0)
    assert out.rates["permute"].point < out.rates["shift"].point
    assert out.disagree


def test_the_nulls_agree_when_the_log_has_no_turn_taking_and_no_characteristic_subjects():
    """The negative control for the test above. If the nulls disagreed on every log the comparison
    would be measuring the harness rather than the data."""
    ev = synth_log(n_events=2000, n_actors=6, turn_taking=0.0, drift=0.9, common_cause=1.0,
                   seed=5)["events"]
    out = compare_nulls(ev, detectors.CopyRateDetector(threshold=0.4),
                        nulls=("permute", "shift"), n_null=60, seed=0)
    assert not out.disagree


def test_comparing_a_surrogate_against_itself_is_rejected():
    with pytest.raises(ValueError, match="at least two distinct"):
        compare_nulls(turn_ordered(400), detectors.no_edges, nulls=("permute", "permute"),
                      n_null=5)


def test_choosing_the_shift_null_warns_the_caller():
    with pytest.warns(UserWarning, match="0.90"):
        calibrate(turn_ordered(400), detectors.no_edges, null="shift", n_null=5, seed=0)


def test_the_report_carries_a_null_audit_that_names_the_damage():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shifted = calibrate(turn_ordered(1500), detectors.no_edges, null="shift", n_null=5, seed=0)
    permuted = calibrate(turn_ordered(1500), detectors.no_edges, null="permute", n_null=5, seed=0)
    assert shifted.null_audit["preserves_actor_sequence"] is False
    assert shifted.null_audit["widened_adjacency"] is True
    assert permuted.null_audit["preserves_actor_sequence"] is True
    assert "not a null for this log" in flat(shifted.null_audit["verdict"])


# --------------------------------------------------------------------- reproducibility


def test_two_runs_at_the_same_seed_produce_the_same_report():
    ev = turn_ordered(1000)
    detector = detectors.SurrogateCopyDetector(n_surrogates=40, null="permute", seed=0)
    spec = PlantSpec(strengths=(0.1, 0.4), volume=150, n_seeds=5)
    with warnings.catch_warnings():
        # The coarse-seed warning is asserted in its own test; here it is incidental.
        warnings.simplefilter("ignore")
        a = calibrate(ev, detector, n_null=30, planted=spec, seed=7)
        b = calibrate(ev, detector, n_null=30, planted=spec, seed=7)
    assert a.to_dict() == b.to_dict()


def test_different_seeds_change_the_replicates_without_changing_the_conclusion():
    ev = turn_ordered(1000)
    a = calibrate(ev, detectors.CopyRateDetector(threshold=0.4), n_null=40, seed=1)
    b = calibrate(ev, detectors.CopyRateDetector(threshold=0.4), n_null=40, seed=2)
    assert a.familywise_error < 0.2 and b.familywise_error < 0.2


def test_adding_a_volume_sweep_does_not_change_the_numbers_it_shares_with_a_run_without_one():
    """Each part of the run draws from its own spawned stream, so the sweeps are independent."""
    ev = turn_ordered(900)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        narrow = calibrate(ev, detectors.no_edges, n_null=20,
                           planted=PlantSpec(strengths=(0.4,), volume=100, n_seeds=3), seed=4)
        wide = calibrate(ev, detectors.no_edges, n_null=20,
                         planted=PlantSpec(strengths=(0.4,), volume=100, volumes=(50, 200),
                                           n_seeds=3), seed=4)
    assert narrow.familywise_error == wide.familywise_error
    assert narrow.realised_strength == wide.realised_strength
    assert wide.volume_curve and not narrow.volume_curve


# --------------------------------------------------------------------- contract and warnings


def test_a_detector_returning_a_list_is_rejected_with_a_message_naming_the_detector():
    def listy(events):
        return [("a", "b")]

    with pytest.raises(TypeError, match="listy"):
        calibrate(turn_ordered(300), listy, n_null=1, seed=0)


def test_too_few_power_seeds_warns_that_the_threshold_is_not_estimable():
    with pytest.warns(UserWarning, match="multiples of 1/n"):
        calibrate(turn_ordered(400), detectors.no_edges, n_null=0,
                  planted=PlantSpec(strengths=(0.4,), volume=50, n_seeds=3), seed=0)


def test_requesting_no_nulls_is_allowed_and_the_report_says_there_is_no_error_rate():
    report = calibrate(turn_ordered(300), detectors.no_edges, n_null=0, seed=0)
    assert report.familywise_error != report.familywise_error, "should be NaN"
    assert report.familywise_error_n == 0
    assert "no measured error rate" in " ".join(report.notes)
    assert "uncalibrated" in flat(report.summary())


def test_omitting_the_spike_in_leaves_an_explicit_no_data_floor_rather_than_a_blank():
    report = calibrate(turn_ordered(300), detectors.no_edges, n_null=5, seed=0)
    assert report.power_curve == {}
    assert report.detection_floor["status"] == "no_data"
    assert "half a result" in flat(report.summary())


def test_symbols_without_labels_is_rejected_because_one_partition_scores_against_nothing():
    with pytest.raises(ValueError, match="only meaningful together"):
        calibrate(turn_ordered(300), detectors.no_edges, n_null=1, symbols=[0, 1], seed=0)


def test_a_supplied_symbolisation_is_scored_and_scoped_in_the_summary():
    ev = turn_ordered(400)
    truth = [e[2] for e in ev]
    report = calibrate(ev, detectors.no_edges, n_null=1, symbols=truth, labels=truth, seed=0)
    assert report.symbolisation["adjusted_rand_index"] == 1.0
    assert "says nothing about what your detector does internally" in flat(report.summary())


def test_a_plant_name_colliding_with_a_real_actor_is_rejected():
    with pytest.raises(ValueError, match="already an actor"):
        calibrate(turn_ordered(300), detectors.no_edges, n_null=0,
                  planted=PlantSpec(name="actor00", strengths=(0.4,), n_seeds=1), seed=0)


def test_an_unknown_null_is_rejected_before_any_work_is_done():
    with pytest.raises(ValueError, match="unknown null"):
        calibrate(turn_ordered(300), detectors.no_edges, null="bootstrap", n_null=5, seed=0)


# --------------------------------------------------------------------- the printable block


def test_the_summary_names_the_surrogate_the_detector_and_every_interval():
    report = calibrate(turn_ordered(600), detectors.CopyRateDetector(threshold=0.4), n_null=30,
                       planted=PlantSpec(strengths=(0.4,), volume=100, n_seeds=10), seed=0)
    text = flat(report.summary())
    assert "permute" in text
    assert "CopyRateDetector" in text
    assert "Wilson" in text
    assert "DETECTION FLOOR" in text
    assert "FAMILY-WISE ERROR" in text
    assert "/30)" in text, "the family-wise error must print its denominator"


def test_the_summary_flags_a_turn_ordered_log_before_anyone_reports_a_shift_based_rate():
    report = calibrate(turn_ordered(600), detectors.no_edges, n_null=10, seed=0)
    assert "turn-taking index" in flat(report.summary())
    assert "Do not report a shift-based error rate" in flat(report.summary())


def test_the_report_is_json_serialisable_with_every_rate_expanded():
    import json
    report = calibrate(turn_ordered(600), detectors.no_edges, n_null=10,
                       planted=PlantSpec(strengths=(0.4,), volume=50, n_seeds=10), seed=0)
    blob = json.loads(json.dumps(report.to_dict()))
    assert blob["familywise_error"]["n"] == 10
    assert "lo" in blob["familywise_error"] and "hi" in blob["familywise_error"]
    assert blob["power_curve"]["0.4"]["n"] == 10


def test_the_summary_distinguishes_a_conservative_detector_from_an_anti_conservative_one():
    """Below nominal and above nominal are different findings. Conflating them is how a detector
    that certifies too little gets reported in the same words as one that certifies too much."""
    quiet = calibrate(turn_ordered(600), detectors.no_edges, n_null=200, seed=0)
    loud = calibrate(turn_ordered(600), detectors.all_pairs, n_null=200, seed=0)
    assert "lies BELOW" in flat(quiet.summary())
    assert "conservative rather than miscalibrated" in flat(quiet.summary())
    assert "lies ABOVE" in flat(loud.summary())
    assert "exceeds the one it claims" in flat(loud.summary())


def test_calibrated_is_about_the_interval_and_not_the_point_estimate():
    """An observed 0.03 and an observed 0.09 over 200 replicates are both consistent with 0.05, and
    a harness that called one of them broken would be reporting noise."""
    report = calibrate(turn_ordered(600), detectors.no_edges, n_null=20, seed=0)
    assert report.familywise_error == 0.0
    assert report.calibrated(), "zero out of twenty cannot exclude a nominal 0.05"


def test_progress_callback_is_invoked_once_per_null_replicate():
    seen = []
    calibrate(turn_ordered(300), detectors.no_edges, n_null=7, seed=0,
              progress=lambda *a: seen.append(a))
    assert len([s for s in seen if s[0] == "null"]) == 7
