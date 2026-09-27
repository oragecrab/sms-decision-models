import math
import pytest
from system_one_models.filter_comparison import matched, curve, metrics, labels, analyze


def row(id, confidence, correct, opposition):
    return dict(id=id, question="flag", language="en", confidence=confidence,
                expected=True, predicted=correct, correct=correct, opposition=opposition)


def test_matched_exact_budget_and_error_conservation():
    rows = [row("a", .9, False, False), row("b", .8, True, True), row("c", .7, False, True)]
    result = matched(rows)
    assert result["confidence_top_k"]["accepted"] == result["opposition"]["accepted"] == 2
    assert result["overlap"] == 1
    assert result["confidence_only"] == result["opposition_only"] == 1
    for policy in ["confidence_top_k", "opposition"]:
        assert result[policy]["accepted_errors"] + result[policy]["rejected_errors"] == 2


def test_boundary_ties_are_stable_and_report_error_bounds():
    rows = [row("a", .9, True, False), row("b", .8, False, True), row("c", .8, True, True)]
    first = matched(rows)
    assert first == matched(list(reversed(rows)))
    assert first["boundary_tie"]["accepted_errors_min"] == 0
    assert first["boundary_tie"]["accepted_errors_max"] == 1
    assert [p["accepted"] for p in curve(rows)] == [0, 1, 3]


@pytest.mark.parametrize("accept", [False, True])
def test_zero_and_full_coverage(accept):
    rows = [row("a", .9, False, accept), row("b", .8, True, accept)]
    result = matched(rows)
    assert result["confidence_top_k"]["accepted"] == (2 if accept else 0)
    assert result["boundary_tie"] is None
    if not accept:
        assert result["confidence_top_k"]["selective_accuracy"] is None


def fixture():
    questions = {"category": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
                 "flag": {"type": "noul", "criteria": {"true": "T", "false": "F"}}}
    report = {"questions": questions}
    prediction = dict(id="one", language="fr", positive={
        "category": {"choice": "b", "probabilities": {"a": .3, "b": .7}, "confidence": .4},
        "flag": {"noul": .1, "confidence": .9}},
        negative={"category": {"noul": .1}, "flag": {"noul": .9}},
        expected={"category": "a", "flag": False},
        matches={"category": False, "flag": True}, accepted={"category": True, "flag": True})
    return report, {"model": {}, "predictions": [prediction]}


def test_selected_probability_and_false_boolean_confidence():
    report, run = fixture()
    results = labels(report, run)
    assert results[0]["confidence"] == .7
    assert results[1]["confidence"] == .9
    assert results[1]["predicted"] is False
    result = analyze(report, run)
    assert result["stratified_matched"]["confidence_top_k"]["accepted_errors"] == 1


def test_stored_masks_are_verified():
    report, run = fixture()
    run["predictions"][0]["accepted"]["flag"] = False
    with pytest.raises(ValueError, match="Saved scores"):
        labels(report, run)


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -1, 2])
def test_invalid_native_confidence(value):
    report, run = fixture()
    run["predictions"][0]["positive"]["category"]["confidence"] = value
    with pytest.raises(ValueError, match="Invalid or missing"):
        analyze(report, run, score="native_confidence")


def test_scam_errors_and_boolean_errors_are_separate():
    rows = [dict(row("a", .9, False, True), question="classification", expected="legitimate", predicted="suspected_scam"),
            dict(row("b", .9, False, True), question="classification", expected="suspected_scam", predicted="marketing"),
            row("c", .9, False, True)]
    result = metrics(rows, rows)
    assert result["accepted_false_scam_flags"] == 1
    assert result["accepted_missed_scams"] == 1
    assert result["accepted_false_negatives"] == 1
