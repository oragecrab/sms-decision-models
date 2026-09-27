import pytest
from system_one_models.action_decomposition import (
    PAIR_CRITERIA, confirm, decompose, pair_questions, propose_spans, summarize,
)


def candidate(id):
    return dict(id=id, text=id)


def relationship(choice):
    return dict(choice=choice, probabilities={k: float(k == choice) for k in PAIR_CRITERIA})


def test_proposals_preserve_source_offsets_and_do_not_split_decimal_or_url():
    state={"body": "Pay $38.20 at https://example.invalid. Then call support or reply STOP.",
           "subject": "Account update"}
    spans=propose_spans(state)
    for span in spans:
        assert state[span["field"]][span["start"]:span["end"]] == span["text"]
    assert [s["text"] for s in spans] == ["Account update", "Pay $38.20 at https://example.invalid", "call support", "reply STOP"]


@pytest.mark.parametrize("relation", ["same_task_steps", "alternatives", "unsupported"])
def test_multiple_categories_or_alternatives_do_not_establish_two_tasks(relation):
    candidates=[candidate("c0"),candidate("c1")]
    gates={c["id"]:{"noul":.9} for c in candidates}
    result=decompose(candidates,gates,{"c0__c1":relationship(relation)})
    assert result["multiple_actions"] is False
    assert result["task_count_lower_bound"] is None


def test_supported_pair_establishes_two_tasks_without_exact_count():
    candidates=[candidate("c0"),candidate("c1"),candidate("c2")]
    gates={c["id"]:{"noul":.9} for c in candidates}
    pairs={key:relationship("independent_together") for key in pair_questions(candidates)}
    result=decompose(candidates,gates,pairs)
    assert result["multiple_actions"] is True
    assert result["task_count_lower_bound"] == 2
    assert result["independent_pair_count"] == 3


def test_rejected_candidates_cannot_supply_pair_witness():
    candidates=[candidate("c0"),candidate("c1")]
    result=decompose(candidates,{"c0":{"noul":.9},"c1":{"noul":.1}}, {})
    assert result["selected_candidate_ids"] == ["c0"]
    assert not result["multiple_actions"]


def test_ties_flag_review_and_do_not_silently_choose_pair():
    candidates=[candidate("c0"),candidate("c1")]
    tied={"choice":"independent_together", "probabilities":{k: (.5 if k in ("independent_together","same_task_steps") else 0) for k in PAIR_CRITERIA}}
    result=decompose(candidates,{"c0":{"noul":.9},"c1":{"noul":.9}}, {"c0__c1":tied})
    assert result["unresolved"] and not result["multiple_actions"]
    result=decompose(candidates,{"c0":{"noul":.5},"c1":{"noul":.9}}, {})
    assert result["candidate_gate_ties"] == ["c0"]


def test_missing_pair_judgment_is_an_error():
    candidates=[candidate("c0"),candidate("c1")]
    with pytest.raises(ValueError,match="Incomplete selected-pair"):
        decompose(candidates,{"c0":{"noul":.9},"c1":{"noul":.9}}, {})


@pytest.mark.parametrize("predicted,p,accepted", [(True,.1,True),(False,.9,True),(True,.9,False),(False,.1,False),(True,.5,False)])
def test_actual_complement_decision(predicted,p,accepted):
    assert confirm(predicted,{"noul":p}) is accepted


def test_summary_error_conservation():
    rows=[dict(expected=False,policies={"decomposed":dict(predicted=True,accepted=True,positive_ms=1)}),
          dict(expected=True,policies={"decomposed":dict(predicted=False,accepted=False,positive_ms=2)})]
    result=summarize(rows,"decomposed")
    assert result["false_positives"] == result["false_negatives"] == 1
    assert result["accepted_errors"] == result["rejected_errors"] == 1


def test_declared_relationship_choice_must_match_distribution():
    from system_one_models.action_decomposition import decisive_choice
    answer=relationship("same_task_steps")
    answer["choice"]="independent_together"
    with pytest.raises(ValueError,match="conflicts"):
        decisive_choice(answer)


def test_zero_batch_size_rejected():
    from system_one_models.action_decomposition import predict_stage
    with pytest.raises(ValueError,match="Batch size"):
        predict_stage(None,{}, {},batch_size=0)


def test_split_factors_require_both_conditions():
    from system_one_models.action_decomposition_variants import factor_result
    answers={"c0__c1__joint":{"noul":.9},"c0__c1__independent":{"noul":.1}}
    assert not factor_result(answers)["multiple_actions"]
    answers["c0__c1__independent"]={"noul":.9}
    assert factor_result(answers)["multiple_actions"]
    answers["c0__c1__joint"]={"noul":.5}
    assert factor_result(answers)["unresolved"]
    assert not factor_result(answers)["multiple_actions"]


def test_choice_gate_tie_or_nonwinning_label_not_accepted():
    from system_one_models.action_decomposition_variants import selection
    answer={"choice":"requested","probabilities":{"requested":.5,"not_requested":.5}}
    assert not selection(answer)
    answer["probabilities"]={"requested":.1,"not_requested":.9}
    with pytest.raises(ValueError,match="conflicts"):
        selection(answer)


def test_audit_separates_gate_and_relationship_failures():
    from system_one_models.action_decomposition_audit import diagnose
    audit={"one":dict(requested_candidate_ids=["c0","c1"],candidate_task_groups={"c0":"call","c1":"install"},multiple_actions=True,notes="")}
    row=dict(id="one",language="en",expected=True,candidates=[candidate("c0"),candidate("c1")],
             decomposition=dict(selected_candidate_ids=["c0"],independent_pair_witnesses=[],multiple_actions=False,unresolved=False))
    r=diagnose([row],audit,variant="noul_gate_choice_pairs")["all"]["summary"]
    assert r["gate_loses_all_true_independent_pairs"]==1
    assert r["relationship_false_negative_with_true_pair_available"]==0
    row["decomposition"]["selected_candidate_ids"]=["c0","c1"]
    r=diagnose([row],audit,variant="noul_gate_choice_pairs")["all"]["summary"]
    assert r["relationship_false_negative_with_true_pair_available"]==1
