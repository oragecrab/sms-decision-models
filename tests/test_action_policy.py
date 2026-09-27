import json
import pytest
from system_one_models.action_policy import TaskCountPolicy,validate_policy
from system_one_models.action_improvement import config_hash


def file(tmp_path):
    execution=dict(head_key="count",multiple_labels=["multiple_tasks"],
        positive_questions={"count":dict(type="choice",instructions="Count tasks",criteria={"zero_tasks":"None","one_task":"One","multiple_tasks":"Two or more"})},
        negative_questions={"count":dict(type="noul",instructions="At most one?",criteria={"true":"Yes","false":"No"})})
    policy=dict(version="task-count-policy-v1",execution=execution,execution_sha256=config_hash(execution))
    path=tmp_path/"policy.json";path.write_text(json.dumps(policy));return path,policy


class FakeBackend:
    metadata={"model":"fake"}
    def __init__(self):self.calls=[]
    def predict(self,state,*,questions):
        self.calls.append((state,questions))
        if questions["count"]["type"]=="choice":
            return {"count":dict(choice="multiple_tasks",probabilities={"zero_tasks":.25,"one_task":.35,"multiple_tasks":.4})}
        return {"count":{"noul":.9}}


def test_categorical_winner_preserved_without_fabricating_binary_probability(tmp_path):
    path,_=file(tmp_path);backend=FakeBackend()
    result=TaskCountPolicy(path,backend=backend).predict({"body":"message"})
    assert result["multiple_actions"] is True
    assert result["selected_category_probability"]==.4
    assert "noul" not in result
    assert len(backend.calls)==1


def test_opposition_is_actual_separate_request_and_does_not_override_answer(tmp_path):
    path,policy=file(tmp_path);backend=FakeBackend()
    result=TaskCountPolicy(path,backend=backend).predict({"body":"message"},check_opposition=True)
    assert result["multiple_actions"] is True
    assert result["accepted"] is False
    assert len(backend.calls)==2
    assert backend.calls[1][1]==policy["execution"]["negative_questions"]


def test_policy_hash_guards_choice_order(tmp_path):
    _,policy=file(tmp_path)
    policy["execution"]["positive_questions"]["count"]["criteria"]={"multiple_tasks":"Two or more","one_task":"One","zero_tasks":"None"}
    with pytest.raises(ValueError,match="hash mismatch"):validate_policy(policy)


def test_cli_task_policy_mode_uses_frozen_policy(monkeypatch,tmp_path,capsys):
    import sys
    import system_one_models.action_policy as policy_module
    from system_one_models.cli import main
    class FakePolicy:
        def __init__(self,path,*,device):assert path==tmp_path/"policy.json"
        def predict(self,state,*,check_opposition):
            assert state["body"]=="Call and install"
            assert check_opposition
            return dict(multiple_actions=True,task_count_band="multiple_tasks",positive_answers={})
    monkeypatch.setattr(policy_module,"TaskCountPolicy",FakePolicy)
    monkeypatch.setattr(sys,"argv",["system-one-models","--task-count-policy",str(tmp_path/"policy.json"),"--body","Call and install","--task-count-opposition"])
    main()
    assert json.loads(capsys.readouterr().out)["multiple_actions"] is True


def test_cli_task_mode_rejects_unfrozen_model_override(monkeypatch,tmp_path):
    import sys
    from system_one_models.cli import main
    monkeypatch.setattr(sys,"argv",["system-one-models","--task-count-policy",str(tmp_path/"policy.json"),"--model","laya","--body","hello"])
    with pytest.raises(SystemExit):main()


def test_binary_mass_decoder_preserves_probability_and_differs_from_category_winner(tmp_path):
    path,policy=file(tmp_path)
    policy["execution"].update(decision="binary_probability",positive_threshold=.5)
    policy["execution_sha256"]=config_hash(policy["execution"])
    path.write_text(json.dumps(policy))
    result=TaskCountPolicy(path,backend=FakeBackend()).predict({"body":"message"})
    assert result["task_count_band"]=="multiple_tasks"
    assert result["multiple_actions"] is False
    assert result["multiple_actions_probability"]==.4
    assert result["probabilities"]=={"zero_tasks":.25,"one_task":.35,"multiple_tasks":.4}
