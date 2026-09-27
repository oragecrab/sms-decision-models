import copy
import hashlib
import json
import pytest
from system_one_models.action_binary_rescore import rescore, validate_source
from system_one_models.action_improvement import config_hash
from system_one_models.action_policy import validate_policy


def fixture(tmp_path):
    questions={'count':{'type':'choice','instructions':'Count tasks','criteria':{'zero':'None','one':'One','multiple':'Two'}}}
    negative={'count':{'type':'noul','instructions':'At most one','criteria':{'true':'Yes','false':'No'}}}
    execution=dict(head_key='count',positive_questions=questions,negative_questions=negative,multiple_labels=['multiple'],decision='binary_probability',positive_threshold=.6)
    model=dict(model='fake',revision='commit',checkpoints={'hash':'frozen'},parameter_dtype='float32')
    policy=dict(version='task-count-policy-v1',execution=execution,execution_sha256=config_hash(execution),model=copy.deepcopy(model),packages={'fake':'1'},selected='binary',previous_selection='categorical')
    dataset=tmp_path/'data.jsonl';dataset.write_text('test fixture dataset')
    answer=dict(choice='multiple',probabilities={'zero':.2,'one':.3,'multiple':.5})
    value=dict(answers={'multiple_actions':answer},negative={'multiple_actions':{'noul':.8}},positive_ms=1,negative_ms=1,predicted=True,accepted=False)
    row=dict(id='one',language='en',expected=False,policies={'categorical':value},batch_questions=questions,batch_negative_questions=negative)
    report=dict(dataset=str(dataset),dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),model=model,packages={'fake':'1'},predictions=[row,dict(copy.deepcopy(row),id='two',language='fr')])
    return report,policy


def test_rescore_validates_probabilities_and_reports_actual_threshold(tmp_path):
    report,policy=fixture(tmp_path)
    source=tmp_path/'source.json';frozen=tmp_path/'policy.json';output=tmp_path/'out.json'
    source.write_text(json.dumps(report));frozen.write_text(json.dumps(policy))
    result=rescore(source,frozen,output)
    assert '0.6' in result['note']
    value=result['predictions'][0]['policies']['binary']
    assert value['predicted'] is False and value['accepted'] is True
    assert value['multiple_actions_probability']==.5
    report['predictions'][0]['policies']['categorical']['answers']['multiple_actions']['probabilities']['multiple']=.9
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Invalid Choice'):rescore(source,frozen,output)


@pytest.mark.parametrize('mutation',[lambda r:r['packages'].update(fake='changed'),lambda r:r['model'].update(revision='changed'),lambda r:r['predictions'][0].pop('batch_negative_questions'),lambda r:r.update(dataset_sha256='changed')])
def test_rescore_rejects_incompatible_saved_provenance(tmp_path,mutation):
    report,policy=fixture(tmp_path);mutation(report)
    with pytest.raises(ValueError):validate_source(report,policy)


def test_frozen_legacy_source_requires_prediction_execution_hash(tmp_path):
    report,policy=fixture(tmp_path)
    frozen=copy.deepcopy(policy);frozen['selected']='categorical';frozen['execution'].pop('decision');frozen['execution'].pop('positive_threshold');frozen['execution_sha256']=config_hash(frozen['execution'])
    report['frozen_policy']=frozen
    for row in report['predictions']:
        row['policies']['categorical'].update(policy_sha256=frozen['execution_sha256'],model=copy.deepcopy(policy['model']))
    value=report['predictions'][0]['policies']['categorical']
    validate_source(report,policy)
    value['policy_sha256']='changed'
    with pytest.raises(ValueError,match='execution hash'):validate_source(report,policy)


@pytest.mark.parametrize('labels',[[],['multiple','multiple'],['zero','one','multiple']])
def test_multiple_labels_are_nonempty_unique_and_have_negative_complement(tmp_path,labels):
    _,policy=fixture(tmp_path);policy['execution']['multiple_labels']=labels;policy['execution_sha256']=config_hash(policy['execution'])
    with pytest.raises(ValueError,match='unique nonempty'):validate_policy(policy)


def test_opposition_requires_boolean_head(tmp_path):
    _,policy=fixture(tmp_path);policy['execution']['negative_questions']['count']['type']='choice';policy['execution_sha256']=config_hash(policy['execution'])
    with pytest.raises(ValueError,match='Boolean'):validate_policy(policy)
