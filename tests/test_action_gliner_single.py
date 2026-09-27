from system_one_models.action_gliner_single import evaluate,HEAD_KEY,POLICY,CONFIG
from system_one_models.action_gliner import decode

class FakeBackend:
    def __init__(self):self.calls=[]
    def predict(self,state,*,questions):
        self.calls.append(questions)
        if len(self.calls)==1:
            return {HEAD_KEY:{'choice':'one_task','probabilities':{'zero_tasks':.1,'one_task':.6,'multiple_tasks':.3}}}
        return {HEAD_KEY:{'noul':.8}}

def test_single_model_head_and_actual_opposite_keep_exact_key():
    backend=FakeBackend();row=evaluate(backend,{'body':'One requested operation.'})
    assert len(backend.calls)==2
    assert all(list(questions)==[HEAD_KEY] for questions in backend.calls)
    assert backend.calls[0][HEAD_KEY]==CONFIG['questions']['multiple_actions']
    assert 'at most ONE' in backend.calls[1][HEAD_KEY]['instructions']
    result=row['policies'][POLICY]
    assert result['predicted'] is False
    assert result['answer_confidence']==.6
    assert result['accepted'] is True
    assert row['model_answers'][HEAD_KEY]['probabilities']['multiple_tasks']==.3

def test_categorical_prediction_uses_argmax_not_boolean_threshold():
    result=decode({'choice':'multiple_tasks','probabilities':{'zero_tasks':.28,'one_task':.32,'multiple_tasks':.40}},CONFIG['questions']['multiple_actions'])
    assert result['predicted'] is True
    assert result['answer_confidence']==.40
    assert result['unresolved'] is False

def test_categorical_tie_remains_unresolved():
    result=decode({'choice':'one_task','probabilities':{'zero_tasks':.1,'one_task':.45,'multiple_tasks':.45}},CONFIG['questions']['multiple_actions'])
    assert result['unresolved'] is True
