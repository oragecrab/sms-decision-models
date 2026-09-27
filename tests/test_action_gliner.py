from system_one_models.action_gliner import NAMES,decode,evaluate,positive_schema
from system_one_models.action_improvement import configurations

class FakeBackend:
    def __init__(self):self.calls=[]
    def predict(self,state,*,questions):
        self.calls.append(questions)
        if len(self.calls)==1:
            result={}
            for key,spec in questions.items():
                if spec['type']=='noul':result[key]={'noul':.8}
                else:
                    chosen='multiple' if 'multiple' in spec['criteria'] else 'multiple_tasks'
                    n=len(spec['criteria'])
                    result[key]={'choice':chosen,'probabilities':{k:.8 if k==chosen else .2/(n-1) for k in spec['criteria']}}
            return result
        return {key:{'noul':.2} for key in questions}

def test_gliner_passes_five_heads_and_keeps_actual_opposition():
    configs=configurations();backend=FakeBackend()
    row=evaluate(backend,{'body':'First operation and second operation.'},configs)
    assert len(backend.calls)==2
    assert all(len(call)==5 for call in backend.calls)
    assert set(row['policies'])=={f'gliner_{name}' for name in NAMES}
    for name in NAMES:
        policy=row['policies'][f'gliner_{name}']
        assert policy['predicted'] and policy['accepted']
        assert policy['positive_ms']==row['batch_positive_ms']
        assert policy['negative_ms']==row['batch_negative_ms']
        spec=policy['negative_questions']['multiple_actions']
        assert spec['type']=='noul'
        if configs[name]['questions']['multiple_actions']['type']=='noul':
            positive=configs[name]['questions']['multiple_actions']
            assert spec['criteria']['true']==positive['criteria']['false']
        else:assert 'at most ONE' in spec['instructions']

def test_gliner_decode_does_not_accept_boolean_tie():
    spec=positive_schema(configurations())['gliner_original']
    assert decode({'noul':.5},spec)['unresolved']
