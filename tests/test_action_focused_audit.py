from system_one_models.action_focused_audit import relationship, summarize
import pytest


def test_reviewed_relationship_uses_explicit_alternative_not_task_partition():
    audit = {'requested_candidate_ids': ['c0', 'c1'],
             'candidate_pair_relationships': {'c0__c1': 'alternatives'}}
    assert relationship(audit, 'c0__c1', {'alternatives': '', 'unsupported': ''}) == 'alternatives'
    assert relationship(audit, 'c0__c1', {'one_task_or_alternative': '', 'unsupported': ''}) == 'one_task_or_alternative'
    assert relationship(audit, 'c0__c2', {}) == 'unsupported'
    with pytest.raises(ValueError, match='Missing reviewed'):
        relationship(dict(audit, candidate_pair_relationships={}), 'c0__c1', {})


def test_component_summary_separates_gate_errors_from_supplied_span_diagnostic():
    row = dict(language='en', expected=True, requested_candidate_ids=['c0', 'c1'], candidate_ids=['c0', 'c1', 'c2'],
        variants={'test': dict(selected_candidate_ids=['c0', 'c2'],
            saved_pair_diagnostics={'c0__c2': dict(expected='unsupported', predicted='independent_together')},
            supplied_pair_diagnostics={'c0__c1': dict(expected='independent_together', predicted='independent_together')},
            supplied_multiple_actions=True, cached_pair_ids=[], inferred_pair_ids=['c0__c1'])})
    report = summarize([row], 'test', 'en')
    assert report['gate']['precision'] == .5
    assert report['gate']['recall'] == .5
    assert report['saved_pairs']['witness_precision'] == 0
    assert report['supplied_pairs']['witness_precision'] == 1
    assert report['supplied_span_messages']['accuracy'] == 1
    assert report['inferred_pairs'] == 1
