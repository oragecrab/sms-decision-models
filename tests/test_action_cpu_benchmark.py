import pytest
from system_one_models.action_cpu_benchmark import percentile, select_sample, portable_identity


def test_percentile_nearest_rank():
    assert percentile([4, 1, 2, 3], .95) == 4
    assert percentile([4, 1, 2, 3], .5) == 2
    with pytest.raises(ValueError): percentile([], .95)


def test_sample_preserves_strata_and_pairs():
    rows=[]
    for risk in ('routine','suspicious'):
        for expected in (False,True):
            for n in range(3):
                for lang in ('en','fr'):
                    rows.append(dict(pair_id=f'{risk}-{expected}-{n}',language=lang,risk_stratum=risk,expected=expected))
    selected=select_sample(rows)
    assert selected == select_sample(list(reversed(rows)))
    assert len(selected)==16
    assert len({r['pair_id'] for r in selected})==8
    for risk in ('routine','suspicious'):
        for expected in (False,True):
            assert sum(r['risk_stratum']==risk and r['expected']==expected for r in selected)==4


def test_portable_identity_retains_revision_and_content():
    assert portable_identity(dict(snapshot='/a/rev',weights_blob='/a/blob',weights_bytes=8)) == portable_identity(dict(snapshot='/b/rev',weights_blob='/b/blob',weights_bytes=8))
    assert portable_identity(dict(snapshot='/a/rev',weights_bytes=8)) != portable_identity(dict(snapshot='/a/new',weights_bytes=8))
