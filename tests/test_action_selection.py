from system_one_models.action_selection import rank,promotion


def sample_rows():
    rows=[]
    for lang in ("en","fr"):
        for i,gold in enumerate((True,True,False,False)):
            policies={"balanced":dict(predicted=True,positive_ms=1),
                      "always_false":dict(predicted=False,positive_ms=.1),
                      "better":dict(predicted=gold,positive_ms=5)}
            rows.append(dict(id=lang+str(i),language=lang,expected=gold,policies=policies))
    return rows


def test_balanced_selection_does_not_reward_always_false():
    rows=sample_rows()
    assert rank(rows,"better")>rank(rows,"balanced")>rank(rows,"always_false")


def test_promotion_requires_positive_recall_and_each_language_accuracy():
    rows=sample_rows()
    assert promotion(rows,"better")["passes"]
    assert not promotion(rows,"always_false")["passes"]
    # A perfect negative class does not justify losing the reference positives.
    for r in rows:
        if r["language"]=="fr" and r["expected"]:r["policies"]["better"]["predicted"]=False
    assert not promotion(rows,"better")["passes"]
