import pytest
from system_one_models.action_improvement import focused_state,choice_decision,load_cases,configurations
from system_one_models.action_decomposition import propose_spans


def test_markers_preserve_original_source_and_target_offsets():
    state={"subject":"Update", "body":"Call the office and then upload the file."}
    spans=propose_spans(state)
    bodyspans=[c for c in spans if c["field"]=="body"]
    result=focused_state(state,bodyspans[:2],marked=True)
    assert result["body"].replace("[[TARGET_1]]","").replace("[[/TARGET_1]]","").replace("[[TARGET_2]]","").replace("[[/TARGET_2]]","")==state["body"]
    assert result["target_1"]==bodyspans[0]["text"]
    assert state["body"]=="Call the office and then upload the file."


def test_target_field_control_preserves_full_source():
    state={"body":"Do not reply. You can view the map."}
    span=propose_spans(state)[0]
    result=focused_state(state,[span],marked=False)
    assert result["body"]==state["body"]
    assert result["target_span"]==span["text"]


def test_wrong_offsets_fail():
    with pytest.raises(ValueError,match="offsets"):
        focused_state({"body":"hello"},[dict(field="body",start=0,end=2,text="no")],marked=True)


def test_decisive_choice_rejects_tie_and_inconsistent_label():
    assert choice_decision({"choice":"a","probabilities":{"a":.5,"b":.5}},{"a":"A","b":"B"}) is None
    with pytest.raises(ValueError,match="conflicts"):
        choice_decision({"choice":"a","probabilities":{"a":.1,"b":.9}},{"a":"A","b":"B"})


def test_translations_cannot_leak_between_splits(tmp_path):
    import json
    rows=[dict(id="one_en",pair_id="one",expected=True,language="en",split="development"),
          dict(id="one_fr",pair_id="one",expected=True,language="fr",split="heldout")]
    p=tmp_path/"cases.jsonl";p.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match="Translations|Related templates"):
        load_cases(p,"development")


def test_configuration_set_is_explicit_and_no_schema_replacement():
    configs=configurations()
    assert len(configs)==8
    assert configs["target_field"]["gate"]==configs["marked_target"]["gate"]
    assert configs["target_field"]["pair"]==configs["marked_target"]["pair"]


def test_full_policy_hash_is_order_sensitive_and_accepts_pipeline_definition():
    from system_one_models.action_improvement import config_hash
    config=configurations()["marked_compact"]
    assert len(config_hash(config))==64
    other=dict(config, marked=False)
    assert config_hash(config)!=config_hash(other)


def test_related_templates_cannot_cross_splits(tmp_path):
    import json
    rows=[]
    for pair,split in (("one","development"),("two","heldout")):
        for language in ("en","fr"):
            rows.append(dict(id=pair+language,pair_id=pair,template_group="same",expected=True,language=language,split=split))
    p=tmp_path/"cases.jsonl";p.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match="Related templates"):
        load_cases(p,"development")


@pytest.fixture
def development_report(tmp_path):
    import hashlib,json
    from system_one_models.action_improvement import config_hash
    examples=[]
    predictions=[]
    for pair,value in (("yes",True),("no",False)):
        for language in ("en","fr"):
            r=dict(id=pair+language,pair_id=pair,language=language,expected=value,split="development")
            examples.append(r)
            predictions.append(dict(r,policies={name:dict(predicted=value,accepted=False,positive_ms=1) for name in ("original","balanced")}))
    dataset=tmp_path/"cases.jsonl";dataset.write_text("\n".join(json.dumps(r) for r in examples))
    configs={name:configurations()[name] for name in ("original","balanced")}
    report=dict(split="development",dataset=str(dataset),dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                predictions=predictions,configs=configs,config_hashes={name:config_hash(v) for name,v in configs.items()},
                model={"checkpoints":{},"torch_threads":2},packages={})
    path=tmp_path/"report.json";path.write_text(json.dumps(report))
    return path,report


def test_selection_rejects_duplicate_rows(development_report,tmp_path):
    import json
    from system_one_models.action_improvement import freeze_selection
    path,report=development_report
    report["predictions"][1]=report["predictions"][0]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="duplicate"):
        freeze_selection(path,tmp_path/"selection.json")


def test_selection_rejects_changed_labels(development_report,tmp_path):
    import json
    from system_one_models.action_improvement import freeze_selection
    path,report=development_report
    report["predictions"][0]["expected"]=False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="metadata changed"):
        freeze_selection(path,tmp_path/"selection.json")


def test_selection_rejects_changed_dataset(development_report,tmp_path):
    from system_one_models.action_improvement import freeze_selection
    path,report=development_report
    Path=__import__("pathlib").Path
    Path(report["dataset"]).write_text("changed")
    with pytest.raises(ValueError,match="dataset changed"):
        freeze_selection(path,tmp_path/"selection.json")


def test_selection_recomputes_summary_and_freezes_provenance(development_report,tmp_path):
    import json
    from system_one_models.action_improvement import freeze_selection
    path,report=development_report
    report["summary"]={"balanced":"incorrect stored summary"}
    path.write_text(json.dumps(report))
    result=freeze_selection(path,tmp_path/"selection.json")
    assert result["heldout_used"] is False
    assert set(result["comparator_hashes"])=={"original","balanced"}
    assert result["development_model"]["torch_threads"]==2
