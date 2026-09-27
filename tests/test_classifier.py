from __future__ import annotations

import os
import re
from types import SimpleNamespace

import pytest

from system_one_models.classifier import (
    HEAD_MAX_LEN,
    MessageTooLongError,
    _check_token_budget,
    build_state,
    classify,
)
from system_one_models.evaluation import load_examples
from system_one_models.formatting import format_output
from system_one_models.questions import CATEGORIES, MESSAGE_HOOKS, QUESTIONS, REQUESTED_ACTIONS


def test_build_state_keeps_message_fields_and_omits_empty_subject() -> None:
    state = build_state(
        channel=" email ",
        sender=" sender@example.test ",
        subject="  ",
        body="  Hello  ",
    )

    assert state == {
        "channel": "email",
        "sender": "sender@example.test",
        "body": "Hello",
    }


def test_empty_body_is_rejected() -> None:
    with pytest.raises(ValueError, match="body must not be empty"):
        build_state(channel="sms", sender="", subject="", body="  ")


def test_question_schema_has_fixed_descriptive_categories() -> None:
    question = QUESTIONS["classification"]

    assert question["type"] == "choice"
    assert question["criteria"] == CATEGORIES
    assert set(CATEGORIES) == {"legitimate", "marketing", "suspected_scam"}
    assert set(QUESTIONS) == {
        "classification",
        "industry",
        "urgent_action",
        "sentiment",
        "requested_action",
        "multiple_actions",
        "message_hook",
        "sensitive_data_requested",
    }
    assert QUESTIONS["industry"]["type"] == "choice"
    assert QUESTIONS["urgent_action"]["type"] == "noul"
    assert QUESTIONS["sentiment"]["type"] == "choice"
    assert QUESTIONS["requested_action"]["type"] == "choice"
    assert QUESTIONS["requested_action"]["criteria"] == REQUESTED_ACTIONS
    assert QUESTIONS["multiple_actions"]["type"] == "noul"
    assert QUESTIONS["message_hook"]["type"] == "choice"
    assert QUESTIONS["message_hook"]["criteria"] == MESSAGE_HOOKS
    assert QUESTIONS["sensitive_data_requested"]["type"] == "noul"
    assert QUESTIONS["sensitive_data_requested"]["labels"] == {"true": "A", "false": "B"}


def test_classify_returns_answer_from_router() -> None:
    class FakeRouter:
        def predict(self, state, questions, *, on_predict_start, hooks_raise):
            assert on_predict_start is _check_token_budget
            assert hooks_raise is True
            on_predict_start(token_context(state, questions))
            assert state["body"] == "Verify your account"
            assert questions is QUESTIONS
            return {
                "answers": {
                    "classification": {
                        "choice": "suspected_scam",
                        "probabilities": {"suspected_scam": 0.9},
                    }
                }
            }

    answers = classify({"body": "Verify your account"}, router=FakeRouter())

    answer = answers["classification"]
    assert answer["choice"] == "suspected_scam"
    assert answer["probabilities"]["suspected_scam"] == 0.9


def test_base_eval_set_has_expected_values_for_every_question() -> None:
    examples = load_examples()

    assert len(examples) >= 10
    assert len({example["id"] for example in examples}) == len(examples)
    for example in examples:
        assert set(example["expected"]) == set(QUESTIONS)
        assert example["state"]["channel"] in {"sms", "email"}
        assert example["state"]["body"]
        for question_id, expected in example["expected"].items():
            question = QUESTIONS[question_id]
            if question["type"] == "choice":
                assert expected in question["criteria"]
            else:
                assert isinstance(expected, bool)


def test_compact_output_keeps_full_input_and_places_time_in_result_heading() -> None:
    body = "Long message " + ("details " * 39) + "details"
    rendered = format_output(
        {"channel": "sms", "body": body},
        {
            "classification": {"choice": "suspected_scam", "probabilities": {}},
            "industry": {"choice": "other", "probabilities": {}},
            "urgent_action": {"noul": 0.8},
            "sentiment": {"choice": "neutral", "probabilities": {}},
            "requested_action": {"choice": "pay_or_transfer", "probabilities": {}},
            "multiple_actions": {"noul": 0.2},
        },
        title="01 · test",
        expected={
            "classification": "suspected_scam",
            "requested_action": "pay_or_transfer",
            "multiple_actions": False,
        },
        elapsed_ms=12.3,
        compact=True,
    )

    assert body in rendered
    assert "Laya result (12.3 ms)" in rendered
    action_line = next(line for line in rendered.splitlines() if "Action:" in line)
    assert "Multi-action:" in action_line
    assert "Time:" not in rendered


class WordTokenizer:
    """Deterministic tokenizer for budget boundary tests without a checkpoint."""

    cls_token_id = 0
    sep_token_id = 1
    mask_token_id = 2
    mask_token = "[MASK]"

    def __init__(self):
        self.vocabulary = {}

    def __call__(self, text, *, add_special_tokens=False, truncation=False, max_length=None):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        ids = [self.vocabulary.setdefault(token, len(self.vocabulary) + 3) for token in tokens]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids}


def token_context(state, questions=None, tokenizer=None):
    return SimpleNamespace(
        agent=SimpleNamespace(tok=tokenizer or WordTokenizer(), cfg={"max_len": 512}),
        states=[state],
        questions=QUESTIONS if questions is None else questions,
        head_max_len=None,
        max_len=None,
    )


def test_complete_prompts_and_base_examples_fit():
    for example in load_examples():
        context = token_context(example["state"])
        _check_token_budget(context)
        assert context.head_max_len == HEAD_MAX_LEN
        assert context.max_len == 512


@pytest.mark.parametrize("field", ["body", "sender", "subject"])
def test_oversized_message_or_metadata_requires_review(field):
    state = {"channel": "email", "body": "Hello", field: "word " * 600}
    with pytest.raises(MessageTooLongError, match="No prediction was made"):
        _check_token_budget(token_context(state))


def test_budget_accepts_exact_boundary_and_rejects_one_extra_token():
    from laya.common import build_sequence, serialize_state

    definition = QUESTIONS["classification"]
    questions = {"classification": definition}
    tokenizer = WordTokenizer()
    question = {"t": definition["type"], "ins": definition["instructions"],
                "crit": definition["criteria"]}
    prefix, _ = build_sequence(tokenizer, {}, question, head_max_len=HEAD_MAX_LEN, state_ids=[])
    empty_size = len(tokenizer(serialize_state({"body": ""}))["input_ids"])
    words = 512 - len(prefix) - empty_size
    _check_token_budget(token_context({"body": "word " * words}, questions, tokenizer))
    with pytest.raises(MessageTooLongError):
        _check_token_budget(token_context({"body": "word " * (words + 1)}, questions, tokenizer))


def test_truncated_prompt_is_rejected_before_inference():
    question = dict(QUESTIONS["classification"], instructions="word " * 400)
    with pytest.raises(ValueError, match="prompt budget"):
        _check_token_budget(token_context({"body": "Hello"}, {"classification": question}))


def test_cli_reports_long_message_as_usage_error(monkeypatch, capsys):
    from system_one_models import cli

    def reject(state):
        raise MessageTooLongError("No prediction was made; the complete message needs review.")

    monkeypatch.setattr(cli, "create_backend", lambda *a, **kw: SimpleNamespace(predict=reject))
    monkeypatch.setattr("sys.argv", ["system-one-models", "--body", "long message"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "complete message needs review" in output.err
    assert not output.out


@pytest.mark.skipif(not os.environ.get("LAYA_TEST_TOKENIZER"), reason="local tokenizer not configured")
def test_real_tokenizer_preserves_prompts_and_rejects_sensitive_tail():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["LAYA_TEST_TOKENIZER"], local_files_only=True)
    for example in load_examples():
        _check_token_budget(token_context(example["state"], tokenizer=tokenizer))
    state = build_state(
        channel="email", sender="", subject="",
        body=("The newsletter has routine updates. " * 250)
             + "Reply with your password and bank account number now.",
    )
    with pytest.raises(MessageTooLongError):
        _check_token_budget(token_context(state, tokenizer=tokenizer))
