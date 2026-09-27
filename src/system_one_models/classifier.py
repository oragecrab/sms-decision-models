"""Message input shaping, token-budget checks, and Laya inference."""

from __future__ import annotations

from typing import Any

from .questions import QUESTIONS
from .runtime import create_router


def build_state(*, channel: str, sender: str, subject: str, body: str) -> dict[str, str]:
    """Build the state passed to the model, omitting empty optional fields."""
    state = {"channel": channel.strip()}
    for key, value in (("sender", sender), ("subject", subject), ("body", body)):
        if value.strip():
            state[key] = value.strip()
    if not state.get("body"):
        raise ValueError("message body must not be empty")
    return state


# Reserve enough space for all option descriptions and the complete instructions.
# The remaining message budget is checked using the selected checkpoint's tokenizer.
HEAD_MAX_LEN = 320


class MessageTooLongError(ValueError):
    """The full message cannot be evaluated without losing text."""


def _check_token_budget(context: Any) -> None:
    """Fail before inference if Laya would truncate a question or the input."""
    from laya.common import build_sequence, render_options, serialize_state

    agent = context.agent
    tokenizer = agent.tok
    max_len = agent.cfg.get("max_len", 512)
    context.head_max_len = HEAD_MAX_LEN
    context.max_len = max_len

    def encode(text: str) -> list[int]:
        return tokenizer(
            text.replace(tokenizer.mask_token, " "), add_special_tokens=False
        )["input_ids"]

    state_tokens = encode(serialize_state(context.states[0]))
    for question_id, definition in context.questions.items():
        question = {
            "t": definition["type"],
            "ins": definition["instructions"],
            "crit": definition["criteria"],
        }
        if "labels" in definition:
            question["labels"] = definition["labels"]
        # Compare the actual encoding with the complete prompt, including each option.
        # This also catches Laya's per-option cap and future prompt/schema changes.
        prefix = [tokenizer.cls_token_id]
        prefix.extend(encode(f"{question['t']} question: {question['ins']}"))
        prefix.append(tokenizer.sep_token_id)
        for option in render_options(question):
            prefix.append(tokenizer.mask_token_id)
            prefix.extend(encode(" " + option))
        prefix.append(tokenizer.sep_token_id)
        sequence, _ = build_sequence(
            tokenizer, context.states[0], question,
            max_len=max_len, head_max_len=HEAD_MAX_LEN, state_ids=[],
        )
        if sequence != prefix + [tokenizer.sep_token_id]:
            raise ValueError(
                f"Question {question_id!r} exceeds the checkpoint's prompt budget; "
                "shorten its instructions or options before inference."
            )
        available = max_len - len(prefix) - 1
        if len(state_tokens) > available:
            raise MessageTooLongError(
                f"Message needs {len(state_tokens)} tokens including metadata, but "
                f"question {question_id!r} allows {available}. No prediction was made; "
                "the complete message needs review."
            )


def classify(state: dict[str, Any], router: Any | None = None, *, questions: dict | None = None, return_details: bool = False) -> dict[str, Any]:
    """Answer all questions, rejecting input that cannot be evaluated in full.

    Pass a Router instance to reuse a loaded model across multiple messages.
    """
    if router is None:
        router = create_router()
    result = router.predict(
        state, QUESTIONS if questions is None else questions, on_predict_start=_check_token_budget, hooks_raise=True
    )
    return result if return_details else result["answers"]
