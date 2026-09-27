"""Human-readable CLI output and display labels."""

from __future__ import annotations

import json
import os
import sys
from typing import Any


QUESTION_LABELS = {
    "classification": "Scam category",
    "industry": "Industry or service",
    "urgent_action": "Urgency",
    "sentiment": "Sentiment",
    "requested_action": "Requested action",
    "multiple_actions": "Multiple actions",
    "message_hook": "Message hook",
    "sensitive_data_requested": "Sensitive data requested",
}

COMPACT_LABELS = {
    "classification": "Scam",
    "industry": "Industry",
    "urgent_action": "Urgent",
    "sentiment": "Tone",
    "requested_action": "Action",
    "multiple_actions": "Multi-action",
    "message_hook": "Hook",
    "sensitive_data_requested": "Sensitive data",
}


def format_output(
    state: dict[str, Any],
    answer: dict[str, Any],
    *,
    title: str = "Input message",
    expected: dict[str, Any] | None = None,
    elapsed_ms: float | None = None,
    details: bool = False,
    compact: bool = False,
    model_name: str = "Laya",
) -> str:
    """Render the message first, followed by Laya's decision data."""
    use_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    def paint(value: str, code: str) -> str:
        if not use_color:
            return value
        return f"\033[{code}m{value}\033[0m"

    colors = {
        "legitimate": "32",
        "marketing": "33",
        "suspected_scam": "31",
        "happy": "32",
        "neutral": "36",
        "sad": "35",
        "pay_or_transfer": "31",
        "share_sensitive_info_or_documents": "31",
        "install_or_grant_access": "31",
        "approve_login_or_transaction": "31",
        "sign_in_or_verify": "33",
        "click_or_scan": "33",
    }

    def display_value(value: Any) -> str:
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value).replace("_", " ").title()

    def prediction(question_id: str, question_answer: dict[str, Any]) -> Any:
        if "choice" in question_answer:
            return question_answer["choice"]
        if "noul" in question_answer:
            return question_answer["noul"] >= 0.5
        return None

    lines = [paint(title, "1;36")]
    if compact:
        source = state.get("sender", "")
        subject = state.get("subject", "")
        body = state.get("body", "")
        prefix = f"{state.get('channel', 'unknown').upper()}"
        if source:
            prefix += f" · {source}"
        if subject:
            prefix += f" · {subject}"
        snippet = f"{prefix} · {body}".replace("\n", " ").strip()
        lines.append(f"  {paint('Input', '36')}: {snippet}")
    else:
        lines.append(f"  Channel: {state.get('channel', 'unknown')}")
        if state.get("sender"):
            lines.append(f"  From: {state['sender']}")
        if state.get("subject"):
            lines.append(f"  Subject: {state['subject']}")
        lines.append(f"  Message: {state.get('body', '')}")
    result_heading = f"{model_name} result"
    if compact and elapsed_ms is not None:
        result_heading += f" ({elapsed_ms:.1f} ms)"
    lines.extend(("", paint(result_heading, "1;36")))

    result_lines: list[str] = []
    detail_lines: list[list[str]] = []
    for question_id, question_answer in answer.items():
        label = (
            COMPACT_LABELS.get(question_id, question_id)
            if compact
            else QUESTION_LABELS.get(question_id, question_id.replace("_", " ").title())
        )
        uncalibrated = question_answer.get("score_semantics") == "uncalibrated_relative_weights"
        choice = question_answer.get("choice")
        probabilities = question_answer.get("probabilities", {})
        predicted = prediction(question_id, question_answer)
        expected_value = expected.get(question_id) if expected is not None else None
        has_expected = expected is not None and question_id in expected
        matched = has_expected and predicted == expected_value
        if choice is not None:
            display_choice = choice.replace("_", " ").title()
            colored_choice = paint(display_choice, colors.get(choice, "1"))
            confidence = probabilities.get(choice)
            suffix = (f" · weight {confidence:.1%}" if uncalibrated else f" · {confidence:.1%}") if confidence is not None else ""
            expected_suffix = (
                f" · exp {display_value(expected_value)} "
                f"{paint('✓' if matched else '✗', '32' if matched else '31')}"
                if has_expected
                else ""
            )
            result_lines.append(
                f"{paint(label, '36')}: {colored_choice}{suffix}{expected_suffix}"
            )
            question_details = []
            if details and probabilities:
                distribution = sorted(
                    probabilities.items(), key=lambda item: item[1], reverse=True
                )
                for category, probability in distribution:
                    colored_category = paint(
                        category.replace("_", " ").title(), colors.get(category, "0")
                    )
                    question_details.append(f"    {colored_category}: {probability:.1%}")
            detail_lines.append(question_details)
        elif "noul" in question_answer:
            yes_probability = question_answer["noul"]
            likely_answer = "yes" if yes_probability >= 0.5 else "no"
            color = "33" if yes_probability >= 0.5 else "32"
            confidence = (f" · relative yes weight {yes_probability:.0%}" if uncalibrated else f" · P(yes) {yes_probability:.0%}") if not compact else ""
            expected_mark = paint("✓" if matched else "✗", "32" if matched else "31")
            expected_suffix = (
                f" · exp {display_value(expected_value)} {expected_mark}"
                if has_expected
                else ""
            )
            result_lines.append(
                f"{paint(label, '36')}: {paint(likely_answer.title(), color)}"
                f"{confidence}{expected_suffix}"
            )
            detail_lines.append([])
        else:
            result_lines.append(
                f"{paint(label, '36')}: {json.dumps(question_answer, ensure_ascii=False)}"
            )
            detail_lines.append([])

    if compact and not details:
        for index in range(0, len(result_lines), 2):
            row = "  │  ".join(result_lines[index : index + 2])
            lines.append(f"  {row}")
    else:
        for result, question_details in zip(result_lines, detail_lines):
            lines.append(f"  {result}")
            lines.extend(question_details)

    if elapsed_ms is not None and not compact:
        lines.append(f"  Inference time: {elapsed_ms:.1f} ms")
    return "\n".join(lines)
