"""Structured prompt templates for each EPOCH benchmark question type."""

from __future__ import annotations

from epoch_bench.schema import Question, QuestionType

SYSTEM_PROMPT = (
    "You are being evaluated on causal reasoning about technology dependencies. "
    "Answer precisely in the requested format. Do not include explanations, "
    "commentary, or any text beyond what is explicitly requested."
)


def format_chain_prompt(question_prompt: str) -> str:
    """Wrap a CHAIN question: model must return items in dependency order."""
    return (
        f"{question_prompt}\n\n"
        "List the items in dependency order (earliest dependency first), "
        "one per line. Output ONLY the ordered list, nothing else."
    )


def format_gate_prompt(question_prompt: str) -> str:
    """Wrap a GATE question: model must answer Yes or No."""
    return (
        f"{question_prompt}\n\n"
        'Answer exactly "Yes" or "No". Nothing else.'
    )


def format_ripple_prompt(question_prompt: str) -> str:
    """Wrap a RIPPLE question: model must list all affected technologies."""
    return (
        f"{question_prompt}\n\n"
        "List all affected technologies or systems, one per line. "
        "Output ONLY the list, nothing else."
    )


def format_bridge_prompt(question_prompt: str, choices: list[str]) -> str:
    """Wrap a BRIDGE question with lettered choices: model must answer with a letter."""
    labels = "ABCD"
    options = "\n".join(
        f"{labels[i]}. {choice}" for i, choice in enumerate(choices)
    )
    return (
        f"{question_prompt}\n\n"
        f"{options}\n\n"
        "Answer with ONLY the letter (A, B, C, or D)."
    )


def format_prompt(question: Question) -> tuple[str, str]:
    """Dispatch to the correct formatter based on question type.

    Returns:
        (system_prompt, user_prompt) tuple ready for the model.
    """
    if question.type == QuestionType.CHAIN:
        user_prompt = format_chain_prompt(question.prompt)
    elif question.type == QuestionType.GATE:
        user_prompt = format_gate_prompt(question.prompt)
    elif question.type == QuestionType.RIPPLE:
        user_prompt = format_ripple_prompt(question.prompt)
    elif question.type == QuestionType.BRIDGE:
        if question.choices is None:
            raise ValueError(
                f"BRIDGE question {question.id} is missing choices"
            )
        user_prompt = format_bridge_prompt(question.prompt, question.choices)
    else:
        raise ValueError(f"Unknown question type: {question.type}")

    return SYSTEM_PROMPT, user_prompt
