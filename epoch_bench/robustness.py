"""Prompt robustness testing: measure score sensitivity to paraphrased prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean, stdev

from epoch_bench.schema import Question, QuestionType


_CHAIN_TEMPLATES = [
    "Order these technologies by dependency (earliest dependency first): {items}",
    "Arrange the following technologies from earliest prerequisite to latest dependent: {items}",
    "List these technologies in order of their historical dependency chain (foundational first): {items}",
]

_GATE_TEMPLATES = [
    "Could {target} have existed without {prereq}?",
    "Was {prereq} a necessary prerequisite for the development of {target}?",
    "Would {target} have been possible to create if {prereq} had never existed?",
]

_RIPPLE_TEMPLATES = [
    "If {removed} had never been created, which of these technologies would not exist in their known form: {options}?",
    "Which of these technologies depend on {removed} and would not exist without it: {options}?",
    "Assuming {removed} was never invented, which of the following would be affected: {options}?",
]

_BRIDGE_TEMPLATES = [
    "What technology bridges the gap between {pred} and {succ}?",
    "Which technology was the key enabler that connected {pred} to {succ}?",
    "What intermediate technology made the transition from {pred} to {succ} possible?",
]


@dataclass
class ParaphrasedQuestion:
    """A question with an alternative prompt phrasing."""

    original: Question
    paraphrased_prompt: str
    template_index: int


@dataclass
class RobustnessResult:
    """Score variance across prompt paraphrases for one question."""

    question_id: str
    question_type: str
    scores: list[float]
    mean_score: float
    score_std: float
    is_robust: bool  # std < 0.1


@dataclass
class RobustnessReport:
    """Aggregate robustness metrics across all questions."""

    n_questions: int
    n_robust: int  # questions where all paraphrases give same score
    robustness_rate: float  # n_robust / n_questions
    mean_std: float  # average score std across questions
    per_type: dict[str, float]  # robustness rate per question type
    fragile_questions: list[str]  # question IDs with highest variance


def _extract_chain_items(prompt: str) -> str | None:
    """Extract the item list from a CHAIN prompt."""
    m = re.search(r"(?:first|chain):\s*(.+?)$", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip("?")
    # Fallback: everything after the last colon
    if ":" in prompt:
        return prompt.rsplit(":", 1)[1].strip()
    return None


def _extract_gate_parts(prompt: str) -> tuple[str, str] | None:
    """Extract target and prereq from a GATE prompt."""
    patterns = [
        r"[Cc]ould\s+(.+?)\s+(?:have\s+)?(?:existed|been\s+\w+)\s+without\s+(.+?)\?",
        r"[Ww]as\s+(?:it\s+)?(?:possible\s+)?.*?(?:without|before)\s+(.+?)\?",
    ]
    for pat in patterns:
        m = re.search(pat, prompt)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


def _extract_ripple_parts(prompt: str) -> tuple[str, str] | None:
    """Extract removed tech and options from a RIPPLE prompt."""
    m = re.search(r"[Ii]f\s+(.+?)\s+had never been (?:created|invented|developed)", prompt)
    removed = m.group(1).strip() if m else None
    m2 = re.search(r"(?:form|affected):\s*(.+?)\?", prompt)
    options = m2.group(1).strip() if m2 else None
    if removed and options:
        return removed, options
    return None


def _extract_bridge_parts(prompt: str) -> tuple[str, str] | None:
    """Extract predecessor and successor from a BRIDGE prompt."""
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)\?", prompt)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def generate_paraphrases(question: Question) -> list[ParaphrasedQuestion]:
    """Generate paraphrased versions of a question prompt."""
    paraphrases = []

    if question.type == QuestionType.CHAIN:
        items = _extract_chain_items(question.prompt)
        if items:
            for i, tmpl in enumerate(_CHAIN_TEMPLATES):
                new_prompt = tmpl.format(items=items)
                if new_prompt != question.prompt:
                    paraphrases.append(ParaphrasedQuestion(
                        original=question,
                        paraphrased_prompt=new_prompt,
                        template_index=i,
                    ))

    elif question.type == QuestionType.GATE:
        parts = _extract_gate_parts(question.prompt)
        if parts:
            target, prereq = parts
            for i, tmpl in enumerate(_GATE_TEMPLATES):
                new_prompt = tmpl.format(target=target, prereq=prereq)
                if new_prompt != question.prompt:
                    paraphrases.append(ParaphrasedQuestion(
                        original=question,
                        paraphrased_prompt=new_prompt,
                        template_index=i,
                    ))

    elif question.type == QuestionType.RIPPLE:
        parts = _extract_ripple_parts(question.prompt)
        if parts:
            removed, options = parts
            for i, tmpl in enumerate(_RIPPLE_TEMPLATES):
                new_prompt = tmpl.format(removed=removed, options=options)
                if new_prompt != question.prompt:
                    paraphrases.append(ParaphrasedQuestion(
                        original=question,
                        paraphrased_prompt=new_prompt,
                        template_index=i,
                    ))

    elif question.type == QuestionType.BRIDGE:
        parts = _extract_bridge_parts(question.prompt)
        if parts:
            pred, succ = parts
            for i, tmpl in enumerate(_BRIDGE_TEMPLATES):
                new_prompt = tmpl.format(pred=pred, succ=succ)
                if new_prompt != question.prompt:
                    paraphrases.append(ParaphrasedQuestion(
                        original=question,
                        paraphrased_prompt=new_prompt,
                        template_index=i,
                    ))

    return paraphrases


def compute_robustness_report(
    results_per_paraphrase: dict[str, list[float]],
    questions: list[Question],
) -> RobustnessReport:
    """Compute robustness metrics from per-question score lists.

    Args:
        results_per_paraphrase: {question_id: [score_original, score_para1, score_para2, ...]}
        questions: list of Question objects for type lookup
    """
    q_type_map = {q.id: q.type.value for q in questions}

    robustness_results = []
    type_robust: dict[str, list[bool]] = {}

    for qid, scores in results_per_paraphrase.items():
        qt = q_type_map.get(qid, "")
        s = stdev(scores) if len(scores) >= 2 else 0.0
        is_robust = s < 0.1
        robustness_results.append(RobustnessResult(
            question_id=qid,
            question_type=qt,
            scores=scores,
            mean_score=mean(scores),
            score_std=s,
            is_robust=is_robust,
        ))
        type_robust.setdefault(qt, []).append(is_robust)

    n = len(robustness_results)
    n_robust = sum(1 for r in robustness_results if r.is_robust)

    fragile = sorted(
        [r for r in robustness_results if not r.is_robust],
        key=lambda r: r.score_std,
        reverse=True,
    )

    per_type = {
        qt: sum(v) / len(v) if v else 1.0
        for qt, v in type_robust.items()
    }

    return RobustnessReport(
        n_questions=n,
        n_robust=n_robust,
        robustness_rate=n_robust / n if n > 0 else 1.0,
        mean_std=mean(r.score_std for r in robustness_results) if robustness_results else 0.0,
        per_type=per_type,
        fragile_questions=[r.question_id for r in fragile[:20]],
    )
