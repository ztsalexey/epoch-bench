"""Interactive Rich terminal quiz for human baseline evaluation."""

from __future__ import annotations

import random
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from epoch_bench.evaluate import compute_overall, compute_type_scores, score_question
from epoch_bench.runner import load_questions, parse_response
from epoch_bench.schema import BenchmarkResult, Question, QuestionType, Result


def present_question(console: Console, question: Question, index: int, total: int) -> Result:
    """Present a single question and collect the human's answer."""
    console.print()
    console.print(f"[bold]Question {index}/{total}[/bold]  [dim]{question.type.value} · {question.variant}[/dim]")

    panel_content = question.prompt
    if question.choices:
        panel_content += "\n"
        for letter, choice in zip("ABCD", question.choices):
            panel_content += f"\n  {letter}) {choice}"

    console.print(Panel(panel_content, title=question.type.value, expand=False))

    start = time.perf_counter()
    raw = Prompt.ask("[bold]Your answer[/bold]")
    latency_ms = (time.perf_counter() - start) * 1000.0

    parsed = parse_response(question.type, raw)
    score = score_question(question.type.value, parsed, question.answer)

    style = "green" if score >= 0.5 else "red"
    console.print(f"  Score: [{style}]{score:.2f}[/{style}]  Expected: {question.answer}")

    return Result(
        question_id=question.id,
        question_type=question.type,
        variant=question.variant,
        pair_id=question.pair_id,
        model_response=raw,
        parsed_answer=parsed,
        expected_answer=question.answer,
        score=score,
        latency_ms=latency_ms,
    )


def run_human_session(
    types: list[QuestionType] | None = None,
    max_questions: int | None = None,
    shuffle: bool = True,
) -> BenchmarkResult:
    """Run an interactive human baseline session."""
    console = Console()
    questions = load_questions(types)

    if shuffle:
        random.shuffle(questions)

    if max_questions is not None:
        questions = questions[:max_questions]

    if not questions:
        console.print("[red]No questions found.[/red]")
        raise SystemExit(1)

    console.print()
    console.print("[bold]EPOCH Human Baseline Session[/bold]")
    console.print(f"  {len(questions)} questions · type 'quit' to stop early")
    console.print()

    results: list[Result] = []
    start_time = time.perf_counter()

    for i, question in enumerate(questions, start=1):
        try:
            result = present_question(console, question, i, len(questions))
            results.append(result)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended early.[/yellow]")
            break

    duration = time.perf_counter() - start_time

    type_scores = compute_type_scores(results)
    of, ocf, og, oe = compute_overall(type_scores)

    return BenchmarkResult(
        model="human",
        provider="HumanBaseline",
        results=results,
        type_scores=type_scores,
        overall_factual=of,
        overall_counterfactual=ocf,
        overall_reasoning_gap=og,
        overall_epoch_score=oe,
        evaluation_duration_seconds=round(duration, 2),
        total_questions=len(questions),
        failed_questions=0,
    )
