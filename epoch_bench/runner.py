"""Orchestration: load data, prompt model, parse response, score, report."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from epoch_bench.evaluate import compute_overall, compute_type_scores, score_question
from epoch_bench.models.base import ModelProvider
from epoch_bench.prompts import format_prompt
from epoch_bench.schema import BenchmarkResult, Question, QuestionType, Result

DATA_DIR = Path(__file__).parent / "data"

QUESTION_FILES = {
    QuestionType.CHAIN: "chain.jsonl",
    QuestionType.GATE: "gate.jsonl",
    QuestionType.RIPPLE: "ripple.jsonl",
    QuestionType.BRIDGE: "bridge.jsonl",
}


def load_questions(types: list[QuestionType] | None = None) -> list[Question]:
    """Load questions from JSONL files."""
    if types is None:
        types = list(QuestionType)

    questions: list[Question] = []
    for qt in types:
        path = DATA_DIR / QUESTION_FILES[qt]
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                questions.append(Question.model_validate_json(line))
    return questions


def _parse_list(raw: str) -> list[str]:
    """Parse a list from various formats: numbered, dashed, bulleted, comma-separated, newline."""
    raw = raw.strip()
    lines = raw.splitlines()

    # If multi-line, parse each line.
    if len(lines) > 1:
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip numbered prefixes: "1.", "1)", "1:"
            line = re.sub(r"^\d+[.):\s]+", "", line)
            # Strip bullet prefixes: "- ", "* ", "• "
            line = re.sub(r"^[-*•]\s+", "", line)
            line = line.strip()
            if line:
                items.append(line)
        return items

    # Single line: try comma-separated.
    if "," in raw:
        return [item.strip() for item in raw.split(",") if item.strip()]

    # Fallback: return as single-item list.
    return [raw] if raw else []


def parse_response(question_type: QuestionType, raw: str) -> str | list[str]:
    """Parse model response into structured answer."""
    raw = raw.strip()
    if question_type == QuestionType.CHAIN:
        return _parse_list(raw)
    if question_type == QuestionType.GATE:
        first_word = raw.split()[0] if raw.split() else raw
        # Strip punctuation
        return first_word.strip(".,!;:")
    if question_type == QuestionType.RIPPLE:
        return _parse_list(raw)
    if question_type == QuestionType.BRIDGE:
        # Extract first letter A-D
        for char in raw:
            if char.upper() in "ABCD":
                return char.upper()
        return raw[:1]
    raise ValueError(f"Unknown question type: {question_type}")


async def evaluate_question(
    provider: ModelProvider,
    question: Question,
    verbose: bool = False,
) -> Result:
    """Evaluate a single question against the model with error handling and timing."""
    system_prompt, user_prompt = format_prompt(question)

    latency_ms = None
    error = None
    raw_response = ""
    parsed = "" if question.type in (QuestionType.GATE, QuestionType.BRIDGE) else []
    score = 0.0

    try:
        start = time.perf_counter()
        raw_response = await provider.generate(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0

        parsed = parse_response(question.type, raw_response)
        score = score_question(question.type.value, parsed, question.answer)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if verbose:
            console = Console(stderr=True)
            console.print(f"[red]Error on {question.id}: {error}[/red]")

    return Result(
        question_id=question.id,
        question_type=question.type,
        variant=question.variant,
        pair_id=question.pair_id,
        model_response=raw_response,
        parsed_answer=parsed,
        expected_answer=question.answer,
        score=score,
        latency_ms=latency_ms,
        error=error,
    )


async def run_benchmark(
    provider: ModelProvider,
    types: list[QuestionType] | None = None,
    concurrency: int = 5,
    verbose: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BenchmarkResult:
    """Run the full benchmark."""
    console = Console()
    questions = load_questions(types)

    if not questions:
        console.print("[red]No questions found. Check data directory.[/red]")
        raise SystemExit(1)

    console.print(f"Loaded {len(questions)} questions")

    # Configure provider params if specified.
    if temperature is not None:
        provider.temperature = temperature
    if max_tokens is not None:
        provider.max_tokens = max_tokens

    results: list[Result] = []
    semaphore = asyncio.Semaphore(concurrency)
    failed_count = 0

    async def bounded_eval(q: Question) -> Result:
        async with semaphore:
            return await evaluate_question(provider, q, verbose=verbose)

    start_time = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(questions))
        tasks = [bounded_eval(q) for q in questions]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if result.error:
                failed_count += 1
            progress.advance(task)

    duration = time.perf_counter() - start_time

    type_scores = compute_type_scores(results)
    overall_f, overall_cf, overall_gap, overall_epoch = compute_overall(type_scores)

    return BenchmarkResult(
        model=provider.name,
        provider=type(provider).__name__,
        results=results,
        type_scores=type_scores,
        overall_factual=overall_f,
        overall_counterfactual=overall_cf,
        overall_reasoning_gap=overall_gap,
        overall_epoch_score=overall_epoch,
        evaluation_duration_seconds=round(duration, 2),
        total_questions=len(questions),
        failed_questions=failed_count,
    )
