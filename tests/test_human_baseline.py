"""Tests for human baseline module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from epoch_bench.schema import BenchmarkResult, Question, QuestionType


class TestPresentQuestion:
    def test_returns_result(self, sample_questions: list[Question]) -> None:
        from epoch_bench.human_baseline import present_question
        from rich.console import Console

        console = Console()
        question = sample_questions[2]  # gate_f_01, answer="No"

        with patch("epoch_bench.human_baseline.Prompt.ask", return_value="No"):
            result = present_question(console, question, 1, 1)

        assert result.question_id == "gate_f_01"
        assert result.score == 1.0
        assert result.latency_ms is not None

    def test_wrong_answer(self, sample_questions: list[Question]) -> None:
        from epoch_bench.human_baseline import present_question
        from rich.console import Console

        console = Console()
        question = sample_questions[2]  # gate_f_01, answer="No"

        with patch("epoch_bench.human_baseline.Prompt.ask", return_value="Yes"):
            result = present_question(console, question, 1, 1)

        assert result.score == 0.0


class TestRunHumanSession:
    def test_returns_benchmark_result(self, sample_questions: list[Question]) -> None:
        from epoch_bench.human_baseline import run_human_session

        answers = iter(["A\nB\nC", "B\nA\nC", "No", "Yes"])
        with (
            patch("epoch_bench.human_baseline.load_questions", return_value=sample_questions),
            patch("epoch_bench.human_baseline.Prompt.ask", side_effect=answers),
        ):
            result = run_human_session(shuffle=False)

        assert isinstance(result, BenchmarkResult)
        assert result.model == "human"
        assert result.provider == "HumanBaseline"
        assert len(result.results) == 4
        assert result.evaluation_duration_seconds is not None

    def test_max_questions(self, sample_questions: list[Question]) -> None:
        from epoch_bench.human_baseline import run_human_session

        answers = iter(["A\nB\nC", "B\nA\nC"])
        with (
            patch("epoch_bench.human_baseline.load_questions", return_value=sample_questions),
            patch("epoch_bench.human_baseline.Prompt.ask", side_effect=answers),
        ):
            result = run_human_session(max_questions=2, shuffle=False)

        assert len(result.results) == 2
