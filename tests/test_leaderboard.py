"""Tests for leaderboard generation."""

from __future__ import annotations

import pytest

from epoch_bench.leaderboard import (
    LeaderboardEntry,
    build_leaderboard,
    leaderboard_to_latex,
    leaderboard_to_markdown,
    print_leaderboard,
)
from epoch_bench.schema import BenchmarkResult


class TestBuildLeaderboard:
    def test_sorts_by_epoch_score_descending(
        self, sample_result: BenchmarkResult, sample_result_b: BenchmarkResult
    ) -> None:
        entries = build_leaderboard([sample_result, sample_result_b])
        assert len(entries) == 2
        assert entries[0].epoch_score >= entries[1].epoch_score
        assert entries[0].rank == 1
        assert entries[1].rank == 2

    def test_single_result(self, sample_result: BenchmarkResult) -> None:
        entries = build_leaderboard([sample_result])
        assert len(entries) == 1
        assert entries[0].rank == 1
        assert entries[0].model == "model-a"

    def test_empty_results(self) -> None:
        entries = build_leaderboard([])
        assert entries == []

    def test_entry_fields(self, sample_result: BenchmarkResult) -> None:
        entries = build_leaderboard([sample_result])
        e = entries[0]
        assert isinstance(e, LeaderboardEntry)
        assert e.model == sample_result.model
        assert e.provider == sample_result.provider
        assert e.epoch_score == sample_result.overall_epoch_score
        assert e.factual == sample_result.overall_factual
        assert e.counterfactual == sample_result.overall_counterfactual
        assert e.reasoning_gap == sample_result.overall_reasoning_gap


class TestPrintLeaderboard:
    def test_no_errors(
        self, sample_result: BenchmarkResult, sample_result_b: BenchmarkResult
    ) -> None:
        entries = build_leaderboard([sample_result, sample_result_b])
        print_leaderboard(entries)  # should not raise


class TestLeaderboardToMarkdown:
    def test_contains_header(self, sample_result: BenchmarkResult) -> None:
        entries = build_leaderboard([sample_result])
        md = leaderboard_to_markdown(entries)
        assert "# EPOCH Benchmark Leaderboard" in md
        assert "model-a" in md
        assert "|" in md

    def test_multiple_models(
        self, sample_result: BenchmarkResult, sample_result_b: BenchmarkResult
    ) -> None:
        entries = build_leaderboard([sample_result, sample_result_b])
        md = leaderboard_to_markdown(entries)
        assert "model-a" in md
        assert "model-b" in md


class TestLeaderboardToLatex:
    def test_contains_tabular(self, sample_result: BenchmarkResult) -> None:
        entries = build_leaderboard([sample_result])
        latex = leaderboard_to_latex(entries)
        assert r"\begin{tabular}" in latex
        assert r"\end{table}" in latex
        assert "model-a" in latex
