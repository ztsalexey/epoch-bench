"""Tests for contamination analysis module."""

from __future__ import annotations

import pytest

from epoch_bench.contamination import (
    compare_contamination,
    compute_contamination_profile,
    compute_pair_contamination,
    contamination_summary,
)
from epoch_bench.schema import BenchmarkResult, Question, QuestionType

from tests.conftest import _make_results


@pytest.fixture
def contamination_questions() -> list[Question]:
    return [
        Question(id="c_f_01", type=QuestionType.CHAIN, variant="factual", pair_id="c_01",
                 prompt="Order A, B", answer=["A", "B"], domains=["computing", "web"]),
        Question(id="c_cf_01", type=QuestionType.CHAIN, variant="counterfactual", pair_id="c_01",
                 prompt="If X, order A, B", answer=["B", "A"], domains=["computing", "web"]),
        Question(id="g_f_01", type=QuestionType.GATE, variant="factual", pair_id="g_01",
                 prompt="Could X?", answer="No", domains=["computing"]),
        Question(id="g_cf_01", type=QuestionType.GATE, variant="counterfactual", pair_id="g_01",
                 prompt="If Z, could X?", answer="Yes", domains=["computing"]),
        Question(id="r_f_01", type=QuestionType.RIPPLE, variant="factual", pair_id="r_01",
                 prompt="If X removed?", answer=["A", "B"], domains=["hardware"]),
        Question(id="r_cf_01", type=QuestionType.RIPPLE, variant="counterfactual", pair_id="r_01",
                 prompt="If Y, X removed?", answer=["A"], domains=["hardware"]),
    ]


@pytest.fixture
def high_gap_result() -> BenchmarkResult:
    """Model with high factual scores and low counterfactual (contaminated)."""
    return _make_results("contaminated-model", "TestProvider", [
        ("c_f_01", "CHAIN", "factual", "c_01", 0.9),
        ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.3),
        ("g_f_01", "GATE", "factual", "g_01", 1.0),
        ("g_cf_01", "GATE", "counterfactual", "g_01", 0.0),
        ("r_f_01", "RIPPLE", "factual", "r_01", 0.8),
        ("r_cf_01", "RIPPLE", "counterfactual", "r_01", 0.2),
    ])


@pytest.fixture
def low_gap_result() -> BenchmarkResult:
    """Model with similar factual/counterfactual scores (clean)."""
    return _make_results("clean-model", "TestProvider", [
        ("c_f_01", "CHAIN", "factual", "c_01", 0.7),
        ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.65),
        ("g_f_01", "GATE", "factual", "g_01", 1.0),
        ("g_cf_01", "GATE", "counterfactual", "g_01", 1.0),
        ("r_f_01", "RIPPLE", "factual", "r_01", 0.5),
        ("r_cf_01", "RIPPLE", "counterfactual", "r_01", 0.45),
    ])


class TestPairContamination:
    def test_known_scores(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        pairs = compute_pair_contamination(high_gap_result, contamination_questions)
        assert len(pairs) == 3

        # c_01: 0.9 - 0.3 = 0.6
        c01 = next(p for p in pairs if p.pair_id == "c_01")
        assert c01.factual_score == pytest.approx(0.9)
        assert c01.counterfactual_score == pytest.approx(0.3)
        assert c01.contamination_signal == pytest.approx(0.6)

    def test_domains_attached(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        pairs = compute_pair_contamination(high_gap_result, contamination_questions)
        c01 = next(p for p in pairs if p.pair_id == "c_01")
        assert "computing" in c01.domains
        assert "web" in c01.domains

    def test_normalized_signal(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        pairs = compute_pair_contamination(high_gap_result, contamination_questions)
        c01 = next(p for p in pairs if p.pair_id == "c_01")
        # normalized = 0.6 / 0.9
        assert c01.normalized_signal == pytest.approx(0.6 / 0.9, abs=0.01)


class TestContaminationProfile:
    def test_overall_index(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        profile = compute_contamination_profile(high_gap_result, contamination_questions)
        # Mean of (0.6, 1.0, 0.6) = 0.733...
        assert profile.overall_index == pytest.approx(0.7333, abs=0.01)

    def test_per_type_breakdown(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        profile = compute_contamination_profile(high_gap_result, contamination_questions)
        assert "CHAIN" in profile.per_type
        assert "GATE" in profile.per_type
        assert profile.per_type["GATE"] == pytest.approx(1.0)

    def test_per_domain_breakdown(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        profile = compute_contamination_profile(high_gap_result, contamination_questions)
        assert "computing" in profile.per_domain
        # computing has c_01 (0.6) and g_01 (1.0) → mean 0.8
        assert profile.per_domain["computing"] == pytest.approx(0.8)

    def test_threshold_classification(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        profile = compute_contamination_profile(high_gap_result, contamination_questions, threshold=0.3)
        # All 3 pairs have signal > 0.3
        assert profile.n_contaminated_pairs == 3
        assert profile.n_clean_pairs == 0

    def test_clean_model_threshold(self, low_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        profile = compute_contamination_profile(low_gap_result, contamination_questions, threshold=0.3)
        # All signals < 0.3
        assert profile.n_contaminated_pairs == 0
        assert profile.n_clean_pairs == 3

    def test_distribution(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        profile = compute_contamination_profile(high_gap_result, contamination_questions)
        assert len(profile.distribution) == 3

    def test_significance_attached(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        profile = compute_contamination_profile(high_gap_result, contamination_questions)
        assert profile.significance.n_pairs == 3


class TestContaminationComparison:
    def test_multi_model(
        self,
        high_gap_result: BenchmarkResult,
        low_gap_result: BenchmarkResult,
        contamination_questions: list[Question],
    ) -> None:
        comparison = compare_contamination([high_gap_result, low_gap_result], contamination_questions)
        assert len(comparison.models) == 2
        assert len(comparison.profiles) == 2

    def test_most_contaminated_domains(
        self,
        high_gap_result: BenchmarkResult,
        low_gap_result: BenchmarkResult,
        contamination_questions: list[Question],
    ) -> None:
        comparison = compare_contamination([high_gap_result, low_gap_result], contamination_questions)
        assert len(comparison.most_contaminated_domains) > 0
        # Domains are sorted by contamination (descending)
        for i in range(len(comparison.most_contaminated_domains) - 1):
            assert comparison.most_contaminated_domains[i][1] >= comparison.most_contaminated_domains[i + 1][1]


class TestContaminationSummary:
    def test_summary_text(self, high_gap_result: BenchmarkResult, contamination_questions: list[Question]) -> None:
        profile = compute_contamination_profile(high_gap_result, contamination_questions)
        text = contamination_summary(profile)
        assert "contaminated-model" in text
        assert "contamination index" in text.lower()
        assert "per-type" in text.lower()
