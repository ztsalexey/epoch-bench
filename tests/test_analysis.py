"""Tests for epoch_bench.analysis statistical functions."""

from __future__ import annotations

import pytest

from epoch_bench.analysis import (
    GapSignificance,
    StratifiedScore,
    correlation_matrix,
    difficulty_stratified,
    domain_stratified,
    gap_significance,
    item_discrimination,
    weight_sensitivity,
)
from epoch_bench.schema import BenchmarkResult


class TestGapSignificance:
    def test_returns_dataclass(self, sample_result):
        gs = gap_significance(sample_result)
        assert isinstance(gs, GapSignificance)
        assert gs.n_pairs == 4

    def test_cohens_d_sign(self, sample_result):
        gs = gap_significance(sample_result)
        # factual scores are generally higher, so d should be positive
        assert gs.cohens_d > 0

    def test_single_pair_returns_defaults(self):
        """With fewer than 2 pairs, should return defaults."""
        from tests.conftest import _make_results

        result = _make_results(
            "test",
            "test",
            [
                ("q1", "GATE", "factual", "p1", 1.0),
                ("q2", "GATE", "counterfactual", "p1", 0.0),
            ],
        )
        gs = gap_significance(result)
        assert gs.n_pairs == 1
        assert gs.t_pvalue == 1.0


class TestCorrelationMatrix:
    def test_two_models(self, sample_result, sample_result_b):
        entries = correlation_matrix([sample_result, sample_result_b])
        assert len(entries) == 1
        entry = entries[0]
        assert entry.model_a == "model-a"
        assert entry.model_b == "model-b"
        assert -1.0 <= entry.pearson_r <= 1.0
        assert -1.0 <= entry.spearman_rho <= 1.0

    def test_single_model_returns_empty(self, sample_result):
        entries = correlation_matrix([sample_result])
        assert entries == []


class TestDifficultyStratified:
    def test_returns_strata(self, sample_result, sample_questions):
        strata = difficulty_stratified(sample_result, sample_questions)
        assert all(isinstance(s, StratifiedScore) for s in strata)

    def test_no_metadata_returns_empty(self, sample_result):
        from epoch_bench.schema import Question, QuestionType

        qs = [
            Question(
                id="chain_f_01",
                type=QuestionType.CHAIN,
                variant="factual",
                pair_id="chain_01",
                prompt="test",
                answer=["A"],
            )
        ]
        strata = difficulty_stratified(sample_result, qs)
        assert strata == []


class TestDomainStratified:
    def test_returns_strata(self, sample_result, sample_questions):
        strata = domain_stratified(sample_result, sample_questions)
        assert len(strata) > 0
        domain_names = [s.stratum for s in strata]
        assert "computing" in domain_names


class TestWeightSensitivity:
    def test_basic(self, sample_result, sample_result_b):
        ws = weight_sensitivity([sample_result, sample_result_b])
        assert len(ws.weights) == 11
        assert len(ws.rankings) == 11
        assert 0.0 <= ws.kendalls_w <= 1.0

    def test_single_model(self, sample_result):
        ws = weight_sensitivity([sample_result])
        assert ws.kendalls_w == 1.0

    def test_custom_steps(self, sample_result, sample_result_b):
        ws = weight_sensitivity([sample_result, sample_result_b], steps=5)
        assert len(ws.weights) == 5


class TestItemDiscrimination:
    def test_few_models(self, sample_result, sample_result_b):
        entries = item_discrimination([sample_result, sample_result_b])
        assert len(entries) > 0
        # With < 3 models, correlations default to 0.0
        for e in entries:
            assert e.correlation == 0.0
            assert e.n_models == 2

    def test_three_models(self, sample_result, sample_result_b):
        from tests.conftest import _make_results

        result_c = _make_results(
            "model-c",
            "TestProvider",
            [
                ("chain_f_01", "CHAIN", "factual", "chain_01", 0.5),
                ("chain_cf_01", "CHAIN", "counterfactual", "chain_01", 0.4),
                ("chain_f_02", "CHAIN", "factual", "chain_02", 0.3),
                ("chain_cf_02", "CHAIN", "counterfactual", "chain_02", 0.2),
                ("gate_f_01", "GATE", "factual", "gate_01", 0.0),
                ("gate_cf_01", "GATE", "counterfactual", "gate_01", 0.0),
                ("gate_f_02", "GATE", "factual", "gate_02", 1.0),
                ("gate_cf_02", "GATE", "counterfactual", "gate_02", 0.0),
            ],
        )
        entries = item_discrimination([sample_result, sample_result_b, result_c])
        assert len(entries) > 0
        for e in entries:
            assert e.n_models == 3
