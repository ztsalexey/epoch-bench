"""Tests for epoch_bench.evaluate scoring functions."""

from __future__ import annotations

import pytest

from epoch_bench.evaluate import (
    compute_overall,
    compute_pair_analysis,
    compute_type_scores,
    score_bridge,
    score_chain,
    score_gate,
    score_ripple,
    score_question,
)
from epoch_bench.schema import QuestionType, Result


class TestScoreChain:
    def test_perfect_order(self):
        assert score_chain(["A", "B", "C"], ["A", "B", "C"]) == 1.0

    def test_reversed_order(self):
        score = score_chain(["C", "B", "A"], ["A", "B", "C"])
        assert score == 0.0

    def test_empty_predicted(self):
        assert score_chain([], ["A", "B", "C"]) == 0.0

    def test_length_mismatch(self):
        assert score_chain(["A", "B"], ["A", "B", "C"]) == 0.0

    def test_single_item(self):
        assert score_chain(["A"], ["A"]) == 1.0

    def test_case_insensitive(self):
        assert score_chain(["a", "b", "c"], ["A", "B", "C"]) == 1.0

    def test_partial_order(self):
        score = score_chain(["A", "C", "B"], ["A", "B", "C"])
        assert 0.0 < score < 1.0


class TestScoreGate:
    def test_exact_match(self):
        assert score_gate("Yes", "Yes") == 1.0

    def test_case_insensitive(self):
        assert score_gate("yes", "Yes") == 1.0

    def test_synonym_yes(self):
        assert score_gate("true", "yes") == 1.0

    def test_synonym_no(self):
        assert score_gate("false", "no") == 1.0

    def test_wrong_answer(self):
        assert score_gate("yes", "no") == 0.0

    def test_affirmative(self):
        assert score_gate("affirmative", "yes") == 1.0


class TestScoreBridge:
    def test_correct(self):
        assert score_bridge("B", "B") == 1.0

    def test_case_insensitive(self):
        assert score_bridge("b", "B") == 1.0

    def test_wrong(self):
        assert score_bridge("A", "B") == 0.0


class TestScoreRipple:
    def test_perfect_match(self):
        assert score_ripple(["A", "B"], ["A", "B"]) == 1.0

    def test_partial_match(self):
        score = score_ripple(["A", "B", "C"], ["A", "B"])
        assert 0.0 < score < 1.0  # precision < 1, recall = 1

    def test_no_overlap(self):
        assert score_ripple(["X"], ["A", "B"]) == 0.0

    def test_empty(self):
        assert score_ripple([], []) == 1.0

    def test_empty_predicted(self):
        assert score_ripple([], ["A"]) == 0.0


class TestScoreQuestion:
    def test_dispatch_chain(self):
        assert score_question("CHAIN", ["A", "B"], ["A", "B"]) == 1.0

    def test_dispatch_gate(self):
        assert score_question("GATE", "yes", "Yes") == 1.0

    def test_dispatch_bridge(self):
        assert score_question("BRIDGE", "C", "C") == 1.0

    def test_dispatch_ripple(self):
        assert score_question("RIPPLE", ["X"], ["X"]) == 1.0

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            score_question("UNKNOWN", "x", "x")


class TestComputeTypeScores:
    def test_basic_computation(self, sample_result):
        ts = sample_result.type_scores
        assert len(ts) == 2  # CHAIN and GATE

        chain_ts = next(t for t in ts if t.question_type == QuestionType.CHAIN)
        assert chain_ts.n_pairs == 2
        assert chain_ts.factual_score == pytest.approx(0.85, abs=0.01)
        assert chain_ts.counterfactual_score == pytest.approx(0.55, abs=0.01)

    def test_empty_results(self):
        ts = compute_type_scores([])
        assert ts == []


class TestComputeOverall:
    def test_basic(self, sample_result):
        ts = sample_result.type_scores
        f, cf, gap, epoch = compute_overall(ts)
        assert f > 0
        assert epoch > 0

    def test_empty(self):
        f, cf, gap, epoch = compute_overall([])
        assert f == 0.0
        assert epoch == 0.0


class TestComputePairAnalysis:
    def test_basic(self, sample_result):
        analysis = compute_pair_analysis(sample_result.results)
        assert len(analysis) == 4  # 4 pairs
        for item in analysis:
            assert "epoch_score" in item
            assert "reasoning_gap" in item
