"""Tests for V2 features: Jaccard scoring, copy-factual baseline, adjusted contamination, splits, robustness."""

from __future__ import annotations

import json
import pytest

from epoch_bench.evaluate import score_ripple_jaccard
from epoch_bench.schema import Question, QuestionType
from tests.conftest import _make_results


class TestScoreRippleJaccard:
    def test_perfect_match(self) -> None:
        assert score_ripple_jaccard(["A", "B", "C"], ["A", "B", "C"]) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert score_ripple_jaccard(["A", "B"], ["C", "D"]) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # Jaccard = 1/3 (intersection=1, union=3)
        assert score_ripple_jaccard(["A", "B"], ["A", "C"]) == pytest.approx(1 / 3)

    def test_subset(self) -> None:
        # Jaccard = 2/4 = 0.5 (intersection=2, union=4)
        assert score_ripple_jaccard(["A", "B"], ["A", "B", "C", "D"]) == pytest.approx(0.5)

    def test_empty_both(self) -> None:
        assert score_ripple_jaccard([], []) == pytest.approx(1.0)

    def test_empty_predicted(self) -> None:
        assert score_ripple_jaccard([], ["A"]) == pytest.approx(0.0)


class TestCopyFactualBaseline:
    def test_returns_per_type(self) -> None:
        from epoch_bench.analysis import copy_factual_baseline

        result = _make_results("test", "Test", [
            ("chain_f_01", "CHAIN", "factual", "chain_01", 0.9),
            ("chain_cf_01", "CHAIN", "counterfactual", "chain_01", 0.6),
            ("gate_f_01", "GATE", "factual", "gate_01", 1.0),
            ("gate_cf_01", "GATE", "counterfactual", "gate_01", 0.0),
        ])
        questions = [
            Question(id="chain_f_01", type=QuestionType.CHAIN, variant="factual",
                     pair_id="chain_01", prompt="Order A, B, C", answer=["A", "B", "C"]),
            Question(id="chain_cf_01", type=QuestionType.CHAIN, variant="counterfactual",
                     pair_id="chain_01", prompt="If X, order A, B, C", answer=["B", "A", "C"]),
            Question(id="gate_f_01", type=QuestionType.GATE, variant="factual",
                     pair_id="gate_01", prompt="Could X?", answer="No"),
            Question(id="gate_cf_01", type=QuestionType.GATE, variant="counterfactual",
                     pair_id="gate_01", prompt="If Z?", answer="Yes"),
        ]
        baselines = copy_factual_baseline(result, questions)
        assert len(baselines) == 2
        types = {b.question_type for b in baselines}
        assert "CHAIN" in types
        assert "GATE" in types


class TestDifficultyAdjustedContamination:
    def test_returns_per_model(self) -> None:
        from epoch_bench.contamination import difficulty_adjusted_contamination

        r1 = _make_results("model-a", "Test", [
            ("c_f_01", "CHAIN", "factual", "c_01", 0.9),
            ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.3),
        ])
        r2 = _make_results("model-b", "Test", [
            ("c_f_01", "CHAIN", "factual", "c_01", 0.7),
            ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.6),
        ])
        questions = [
            Question(id="c_f_01", type=QuestionType.CHAIN, variant="factual",
                     pair_id="c_01", prompt="Q", answer=["A", "B"]),
            Question(id="c_cf_01", type=QuestionType.CHAIN, variant="counterfactual",
                     pair_id="c_01", prompt="Q2", answer=["B", "A"]),
        ]
        adj = difficulty_adjusted_contamination([r1, r2], questions)
        assert "model-a" in adj
        assert "model-b" in adj
        # model-a has bigger gap, should have higher contamination
        assert adj["model-a"] >= adj["model-b"]


class TestSplitFiltering:
    def test_load_open_questions(self) -> None:
        from epoch_bench.runner import load_questions

        open_qs = load_questions(split="open")
        closed_qs = load_questions(split="closed")
        all_qs = load_questions()
        assert len(open_qs) + len(closed_qs) == len(all_qs)
        assert len(open_qs) == 260
        assert len(closed_qs) == 80

    def test_split_field_present(self) -> None:
        from epoch_bench.runner import load_questions

        for q in load_questions():
            assert q.split in ("open", "closed")


class TestRobustnessModule:
    def test_generate_paraphrases_chain(self) -> None:
        from epoch_bench.robustness import generate_paraphrases

        q = Question(
            id="test_f_01", type=QuestionType.CHAIN, variant="factual",
            pair_id="test_01", prompt="Order these technologies by dependency (earliest dependency first): A, B, C",
            answer=["A", "B", "C"],
        )
        paras = generate_paraphrases(q)
        assert len(paras) >= 1
        for p in paras:
            assert p.paraphrased_prompt != q.prompt
            assert "A, B, C" in p.paraphrased_prompt

    def test_generate_paraphrases_gate(self) -> None:
        from epoch_bench.robustness import generate_paraphrases

        q = Question(
            id="test_f_01", type=QuestionType.GATE, variant="factual",
            pair_id="test_01", prompt="Could X have existed without Y?",
            answer="No",
        )
        paras = generate_paraphrases(q)
        assert len(paras) >= 1

    def test_compute_robustness_report(self) -> None:
        from epoch_bench.robustness import compute_robustness_report

        results = {
            "q1": [1.0, 1.0, 1.0],  # robust
            "q2": [1.0, 0.0, 0.5],  # fragile
        }
        questions = [
            Question(id="q1", type=QuestionType.GATE, variant="factual",
                     pair_id="p1", prompt="Q", answer="No"),
            Question(id="q2", type=QuestionType.GATE, variant="factual",
                     pair_id="p2", prompt="Q2", answer="Yes"),
        ]
        report = compute_robustness_report(results, questions)
        assert report.n_questions == 2
        assert report.n_robust == 1  # q1 is robust
        assert report.robustness_rate == pytest.approx(0.5)
        assert "q2" in report.fragile_questions


class TestGateNonFlipping:
    def test_new_gate_pairs_dont_flip(self) -> None:
        from epoch_bench.runner import load_questions

        qs = load_questions([QuestionType.GATE])
        pairs = {}
        for q in qs:
            pairs.setdefault(q.pair_id, {})[q.variant] = q.answer

        # Pairs 41-50 should all be non-flipping
        for i in range(41, 51):
            pid = f"gate_{i:02d}" if i < 100 else f"gate_{i}"
            if pid in pairs:
                f_ans = pairs[pid].get("factual")
                cf_ans = pairs[pid].get("counterfactual")
                assert f_ans == cf_ans, f"{pid}: F={f_ans} CF={cf_ans} should match"
