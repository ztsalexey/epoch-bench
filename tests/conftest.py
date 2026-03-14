"""Shared fixtures for EPOCH benchmark tests."""

from __future__ import annotations

import pytest

from epoch_bench.schema import BenchmarkResult, Question, QuestionType, Result, TypeScore


@pytest.fixture
def sample_questions() -> list[Question]:
    """Minimal set of questions for testing."""
    return [
        Question(
            id="chain_f_01",
            type=QuestionType.CHAIN,
            variant="factual",
            pair_id="chain_01",
            prompt="Order A, B, C",
            answer=["A", "B", "C"],
            difficulty=2,
            domains=["computing", "web"],
            source="test",
        ),
        Question(
            id="chain_cf_01",
            type=QuestionType.CHAIN,
            variant="counterfactual",
            pair_id="chain_01",
            prompt="If X, order A, B, C",
            answer=["B", "A", "C"],
            difficulty=2,
            domains=["computing", "web"],
            source="test",
        ),
        Question(
            id="gate_f_01",
            type=QuestionType.GATE,
            variant="factual",
            pair_id="gate_01",
            prompt="Could X without Y?",
            answer="No",
            difficulty=3,
            domains=["computing"],
            source="test",
        ),
        Question(
            id="gate_cf_01",
            type=QuestionType.GATE,
            variant="counterfactual",
            pair_id="gate_01",
            prompt="If Z, could X without Y?",
            answer="Yes",
            difficulty=3,
            domains=["computing"],
            source="test",
        ),
    ]


def _make_results(
    model: str,
    provider: str,
    scores: list[tuple[str, str, str, str, float]],
) -> BenchmarkResult:
    """Build a BenchmarkResult from (question_id, question_type, variant, pair_id, score) tuples."""
    results = []
    for qid, qt, variant, pair_id, score in scores:
        results.append(
            Result(
                question_id=qid,
                question_type=QuestionType(qt),
                variant=variant,
                pair_id=pair_id,
                model_response="test",
                parsed_answer="test",
                expected_answer="test",
                score=score,
            )
        )

    from epoch_bench.evaluate import compute_overall, compute_type_scores

    type_scores = compute_type_scores(results)
    of, ocf, og, oe = compute_overall(type_scores)

    return BenchmarkResult(
        model=model,
        provider=provider,
        results=results,
        type_scores=type_scores,
        overall_factual=of,
        overall_counterfactual=ocf,
        overall_reasoning_gap=og,
        overall_epoch_score=oe,
    )


@pytest.fixture
def sample_result() -> BenchmarkResult:
    """A single benchmark result with known scores."""
    return _make_results(
        "model-a",
        "TestProvider",
        [
            ("chain_f_01", "CHAIN", "factual", "chain_01", 0.9),
            ("chain_cf_01", "CHAIN", "counterfactual", "chain_01", 0.6),
            ("chain_f_02", "CHAIN", "factual", "chain_02", 0.8),
            ("chain_cf_02", "CHAIN", "counterfactual", "chain_02", 0.5),
            ("gate_f_01", "GATE", "factual", "gate_01", 1.0),
            ("gate_cf_01", "GATE", "counterfactual", "gate_01", 0.0),
            ("gate_f_02", "GATE", "factual", "gate_02", 1.0),
            ("gate_cf_02", "GATE", "counterfactual", "gate_02", 1.0),
        ],
    )


@pytest.fixture
def sample_result_b() -> BenchmarkResult:
    """A second benchmark result for comparison tests."""
    return _make_results(
        "model-b",
        "TestProvider",
        [
            ("chain_f_01", "CHAIN", "factual", "chain_01", 0.7),
            ("chain_cf_01", "CHAIN", "counterfactual", "chain_01", 0.8),
            ("chain_f_02", "CHAIN", "factual", "chain_02", 0.6),
            ("chain_cf_02", "CHAIN", "counterfactual", "chain_02", 0.7),
            ("gate_f_01", "GATE", "factual", "gate_01", 1.0),
            ("gate_cf_01", "GATE", "counterfactual", "gate_01", 1.0),
            ("gate_f_02", "GATE", "factual", "gate_02", 0.0),
            ("gate_cf_02", "GATE", "counterfactual", "gate_02", 0.0),
        ],
    )
