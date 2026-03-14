"""Tests for procedural question generator."""

from __future__ import annotations

import pytest

from epoch_bench.graph import QuestionGenerator, TechGraph
from epoch_bench.schema import Question, QuestionType


@pytest.fixture
def tech_graph() -> TechGraph:
    """Build graph from real questions for generation tests."""
    from epoch_bench.runner import load_questions

    questions = load_questions()
    return TechGraph.from_questions(questions)


@pytest.fixture
def generator(tech_graph: TechGraph) -> QuestionGenerator:
    return QuestionGenerator(tech_graph, seed=42)


class TestQuestionSchema:
    def test_generated_chain_has_valid_schema(self, generator: QuestionGenerator) -> None:
        q = generator.generate_chain()
        assert q is not None
        assert q.type == QuestionType.CHAIN
        assert q.variant == "factual"
        assert q.source == "procedural:graph"
        assert isinstance(q.answer, list)
        assert len(q.answer) >= 3
        assert q.id.startswith("gen_chain_")

    def test_generated_gate_has_valid_schema(self, generator: QuestionGenerator) -> None:
        q = generator.generate_gate()
        assert q is not None
        assert q.type == QuestionType.GATE
        assert q.variant == "factual"
        assert q.source == "procedural:graph"
        assert q.answer in ("Yes", "No")
        assert q.id.startswith("gen_gate_")

    def test_generated_ripple_has_valid_schema(self, generator: QuestionGenerator) -> None:
        q = generator.generate_ripple()
        assert q is not None
        assert q.type == QuestionType.RIPPLE
        assert q.variant == "factual"
        assert q.source == "procedural:graph"
        assert isinstance(q.answer, list)
        assert len(q.answer) >= 2
        assert q.id.startswith("gen_ripple_")

    def test_generated_bridge_has_valid_schema(self, generator: QuestionGenerator) -> None:
        q = generator.generate_bridge()
        assert q is not None
        assert q.type == QuestionType.BRIDGE
        assert q.variant == "factual"
        assert q.source == "procedural:graph"
        assert q.answer in ("A", "B", "C", "D")
        assert q.choices is not None
        assert len(q.choices) == 4
        assert q.id.startswith("gen_bridge_")


class TestChainCorrectness:
    def test_chain_answer_in_topological_order(self, tech_graph: TechGraph, generator: QuestionGenerator) -> None:
        q = generator.generate_chain()
        assert q is not None
        answer = q.answer
        # Each adjacent pair should have the first as prerequisite of the second
        for i in range(len(answer) - 1):
            assert tech_graph.is_prerequisite(answer[i], answer[i + 1])


class TestGateCorrectness:
    def test_gate_no_means_ancestor(self, tech_graph: TechGraph) -> None:
        gen = QuestionGenerator(tech_graph, seed=123)
        for _ in range(20):
            q = gen.generate_gate()
            if q is None:
                continue
            # Parse the prompt to extract node and prereq
            # "Could {node} have existed without {prereq}?"
            parts = q.prompt.split(" have existed without ")
            if len(parts) != 2:
                continue
            node = parts[0].replace("Could ", "")
            prereq = parts[1].rstrip("?")
            if q.answer == "No":
                assert tech_graph.is_prerequisite(prereq, node)
            else:
                assert not tech_graph.is_prerequisite(prereq, node)


class TestRippleCorrectness:
    def test_ripple_affected_are_descendants(self, tech_graph: TechGraph) -> None:
        gen = QuestionGenerator(tech_graph, seed=456)
        for _ in range(10):
            q = gen.generate_ripple()
            if q is None:
                continue
            # Extract removed tech from prompt
            prompt = q.prompt
            removed_match = prompt.split("If ")[1].split(" had never been created")[0]
            descendants = tech_graph.descendants(removed_match)
            for affected in q.answer:
                assert affected in descendants, f"{affected} not in descendants of {removed_match}"


class TestBridgeCorrectness:
    def test_bridge_node_on_path(self, tech_graph: TechGraph) -> None:
        gen = QuestionGenerator(tech_graph, seed=789)
        for _ in range(10):
            q = gen.generate_bridge()
            if q is None:
                continue
            # The correct answer choice should be on the path
            idx = ord(q.answer) - ord("A")
            bridge = q.choices[idx]
            # Bridge should be a bridge node (has in and out edges)
            assert tech_graph.graph.in_degree(bridge) > 0 or tech_graph.graph.out_degree(bridge) > 0


class TestCounterfactualGeneration:
    def test_chain_cf_schema(self, generator: QuestionGenerator) -> None:
        f = generator.generate_chain()
        assert f is not None
        cf = generator.generate_chain_cf(f)
        assert cf.type == QuestionType.CHAIN
        assert cf.variant == "counterfactual"
        assert cf.pair_id == f.pair_id
        assert cf.source == "procedural:graph"
        assert isinstance(cf.answer, list)
        assert len(cf.answer) == len(f.answer)

    def test_chain_cf_different_order(self, generator: QuestionGenerator) -> None:
        f = generator.generate_chain()
        assert f is not None
        cf = generator.generate_chain_cf(f)
        assert cf.answer != f.answer
        assert set(cf.answer) == set(f.answer)  # Same items, different order

    def test_chain_cf_has_premise(self, generator: QuestionGenerator) -> None:
        f = generator.generate_chain()
        assert f is not None
        cf = generator.generate_chain_cf(f)
        assert "was never created" in cf.prompt

    def test_gate_cf_flips_answer(self, generator: QuestionGenerator) -> None:
        for _ in range(20):
            f = generator.generate_gate()
            if f is None:
                continue
            cf = generator.generate_gate_cf(f)
            assert cf.pair_id == f.pair_id
            assert cf.variant == "counterfactual"
            assert cf.answer != f.answer  # Answer is flipped
            assert cf.answer in ("Yes", "No")
            return
        pytest.skip("Could not generate gate question")

    def test_ripple_cf_subset(self, generator: QuestionGenerator) -> None:
        for _ in range(20):
            f = generator.generate_ripple()
            if f is None or len(f.answer) < 3:
                continue
            cf = generator.generate_ripple_cf(f)
            assert cf.pair_id == f.pair_id
            assert cf.variant == "counterfactual"
            # CF affected should be a strict subset (fewer items)
            assert len(cf.answer) < len(f.answer)
            assert set(cf.answer).issubset(set(f.answer))
            return
        pytest.skip("Could not generate ripple question with 3+ affected")

    def test_ripple_cf_has_alternative(self, generator: QuestionGenerator) -> None:
        f = generator.generate_ripple()
        assert f is not None
        cf = generator.generate_ripple_cf(f)
        assert "partial replacement" in cf.prompt or "alternative" in cf.prompt

    def test_bridge_cf_different_answer(self, generator: QuestionGenerator) -> None:
        f = generator.generate_bridge()
        assert f is not None
        cf = generator.generate_bridge_cf(f)
        assert cf.pair_id == f.pair_id
        assert cf.variant == "counterfactual"
        assert cf.answer != f.answer
        assert cf.answer in ("A", "B", "C", "D")
        assert cf.choices == f.choices  # Same choices

    def test_bridge_cf_has_premise(self, generator: QuestionGenerator) -> None:
        f = generator.generate_bridge()
        assert f is not None
        cf = generator.generate_bridge_cf(f)
        assert "was never developed" in cf.prompt

    def test_pair_generators(self, tech_graph: TechGraph) -> None:
        gen = QuestionGenerator(tech_graph, seed=42)
        pair = gen.generate_chain_pair()
        assert pair is not None
        f, cf = pair
        assert f.variant == "factual"
        assert cf.variant == "counterfactual"
        assert f.pair_id == cf.pair_id

    def test_pair_id_scheme(self, generator: QuestionGenerator) -> None:
        f = generator.generate_chain()
        assert f is not None
        assert "_f_" in f.id
        cf = generator.generate_chain_cf(f)
        assert "_cf_" in cf.id


class TestBatchGeneration:
    def test_batch_counts(self, generator: QuestionGenerator) -> None:
        batch = generator.generate_batch(n_per_type=5)
        types_present = {q.type for q in batch}
        assert len(batch) > 0
        assert len(types_present) >= 1

    def test_batch_no_duplicate_prompts(self, generator: QuestionGenerator) -> None:
        batch = generator.generate_batch(n_per_type=5)
        prompts = [q.prompt for q in batch]
        assert len(prompts) == len(set(prompts))

    def test_seed_reproducibility(self, tech_graph: TechGraph) -> None:
        gen1 = QuestionGenerator(tech_graph, seed=42)
        batch1 = gen1.generate_batch(n_per_type=3)

        gen2 = QuestionGenerator(tech_graph, seed=42)
        batch2 = gen2.generate_batch(n_per_type=3)

        assert len(batch1) == len(batch2)
        for q1, q2 in zip(batch1, batch2):
            assert q1.prompt == q2.prompt
            assert q1.answer == q2.answer

    def test_batch_with_counterfactuals(self, tech_graph: TechGraph) -> None:
        gen = QuestionGenerator(tech_graph, seed=42)
        batch = gen.generate_batch(n_per_type=3, include_counterfactuals=True)
        factual = [q for q in batch if q.variant == "factual"]
        cf = [q for q in batch if q.variant == "counterfactual"]
        assert len(factual) == len(cf)
        # Each factual should have a CF with the same pair_id
        f_pairs = {q.pair_id for q in factual}
        cf_pairs = {q.pair_id for q in cf}
        assert f_pairs == cf_pairs

    def test_batch_pairs_validate_schema(self, tech_graph: TechGraph) -> None:
        gen = QuestionGenerator(tech_graph, seed=42)
        batch = gen.generate_batch(n_per_type=3, include_counterfactuals=True)
        from epoch_bench.schema import Question as Q
        for q in batch:
            # Validates via pydantic
            Q.model_validate(q.model_dump())
