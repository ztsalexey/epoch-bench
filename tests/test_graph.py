"""Tests for technology dependency graph."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from epoch_bench.graph import TechGraph, TechNormalizer
from epoch_bench.schema import Question, QuestionType


@pytest.fixture
def aliases_path(tmp_path: Path) -> Path:
    """Create a temporary aliases file."""
    aliases = {
        "UNIX": ["Unix", "unix"],
        "Transistor": ["Transistors", "transistors"],
        "TCP/IP": ["TCP/IP protocol suite"],
        "C language": ["C", "the C programming language"],
    }
    path = tmp_path / "aliases.json"
    with open(path, "w") as f:
        json.dump(aliases, f)
    return path


@pytest.fixture
def normalizer(aliases_path: Path) -> TechNormalizer:
    return TechNormalizer(aliases_path)


@pytest.fixture
def graph_questions() -> list[Question]:
    """Small question set for graph tests. Uses names that won't collide with alias table."""
    return [
        # CHAIN: Alpha -> Beta -> Gamma -> Delta
        Question(
            id="chain_f_01",
            type=QuestionType.CHAIN,
            variant="factual",
            pair_id="chain_01",
            prompt="Order: Delta, Alpha, Beta, Gamma",
            answer=["Alpha", "Beta", "Gamma", "Delta"],
            domains=["computing"],
        ),
        # GATE: Epsilon is prerequisite for Zeta (answer No)
        Question(
            id="gate_f_01",
            type=QuestionType.GATE,
            variant="factual",
            pair_id="gate_01",
            prompt="Could Zeta have been created without Epsilon?",
            answer="No",
            domains=["computing"],
        ),
        # GATE: Eta is NOT prerequisite for Theta (answer Yes)
        Question(
            id="gate_f_02",
            type=QuestionType.GATE,
            variant="factual",
            pair_id="gate_02",
            prompt="Could Theta have been created without Eta?",
            answer="Yes",
            domains=["computing"],
        ),
        # RIPPLE: removing Beta affects Gamma and Delta
        Question(
            id="ripple_f_01",
            type=QuestionType.RIPPLE,
            variant="factual",
            pair_id="ripple_01",
            prompt="If Beta had never been created, which would not exist: Gamma, Delta, Epsilon?",
            answer=["Gamma", "Delta"],
            domains=["computing"],
        ),
        # BRIDGE: Kappa bridges Iota and Lambda
        Question(
            id="bridge_f_01",
            type=QuestionType.BRIDGE,
            variant="factual",
            pair_id="bridge_01",
            prompt="What bridges the gap between Iota and Lambda?",
            choices=["Mu", "Kappa", "Nu", "Xi"],
            answer="B",
            domains=["computing"],
        ),
        # Counterfactual — should be skipped
        Question(
            id="chain_cf_01",
            type=QuestionType.CHAIN,
            variant="counterfactual",
            pair_id="chain_01",
            prompt="If X, order: Delta, Alpha, Beta, Gamma",
            answer=["Beta", "Alpha", "Gamma", "Delta"],
            domains=["computing"],
        ),
    ]


class TestTechNormalizer:
    def test_exact_alias(self, normalizer: TechNormalizer) -> None:
        assert normalizer.normalize("Unix") == "UNIX"
        assert normalizer.normalize("unix") == "UNIX"
        assert normalizer.normalize("UNIX") == "UNIX"

    def test_plural_fallback(self, normalizer: TechNormalizer) -> None:
        assert normalizer.normalize("Transistors") == "Transistor"
        assert normalizer.normalize("transistors") == "Transistor"

    def test_article_stripping(self, normalizer: TechNormalizer) -> None:
        assert normalizer.normalize("the C programming language") == "C language"

    def test_unknown_passthrough(self, normalizer: TechNormalizer) -> None:
        assert normalizer.normalize("SomethingNew") == "SomethingNew"

    def test_no_aliases_file(self, tmp_path: Path) -> None:
        n = TechNormalizer(tmp_path / "nonexistent.json")
        assert n.normalize("whatever") == "whatever"


class TestTechGraph:
    def test_from_questions_creates_nodes(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        # CHAIN adds A, B, C, D; GATE adds E, F; RIPPLE adds B->C, B->D; BRIDGE adds P->X->Q
        assert g.graph.number_of_nodes() > 0

    def test_chain_edges(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        assert g.is_prerequisite("Alpha", "Beta")
        assert g.is_prerequisite("Alpha", "Delta")
        assert g.is_prerequisite("Beta", "Gamma")

    def test_gate_edge(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        assert g.is_prerequisite("Epsilon", "Zeta")

    def test_gate_yes_no_edge(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        assert not g.is_prerequisite("Eta", "Theta")

    def test_ripple_edges(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        assert g.is_prerequisite("Beta", "Gamma")
        assert g.is_prerequisite("Beta", "Delta")

    def test_bridge_edges(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        assert g.is_prerequisite("Iota", "Kappa")
        assert g.is_prerequisite("Kappa", "Lambda")
        assert g.is_prerequisite("Iota", "Lambda")

    def test_counterfactual_ignored(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        assert not g.is_prerequisite("Beta", "Alpha")

    def test_ancestors(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        anc = g.ancestors("Delta")
        assert "Alpha" in anc
        assert "Beta" in anc
        assert "Gamma" in anc

    def test_descendants(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        desc = g.descendants("Alpha")
        assert "Beta" in desc
        assert "Gamma" in desc
        assert "Delta" in desc

    def test_critical_path(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        cp = g.critical_path()
        assert len(cp) >= 4  # At least Alpha->Beta->Gamma->Delta

    def test_ancestors_nonexistent(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        assert g.ancestors("nonexistent") == set()

    def test_descendants_nonexistent(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        assert g.descendants("nonexistent") == set()

    def test_bridge_nodes(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        bridges = g.bridge_nodes()
        # Beta, Gamma, Kappa should be bridges (have in and out edges)
        assert len(bridges) > 0

    def test_degree_centrality(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        centrality = g.degree_centrality()
        assert len(centrality) > 0
        assert all(0.0 <= v <= 1.0 for v in centrality.values())

    def test_stats(self, graph_questions: list[Question]) -> None:
        g = TechGraph.from_questions(graph_questions)
        s = g.stats()
        assert s["n_nodes"] > 0
        assert s["n_edges"] > 0
        assert "longest_path_len" in s

    def test_save_load_roundtrip(self, graph_questions: list[Question], tmp_path: Path) -> None:
        g1 = TechGraph.from_questions(graph_questions)
        save_path = tmp_path / "graph.json"
        g1.save(save_path)

        g2 = TechGraph.load(save_path)
        assert g2.graph.number_of_nodes() == g1.graph.number_of_nodes()
        assert g2.graph.number_of_edges() == g1.graph.number_of_edges()
        assert set(g2.graph.nodes) == set(g1.graph.nodes)

    def test_cycle_resolution(self) -> None:
        """Graph with cyclic questions should resolve to a DAG."""
        questions = [
            Question(
                id="chain_f_cyc1",
                type=QuestionType.CHAIN,
                variant="factual",
                pair_id="cyc1",
                prompt="Order: X, Y",
                answer=["X", "Y"],
                domains=["test"],
            ),
            Question(
                id="chain_f_cyc2",
                type=QuestionType.CHAIN,
                variant="factual",
                pair_id="cyc2",
                prompt="Order: Y, X",
                answer=["Y", "X"],
                domains=["test"],
            ),
        ]
        g = TechGraph.from_questions(questions)
        import networkx as nx

        assert nx.is_directed_acyclic_graph(g.graph)


class TestTechGraphIntegration:
    def test_from_real_questions(self) -> None:
        """Integration: build graph from actual question data."""
        from epoch_bench.runner import load_questions

        questions = load_questions()
        g = TechGraph.from_questions(questions)
        s = g.stats()
        # Should have a reasonable number of nodes and edges
        assert s["n_nodes"] >= 20
        assert s["n_edges"] >= 10
        # Must be a DAG
        import networkx as nx

        assert nx.is_directed_acyclic_graph(g.graph)
