"""Technology dependency graph and procedural question generator."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from epoch_bench.schema import Question, QuestionType

_DATA_DIR = Path(__file__).parent / "data"
_ALIASES_PATH = _DATA_DIR / "tech_aliases.json"


class TechNormalizer:
    """Normalize technology names to canonical forms."""

    def __init__(self, aliases_path: str | Path | None = None) -> None:
        self._alias_map: dict[str, str] = {}
        path = Path(aliases_path) if aliases_path else _ALIASES_PATH
        if path.exists():
            with open(path) as f:
                raw: dict[str, list[str]] = json.load(f)
            for canonical, variants in raw.items():
                key = self._key(canonical)
                self._alias_map[key] = canonical
                for v in variants:
                    self._alias_map[self._key(v)] = canonical

    @staticmethod
    def _key(name: str) -> str:
        """Lowercase, strip articles and trailing plurals for lookup."""
        s = name.lower().strip()
        # Strip leading articles
        for article in ("the ", "a ", "an "):
            if s.startswith(article):
                s = s[len(article):]
                break
        return s

    def normalize(self, name: str) -> str:
        """Return canonical name for a tech string."""
        key = self._key(name)
        if key in self._alias_map:
            return self._alias_map[key]
        # Try without trailing 's' for plurals
        if key.endswith("s") and key[:-1] in self._alias_map:
            return self._alias_map[key[:-1]]
        # Return title-cased original if no alias
        return name.strip()


@dataclass
class TechNode:
    """A technology node in the dependency graph."""

    canonical_name: str
    raw_names: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)


@dataclass
class TechEdge:
    """A directed dependency edge: source must exist before target."""

    source: str  # canonical name
    target: str  # canonical name
    relationship: str  # chain_adjacency | gate_prerequisite | ripple_dependency | bridge_enabler
    source_question_ids: set[str] = field(default_factory=set)


class TechGraph:
    """Wraps a networkx DiGraph of technology dependencies."""

    def __init__(self) -> None:
        self._g = nx.DiGraph()
        self._nodes: dict[str, TechNode] = {}
        self._edges: dict[tuple[str, str], TechEdge] = {}
        self._normalizer = TechNormalizer()

    @property
    def graph(self) -> nx.DiGraph:
        return self._g

    @property
    def nodes(self) -> dict[str, TechNode]:
        return self._nodes

    @property
    def edges(self) -> dict[tuple[str, str], TechEdge]:
        return self._edges

    def _add_node(self, raw_name: str, domains: list[str] | None = None) -> str:
        """Add or update a node, returning canonical name."""
        canonical = self._normalizer.normalize(raw_name)
        if canonical not in self._nodes:
            self._nodes[canonical] = TechNode(canonical_name=canonical)
            self._g.add_node(canonical)
        self._nodes[canonical].raw_names.add(raw_name)
        if domains:
            self._nodes[canonical].domains.update(domains)
        return canonical

    def _add_edge(
        self,
        source_raw: str,
        target_raw: str,
        relationship: str,
        question_id: str,
        domains: list[str] | None = None,
    ) -> None:
        """Add or update an edge."""
        src = self._add_node(source_raw, domains)
        tgt = self._add_node(target_raw, domains)
        if src == tgt:
            return
        key = (src, tgt)
        if key not in self._edges:
            self._edges[key] = TechEdge(source=src, target=tgt, relationship=relationship)
            self._g.add_edge(src, tgt, relationship=relationship)
        self._edges[key].source_question_ids.add(question_id)

    @classmethod
    def from_questions(
        cls,
        questions: list[Question],
        aliases_path: str | Path | None = None,
    ) -> TechGraph:
        """Build graph from factual-variant questions."""
        graph = cls()
        if aliases_path:
            graph._normalizer = TechNormalizer(aliases_path)

        for q in questions:
            if q.variant != "factual":
                continue
            domains = q.domains or []

            if q.type == QuestionType.CHAIN:
                graph._extract_chain(q, domains)
            elif q.type == QuestionType.GATE:
                graph._extract_gate(q, domains)
            elif q.type == QuestionType.RIPPLE:
                graph._extract_ripple(q, domains)
            elif q.type == QuestionType.BRIDGE:
                graph._extract_bridge(q, domains)

        graph._resolve_cycles()
        return graph

    def _extract_chain(self, q: Question, domains: list[str]) -> None:
        """CHAIN: adjacent pairs in answer order are edges."""
        if not isinstance(q.answer, list) or len(q.answer) < 2:
            return
        for i in range(len(q.answer) - 1):
            self._add_edge(
                q.answer[i],
                q.answer[i + 1],
                "chain_adjacency",
                q.id,
                domains,
            )

    def _extract_gate(self, q: Question, domains: list[str]) -> None:
        """GATE: 'Could X without Y?' → No means Y→X prerequisite."""
        if not isinstance(q.answer, str):
            return
        answer_lower = q.answer.strip().lower()
        if answer_lower != "no":
            return

        # Parse "Could X have been created/built without Y?"
        prompt = q.prompt
        # Try common patterns
        patterns = [
            r"[Cc]ould\s+(.+?)\s+(?:have been|be)\s+\w+\s+without\s+(?:the\s+)?(?:prior\s+)?(?:development of\s+|invention of\s+)?(.+?)(?:\?|$)",
            r"[Ww]as it possible to (?:build|create)\s+(.+?)\s+before\s+(?:the\s+)?(?:invention of\s+)?(.+?)(?:\?|$)",
            r"[Cc]ould\s+(.+?)\s+(?:have been\s+)?(?:created|invented|built|developed)\s+(?:before|without)\s+(?:the\s+)?(.+?)(?:\?|$)",
        ]
        for pat in patterns:
            m = re.search(pat, prompt)
            if m:
                target_raw = m.group(1).strip().rstrip("?")
                source_raw = m.group(2).strip().rstrip("?")
                self._add_edge(source_raw, target_raw, "gate_prerequisite", q.id, domains)
                return

    def _extract_ripple(self, q: Question, domains: list[str]) -> None:
        """RIPPLE: 'If X removed' → X→each affected."""
        if not isinstance(q.answer, list):
            return

        # Extract the removed technology from prompt
        patterns = [
            r"[Ii]f\s+(?:the\s+)?(.+?)\s+had never been (?:created|invented|developed|launched|built)",
            r"[Ii]f\s+(.+?)\s+had never been (?:created|invented|developed|launched|built)",
        ]
        removed = None
        for pat in patterns:
            m = re.search(pat, q.prompt)
            if m:
                removed = m.group(1).strip()
                break

        if removed is None:
            return

        self._add_node(removed, domains)
        for affected in q.answer:
            self._add_edge(removed, affected, "ripple_dependency", q.id, domains)

    def _extract_bridge(self, q: Question, domains: list[str]) -> None:
        """BRIDGE: predecessor→bridge→successor from prompt + correct answer."""
        if not isinstance(q.answer, str) or not q.choices:
            return

        # Get bridge technology from correct answer
        idx = ord(q.answer.upper()) - ord("A")
        if idx < 0 or idx >= len(q.choices):
            return
        bridge_tech = q.choices[idx]

        # Parse "What technology bridges X and Y?"
        patterns = [
            r"bridges?\s+(?:the\s+)?(?:gap\s+)?between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        ]
        for pat in patterns:
            m = re.search(pat, q.prompt)
            if m:
                predecessor = m.group(1).strip().rstrip("?")
                successor = m.group(2).strip().rstrip("?")
                self._add_edge(predecessor, bridge_tech, "bridge_enabler", q.id, domains)
                self._add_edge(bridge_tech, successor, "bridge_enabler", q.id, domains)
                return

    def _resolve_cycles(self) -> None:
        """Detect and break cycles by removing lowest-evidence edges."""
        while True:
            try:
                cycle = nx.find_cycle(self._g, orientation="original")
            except nx.NetworkXNoCycle:
                break

            # Find weakest edge in cycle (fewest source questions)
            weakest = None
            min_evidence = float("inf")
            for u, v, _ in cycle:
                key = (u, v)
                edge = self._edges.get(key)
                evidence = len(edge.source_question_ids) if edge else 0
                if evidence < min_evidence:
                    min_evidence = evidence
                    weakest = key

            if weakest:
                self._g.remove_edge(*weakest)
                del self._edges[weakest]

    # --- Query API ---

    def ancestors(self, node: str) -> set[str]:
        """All transitive predecessors of node."""
        canonical = self._normalizer.normalize(node)
        if canonical not in self._g:
            return set()
        return nx.ancestors(self._g, canonical)

    def descendants(self, node: str) -> set[str]:
        """All transitive successors of node."""
        canonical = self._normalizer.normalize(node)
        if canonical not in self._g:
            return set()
        return nx.descendants(self._g, canonical)

    def critical_path(self) -> list[str]:
        """Longest path in the DAG (critical path)."""
        if not self._g.nodes:
            return []
        return nx.dag_longest_path(self._g)

    def is_prerequisite(self, source: str, target: str) -> bool:
        """True if source is a transitive ancestor of target."""
        src = self._normalizer.normalize(source)
        tgt = self._normalizer.normalize(target)
        if src not in self._g or tgt not in self._g:
            return False
        return nx.has_path(self._g, src, tgt)

    def bridge_nodes(self) -> list[str]:
        """Nodes with both in-edges and out-edges (potential bridges)."""
        return [
            n for n in self._g.nodes
            if self._g.in_degree(n) > 0 and self._g.out_degree(n) > 0
        ]

    def degree_centrality(self) -> dict[str, float]:
        """Degree centrality for each node."""
        return nx.degree_centrality(self._g)

    def stats(self) -> dict[str, int | float]:
        """Summary statistics for the graph."""
        centrality = self.degree_centrality()
        return {
            "n_nodes": self._g.number_of_nodes(),
            "n_edges": self._g.number_of_edges(),
            "n_components": nx.number_weakly_connected_components(self._g),
            "longest_path_len": len(self.critical_path()),
            "n_bridge_nodes": len(self.bridge_nodes()),
            "max_centrality": max(centrality.values()) if centrality else 0.0,
        }

    # --- Persistence ---

    def save(self, path: str | Path) -> None:
        """Save graph to JSON."""
        data = {
            "nodes": {
                name: {
                    "raw_names": sorted(node.raw_names),
                    "domains": sorted(node.domains),
                }
                for name, node in self._nodes.items()
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relationship": edge.relationship,
                    "source_question_ids": sorted(edge.source_question_ids),
                }
                for edge in self._edges.values()
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TechGraph:
        """Load graph from JSON."""
        graph = cls()
        with open(path) as f:
            data = json.load(f)

        for name, info in data["nodes"].items():
            graph._nodes[name] = TechNode(
                canonical_name=name,
                raw_names=set(info["raw_names"]),
                domains=set(info["domains"]),
            )
            graph._g.add_node(name)

        for edge_data in data["edges"]:
            src, tgt = edge_data["source"], edge_data["target"]
            edge = TechEdge(
                source=src,
                target=tgt,
                relationship=edge_data["relationship"],
                source_question_ids=set(edge_data["source_question_ids"]),
            )
            graph._edges[(src, tgt)] = edge
            graph._g.add_edge(src, tgt, relationship=edge_data["relationship"])

        return graph


class QuestionGenerator:
    """Generate new questions from the technology dependency graph."""

    def __init__(self, graph: TechGraph, seed: int | None = None) -> None:
        self._graph = graph
        self._rng = random.Random(seed)
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"gen_{prefix}_{self._counter:04d}"

    def _next_pair(self, qtype: str) -> tuple[str, str, str]:
        """Return (pair_id, factual_id, cf_id) for a new question pair."""
        self._counter += 1
        num = f"{self._counter:04d}"
        pair_id = f"gen_{qtype}_{num}"
        return pair_id, f"gen_{qtype}_f_{num}", f"gen_{qtype}_cf_{num}"

    # --- Factual generators (unchanged API, now use _next_pair internally) ---

    def generate_chain(self, min_len: int = 3, max_len: int = 6) -> Question | None:
        """Generate a CHAIN question from a DAG subpath."""
        g = self._graph.graph
        nodes = list(g.nodes)
        if len(nodes) < min_len:
            return None

        for _ in range(100):
            start = self._rng.choice(nodes)
            desc = list(self._graph.descendants(start))
            if not desc:
                continue

            end = self._rng.choice(desc)
            try:
                path = nx.shortest_path(g, start, end)
            except nx.NetworkXNoPath:
                continue

            if len(path) < min_len:
                all_paths = list(nx.all_simple_paths(g, start, end, cutoff=max_len))
                if all_paths:
                    path = max(all_paths, key=len)

            if min_len <= len(path) <= max_len:
                domains = set()
                for n in path:
                    node = self._graph.nodes.get(n)
                    if node:
                        domains.update(node.domains)

                shuffled = list(path)
                self._rng.shuffle(shuffled)
                prompt = (
                    "Order these technologies by dependency (earliest dependency first): "
                    + ", ".join(shuffled)
                )
                pair_id, fid, _ = self._next_pair("chain")
                return Question(
                    id=fid,
                    type=QuestionType.CHAIN,
                    variant="factual",
                    pair_id=pair_id,
                    prompt=prompt,
                    answer=list(path),
                    domains=sorted(domains) if domains else None,
                    source="procedural:graph",
                )
        return None

    def generate_gate(self) -> Question | None:
        """Generate a GATE question: 'Could {node} exist without {ancestor}?'"""
        g = self._graph.graph
        nodes = list(g.nodes)
        if len(nodes) < 2:
            return None

        for _ in range(100):
            node = self._rng.choice(nodes)
            anc = self._graph.ancestors(node)
            non_anc = set(nodes) - anc - {node}

            if not anc or not non_anc:
                continue

            if self._rng.random() < 0.5 and anc:
                prereq = self._rng.choice(sorted(anc))
                answer = "No"
            else:
                prereq = self._rng.choice(sorted(non_anc))
                answer = "Yes"

            domains = set()
            for n in (node, prereq):
                nd = self._graph.nodes.get(n)
                if nd:
                    domains.update(nd.domains)

            pair_id, fid, _ = self._next_pair("gate")
            prompt = f"Could {node} have existed without {prereq}?"
            return Question(
                id=fid,
                type=QuestionType.GATE,
                variant="factual",
                pair_id=pair_id,
                prompt=prompt,
                answer=answer,
                domains=sorted(domains) if domains else None,
                source="procedural:graph",
            )
        return None

    def generate_ripple(
        self, min_affected: int = 2, max_affected: int = 6
    ) -> Question | None:
        """Generate a RIPPLE question by removing a node and finding affected descendants."""
        g = self._graph.graph
        nodes = list(g.nodes)
        if len(nodes) < 3:
            return None

        for _ in range(100):
            removed = self._rng.choice(nodes)
            desc = self._graph.descendants(removed)
            non_desc = set(nodes) - desc - {removed}

            if len(desc) < min_affected or len(non_desc) < 1:
                continue

            affected = sorted(desc)
            if len(affected) > max_affected:
                affected = self._rng.sample(affected, max_affected)

            n_distractors = min(len(non_desc), self._rng.randint(1, 3))
            distractors = self._rng.sample(sorted(non_desc), n_distractors)

            all_options = affected + distractors
            self._rng.shuffle(all_options)

            domains = set()
            nd = self._graph.nodes.get(removed)
            if nd:
                domains.update(nd.domains)

            pair_id, fid, _ = self._next_pair("ripple")
            prompt = (
                f"If {removed} had never been created, which of these technologies "
                f"would not exist in their known form: {', '.join(all_options)}?"
            )
            return Question(
                id=fid,
                type=QuestionType.RIPPLE,
                variant="factual",
                pair_id=pair_id,
                prompt=prompt,
                answer=sorted(affected),
                domains=sorted(domains) if domains else None,
                source="procedural:graph",
            )
        return None

    def generate_bridge(self, n_distractors: int = 3) -> Question | None:
        """Generate a BRIDGE question from a node with both in- and out-edges."""
        bridges = self._graph.bridge_nodes()
        if not bridges:
            return None

        for _ in range(100):
            bridge = self._rng.choice(bridges)
            predecessors = list(self._graph.graph.predecessors(bridge))
            successors = list(self._graph.graph.successors(bridge))

            if not predecessors or not successors:
                continue

            pred = self._rng.choice(predecessors)
            succ = self._rng.choice(successors)

            bridge_node = self._graph.nodes.get(bridge)
            bridge_domains = bridge_node.domains if bridge_node else set()

            candidates = [
                n for n in self._graph.graph.nodes
                if n != bridge and n != pred and n != succ
            ]
            if bridge_domains:
                domain_candidates = [
                    n for n in candidates
                    if self._graph.nodes.get(n)
                    and self._graph.nodes[n].domains & bridge_domains
                ]
                if len(domain_candidates) >= n_distractors:
                    candidates = domain_candidates

            if len(candidates) < n_distractors:
                continue

            distractors = self._rng.sample(candidates, n_distractors)
            choices = [bridge] + distractors
            self._rng.shuffle(choices)
            correct_idx = choices.index(bridge)
            answer_letter = chr(ord("A") + correct_idx)

            domains = set()
            if bridge_node:
                domains.update(bridge_node.domains)

            pair_id, fid, _ = self._next_pair("bridge")
            prompt = (
                f"What technology bridges the gap between {pred} and {succ}?"
            )
            return Question(
                id=fid,
                type=QuestionType.BRIDGE,
                variant="factual",
                pair_id=pair_id,
                prompt=prompt,
                answer=answer_letter,
                choices=choices,
                domains=sorted(domains) if domains else None,
                source="procedural:graph",
            )
        return None

    # --- Counterfactual generators ---

    def generate_chain_cf(self, factual: Question) -> Question:
        """Generate counterfactual twin for a CHAIN question.

        Strategy: pick one node from the chain, premise that it was never created,
        move it to the end of the ordering (developed independently later).
        """
        assert factual.type == QuestionType.CHAIN
        assert isinstance(factual.answer, list) and len(factual.answer) >= 3

        path = list(factual.answer)
        # Pick a node from the first half to remove (more disruptive)
        remove_idx = self._rng.randint(0, len(path) // 2)
        removed = path[remove_idx]

        # CF order: remove node from its position, append at end
        cf_answer = [n for n in path if n != removed] + [removed]

        items = list(path)
        self._rng.shuffle(items)

        cf_id = factual.pair_id.replace("gen_chain_", "gen_chain_cf_")
        if cf_id == factual.pair_id:
            cf_id = factual.id.replace("_f_", "_cf_")

        prompt = (
            f"In a world where {removed} was never created and its function "
            f"was only fulfilled much later by an independent effort, order these "
            f"technologies by dependency (earliest dependency first): "
            + ", ".join(items)
        )
        return Question(
            id=cf_id,
            type=QuestionType.CHAIN,
            variant="counterfactual",
            pair_id=factual.pair_id,
            prompt=prompt,
            answer=cf_answer,
            difficulty=factual.difficulty,
            domains=factual.domains,
            source="procedural:graph",
        )

    def generate_gate_cf(self, factual: Question) -> Question:
        """Generate counterfactual twin for a GATE question.

        Strategy:
        - If factual is No (Y is prereq of X): CF premise introduces an alternative
          to Y, flipping answer to Yes.
        - If factual is Yes (Y is not prereq of X): CF premise makes X depend on Y,
          flipping answer to No.
        """
        assert factual.type == QuestionType.GATE

        # Parse node and prereq from prompt "Could {node} have existed without {prereq}?"
        m = re.match(r"Could (.+?) have existed without (.+?)\?", factual.prompt)
        node = m.group(1) if m else "the technology"
        prereq = m.group(2) if m else "the prerequisite"

        cf_id = factual.id.replace("_f_", "_cf_")

        if factual.answer == "No":
            # Factual: prereq IS required. CF: alternative exists, so not required.
            # Find a sibling tech in same domain for plausibility
            alt = self._find_alternative(prereq)
            prompt = (
                f"If {alt} had been developed as a drop-in replacement for "
                f"{prereq} before {node} was created, could {node} have existed "
                f"without {prereq}?"
            )
            cf_answer = "Yes"
        else:
            # Factual: prereq is NOT required. CF: make it required.
            prompt = (
                f"If {node} had been architecturally designed to require "
                f"{prereq} as its core foundation, could {node} have existed "
                f"without {prereq}?"
            )
            cf_answer = "No"

        return Question(
            id=cf_id,
            type=QuestionType.GATE,
            variant="counterfactual",
            pair_id=factual.pair_id,
            prompt=prompt,
            answer=cf_answer,
            difficulty=factual.difficulty,
            domains=factual.domains,
            source="procedural:graph",
        )

    def generate_ripple_cf(self, factual: Question) -> Question:
        """Generate counterfactual twin for a RIPPLE question.

        Strategy: introduce an alternative to the removed tech that partially
        compensates, reducing the set of affected technologies.
        """
        assert factual.type == QuestionType.RIPPLE
        assert isinstance(factual.answer, list)

        # Parse removed tech from prompt
        m = re.search(r"If (.+?) had never been created", factual.prompt)
        removed = m.group(1) if m else "the technology"

        # CF: alternative saves some of the affected techs
        affected = list(factual.answer)
        # Keep at least 1, drop up to half
        n_saved = self._rng.randint(1, max(1, len(affected) // 2))
        saved = self._rng.sample(affected, min(n_saved, len(affected) - 1))
        cf_affected = sorted(a for a in affected if a not in saved)

        # If we'd end up with zero affected, keep at least one
        if not cf_affected:
            cf_affected = [affected[0]]
            saved = affected[1:]

        alt = self._find_alternative(removed)

        # Reconstruct the full options list from the factual prompt
        options_m = re.search(r"known form: (.+?)\?", factual.prompt)
        all_options = options_m.group(1) if options_m else ", ".join(affected)

        cf_id = factual.id.replace("_f_", "_cf_")
        prompt = (
            f"If {removed} had never been created but {alt} emerged as a "
            f"partial replacement shortly after, which of these technologies "
            f"would still not exist in their known form: {all_options}?"
        )
        return Question(
            id=cf_id,
            type=QuestionType.RIPPLE,
            variant="counterfactual",
            pair_id=factual.pair_id,
            prompt=prompt,
            answer=cf_affected,
            difficulty=factual.difficulty,
            domains=factual.domains,
            source="procedural:graph",
        )

    def generate_bridge_cf(self, factual: Question) -> Question:
        """Generate counterfactual twin for a BRIDGE question.

        Strategy: premise removes the correct bridge tech, pick a distractor
        as the alternative answer.
        """
        assert factual.type == QuestionType.BRIDGE
        assert factual.choices is not None

        correct_idx = ord(factual.answer) - ord("A")
        bridge_tech = factual.choices[correct_idx]

        # Parse predecessor and successor from prompt
        m = re.search(r"between (.+?) and (.+?)\?", factual.prompt)
        pred = m.group(1) if m else "the predecessor"
        succ = m.group(2) if m else "the successor"

        # Pick a distractor as the new correct answer
        distractors_idx = [i for i in range(len(factual.choices)) if i != correct_idx]
        cf_correct_idx = self._rng.choice(distractors_idx)
        cf_answer = chr(ord("A") + cf_correct_idx)

        cf_id = factual.id.replace("_f_", "_cf_")
        prompt = (
            f"In a world where {bridge_tech} was never developed, what technology "
            f"would most plausibly bridge the gap between {pred} and {succ}?"
        )
        return Question(
            id=cf_id,
            type=QuestionType.BRIDGE,
            variant="counterfactual",
            pair_id=factual.pair_id,
            prompt=prompt,
            answer=cf_answer,
            choices=list(factual.choices),
            difficulty=factual.difficulty,
            domains=factual.domains,
            source="procedural:graph",
        )

    def _find_alternative(self, tech: str) -> str:
        """Find a plausible alternative technology from the same domain."""
        node = self._graph.nodes.get(tech)
        if node and node.domains:
            # Find other nodes in same domain
            candidates = [
                n for n, nd in self._graph.nodes.items()
                if n != tech and nd.domains & node.domains
            ]
            if candidates:
                return self._rng.choice(candidates)
        # Fallback: generic alternative
        return f"an alternative to {tech}"

    # --- Pair generators ---

    def generate_chain_pair(self, **kwargs) -> tuple[Question, Question] | None:
        """Generate a factual/counterfactual CHAIN pair."""
        f = self.generate_chain(**kwargs)
        if f is None:
            return None
        return f, self.generate_chain_cf(f)

    def generate_gate_pair(self) -> tuple[Question, Question] | None:
        """Generate a factual/counterfactual GATE pair."""
        f = self.generate_gate()
        if f is None:
            return None
        return f, self.generate_gate_cf(f)

    def generate_ripple_pair(self, **kwargs) -> tuple[Question, Question] | None:
        """Generate a factual/counterfactual RIPPLE pair."""
        f = self.generate_ripple(**kwargs)
        if f is None:
            return None
        return f, self.generate_ripple_cf(f)

    def generate_bridge_pair(self, **kwargs) -> tuple[Question, Question] | None:
        """Generate a factual/counterfactual BRIDGE pair."""
        f = self.generate_bridge(**kwargs)
        if f is None:
            return None
        return f, self.generate_bridge_cf(f)

    # --- Batch generation ---

    def generate_batch(
        self,
        n_per_type: int = 10,
        exclude_pair_ids: set[str] | None = None,
        include_counterfactuals: bool = False,
    ) -> list[Question]:
        """Generate a batch of questions, deduplicating by content.

        If include_counterfactuals is True, each factual question gets a
        paired counterfactual twin.
        """
        if exclude_pair_ids is None:
            exclude_pair_ids = set()

        generated: list[Question] = []
        seen_prompts: set[str] = set()

        generators = [
            ("chain", self.generate_chain),
            ("gate", self.generate_gate),
            ("ripple", self.generate_ripple),
            ("bridge", self.generate_bridge),
        ]

        cf_generators = {
            "chain": self.generate_chain_cf,
            "gate": self.generate_gate_cf,
            "ripple": self.generate_ripple_cf,
            "bridge": self.generate_bridge_cf,
        }

        for name, gen_fn in generators:
            count = 0
            attempts = 0
            while count < n_per_type and attempts < n_per_type * 10:
                attempts += 1
                q = gen_fn()
                if q is None:
                    continue
                if q.pair_id in exclude_pair_ids:
                    continue
                if q.prompt in seen_prompts:
                    continue
                seen_prompts.add(q.prompt)
                generated.append(q)
                if include_counterfactuals:
                    cf = cf_generators[name](q)
                    generated.append(cf)
                count += 1

        return generated
