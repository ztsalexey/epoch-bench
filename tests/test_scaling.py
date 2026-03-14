"""Tests for scaling analysis module."""

from __future__ import annotations

import pytest

from epoch_bench.scaling import (
    DEFAULT_FAMILIES,
    ScalingAnalysis,
    compute_scaling_analysis,
    scaling_headline,
)
from epoch_bench.schema import BenchmarkResult

from tests.conftest import _make_results


@pytest.fixture
def three_model_results() -> list[BenchmarkResult]:
    """Three models with increasing capability and decreasing gap."""
    return [
        _make_results("small-model", "TestProvider", [
            ("c_f_01", "CHAIN", "factual", "c_01", 0.6),
            ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.2),
            ("g_f_01", "GATE", "factual", "g_01", 0.8),
            ("g_cf_01", "GATE", "counterfactual", "g_01", 0.3),
        ]),
        _make_results("medium-model", "TestProvider", [
            ("c_f_01", "CHAIN", "factual", "c_01", 0.7),
            ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.5),
            ("g_f_01", "GATE", "factual", "g_01", 0.9),
            ("g_cf_01", "GATE", "counterfactual", "g_01", 0.6),
        ]),
        _make_results("large-model", "TestProvider", [
            ("c_f_01", "CHAIN", "factual", "c_01", 0.9),
            ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.85),
            ("g_f_01", "GATE", "factual", "g_01", 1.0),
            ("g_cf_01", "GATE", "counterfactual", "g_01", 0.95),
        ]),
    ]


@pytest.fixture
def family_results() -> list[BenchmarkResult]:
    """Results matching default family orderings for family analysis."""
    return [
        _make_results("gpt-4o-mini", "OpenAI", [
            ("c_f_01", "CHAIN", "factual", "c_01", 0.5),
            ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.2),
            ("g_f_01", "GATE", "factual", "g_01", 0.7),
            ("g_cf_01", "GATE", "counterfactual", "g_01", 0.3),
        ]),
        _make_results("gpt-4o", "OpenAI", [
            ("c_f_01", "CHAIN", "factual", "c_01", 0.8),
            ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.6),
            ("g_f_01", "GATE", "factual", "g_01", 0.9),
            ("g_cf_01", "GATE", "counterfactual", "g_01", 0.7),
        ]),
    ]


class TestScalingAnalysis:
    def test_three_model_analysis(self, three_model_results: list[BenchmarkResult]) -> None:
        analysis = compute_scaling_analysis(three_model_results)
        assert len(analysis.entries) == 3
        assert analysis.slope is not None
        assert analysis.r_squared is not None
        assert analysis.p_value is not None

    def test_gap_trend(self, three_model_results: list[BenchmarkResult]) -> None:
        analysis = compute_scaling_analysis(three_model_results)
        # Gaps decrease: small=0.45, medium=0.25, large=0.05 → closes
        assert analysis.gap_trend in ("closes", "widens", "flat")

    def test_regression_values(self, three_model_results: list[BenchmarkResult]) -> None:
        analysis = compute_scaling_analysis(three_model_results)
        assert 0.0 <= analysis.r_squared <= 1.0
        assert analysis.p_value >= 0.0

    def test_headline_format(self, three_model_results: list[BenchmarkResult]) -> None:
        analysis = compute_scaling_analysis(three_model_results)
        h = scaling_headline(analysis)
        assert "Scaling" in h
        assert "slope=" in h
        assert "p=" in h

    def test_family_detection(self, family_results: list[BenchmarkResult]) -> None:
        analysis = compute_scaling_analysis(family_results)
        family_names = [fa.family for fa in analysis.per_family]
        assert "openai" in family_names

    def test_family_models_ordered(self, family_results: list[BenchmarkResult]) -> None:
        analysis = compute_scaling_analysis(family_results)
        openai_fam = next(fa for fa in analysis.per_family if fa.family == "openai")
        assert openai_fam.models == ["gpt-4o-mini", "gpt-4o"]

    def test_family_trend(self, family_results: list[BenchmarkResult]) -> None:
        analysis = compute_scaling_analysis(family_results)
        openai_fam = next(fa for fa in analysis.per_family if fa.family == "openai")
        assert openai_fam.trend in ("closes", "widens", "flat")

    def test_explicit_orderings(self, three_model_results: list[BenchmarkResult]) -> None:
        custom = {"test_family": ["small-model", "medium-model", "large-model"]}
        analysis = compute_scaling_analysis(three_model_results, family_orderings=custom)
        assert len(analysis.per_family) == 1
        assert analysis.per_family[0].family == "test_family"


class TestEdgeCases:
    def test_single_model(self) -> None:
        results = [
            _make_results("solo", "Test", [
                ("c_f_01", "CHAIN", "factual", "c_01", 0.8),
                ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.5),
            ])
        ]
        analysis = compute_scaling_analysis(results)
        assert len(analysis.entries) == 1
        assert analysis.gap_trend == "flat"

    def test_two_models(self) -> None:
        results = [
            _make_results("model-a", "Test", [
                ("c_f_01", "CHAIN", "factual", "c_01", 0.6),
                ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.2),
            ]),
            _make_results("model-b", "Test", [
                ("c_f_01", "CHAIN", "factual", "c_01", 0.9),
                ("c_cf_01", "CHAIN", "counterfactual", "c_01", 0.85),
            ]),
        ]
        analysis = compute_scaling_analysis(results)
        assert len(analysis.entries) == 2

    def test_single_family(self, family_results: list[BenchmarkResult]) -> None:
        analysis = compute_scaling_analysis(family_results)
        assert len(analysis.per_family) >= 1
