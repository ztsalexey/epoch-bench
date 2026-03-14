"""Tests for figure generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from epoch_bench.schema import BenchmarkResult


@pytest.fixture
def _ensure_paper_deps():
    """Skip if matplotlib/seaborn not installed."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")


@pytest.mark.usefixtures("_ensure_paper_deps")
class TestReasoningGapChart:
    def test_single_model(self, sample_result: BenchmarkResult) -> None:
        from epoch_bench.figures import reasoning_gap_chart

        fig = reasoning_gap_chart([sample_result])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_multiple_models(
        self, sample_result: BenchmarkResult, sample_result_b: BenchmarkResult
    ) -> None:
        from epoch_bench.figures import reasoning_gap_chart

        fig = reasoning_gap_chart([sample_result, sample_result_b])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.usefixtures("_ensure_paper_deps")
class TestTypeHeatmap:
    def test_generates_figure(
        self, sample_result: BenchmarkResult, sample_result_b: BenchmarkResult
    ) -> None:
        from epoch_bench.figures import type_heatmap

        fig = type_heatmap([sample_result, sample_result_b])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.usefixtures("_ensure_paper_deps")
class TestDifficultyCurve:
    def test_without_questions(self, sample_result: BenchmarkResult) -> None:
        from epoch_bench.figures import difficulty_curve

        fig = difficulty_curve([sample_result])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_questions(
        self, sample_result: BenchmarkResult, sample_questions
    ) -> None:
        from epoch_bench.figures import difficulty_curve

        fig = difficulty_curve([sample_result], questions=sample_questions)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.usefixtures("_ensure_paper_deps")
class TestWeightSensitivityRibbon:
    def test_generates_figure(self) -> None:
        from epoch_bench.figures import weight_sensitivity_ribbon

        weights = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        rankings = [["a", "b"], ["a", "b"], ["b", "a"]]
        fig = weight_sensitivity_ribbon(weights, rankings)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_empty_data(self) -> None:
        from epoch_bench.figures import weight_sensitivity_ribbon

        fig = weight_sensitivity_ribbon([], [])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.usefixtures("_ensure_paper_deps")
class TestSaveAllFigures:
    def test_saves_files(
        self, sample_result: BenchmarkResult, sample_result_b: BenchmarkResult
    ) -> None:
        from epoch_bench.figures import save_all_figures

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_all_figures(
                [sample_result, sample_result_b], tmpdir
            )
            assert len(saved) >= 2
            for p in saved:
                assert Path(p).exists()
                assert Path(p).stat().st_size > 0
