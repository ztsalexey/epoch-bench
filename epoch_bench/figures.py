"""Publication-quality figures for EPOCH benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from epoch_bench.schema import BenchmarkResult, QuestionType


def reasoning_gap_chart(results: list[BenchmarkResult], ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart: factual vs counterfactual per model with gap annotation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    models = [r.model for r in results]
    factual = [r.overall_factual * 100 for r in results]
    counterfactual = [r.overall_counterfactual * 100 for r in results]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, factual, width, label="Factual", color="#2196F3")
    ax.bar(x + width / 2, counterfactual, width, label="Counterfactual", color="#FF9800")

    for i, (f, c) in enumerate(zip(factual, counterfactual)):
        gap = f - c
        sign = "+" if gap >= 0 else ""
        ax.annotate(
            f"{sign}{gap:.1f}%",
            xy=(i, max(f, c) + 1),
            ha="center",
            fontsize=9,
            color="red" if abs(gap) > 20 else "black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Reasoning Gap: Factual vs Counterfactual")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig


def type_heatmap(results: list[BenchmarkResult], ax: plt.Axes | None = None) -> plt.Figure:
    """Heatmap: models x question types showing EPOCH scores."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.8)))
    else:
        fig = ax.get_figure()

    types = [qt.value for qt in QuestionType]
    models = [r.model for r in results]

    data = np.zeros((len(models), len(types)))
    for i, r in enumerate(results):
        type_map = {ts.question_type.value: ts.epoch_score * 100 for ts in r.type_scores}
        for j, t in enumerate(types):
            data[i, j] = type_map.get(t, 0.0)

    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".1f",
        xticklabels=types,
        yticklabels=models,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "EPOCH Score (%)"},
    )
    ax.set_title("EPOCH Scores by Question Type")
    fig.tight_layout()
    return fig


def difficulty_curve(
    results: list[BenchmarkResult],
    questions: list | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Line plot: score vs difficulty level per model."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    for r in results:
        # Group results by difficulty
        diff_scores: dict[int, list[float]] = {}
        for res in r.results:
            # Find matching question difficulty
            diff = None
            if questions:
                for q in questions:
                    if q.id == res.question_id and q.difficulty is not None:
                        diff = q.difficulty
                        break
            if diff is None:
                continue
            diff_scores.setdefault(diff, []).append(res.score)

        if not diff_scores:
            continue

        levels = sorted(diff_scores.keys())
        means = [np.mean(diff_scores[d]) * 100 for d in levels]
        ax.plot(levels, means, marker="o", label=r.model)

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Score (%)")
    ax.set_title("Score vs Difficulty")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig


def weight_sensitivity_ribbon(
    weights: list[tuple[float, float]],
    rankings: list[list[str]],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Ribbon plot showing how model rankings change with weight variation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    if not rankings or not rankings[0]:
        ax.set_title("Weight Sensitivity (no data)")
        fig.tight_layout()
        return fig

    models = list(dict.fromkeys(m for ranking in rankings for m in ranking))
    w_factual = [w[0] for w in weights]

    for model in models:
        ranks = []
        for ranking in rankings:
            if model in ranking:
                ranks.append(ranking.index(model) + 1)
            else:
                ranks.append(len(models))
        ax.plot(w_factual, ranks, marker=".", label=model, linewidth=2)

    ax.set_xlabel("Factual Weight")
    ax.set_ylabel("Rank")
    ax.set_title("Weight Sensitivity: Model Rankings")
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(models) + 1))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    return fig


def save_all_figures(
    results: list[BenchmarkResult],
    output_dir: str | Path,
    questions: list | None = None,
    weight_data: dict | None = None,
    contamination_profiles: list | None = None,
    scaling_data=None,
    tech_graph=None,
) -> list[Path]:
    """Generate and save all figures. Returns list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Reasoning gap chart
    fig = reasoning_gap_chart(results)
    path = output_dir / "reasoning_gap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # 2. Type heatmap
    fig = type_heatmap(results)
    path = output_dir / "type_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # 3. Difficulty curve
    if questions:
        fig = difficulty_curve(results, questions)
        path = output_dir / "difficulty_curve.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # 4. Weight sensitivity ribbon
    if weight_data and "weights" in weight_data and "rankings" in weight_data:
        fig = weight_sensitivity_ribbon(weight_data["weights"], weight_data["rankings"])
        path = output_dir / "weight_sensitivity.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # 5. Contamination figures
    if contamination_profiles:
        fig = contamination_heatmap(contamination_profiles)
        path = output_dir / "contamination_heatmap.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        fig = contamination_comparison_bar(contamination_profiles)
        path = output_dir / "contamination_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        # Distribution for first profile
        fig = contamination_distribution(contamination_profiles[0])
        path = output_dir / "contamination_distribution.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # 6. Scaling figures
    if scaling_data:
        fig = scaling_gap_plot(scaling_data)
        path = output_dir / "scaling_gap.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        if scaling_data.per_family:
            fig = scaling_family_lines(scaling_data)
            path = output_dir / "scaling_families.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    # 7. Graph centrality
    if tech_graph:
        fig = graph_centrality_bar(tech_graph)
        path = output_dir / "graph_centrality.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


# --- New figure functions for contamination, scaling, and graph ---


def contamination_heatmap(
    profiles: list,
    top_n: int = 15,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Heatmap: models x top-N domains showing contamination index."""
    # Rank domains by mean absolute contamination across models, keep top N
    from collections import defaultdict
    from statistics import mean as _mean

    domain_scores: dict[str, list[float]] = defaultdict(list)
    for p in profiles:
        for d, v in p.per_domain.items():
            domain_scores[d].append(abs(v))
    ranked = sorted(domain_scores.items(), key=lambda x: _mean(x[1]), reverse=True)
    # Filter out domains with too few data points (< 3 models reporting)
    ranked = [(d, scores) for d, scores in ranked if len(scores) >= len(profiles) // 2]
    domains = [d for d, _ in ranked[:top_n]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, len(domains) * 0.7), max(4, len(profiles) * 0.7)))
    else:
        fig = ax.get_figure()

    models = [p.model for p in profiles]

    data = np.zeros((len(models), len(domains)))
    for i, p in enumerate(profiles):
        for j, d in enumerate(domains):
            data[i, j] = p.per_domain.get(d, 0.0) * 100

    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 9},
        xticklabels=domains,
        yticklabels=models,
        cmap="RdYlGn_r",
        center=0,
        vmin=-50,
        vmax=100,
        cbar_kws={"label": "Contamination Signal (%)"},
    )
    ax.set_title(f"Contamination by Domain (top {len(domains)})")
    ax.set_xticklabels(domains, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(models, fontsize=9)
    fig.tight_layout()
    return fig


def contamination_distribution(
    profile,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Histogram of per-pair contamination signals for one model."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    signals = [s * 100 for s in profile.distribution]
    ax.hist(signals, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Contamination Signal (%)")
    ax.set_ylabel("Number of Pairs")
    ax.set_title(f"Contamination Distribution: {profile.model}")
    fig.tight_layout()
    return fig


def contamination_comparison_bar(
    profiles: list,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Grouped bars: contamination by question type across models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    all_types: set[str] = set()
    for p in profiles:
        all_types.update(p.per_type.keys())
    types = sorted(all_types)

    x = np.arange(len(types))
    n_models = len(profiles)
    width = 0.8 / max(n_models, 1)

    for i, p in enumerate(profiles):
        values = [p.per_type.get(t, 0.0) * 100 for t in types]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=p.model)

    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("Contamination Signal (%)")
    ax.set_title("Contamination by Question Type")
    ax.legend()
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    return fig


def scaling_gap_plot(
    analysis,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter + regression line: capability rank vs reasoning gap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ranks = [e.capability_rank for e in analysis.entries]
    gaps = [e.gap * 100 for e in analysis.entries]
    labels = [e.model for e in analysis.entries]

    ax.scatter(ranks, gaps, s=80, zorder=3)

    # Label placement with collision avoidance
    annotations = []
    for r, g, label in zip(ranks, gaps, labels):
        annotations.append((r, g, label))

    # Sort by y-position to handle close points
    annotations.sort(key=lambda x: (-x[1], x[0]))

    placed_boxes: list[tuple[float, float, float, float]] = []  # (x, y, xo, yo)
    candidate_offsets = [
        (7, 8), (7, -16), (-7, 8), (-7, -16),
        (7, 20), (7, -28), (-7, 20), (-7, -28),
        (20, 0), (-20, 0),
    ]

    for r, g, label in annotations:
        best = candidate_offsets[0]
        for xo, yo in candidate_offsets:
            # Check against all placed labels using offset-point scale
            ok = True
            for pr, pg, pxo, pyo in placed_boxes:
                # Estimate if labels would overlap: same offset-point space
                dx = abs((r - pr) * 12 + (xo - pxo))  # scale data to ~offset points
                dy = abs((g - pg) * 3 + (yo - pyo))
                if dx < 55 and dy < 14:
                    ok = False
                    break
            if ok:
                best = (xo, yo)
                break

        ha = "left" if best[0] >= 0 else "right"
        ax.annotate(
            label, (r, g),
            textcoords="offset points", xytext=best,
            fontsize=7.5, ha=ha,
            arrowprops={"arrowstyle": "-", "color": "gray", "alpha": 0.3} if abs(best[0]) > 15 or abs(best[1]) > 20 else None,
        )
        placed_boxes.append((r, g, best[0], best[1]))

    # Regression line
    if len(ranks) >= 2:
        x_line = np.linspace(min(ranks), max(ranks), 100)
        y_line = analysis.slope * 100 * x_line + np.mean(gaps) - analysis.slope * 100 * np.mean(ranks)
        ax.plot(x_line, y_line, "r--", alpha=0.7, label=f"slope={analysis.slope:.4f}")

    ax.set_xlabel("Capability Rank")
    ax.set_ylabel("Reasoning Gap (%)")
    ax.set_title("Scaling: Capability vs Reasoning Gap")
    ax.legend()
    fig.tight_layout()
    return fig


def scaling_family_lines(
    analysis,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Per-family line plot of gap vs capability within each family."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    colors = {"anthropic": "#E74C3C", "openai": "#3498DB", "gemini": "#2ECC71", "deepseek": "#9B59B6"}

    for fa in analysis.per_family:
        ranks = list(range(len(fa.models)))
        gaps = [g * 100 for g in fa.gaps]
        color = colors.get(fa.family, None)
        ax.plot(ranks, gaps, marker="o", label=f"{fa.family} ({fa.trend})",
                linewidth=2, markersize=8, color=color)
        # Alternate label positions to avoid overlap
        for i, (r, g, m) in enumerate(zip(ranks, gaps, fa.models)):
            y_off = 8 if i % 2 == 0 else -14
            ax.annotate(m, (r, g), textcoords="offset points",
                        xytext=(0, y_off), fontsize=7, ha="center")

    ax.set_xlabel("Capability Rank (within family)")
    ax.set_ylabel("Reasoning Gap (%)")
    ax.set_title("Scaling by Model Family")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def graph_centrality_bar(
    graph,
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart of most central technologies in the dependency graph."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    centrality = graph.degree_centrality()
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [n for n, _ in sorted_nodes]
    values = [v for _, v in sorted_nodes]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#2196F3")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Degree Centrality")
    ax.set_title(f"Top {top_n} Most Central Technologies")
    fig.tight_layout()
    return fig
