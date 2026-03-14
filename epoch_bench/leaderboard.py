"""Leaderboard generation from benchmark results."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from epoch_bench.schema import BenchmarkResult


@dataclass
class LeaderboardEntry:
    rank: int
    model: str
    provider: str
    epoch_score: float
    factual: float
    counterfactual: float
    reasoning_gap: float


def build_leaderboard(results: list[BenchmarkResult]) -> list[LeaderboardEntry]:
    """Sort results by EPOCH score descending and assign ranks."""
    sorted_results = sorted(results, key=lambda r: r.overall_epoch_score, reverse=True)
    entries = []
    for i, r in enumerate(sorted_results, start=1):
        entries.append(
            LeaderboardEntry(
                rank=i,
                model=r.model,
                provider=r.provider,
                epoch_score=r.overall_epoch_score,
                factual=r.overall_factual,
                counterfactual=r.overall_counterfactual,
                reasoning_gap=r.overall_reasoning_gap,
            )
        )
    return entries


def print_leaderboard(entries: list[LeaderboardEntry]) -> None:
    """Print leaderboard as a Rich table."""
    console = Console()
    console.print()
    console.print("[bold]EPOCH Benchmark Leaderboard[/bold]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("EPOCH", justify="right")
    table.add_column("Factual", justify="right")
    table.add_column("Counterfactual", justify="right")
    table.add_column("Gap", justify="right")

    for e in entries:
        gap_sign = "+" if e.reasoning_gap >= 0 else ""
        table.add_row(
            str(e.rank),
            e.model,
            e.provider,
            f"{e.epoch_score * 100:.1f}%",
            f"{e.factual * 100:.1f}%",
            f"{e.counterfactual * 100:.1f}%",
            f"{gap_sign}{e.reasoning_gap * 100:.1f}%",
        )

    console.print(table)
    console.print()


def leaderboard_to_markdown(entries: list[LeaderboardEntry]) -> str:
    """Generate Markdown leaderboard table."""
    lines = [
        "# EPOCH Benchmark Leaderboard",
        "",
        "| # | Model | Provider | EPOCH | Factual | Counterfactual | Gap |",
        "|---|-------|----------|-------|---------|----------------|-----|",
    ]
    for e in entries:
        gap_sign = "+" if e.reasoning_gap >= 0 else ""
        lines.append(
            f"| {e.rank} | {e.model} | {e.provider} "
            f"| {e.epoch_score * 100:.1f}% "
            f"| {e.factual * 100:.1f}% "
            f"| {e.counterfactual * 100:.1f}% "
            f"| {gap_sign}{e.reasoning_gap * 100:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def leaderboard_to_latex(entries: list[LeaderboardEntry]) -> str:
    """Generate LaTeX leaderboard table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{EPOCH Benchmark Leaderboard}",
        r"\label{tab:epoch-leaderboard}",
        r"\begin{tabular}{rlrrrrr}",
        r"\toprule",
        r"\# & Model & Provider & EPOCH & Factual & CF & Gap \\",
        r"\midrule",
    ]
    for e in entries:
        gap_sign = "+" if e.reasoning_gap >= 0 else ""
        lines.append(
            f"{e.rank} & {e.model} & {e.provider} & "
            f"{e.epoch_score * 100:.1f}\\% & "
            f"{e.factual * 100:.1f}\\% & "
            f"{e.counterfactual * 100:.1f}\\% & "
            f"{gap_sign}{e.reasoning_gap * 100:.1f}\\% \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"
