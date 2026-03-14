"""Rich console output formatting for EPOCH benchmark results."""

from rich.console import Console
from rich.table import Table
from rich.text import Text

from epoch_bench.schema import BenchmarkResult


def _fmt_pct(value: float) -> str:
    """Format a float as a percentage with 1 decimal place."""
    return f"{value * 100:.1f}%"


def _fmt_gap(value: float) -> Text:
    """Format reasoning gap with sign and color."""
    pct = abs(value) * 100
    sign = "+" if value >= 0 else "-"
    label = f"{sign}{pct:.1f}%"

    if pct <= 10:
        style = "green"
    elif pct <= 20:
        style = "yellow"
    else:
        style = "red"

    return Text(label, style=style)


def _fmt_ci(lower: float | None, upper: float | None) -> str:
    """Format 95% CI range."""
    if lower is None or upper is None:
        return "—"
    return f"[{lower * 100:.1f}%, {upper * 100:.1f}%]"


def print_report(result: BenchmarkResult) -> None:
    """Print a formatted benchmark report to the console."""
    console = Console()

    # -- Header --
    console.print()
    console.print("[bold]EPOCH Benchmark Results[/bold]")
    console.print(f"{result.model}  [dim]({result.provider})[/dim]")
    console.print()

    # -- Per-type table --
    table = Table(show_header=True, header_style="bold")
    table.add_column("Type")
    table.add_column("Factual", justify="right")
    table.add_column("Counterfactual", justify="right")
    table.add_column("Reasoning Gap", justify="right")
    table.add_column("EPOCH Score", justify="right")
    table.add_column("StdDev", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Pairs", justify="right")

    for ts in result.type_scores:
        table.add_row(
            ts.question_type.value,
            _fmt_pct(ts.factual_score),
            _fmt_pct(ts.counterfactual_score),
            _fmt_gap(ts.reasoning_gap),
            _fmt_pct(ts.epoch_score),
            _fmt_pct(ts.std_dev) if ts.std_dev is not None else "—",
            _fmt_ci(ts.ci_lower, ts.ci_upper),
            str(ts.n_pairs),
        )

    console.print(table)

    # -- Overall scores --
    console.rule()
    console.print(f"  Overall Factual:        {_fmt_pct(result.overall_factual)}")
    console.print(f"  Overall Counterfactual:  {_fmt_pct(result.overall_counterfactual)}")
    console.print(f"  Overall Reasoning Gap:   ", end="")
    console.print(_fmt_gap(result.overall_reasoning_gap))
    console.print(f"  Overall EPOCH Score:     [bold]{_fmt_pct(result.overall_epoch_score)}[/bold]")

    # -- Footer: duration and failures --
    if result.evaluation_duration_seconds is not None or result.failed_questions:
        console.print()
        if result.evaluation_duration_seconds is not None:
            console.print(
                f"  Duration: {result.evaluation_duration_seconds:.1f}s"
                f"  ({result.total_questions or 0} questions)"
            )
        if result.failed_questions:
            console.print(f"  [red]Failed: {result.failed_questions} questions[/red]")
    console.print()


def to_markdown(result: BenchmarkResult) -> str:
    """Generate a Markdown report string."""
    lines = [
        f"# EPOCH Benchmark Results",
        f"**Model:** {result.model} ({result.provider})",
        "",
        "| Type | Factual | Counterfactual | Gap | EPOCH | StdDev | 95% CI | Pairs |",
        "|------|---------|----------------|-----|-------|--------|--------|-------|",
    ]

    for ts in result.type_scores:
        gap_sign = "+" if ts.reasoning_gap >= 0 else ""
        ci = _fmt_ci(ts.ci_lower, ts.ci_upper) if ts.ci_lower is not None else "—"
        sd = _fmt_pct(ts.std_dev) if ts.std_dev is not None else "—"
        lines.append(
            f"| {ts.question_type.value} "
            f"| {_fmt_pct(ts.factual_score)} "
            f"| {_fmt_pct(ts.counterfactual_score)} "
            f"| {gap_sign}{ts.reasoning_gap * 100:.1f}% "
            f"| {_fmt_pct(ts.epoch_score)} "
            f"| {sd} "
            f"| {ci} "
            f"| {ts.n_pairs} |"
        )

    gap_sign = "+" if result.overall_reasoning_gap >= 0 else ""
    lines.extend([
        "",
        f"**Overall Factual:** {_fmt_pct(result.overall_factual)}  ",
        f"**Overall Counterfactual:** {_fmt_pct(result.overall_counterfactual)}  ",
        f"**Overall Reasoning Gap:** {gap_sign}{result.overall_reasoning_gap * 100:.1f}%  ",
        f"**Overall EPOCH Score:** {_fmt_pct(result.overall_epoch_score)}  ",
    ])

    if result.evaluation_duration_seconds is not None:
        lines.append(
            f"\n*Duration: {result.evaluation_duration_seconds:.1f}s "
            f"({result.total_questions or 0} questions"
            f"{f', {result.failed_questions} failed' if result.failed_questions else ''})*"
        )

    return "\n".join(lines) + "\n"


def print_analysis(analysis_dict: dict) -> None:
    """Rich-formatted gap significance, stratified tables, weight sensitivity."""
    console = Console()

    if "gap_significance" in analysis_dict:
        gs = analysis_dict["gap_significance"]
        console.print()
        console.print("[bold]Gap Significance Test[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Paired t-statistic", f"{gs['t_statistic']:.4f}")
        table.add_row("t-test p-value", f"{gs['t_pvalue']:.4f}")
        table.add_row("Wilcoxon statistic", f"{gs['wilcoxon_statistic']:.4f}")
        table.add_row("Wilcoxon p-value", f"{gs['wilcoxon_pvalue']:.4f}")
        table.add_row("Cohen's d", f"{gs['cohens_d']:.4f}")
        table.add_row("N pairs", str(gs["n_pairs"]))
        console.print(table)

    if "difficulty_stratified" in analysis_dict:
        console.print()
        console.print("[bold]Scores by Difficulty[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Difficulty")
        table.add_column("Mean", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("N", justify="right")
        for s in analysis_dict["difficulty_stratified"]:
            table.add_row(
                s["stratum"],
                _fmt_pct(s["mean_score"]),
                _fmt_ci(s["ci_lower"], s["ci_upper"]),
                str(s["n"]),
            )
        console.print(table)

    if "domain_stratified" in analysis_dict:
        console.print()
        console.print("[bold]Scores by Domain[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Domain")
        table.add_column("Mean", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("N", justify="right")
        for s in analysis_dict["domain_stratified"]:
            table.add_row(
                s["stratum"],
                _fmt_pct(s["mean_score"]),
                _fmt_ci(s["ci_lower"], s["ci_upper"]),
                str(s["n"]),
            )
        console.print(table)

    if "weight_sensitivity" in analysis_dict:
        ws = analysis_dict["weight_sensitivity"]
        console.print()
        console.print("[bold]Weight Sensitivity[/bold]")
        console.print(f"  Kendall's W: {ws['kendalls_w']:.4f}")
        sig = "stable" if ws["kendalls_w"] >= 0.7 else "unstable"
        style = "green" if ws["kendalls_w"] >= 0.7 else "red"
        console.print(f"  Ranking stability: [{style}]{sig}[/{style}]")
        console.print()


def print_correlation_matrix(correlations: list[dict], model_names: list[str]) -> None:
    """N*N matrix with significance stars."""
    console = Console()
    console.print()
    console.print("[bold]Model Correlation Matrix[/bold]")

    corr_map: dict[tuple[str, str], dict] = {}
    for c in correlations:
        key = (c["model_a"], c["model_b"])
        corr_map[key] = c
        corr_map[(c["model_b"], c["model_a"])] = c

    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    for name in model_names:
        table.add_column(name, justify="right")

    for row_name in model_names:
        cells = []
        for col_name in model_names:
            if row_name == col_name:
                cells.append("1.000")
            else:
                c = corr_map.get((row_name, col_name))
                if c is None:
                    cells.append("—")
                else:
                    r_val = c["pearson_r"]
                    p_val = c["pearson_p"]
                    stars = ""
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"
                    cells.append(f"{r_val:.3f}{stars}")
        table.add_row(row_name, *cells)

    console.print(table)
    console.print("[dim]* p<0.05  ** p<0.01  *** p<0.001[/dim]")
    console.print()


def to_latex_table(result: BenchmarkResult) -> str:
    """Single-model LaTeX tabular."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{EPOCH Benchmark Results — {result.model}}}",
        rf"\label{{tab:epoch-{result.model.replace(' ', '-').lower()}}}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Type & Factual & Counterfactual & Gap & EPOCH & Pairs \\",
        r"\midrule",
    ]

    for ts in result.type_scores:
        gap_sign = "+" if ts.reasoning_gap >= 0 else ""
        lines.append(
            f"{ts.question_type.value} & "
            f"{ts.factual_score * 100:.1f}\\% & "
            f"{ts.counterfactual_score * 100:.1f}\\% & "
            f"{gap_sign}{ts.reasoning_gap * 100:.1f}\\% & "
            f"{ts.epoch_score * 100:.1f}\\% & "
            f"{ts.n_pairs} \\\\"
        )

    gap_sign = "+" if result.overall_reasoning_gap >= 0 else ""
    lines.extend(
        [
            r"\midrule",
            f"Overall & "
            f"{result.overall_factual * 100:.1f}\\% & "
            f"{result.overall_counterfactual * 100:.1f}\\% & "
            f"{gap_sign}{result.overall_reasoning_gap * 100:.1f}\\% & "
            f"{result.overall_epoch_score * 100:.1f}\\% & — \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines) + "\n"


def to_latex_comparison(results: list[BenchmarkResult]) -> str:
    """Multi-model comparison LaTeX table."""
    if not results:
        return ""

    model_cols = " & ".join(f"\\textbf{{{r.model}}}" for r in results)
    col_spec = "l" + "r" * len(results)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{EPOCH Benchmark Comparison}",
        r"\label{tab:epoch-comparison}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Metric & {model_cols} \\\\",
        r"\midrule",
    ]

    all_types = set()
    for r in results:
        for ts in r.type_scores:
            all_types.add(ts.question_type)

    for qt in sorted(all_types, key=lambda x: x.value):
        vals = []
        for r in results:
            ts = next((t for t in r.type_scores if t.question_type == qt), None)
            if ts:
                vals.append(f"{ts.epoch_score * 100:.1f}\\%")
            else:
                vals.append("—")
        val_str = " & ".join(vals)
        lines.append(f"{qt.value} & {val_str} \\\\")

    lines.append(r"\midrule")
    overall_vals = " & ".join(f"{r.overall_epoch_score * 100:.1f}\\%" for r in results)
    lines.append(f"Overall & {overall_vals} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines) + "\n"


def print_comparison(r1: BenchmarkResult, r2: BenchmarkResult) -> None:
    """Print side-by-side comparison of two benchmark results."""
    console = Console()

    console.print()
    console.print("[bold]EPOCH Benchmark Comparison[/bold]")
    console.print(f"  Model A: {r1.model}  [dim]({r1.provider})[/dim]")
    console.print(f"  Model B: {r2.model}  [dim]({r2.provider})[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Type")
    table.add_column("A Factual", justify="right")
    table.add_column("B Factual", justify="right")
    table.add_column("A Counter.", justify="right")
    table.add_column("B Counter.", justify="right")
    table.add_column("A EPOCH", justify="right")
    table.add_column("B EPOCH", justify="right")
    table.add_column("Delta", justify="right")

    # Build lookup for r2 scores.
    r2_by_type = {ts.question_type: ts for ts in r2.type_scores}

    for ts1 in r1.type_scores:
        ts2 = r2_by_type.get(ts1.question_type)
        if ts2 is None:
            continue

        delta = ts1.epoch_score - ts2.epoch_score
        delta_text = _fmt_gap(delta)

        table.add_row(
            ts1.question_type.value,
            _fmt_pct(ts1.factual_score),
            _fmt_pct(ts2.factual_score),
            _fmt_pct(ts1.counterfactual_score),
            _fmt_pct(ts2.counterfactual_score),
            _fmt_pct(ts1.epoch_score),
            _fmt_pct(ts2.epoch_score),
            delta_text,
        )

    console.print(table)

    # -- Overall comparison --
    console.rule()
    delta_overall = r1.overall_epoch_score - r2.overall_epoch_score
    console.print(f"  A Overall EPOCH: [bold]{_fmt_pct(r1.overall_epoch_score)}[/bold]")
    console.print(f"  B Overall EPOCH: [bold]{_fmt_pct(r2.overall_epoch_score)}[/bold]")
    console.print(f"  Delta (A - B):   ", end="")
    console.print(_fmt_gap(delta_overall))
    console.print()
