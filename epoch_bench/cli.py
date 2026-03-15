"""CLI entry point for EPOCH benchmark."""

from __future__ import annotations

import asyncio
import json

import click

from epoch_bench.models import get_provider
from epoch_bench.report import (
    print_analysis,
    print_comparison,
    print_correlation_matrix,
    print_report,
    to_latex_comparison,
    to_latex_table,
    to_markdown,
)
from epoch_bench.runner import load_questions, run_benchmark
from epoch_bench.schema import BenchmarkResult, QuestionType


@click.group()
def main() -> None:
    """EPOCH: Evaluating Progress Origins in Causal History."""


@main.command()
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "gemini", "deepseek"]),
    required=True,
    help="Model provider.",
)
@click.option("--model", required=True, help="Model name (e.g. claude-sonnet-4-6).")
@click.option(
    "--types",
    multiple=True,
    type=click.Choice(["CHAIN", "GATE", "RIPPLE", "BRIDGE"]),
    help="Run only specific question types. Omit for all.",
)
@click.option(
    "--concurrency",
    default=5,
    show_default=True,
    help="Max concurrent API requests.",
)
@click.option("--output", type=click.Path(), help="Save results JSON to file.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["console", "json", "markdown"]),
    default="console",
    show_default=True,
    help="Output format.",
)
@click.option("--verbose", is_flag=True, help="Show per-question errors and details.")
@click.option("--temperature", type=float, default=None, help="Model temperature (default: 0.0).")
@click.option("--max-tokens", type=int, default=None, help="Max tokens per response (default: 2048).")
@click.option("--split", type=click.Choice(["open", "closed", "all"]), default="all", show_default=True, help="Test set split to run.")
def run(
    provider: str,
    model: str,
    types: tuple[str, ...],
    concurrency: int,
    output: str | None,
    fmt: str,
    verbose: bool,
    temperature: float | None,
    max_tokens: int | None,
    split: str,
) -> None:
    """Run the EPOCH benchmark against a model."""
    model_provider = get_provider(provider, model)

    qt_filter = [QuestionType(t) for t in types] if types else None
    split_filter = None if split == "all" else split
    result = asyncio.run(
        run_benchmark(
            model_provider,
            qt_filter,
            concurrency,
            verbose=verbose,
            temperature=temperature,
            max_tokens=max_tokens,
            split=split_filter,
        )
    )

    if fmt == "console":
        print_report(result)
    elif fmt == "json":
        click.echo(json.dumps(result.model_dump(), indent=2))
    elif fmt == "markdown":
        click.echo(to_markdown(result))

    if output:
        with open(output, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        click.echo(f"\nResults saved to {output}")


@main.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
def compare(file1: str, file2: str) -> None:
    """Compare two benchmark result files side-by-side.

    Usage: epoch-bench compare results_a.json results_b.json
    """
    with open(file1) as f:
        r1 = BenchmarkResult.model_validate(json.load(f))
    with open(file2) as f:
        r2 = BenchmarkResult.model_validate(json.load(f))

    print_comparison(r1, r2)


@main.command()
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", type=click.Path(), help="Save analysis JSON to file.")
@click.option("--latex", is_flag=True, help="Output LaTeX tables.")
def analyze(result_files: tuple[str, ...], output: str | None, latex: bool) -> None:
    """Run statistical analysis on one or more result files.

    Usage: epoch-bench analyze result1.json result2.json [--output analysis.json] [--latex]
    """
    from dataclasses import asdict

    from epoch_bench.analysis import (
        copy_factual_baseline,
        correlation_matrix,
        difficulty_stratified,
        domain_stratified,
        gap_significance,
        item_discrimination,
        weight_sensitivity,
    )

    results = []
    for path in result_files:
        with open(path) as f:
            results.append(BenchmarkResult.model_validate(json.load(f)))

    questions = load_questions()

    analysis: dict = {}

    # Per-result gap significance
    for result in results:
        gs = gap_significance(result)
        key = "gap_significance" if len(results) == 1 else f"gap_significance_{result.model}"
        analysis[key] = asdict(gs)

    # Copy-factual baseline (first result)
    baselines = copy_factual_baseline(results[0], questions)
    analysis["copy_factual_baseline"] = [asdict(b) for b in baselines]
    click.echo("\nCopy-factual baseline (first model):")
    for b in baselines:
        marker = "beats baseline" if b.model_beats_baseline else "BELOW baseline"
        click.echo(f"  {b.question_type}: model CF={b.model_cf_score:.1%} vs baseline={b.baseline_cf_score:.1%} ({marker})")

    # Difficulty and domain stratification (first result)
    ds = difficulty_stratified(results[0], questions)
    analysis["difficulty_stratified"] = [asdict(s) for s in ds]

    doms = domain_stratified(results[0], questions)
    analysis["domain_stratified"] = [asdict(s) for s in doms]

    # Multi-model analyses
    if len(results) >= 2:
        corr = correlation_matrix(results)
        analysis["correlation_matrix"] = [asdict(c) for c in corr]

        ws = weight_sensitivity(results)
        analysis["weight_sensitivity"] = asdict(ws)

        disc = item_discrimination(results)
        analysis["item_discrimination"] = [asdict(d) for d in disc]

    # Display
    print_analysis(analysis)

    if len(results) >= 2 and "correlation_matrix" in analysis:
        model_names = [r.model for r in results]
        print_correlation_matrix(analysis["correlation_matrix"], model_names)

    if latex:
        for result in results:
            click.echo(to_latex_table(result))
        if len(results) >= 2:
            click.echo(to_latex_comparison(results))

    if output:
        with open(output, "w") as f:
            json.dump(analysis, f, indent=2)
        click.echo(f"\nAnalysis saved to {output}")


@main.command("export-review")
@click.option("--output", required=True, type=click.Path(), help="Output file path.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "json"]),
    default="csv",
    show_default=True,
    help="Export format.",
)
@click.option(
    "--types",
    multiple=True,
    type=click.Choice(["CHAIN", "GATE", "RIPPLE", "BRIDGE"]),
    help="Export only specific question types. Omit for all.",
)
def export_review(output: str, fmt: str, types: tuple[str, ...]) -> None:
    """Export questions for expert review.

    Usage: epoch-bench export-review --output review.csv [--format csv|json] [--types CHAIN GATE]
    """
    from epoch_bench.validation import export_for_review

    qt_filter = [QuestionType(t) for t in types] if types else None
    questions = load_questions(qt_filter)
    export_for_review(questions, output, format=fmt)
    click.echo(f"Exported {len(questions)} questions to {output}")


@main.command("import-review")
@click.argument("review_file", type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path(), help="Output validated questions JSON.")
@click.option(
    "--min-score",
    default=3,
    show_default=True,
    type=int,
    help="Minimum reviewer score to keep.",
)
def import_review(review_file: str, output: str, min_score: int) -> None:
    """Import completed reviews and filter validated questions.

    Usage: epoch-bench import-review review.csv --output validated.json [--min-score 3]
    """
    from epoch_bench.validation import filter_validated, import_reviews

    reviews = import_reviews(review_file)
    questions = load_questions()
    validated = filter_validated(questions, reviews, min_score=min_score)

    data = [q.model_dump() for q in validated]
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    click.echo(f"Validated {len(validated)} of {len(questions)} questions (min_score={min_score})")
    click.echo(f"Saved to {output}")


@main.command()
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["console", "markdown", "latex"]), default="console", show_default=True, help="Output format.")
def leaderboard(result_files: tuple[str, ...], fmt: str) -> None:
    """Generate a leaderboard from one or more result files.

    Usage: epoch-bench leaderboard results_a.json results_b.json [--format console|markdown|latex]
    """
    from epoch_bench.leaderboard import (
        build_leaderboard,
        leaderboard_to_latex,
        leaderboard_to_markdown,
        print_leaderboard,
    )

    results = []
    for path in result_files:
        with open(path) as f:
            results.append(BenchmarkResult.model_validate(json.load(f)))

    entries = build_leaderboard(results)

    if fmt == "console":
        print_leaderboard(entries)
    elif fmt == "markdown":
        click.echo(leaderboard_to_markdown(entries))
    elif fmt == "latex":
        click.echo(leaderboard_to_latex(entries))


@main.command("human-baseline")
@click.option(
    "--types",
    multiple=True,
    type=click.Choice(["CHAIN", "GATE", "RIPPLE", "BRIDGE"]),
    help="Run only specific question types. Omit for all.",
)
@click.option("--max-questions", type=int, default=None, help="Limit number of questions.")
@click.option("--no-shuffle", is_flag=True, help="Present questions in original order.")
@click.option("--output", type=click.Path(), help="Save results JSON to file.")
def human_baseline(
    types: tuple[str, ...],
    max_questions: int | None,
    no_shuffle: bool,
    output: str | None,
) -> None:
    """Run interactive human baseline session.

    Usage: epoch-bench human-baseline [--types CHAIN GATE] [--max-questions 10] [--output human.json]
    """
    from epoch_bench.human_baseline import run_human_session

    qt_filter = [QuestionType(t) for t in types] if types else None
    result = run_human_session(
        types=qt_filter,
        max_questions=max_questions,
        shuffle=not no_shuffle,
    )

    print_report(result)

    if output:
        with open(output, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        click.echo(f"\nResults saved to {output}")


@main.command()
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path(), default="figures", show_default=True, help="Directory to save figures.")
def figures(result_files: tuple[str, ...], output_dir: str) -> None:
    """Generate publication-quality figures from result files.

    Usage: epoch-bench figures results_a.json results_b.json --output-dir /tmp/figs
    """
    from dataclasses import asdict

    from epoch_bench.contamination import compute_contamination_profile
    from epoch_bench.figures import save_all_figures
    from epoch_bench.graph import TechGraph
    from epoch_bench.scaling import compute_scaling_analysis

    results = []
    for path in result_files:
        with open(path) as f:
            results.append(BenchmarkResult.model_validate(json.load(f)))

    questions = load_questions()

    # Compute all data needed for figures
    weight_data = None
    if len(results) >= 2:
        from epoch_bench.analysis import weight_sensitivity

        ws = weight_sensitivity(results)
        weight_data = asdict(ws)

    profiles = [compute_contamination_profile(r, questions) for r in results]
    scaling = compute_scaling_analysis(results) if len(results) >= 2 else None
    tech_graph = TechGraph.from_questions(questions)

    saved = save_all_figures(
        results, output_dir, questions=questions,
        weight_data=weight_data,
        contamination_profiles=profiles,
        scaling_data=scaling,
        tech_graph=tech_graph,
    )
    for p in saved:
        click.echo(f"  Saved: {p}")
    click.echo(f"\n{len(saved)} figures saved to {output_dir}")


@main.command("run-suite")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path(), default=None, help="Override output directory from config.")
def run_suite(config_file: str, output_dir: str | None) -> None:
    """Run benchmark suite across multiple models from a YAML config.

    Usage: epoch-bench run-suite suite.yaml [--output-dir results/]
    """
    from epoch_bench.suite import load_suite_config, run_suite as _run_suite

    config = load_suite_config(config_file)
    if output_dir:
        config.output_dir = output_dir

    results = asyncio.run(_run_suite(config))
    click.echo(f"\nSuite complete: {len(results)} models evaluated.")


@main.command()
@click.option("--rebuild", is_flag=True, help="Rebuild graph from questions.")
@click.option("--stats", is_flag=True, help="Print graph statistics.")
@click.option("--query", type=str, default=None, help="Query ancestors/descendants of a technology.")
@click.option("--export-dot", type=click.Path(), default=None, help="Export graph as DOT file.")
@click.option("--output", type=click.Path(), default=None, help="Save graph JSON to file.")
def graph(rebuild: bool, stats: bool, query: str | None, export_dot: str | None, output: str | None) -> None:
    """Build and query the technology dependency graph.

    Usage: epoch-bench graph --rebuild --stats
    """
    from epoch_bench.graph import TechGraph

    graph_path = output or "tech_graph.json"

    if rebuild:
        questions = load_questions()
        g = TechGraph.from_questions(questions)
        g.save(graph_path)
        click.echo(f"Graph built and saved to {graph_path}")
    else:
        from pathlib import Path

        if not Path(graph_path).exists():
            click.echo("No graph file found. Use --rebuild to create one.")
            return
        g = TechGraph.load(graph_path)

    if stats:
        s = g.stats()
        click.echo("Graph Statistics:")
        for k, v in s.items():
            click.echo(f"  {k}: {v}")

    if query:
        anc = g.ancestors(query)
        desc = g.descendants(query)
        click.echo(f"\nQuery: {query}")
        click.echo(f"  Ancestors ({len(anc)}): {', '.join(sorted(anc)) if anc else 'none'}")
        click.echo(f"  Descendants ({len(desc)}): {', '.join(sorted(desc)) if desc else 'none'}")

    if export_dot:
        import networkx as nx

        nx.drawing.nx_pydot.write_dot(g.graph, export_dot)
        click.echo(f"DOT file exported to {export_dot}")


@main.command()
@click.option("--n-per-type", type=int, default=10, show_default=True, help="Number of questions per type.")
@click.option("--output", type=click.Path(), default=None, help="Output JSONL file.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--graph-file", type=click.Path(exists=True), default=None, help="Graph JSON file (default: builds from questions).")
@click.option("--pairs", is_flag=True, help="Generate factual/counterfactual pairs.")
def generate(n_per_type: int, output: str | None, seed: int | None, graph_file: str | None, pairs: bool) -> None:
    """Generate new questions from the technology dependency graph.

    Usage: epoch-bench generate --n-per-type 5 --output /tmp/generated.jsonl [--pairs]
    """
    from epoch_bench.graph import QuestionGenerator, TechGraph

    if graph_file:
        g = TechGraph.load(graph_file)
    else:
        questions = load_questions()
        g = TechGraph.from_questions(questions)

    gen = QuestionGenerator(g, seed=seed)
    generated = gen.generate_batch(n_per_type=n_per_type, include_counterfactuals=pairs)

    if output:
        with open(output, "w") as f:
            for q in generated:
                f.write(json.dumps(q.model_dump()) + "\n")
        click.echo(f"Generated {len(generated)} questions, saved to {output}")
    else:
        for q in generated:
            click.echo(json.dumps(q.model_dump()))
    click.echo(f"\nTotal: {len(generated)} questions generated")


@main.command()
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--threshold", type=float, default=0.3, show_default=True, help="Contamination signal threshold.")
@click.option("--output", type=click.Path(), default=None, help="Save analysis JSON to file.")
@click.option("--latex", is_flag=True, help="Output LaTeX-formatted summary.")
def contamination(result_files: tuple[str, ...], threshold: float, output: str | None, latex: bool) -> None:
    """Run contamination analysis on result files.

    Usage: epoch-bench contamination result1.json result2.json --threshold 0.3
    """
    from dataclasses import asdict

    from epoch_bench.contamination import (
        compare_contamination,
        compute_contamination_profile,
        contamination_summary,
        difficulty_adjusted_contamination,
    )

    results = []
    for path in result_files:
        with open(path) as f:
            results.append(BenchmarkResult.model_validate(json.load(f)))

    questions = load_questions()

    for r in results:
        profile = compute_contamination_profile(r, questions, threshold=threshold)
        click.echo(contamination_summary(profile))
        click.echo("")

    if len(results) >= 2:
        comparison = compare_contamination(results, questions, threshold=threshold)
        click.echo("Cross-model comparison:")
        click.echo(f"  Most contaminated domains: {', '.join(d for d, _ in comparison.most_contaminated_domains[:5])}")
        click.echo(f"  Most contaminated types: {', '.join(t for t, _ in comparison.most_contaminated_types)}")

        click.echo("\nDifficulty-adjusted contamination:")
        adj = difficulty_adjusted_contamination(results, questions)
        for model, idx in sorted(adj.items(), key=lambda x: x[1], reverse=True):
            click.echo(f"  {model:<30} {idx:.3f}")

    if output:
        profiles = [
            compute_contamination_profile(r, questions, threshold=threshold)
            for r in results
        ]
        data = {
            "profiles": [
                {
                    "model": p.model,
                    "overall_index": p.overall_index,
                    "per_type": p.per_type,
                    "per_domain": p.per_domain,
                    "n_contaminated_pairs": p.n_contaminated_pairs,
                    "n_clean_pairs": p.n_clean_pairs,
                }
                for p in profiles
            ]
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"\nAnalysis saved to {output}")


@main.command()
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--families", type=click.Path(exists=True), default=None, help="YAML file with family orderings.")
@click.option("--output", type=click.Path(), default=None, help="Save analysis JSON to file.")
def scaling(result_files: tuple[str, ...], families: str | None, output: str | None) -> None:
    """Run scaling analysis on result files.

    Usage: epoch-bench scaling result1.json result2.json [--families families.yaml]
    """
    from dataclasses import asdict

    from epoch_bench.scaling import compute_scaling_analysis, scaling_headline

    results = []
    for path in result_files:
        with open(path) as f:
            results.append(BenchmarkResult.model_validate(json.load(f)))

    family_orderings = None
    if families:
        import yaml

        with open(families) as f:
            family_orderings = yaml.safe_load(f)

    questions = load_questions()
    analysis = compute_scaling_analysis(results, family_orderings=family_orderings, questions=questions)

    click.echo(scaling_headline(analysis))
    click.echo(f"  Slope: {analysis.slope:.4f}")
    click.echo(f"  R-squared: {analysis.r_squared:.4f}")
    click.echo(f"  P-value: {analysis.p_value:.4f}")
    click.echo(f"  Overall trend: {analysis.gap_trend}")

    if analysis.per_family:
        click.echo("\nPer-family trends:")
        for fa in analysis.per_family:
            click.echo(f"  {fa.family}: {fa.trend} ({', '.join(fa.models)})")

    if output:
        data = {
            "slope": analysis.slope,
            "r_squared": analysis.r_squared,
            "p_value": analysis.p_value,
            "gap_trend": analysis.gap_trend,
            "headline": analysis.headline,
            "entries": [asdict(e) for e in analysis.entries],
            "per_family": [asdict(fa) for fa in analysis.per_family],
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"\nAnalysis saved to {output}")


@main.command()
@click.argument("result_file", type=click.Path(exists=True))
@click.option("--provider", type=click.Choice(["anthropic", "openai", "gemini", "deepseek"]), required=True, help="Model provider.")
@click.option("--model", required=True, help="Model name.")
@click.option("--concurrency", default=5, show_default=True, help="Max concurrent API requests.")
@click.option("--output", type=click.Path(), default=None, help="Save robustness report JSON.")
def robustness(result_file: str, provider: str, model: str, concurrency: int, output: str | None) -> None:
    """Test prompt robustness by running paraphrased variants.

    Takes an existing result file as baseline, generates paraphrased prompts,
    re-runs them, and reports score variance.

    Usage: epoch-bench robustness results/gpt-4o.json --provider openai --model gpt-4o
    """
    from epoch_bench.robustness import generate_paraphrases, compute_robustness_report

    with open(result_file) as f:
        baseline = BenchmarkResult.model_validate(json.load(f))

    questions = load_questions()
    model_provider = get_provider(provider, model)

    # Generate paraphrases for factual questions
    paraphrases = []
    for q in questions:
        paraphrases.extend(generate_paraphrases(q))

    click.echo(f"Generated {len(paraphrases)} paraphrased prompts from {len(questions)} questions")

    if not paraphrases:
        click.echo("No paraphrases generated.")
        return

    # Collect baseline scores
    baseline_scores = {r.question_id: r.score for r in baseline.results}

    # Run paraphrased variants
    async def _run_paraphrases():
        import asyncio as _asyncio

        sem = _asyncio.Semaphore(concurrency)
        results_map: dict[str, list[float]] = {}

        async def _eval_one(pq):
            async with sem:
                try:
                    from epoch_bench.prompts import format_prompt
                    system_prompt, _ = format_prompt(pq.original)
                    raw = await model_provider.generate(system_prompt, pq.paraphrased_prompt)
                    from epoch_bench.runner import _parse_response
                    parsed = _parse_response(raw, pq.original.type)
                    from epoch_bench.evaluate import score_question
                    score = score_question(pq.original.type.value, parsed, pq.original.answer)
                except Exception:
                    score = 0.0
                qid = pq.original.id
                results_map.setdefault(qid, []).append(score)

        await _asyncio.gather(*[_eval_one(pq) for pq in paraphrases])
        return results_map

    para_scores = asyncio.run(_run_paraphrases())

    # Merge baseline scores with paraphrase scores
    merged: dict[str, list[float]] = {}
    for qid in baseline_scores:
        merged[qid] = [baseline_scores[qid]] + para_scores.get(qid, [])

    report = compute_robustness_report(merged, questions)

    click.echo(f"\nRobustness Report:")
    click.echo(f"  Questions tested: {report.n_questions}")
    click.echo(f"  Robust (std < 0.1): {report.n_robust} ({report.robustness_rate:.0%})")
    click.echo(f"  Mean score std: {report.mean_std:.3f}")
    click.echo(f"  Per-type robustness:")
    for qt, rate in report.per_type.items():
        click.echo(f"    {qt}: {rate:.0%}")
    if report.fragile_questions:
        click.echo(f"  Most fragile: {', '.join(report.fragile_questions[:5])}")

    if output:
        from dataclasses import asdict
        with open(output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        click.echo(f"\nReport saved to {output}")


if __name__ == "__main__":
    main()
