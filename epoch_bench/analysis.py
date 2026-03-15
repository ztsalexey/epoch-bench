"""Statistical analysis module for EPOCH benchmark (pure functions, no I/O)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev

from scipy.stats import pearsonr, spearmanr, t as t_dist, ttest_rel, wilcoxon

from epoch_bench.schema import BenchmarkResult, Question


@dataclass
class GapSignificance:
    """Paired significance test results for factual vs counterfactual scores."""

    t_statistic: float
    t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    cohens_d: float
    n_pairs: int


@dataclass
class CorrelationEntry:
    """Pairwise correlation between two models."""

    model_a: str
    model_b: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float


@dataclass
class StratifiedScore:
    """Score for a single stratum (difficulty level or domain)."""

    stratum: str
    mean_score: float
    ci_lower: float
    ci_upper: float
    n: int


@dataclass
class WeightSensitivityResult:
    """Result of weight sensitivity analysis."""

    weights: list[tuple[float, float]]  # (factual_weight, cf_weight)
    rankings: list[list[str]]  # model rankings at each weight
    kendalls_w: float


@dataclass
class ItemDiscriminationEntry:
    """Item-total correlation for a single question pair."""

    pair_id: str
    question_type: str
    correlation: float
    n_models: int


def _get_pair_scores(result: BenchmarkResult) -> dict[str, dict[str, float]]:
    """Extract per-pair factual/counterfactual scores from a BenchmarkResult."""
    pairs: dict[str, dict[str, float]] = {}
    for r in result.results:
        if r.pair_id not in pairs:
            pairs[r.pair_id] = {}
        pairs[r.pair_id][r.variant] = r.score
    return pairs


def gap_significance(result: BenchmarkResult) -> GapSignificance:
    """Paired t-test + Wilcoxon on factual vs CF per-pair scores; returns t, p, Cohen's d."""
    pairs = _get_pair_scores(result)

    factual_scores = []
    cf_scores = []
    for pair_id in sorted(pairs.keys()):
        p = pairs[pair_id]
        if "factual" in p and "counterfactual" in p:
            factual_scores.append(p["factual"])
            cf_scores.append(p["counterfactual"])

    n = len(factual_scores)
    if n < 2:
        return GapSignificance(
            t_statistic=0.0,
            t_pvalue=1.0,
            wilcoxon_statistic=0.0,
            wilcoxon_pvalue=1.0,
            cohens_d=0.0,
            n_pairs=n,
        )

    # Paired t-test
    t_stat, t_p = ttest_rel(factual_scores, cf_scores)

    # Wilcoxon signed-rank test (requires non-zero differences)
    diffs = [f - c for f, c in zip(factual_scores, cf_scores)]
    nonzero_diffs = [d for d in diffs if d != 0.0]
    if len(nonzero_diffs) >= 10:
        w_stat, w_p = wilcoxon(factual_scores, cf_scores)
    else:
        w_stat, w_p = 0.0, 1.0

    # Cohen's d for paired samples
    diff_mean = mean(diffs)
    diff_sd = stdev(diffs) if len(diffs) >= 2 else 1.0
    cohens_d = diff_mean / diff_sd if diff_sd > 0 else 0.0

    return GapSignificance(
        t_statistic=float(t_stat),
        t_pvalue=float(t_p),
        wilcoxon_statistic=float(w_stat),
        wilcoxon_pvalue=float(w_p),
        cohens_d=cohens_d,
        n_pairs=n,
    )


def correlation_matrix(results: list[BenchmarkResult]) -> list[CorrelationEntry]:
    """Pairwise Pearson/Spearman on per-pair EPOCH scores across models."""
    model_scores: dict[str, dict[str, float]] = {}
    for result in results:
        pairs = _get_pair_scores(result)
        epoch_by_pair = {}
        for pair_id, scores in pairs.items():
            f = scores.get("factual", 0.0)
            c = scores.get("counterfactual", 0.0)
            epoch_by_pair[pair_id] = 0.4 * f + 0.6 * c
        model_scores[result.model] = epoch_by_pair

    models = list(model_scores.keys())
    entries = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m_a, m_b = models[i], models[j]
            common = sorted(set(model_scores[m_a].keys()) & set(model_scores[m_b].keys()))
            if len(common) < 3:
                continue

            scores_a = [model_scores[m_a][p] for p in common]
            scores_b = [model_scores[m_b][p] for p in common]

            pr, pp = pearsonr(scores_a, scores_b)
            sr, sp = spearmanr(scores_a, scores_b)

            entries.append(
                CorrelationEntry(
                    model_a=m_a,
                    model_b=m_b,
                    pearson_r=float(pr),
                    pearson_p=float(pp),
                    spearman_rho=float(sr),
                    spearman_p=float(sp),
                )
            )

    return entries


def _ci_from_scores(scores: list[float]) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) from a list of scores."""
    if not scores:
        return 0.0, 0.0, 0.0
    m = mean(scores)
    if len(scores) < 2:
        return m, m, m
    sd = stdev(scores)
    se = sd / math.sqrt(len(scores))
    t_val = t_dist.ppf(0.975, df=len(scores) - 1)
    return m, m - t_val * se, m + t_val * se


def difficulty_stratified(
    result: BenchmarkResult,
    questions: list[Question],
) -> list[StratifiedScore]:
    """Scores grouped by difficulty 1-5 with CIs."""
    q_map = {q.id: q for q in questions}

    by_difficulty: dict[int, list[float]] = defaultdict(list)
    for r in result.results:
        q = q_map.get(r.question_id)
        if q is None or q.difficulty is None:
            continue
        by_difficulty[q.difficulty].append(r.score)

    strata = []
    for d in sorted(by_difficulty.keys()):
        scores = by_difficulty[d]
        m, ci_lo, ci_hi = _ci_from_scores(scores)
        strata.append(
            StratifiedScore(
                stratum=str(d),
                mean_score=m,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                n=len(scores),
            )
        )

    return strata


def domain_stratified(
    result: BenchmarkResult,
    questions: list[Question],
) -> list[StratifiedScore]:
    """Scores grouped by domain tag with CIs."""
    q_map = {q.id: q for q in questions}

    by_domain: dict[str, list[float]] = defaultdict(list)
    for r in result.results:
        q = q_map.get(r.question_id)
        if q is None or not q.domains:
            continue
        for domain in q.domains:
            by_domain[domain].append(r.score)

    strata = []
    for domain in sorted(by_domain.keys()):
        scores = by_domain[domain]
        m, ci_lo, ci_hi = _ci_from_scores(scores)
        strata.append(
            StratifiedScore(
                stratum=domain,
                mean_score=m,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                n=len(scores),
            )
        )

    return strata


def weight_sensitivity(
    results: list[BenchmarkResult],
    steps: int = 11,
) -> WeightSensitivityResult:
    """Recompute EPOCH at weights 0.0/1.0->1.0/0.0; Kendall's W for rank stability."""
    weights = [(i / (steps - 1), 1.0 - i / (steps - 1)) for i in range(steps)]

    model_pairs: dict[str, dict[str, tuple[float, float]]] = {}
    for result in results:
        pairs = _get_pair_scores(result)
        pair_fc = {}
        for pair_id, scores in pairs.items():
            f = scores.get("factual", 0.0)
            c = scores.get("counterfactual", 0.0)
            pair_fc[pair_id] = (f, c)
        model_pairs[result.model] = pair_fc

    models = list(model_pairs.keys())
    rankings = []

    for w_f, w_c in weights:
        model_epoch = {}
        for model in models:
            pairs_fc = model_pairs[model]
            if not pairs_fc:
                model_epoch[model] = 0.0
                continue
            epoch_scores = [w_f * f + w_c * c for f, c in pairs_fc.values()]
            model_epoch[model] = mean(epoch_scores)

        ranked = sorted(models, key=lambda m: model_epoch[m], reverse=True)
        rankings.append(ranked)

    # Compute Kendall's W
    if len(models) <= 1 or len(weights) <= 1:
        w = 1.0
    else:
        n_items = len(models)
        k = len(weights)

        ranks_per_weight = []
        for ranking in rankings:
            rank_map = {model: r + 1 for r, model in enumerate(ranking)}
            ranks_per_weight.append(rank_map)

        rank_sums = []
        for model in models:
            total = sum(rpw[model] for rpw in ranks_per_weight)
            rank_sums.append(total)

        mean_rank_sum = mean(rank_sums)
        ss = sum((rs - mean_rank_sum) ** 2 for rs in rank_sums)

        denominator = k * k * (n_items**3 - n_items)
        w = 12.0 * ss / denominator if denominator > 0 else 1.0

    return WeightSensitivityResult(
        weights=weights,
        rankings=rankings,
        kendalls_w=w,
    )


@dataclass
class BaselineComparison:
    """Compares model CF performance against naive copy-factual baseline."""

    question_type: str
    model_cf_score: float
    baseline_cf_score: float  # score if CF answer = factual answer
    model_beats_baseline: bool
    margin: float  # model_cf - baseline_cf


def copy_factual_baseline(
    result: BenchmarkResult,
    questions: list[Question],
) -> list[BaselineComparison]:
    """Compute what a naive 'copy factual answer to CF' strategy would score.

    This establishes a lower bound: if a model doesn't beat this baseline,
    its CF performance may just reflect factual knowledge leaking through.
    """
    from epoch_bench.evaluate import score_chain, score_gate, score_ripple, score_bridge

    q_map = {q.id: q for q in questions}

    # Group questions by pair
    pair_questions: dict[str, dict[str, Question]] = {}
    for q in questions:
        pair_questions.setdefault(q.pair_id, {})[q.variant] = q

    # Group results by pair and type
    pair_results: dict[str, dict[str, float]] = {}
    pair_types: dict[str, str] = {}
    for r in result.results:
        pair_results.setdefault(r.pair_id, {})[r.variant] = r.score
        pair_types[r.pair_id] = r.question_type.value

    # For each pair: score the factual answer against the CF expected answer
    type_model_cf: dict[str, list[float]] = defaultdict(list)
    type_baseline_cf: dict[str, list[float]] = defaultdict(list)

    for pair_id, pq in pair_questions.items():
        if "factual" not in pq or "counterfactual" not in pq:
            continue
        fq = pq["factual"]
        cfq = pq["counterfactual"]
        qt = fq.type.value

        # Model's actual CF score
        model_cf = pair_results.get(pair_id, {}).get("counterfactual", 0.0)
        type_model_cf[qt].append(model_cf)

        # Baseline: score factual answer against CF expected answer
        if qt == "CHAIN" and isinstance(fq.answer, list) and isinstance(cfq.answer, list):
            baseline = score_chain(fq.answer, cfq.answer)
        elif qt == "GATE":
            baseline = score_gate(str(fq.answer), str(cfq.answer))
        elif qt == "RIPPLE" and isinstance(fq.answer, list) and isinstance(cfq.answer, list):
            baseline = score_ripple(fq.answer, cfq.answer)
        elif qt == "BRIDGE":
            baseline = score_bridge(str(fq.answer), str(cfq.answer))
        else:
            baseline = 0.0
        type_baseline_cf[qt].append(baseline)

    comparisons = []
    for qt in sorted(type_model_cf.keys()):
        model_scores = type_model_cf[qt]
        baseline_scores = type_baseline_cf[qt]
        m_cf = mean(model_scores) if model_scores else 0.0
        b_cf = mean(baseline_scores) if baseline_scores else 0.0
        comparisons.append(
            BaselineComparison(
                question_type=qt,
                model_cf_score=m_cf,
                baseline_cf_score=b_cf,
                model_beats_baseline=m_cf > b_cf + 0.01,
                margin=m_cf - b_cf,
            )
        )

    return comparisons


def item_discrimination(
    results: list[BenchmarkResult],
) -> list[ItemDiscriminationEntry]:
    """Item-total correlation per pair across models."""
    pair_model_scores: dict[str, dict[str, float]] = defaultdict(dict)
    pair_types: dict[str, str] = {}

    for result in results:
        pairs = _get_pair_scores(result)
        for pair_id, scores in pairs.items():
            f = scores.get("factual", 0.0)
            c = scores.get("counterfactual", 0.0)
            pair_model_scores[pair_id][result.model] = 0.4 * f + 0.6 * c
        for r in result.results:
            pair_types[r.pair_id] = r.question_type.value

    models = [r.model for r in results]
    all_pairs = sorted(pair_model_scores.keys())

    if len(models) < 3:
        return [
            ItemDiscriminationEntry(
                pair_id=pid,
                question_type=pair_types.get(pid, ""),
                correlation=0.0,
                n_models=len(models),
            )
            for pid in all_pairs
        ]

    model_totals = {}
    for model in models:
        scores = [pair_model_scores[pid].get(model, 0.0) for pid in all_pairs]
        model_totals[model] = mean(scores) if scores else 0.0

    entries = []
    for pid in all_pairs:
        item_scores = [pair_model_scores[pid].get(m, 0.0) for m in models]
        total_scores = [model_totals[m] for m in models]

        try:
            r_val, _ = pearsonr(item_scores, total_scores)
            if math.isnan(r_val):
                r_val = 0.0
        except Exception:
            r_val = 0.0

        entries.append(
            ItemDiscriminationEntry(
                pair_id=pid,
                question_type=pair_types.get(pid, ""),
                correlation=r_val,
                n_models=len(models),
            )
        )

    return entries
