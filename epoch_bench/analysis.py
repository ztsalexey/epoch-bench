"""Statistical analysis module for EPOCH benchmark (pure functions, no I/O)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev

from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon

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
    return m, m - 1.96 * se, m + 1.96 * se


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
