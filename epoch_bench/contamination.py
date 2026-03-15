"""Contamination analysis: reframe factual/counterfactual gap as contamination detector."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev

from epoch_bench.analysis import GapSignificance, gap_significance
from epoch_bench.schema import BenchmarkResult, Question


@dataclass
class PairContamination:
    """Per-pair contamination signal derived from factual/counterfactual gap."""

    pair_id: str
    factual_score: float
    counterfactual_score: float
    contamination_signal: float  # factual - counterfactual
    normalized_signal: float  # signal / max(factual, epsilon)
    domains: list[str]


@dataclass
class ContaminationProfile:
    """Full contamination profile for a single model."""

    model: str
    overall_index: float
    per_type: dict[str, float]  # question_type -> mean contamination signal
    per_domain: dict[str, float]  # domain -> mean contamination signal
    n_contaminated_pairs: int  # signal > threshold
    n_clean_pairs: int  # signal <= threshold
    distribution: list[float]  # all per-pair signals
    significance: GapSignificance


@dataclass
class ContaminationComparison:
    """Cross-model contamination comparison."""

    models: list[str]
    profiles: list[ContaminationProfile]
    most_contaminated_domains: list[tuple[str, float]]  # (domain, mean signal across models)
    most_contaminated_types: list[tuple[str, float]]  # (type, mean signal across models)


def compute_pair_contamination(
    result: BenchmarkResult,
    questions: list[Question],
) -> list[PairContamination]:
    """Compute per-pair contamination signals from benchmark results."""
    q_map = {q.id: q for q in questions}

    # Group scores by pair
    pair_scores: dict[str, dict[str, float]] = {}
    pair_domains: dict[str, list[str]] = {}
    for r in result.results:
        if r.pair_id not in pair_scores:
            pair_scores[r.pair_id] = {}
        pair_scores[r.pair_id][r.variant] = r.score
        q = q_map.get(r.question_id)
        if q and q.domains and r.pair_id not in pair_domains:
            pair_domains[r.pair_id] = list(q.domains)

    pairs = []
    for pair_id in sorted(pair_scores.keys()):
        scores = pair_scores[pair_id]
        f = scores.get("factual", 0.0)
        c = scores.get("counterfactual", 0.0)
        signal = f - c
        normalized = signal / max(f, 1e-9)
        pairs.append(
            PairContamination(
                pair_id=pair_id,
                factual_score=f,
                counterfactual_score=c,
                contamination_signal=signal,
                normalized_signal=normalized,
                domains=pair_domains.get(pair_id, []),
            )
        )

    return pairs


def compute_contamination_profile(
    result: BenchmarkResult,
    questions: list[Question],
    threshold: float = 0.3,
) -> ContaminationProfile:
    """Compute full contamination profile for one model."""
    pairs = compute_pair_contamination(result, questions)
    q_map = {q.id: q for q in questions}

    # Map pair_id -> question type
    pair_type: dict[str, str] = {}
    for r in result.results:
        pair_type[r.pair_id] = r.question_type.value

    # Overall index
    signals = [p.contamination_signal for p in pairs]
    overall = mean(signals) if signals else 0.0

    # Per-type breakdown
    by_type: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        qt = pair_type.get(p.pair_id, "")
        if qt:
            by_type[qt].append(p.contamination_signal)
    per_type = {t: mean(sigs) for t, sigs in sorted(by_type.items())}

    # Per-domain breakdown
    by_domain: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        for d in p.domains:
            by_domain[d].append(p.contamination_signal)
    per_domain = {d: mean(sigs) for d, sigs in sorted(by_domain.items())}

    # Threshold classification
    n_contaminated = sum(1 for p in pairs if p.contamination_signal > threshold)
    n_clean = len(pairs) - n_contaminated

    # Significance
    sig = gap_significance(result)

    return ContaminationProfile(
        model=result.model,
        overall_index=overall,
        per_type=per_type,
        per_domain=per_domain,
        n_contaminated_pairs=n_contaminated,
        n_clean_pairs=n_clean,
        distribution=signals,
        significance=sig,
    )


def compare_contamination(
    results: list[BenchmarkResult],
    questions: list[Question],
    threshold: float = 0.3,
) -> ContaminationComparison:
    """Cross-model contamination comparison."""
    profiles = [
        compute_contamination_profile(r, questions, threshold) for r in results
    ]

    # Aggregate domains across models
    domain_signals: dict[str, list[float]] = defaultdict(list)
    type_signals: dict[str, list[float]] = defaultdict(list)
    for p in profiles:
        for d, v in p.per_domain.items():
            domain_signals[d].append(v)
        for t, v in p.per_type.items():
            type_signals[t].append(v)

    most_contaminated_domains = sorted(
        [(d, mean(sigs)) for d, sigs in domain_signals.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    most_contaminated_types = sorted(
        [(t, mean(sigs)) for t, sigs in type_signals.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    return ContaminationComparison(
        models=[p.model for p in profiles],
        profiles=profiles,
        most_contaminated_domains=most_contaminated_domains,
        most_contaminated_types=most_contaminated_types,
    )


def contamination_summary(profile: ContaminationProfile) -> str:
    """Human-readable contamination summary for paper results section."""
    lines = [
        f"Contamination Analysis: {profile.model}",
        f"  Overall contamination index: {profile.overall_index:.3f}",
        f"  Contaminated pairs (signal > threshold): {profile.n_contaminated_pairs}",
        f"  Clean pairs: {profile.n_clean_pairs}",
        "",
        "  Per-type contamination:",
    ]
    for t, v in profile.per_type.items():
        lines.append(f"    {t}: {v:.3f}")

    lines.append("")
    lines.append("  Per-domain contamination:")
    for d, v in sorted(profile.per_domain.items(), key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"    {d}: {v:.3f}")

    lines.append("")
    sig = profile.significance
    lines.append(
        f"  Significance: t={sig.t_statistic:.3f}, p={sig.t_pvalue:.4f}, "
        f"Cohen's d={sig.cohens_d:.3f} (n={sig.n_pairs})"
    )

    return "\n".join(lines)


def difficulty_adjusted_contamination(
    results: list[BenchmarkResult],
    questions: list[Question],
) -> dict[str, float]:
    """Disentangle contamination from inherent CF difficulty.

    Uses the cross-model mean CF score per pair as a difficulty proxy.
    A pair where ALL models score low on CF is hard, not contaminated.
    A pair where one model has a large gap but others don't suggests
    model-specific contamination.

    Returns per-model adjusted contamination index.
    """
    q_map = {q.id: q for q in questions}

    # Compute per-pair mean CF score across all models (difficulty proxy)
    pair_cf_scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for r in result.results:
            if r.variant == "counterfactual":
                pair_cf_scores[r.pair_id].append(r.score)

    pair_difficulty = {
        pid: 1.0 - mean(scores) for pid, scores in pair_cf_scores.items()
    }  # higher = harder

    # For each model, compute contamination adjusted for difficulty
    adjusted = {}
    for result in results:
        pairs = compute_pair_contamination(result, questions)
        # Adjusted signal = raw signal - cross-model difficulty
        adj_signals = []
        for p in pairs:
            difficulty = pair_difficulty.get(p.pair_id, 0.0)
            # If the pair is universally hard, discount the signal
            adj_signal = p.contamination_signal - difficulty
            adj_signals.append(max(adj_signal, 0.0))
        adjusted[result.model] = mean(adj_signals) if adj_signals else 0.0

    return adjusted
