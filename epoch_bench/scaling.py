"""Scaling analysis: does more compute close the reasoning gap?"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from scipy.stats import linregress

from epoch_bench.schema import BenchmarkResult, Question


DEFAULT_FAMILIES: dict[str, list[str]] = {
    "anthropic": [
        "claude-3-5-haiku-latest",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "o1-mini",
        "o1",
    ],
    "gemini": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
}


@dataclass
class ModelCapabilityEntry:
    """One model's capability rank and scores."""

    model: str
    provider: str
    capability_rank: int  # 0-based within family or global ordering
    factual_score: float
    counterfactual_score: float
    gap: float  # factual - counterfactual
    epoch_score: float


@dataclass
class FamilyAnalysis:
    """Per-family scaling trend."""

    family: str
    models: list[str]
    gaps: list[float]
    trend: str  # "closes", "widens", "flat"


@dataclass
class ScalingAnalysis:
    """Full scaling analysis across all models."""

    entries: list[ModelCapabilityEntry]
    slope: float
    r_squared: float
    p_value: float
    gap_trend: str  # "closes", "widens", "flat"
    headline: str
    per_family: list[FamilyAnalysis]


def _classify_trend(gaps: list[float]) -> str:
    """Classify trend from a list of gaps ordered by capability."""
    if len(gaps) < 2:
        return "flat"
    first_half = mean(gaps[: len(gaps) // 2 + 1])
    second_half = mean(gaps[len(gaps) // 2 :])
    diff = second_half - first_half
    if diff < -0.02:
        return "closes"
    elif diff > 0.02:
        return "widens"
    return "flat"


def _infer_family(model: str) -> str | None:
    """Try to infer family from model name."""
    for family, models in DEFAULT_FAMILIES.items():
        if model in models:
            return family
    lower = model.lower()
    if "claude" in lower:
        return "anthropic"
    if "gpt" in lower or lower.startswith("o1") or lower.startswith("o3"):
        return "openai"
    if "gemini" in lower:
        return "gemini"
    if "deepseek" in lower:
        return "deepseek"
    return None


def compute_scaling_analysis(
    results: list[BenchmarkResult],
    family_orderings: dict[str, list[str]] | None = None,
    questions: list[Question] | None = None,
) -> ScalingAnalysis:
    """Group models by family, regress gap vs capability rank, classify trend."""
    if family_orderings is None:
        family_orderings = DEFAULT_FAMILIES

    # Build model -> family mapping and ordering
    model_family: dict[str, str] = {}
    model_rank_in_family: dict[str, int] = {}
    for family, ordered_models in family_orderings.items():
        for rank, m in enumerate(ordered_models):
            model_family[m] = family
            model_rank_in_family[m] = rank

    # Build entries
    entries: list[ModelCapabilityEntry] = []
    family_entries: dict[str, list[ModelCapabilityEntry]] = defaultdict(list)

    # Assign global capability rank based on epoch score
    sorted_results = sorted(results, key=lambda r: r.overall_epoch_score)
    for global_rank, r in enumerate(sorted_results):
        family = model_family.get(r.model) or _infer_family(r.model)
        rank = model_rank_in_family.get(r.model, global_rank)

        entry = ModelCapabilityEntry(
            model=r.model,
            provider=r.provider,
            capability_rank=rank,
            factual_score=r.overall_factual,
            counterfactual_score=r.overall_counterfactual,
            gap=r.overall_reasoning_gap,
            epoch_score=r.overall_epoch_score,
        )
        entries.append(entry)
        if family:
            family_entries[family].append(entry)

    # Global regression: gap vs capability rank
    if len(entries) >= 3:
        ranks = [e.capability_rank for e in entries]
        gaps = [e.gap for e in entries]
        reg = linregress(ranks, gaps)
        slope = float(reg.slope)
        r_sq = float(reg.rvalue ** 2)
        p_val = float(reg.pvalue)
    elif len(entries) == 2:
        slope = entries[1].gap - entries[0].gap
        r_sq = 1.0
        p_val = 1.0
    else:
        slope = 0.0
        r_sq = 0.0
        p_val = 1.0

    gap_trend = _classify_trend([e.gap for e in entries])

    # Per-family analysis
    per_family: list[FamilyAnalysis] = []
    for family_name, ordered_models in family_orderings.items():
        fam_entries = family_entries.get(family_name, [])
        if not fam_entries:
            continue
        # Sort by rank within family
        fam_entries.sort(key=lambda e: e.capability_rank)
        fa = FamilyAnalysis(
            family=family_name,
            models=[e.model for e in fam_entries],
            gaps=[e.gap for e in fam_entries],
            trend=_classify_trend([e.gap for e in fam_entries]),
        )
        per_family.append(fa)

    headline = scaling_headline_from_values(slope, p_val, gap_trend)

    return ScalingAnalysis(
        entries=entries,
        slope=slope,
        r_squared=r_sq,
        p_value=p_val,
        gap_trend=gap_trend,
        headline=headline,
        per_family=per_family,
    )


def scaling_headline_from_values(slope: float, p_value: float, trend: str) -> str:
    """Build headline string from analysis values."""
    verb = "closes" if trend == "closes" else "doesn't close"
    return f"Scaling {verb} the reasoning gap (slope={slope:.4f}, p={p_value:.4f})"


def scaling_headline(analysis: ScalingAnalysis) -> str:
    """Return headline string for a scaling analysis."""
    return analysis.headline
