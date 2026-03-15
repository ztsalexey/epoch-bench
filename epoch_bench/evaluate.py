"""Scoring logic for the EPOCH benchmark."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

from scipy.stats import kendalltau

from epoch_bench.schema import QuestionType, Result, TypeScore

# Articles and filler words to strip during normalization.
_ARTICLES = {"a", "an", "the"}

# Synonym sets for yes/no gate answers.
_YES_SYNONYMS = {"yes", "y", "true", "correct", "affirmative"}
_NO_SYNONYMS = {"no", "n", "false", "incorrect", "negative"}


def _normalize(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip articles/punctuation."""
    text = text.strip().lower()
    # Remove punctuation (keep alphanumeric and spaces).
    text = re.sub(r"[^\w\s]", "", text)
    # Remove articles.
    words = [w for w in text.split() if w not in _ARTICLES]
    return " ".join(words).strip()


def _load_tech_aliases() -> dict[str, str]:
    """Load tech_aliases.json and build a mapping from each alias to its canonical name."""
    alias_file = Path(__file__).parent / "data" / "tech_aliases.json"
    mapping: dict[str, str] = {}
    if not alias_file.exists():
        return mapping
    with open(alias_file) as f:
        raw: dict[str, list[str]] = json.load(f)
    for canonical, aliases in raw.items():
        canonical_norm = _normalize(canonical)
        mapping[canonical_norm] = canonical_norm
        for alias in aliases:
            mapping[_normalize(alias)] = canonical_norm
    return mapping


# Module-level alias table: loaded once on import.
_TECH_ALIASES: dict[str, str] = _load_tech_aliases()


def _normalize_tech(text: str) -> str:
    """Normalize text and resolve tech aliases to canonical names."""
    normed = _normalize(text)
    return _TECH_ALIASES.get(normed, normed)


def score_chain(predicted: list[str], expected: list[str]) -> float:
    """Kendall's tau correlation normalized to 0-1."""
    if len(expected) <= 1:
        return 1.0 if predicted == expected else 0.0
    if not predicted or len(predicted) != len(expected):
        return 0.0

    # Build rank vectors based on position in each list (normalized).
    expected_norm = {_normalize(item): i for i, item in enumerate(expected)}
    try:
        pred_ranks = [expected_norm[_normalize(item)] for item in predicted]
    except KeyError:
        return 0.0

    exp_ranks = list(range(len(expected)))
    tau, _ = kendalltau(pred_ranks, exp_ranks)

    # kendalltau returns nan for constant inputs; treat as perfect.
    if tau != tau:  # nan check
        return 1.0

    # Normalize from [-1, 1] to [0, 1].
    return (tau + 1.0) / 2.0


def score_gate(predicted: str, expected: str) -> float:
    """Binary accuracy for yes/no answers with synonym support."""
    pred = predicted.strip().lower()
    exp = expected.strip().lower()

    # Direct match.
    if pred == exp:
        return 1.0

    # Synonym matching: both map to the same category.
    pred_yes = pred in _YES_SYNONYMS
    pred_no = pred in _NO_SYNONYMS
    exp_yes = exp in _YES_SYNONYMS
    exp_no = exp in _NO_SYNONYMS

    if (pred_yes and exp_yes) or (pred_no and exp_no):
        return 1.0

    return 0.0


def score_bridge(predicted: str, expected: str) -> float:
    """Binary accuracy for multiple-choice (A/B/C/D)."""
    return 1.0 if predicted.strip().upper() == expected.strip().upper() else 0.0


def score_ripple(predicted: list[str], expected: list[str]) -> float:
    """F1 score between predicted and expected sets of affected technologies."""
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0

    pred_set = {_normalize_tech(s) for s in predicted}
    exp_set = {_normalize_tech(s) for s in expected}

    tp = len(pred_set & exp_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(exp_set)
    return 2.0 * precision * recall / (precision + recall)


def score_ripple_jaccard(predicted: list[str], expected: list[str]) -> float:
    """Jaccard similarity — length-normalized alternative to F1 for RIPPLE.

    Jaccard = |intersection| / |union|, which is symmetric and doesn't
    penalize predictions of different lengths as harshly as F1.
    """
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0

    pred_set = {_normalize_tech(s) for s in predicted}
    exp_set = {_normalize_tech(s) for s in expected}

    intersection = len(pred_set & exp_set)
    union = len(pred_set | exp_set)
    return intersection / union if union > 0 else 0.0


def score_question(question_type: str, predicted, expected) -> float:
    """Dispatch to the right scorer based on question type."""
    qt = QuestionType(question_type)
    if qt is QuestionType.CHAIN:
        return score_chain(predicted, expected)
    if qt is QuestionType.GATE:
        return score_gate(predicted, expected)
    if qt is QuestionType.BRIDGE:
        return score_bridge(predicted, expected)
    if qt is QuestionType.RIPPLE:
        return score_ripple(predicted, expected)
    raise ValueError(f"Unknown question type: {question_type}")


def compute_type_scores(results: list[Result]) -> list[TypeScore]:
    """Group results by question type and compute aggregated scores with statistics."""
    by_type: dict[QuestionType, dict[str, list[float]]] = defaultdict(
        lambda: {"factual": [], "counterfactual": []}
    )

    for r in results:
        by_type[r.question_type][r.variant].append(r.score)

    type_scores: list[TypeScore] = []
    for qt in QuestionType:
        buckets = by_type.get(qt)
        if buckets is None:
            continue
        factual_scores = buckets["factual"]
        counterfactual_scores = buckets["counterfactual"]
        if not factual_scores and not counterfactual_scores:
            continue

        factual_avg = mean(factual_scores) if factual_scores else 0.0
        counterfactual_avg = mean(counterfactual_scores) if counterfactual_scores else 0.0
        reasoning_gap = factual_avg - counterfactual_avg
        epoch_score = 0.4 * factual_avg + 0.6 * counterfactual_avg

        # Compute per-pair EPOCH scores for statistics.
        all_scores = factual_scores + counterfactual_scores
        n_pairs = min(len(factual_scores), len(counterfactual_scores))

        # Per-pair EPOCH scores for median/stddev/CI.
        pair_epoch_scores = [
            0.4 * f + 0.6 * c
            for f, c in zip(factual_scores, counterfactual_scores)
        ]

        std = stdev(pair_epoch_scores) if len(pair_epoch_scores) >= 2 else 0.0
        med = median(pair_epoch_scores) if pair_epoch_scores else 0.0

        # 95% CI using normal approximation.
        n = len(pair_epoch_scores)
        if n >= 2 and std > 0:
            se = std / math.sqrt(n)
            ci_lower = epoch_score - 1.96 * se
            ci_upper = epoch_score + 1.96 * se
        else:
            ci_lower = epoch_score
            ci_upper = epoch_score

        type_scores.append(
            TypeScore(
                question_type=qt,
                factual_score=factual_avg,
                counterfactual_score=counterfactual_avg,
                reasoning_gap=reasoning_gap,
                epoch_score=epoch_score,
                n_pairs=n_pairs,
                std_dev=std,
                median_score=med,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )

    return type_scores


def compute_overall(
    type_scores: list[TypeScore],
) -> tuple[float, float, float, float]:
    """Return (factual, counterfactual, reasoning_gap, epoch_score) averaged across types."""
    if not type_scores:
        return 0.0, 0.0, 0.0, 0.0

    factual = mean(ts.factual_score for ts in type_scores)
    counterfactual = mean(ts.counterfactual_score for ts in type_scores)
    reasoning_gap = mean(ts.reasoning_gap for ts in type_scores)
    epoch_score = mean(ts.epoch_score for ts in type_scores)

    return factual, counterfactual, reasoning_gap, epoch_score


def compute_pair_analysis(results: list[Result]) -> list[dict]:
    """Compute per-pair breakdown: factual score, counterfactual score, gap, EPOCH score."""
    pairs: dict[str, dict[str, float | str]] = {}
    for r in results:
        if r.pair_id not in pairs:
            pairs[r.pair_id] = {
                "pair_id": r.pair_id,
                "question_type": r.question_type.value,
                "factual_score": 0.0,
                "counterfactual_score": 0.0,
            }
        if r.variant == "factual":
            pairs[r.pair_id]["factual_score"] = r.score
        else:
            pairs[r.pair_id]["counterfactual_score"] = r.score

    analysis = []
    for pair in pairs.values():
        f = pair["factual_score"]
        c = pair["counterfactual_score"]
        pair["reasoning_gap"] = f - c
        pair["epoch_score"] = 0.4 * f + 0.6 * c
        analysis.append(pair)

    return sorted(analysis, key=lambda p: p["pair_id"])
