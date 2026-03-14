"""Pydantic models for EPOCH benchmark."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    CHAIN = "CHAIN"
    GATE = "GATE"
    RIPPLE = "RIPPLE"
    BRIDGE = "BRIDGE"


class Question(BaseModel):
    """A single benchmark question."""

    id: str
    type: QuestionType
    variant: Literal["factual", "counterfactual"]
    pair_id: str = Field(description="Shared ID linking factual/counterfactual twins")
    prompt: str
    # CHAIN: list of items in correct dependency order
    # GATE: "yes" or "no"
    # RIPPLE: list of affected technologies
    # BRIDGE: correct option letter (A/B/C/D)
    answer: str | list[str]
    choices: list[str] | None = Field(
        default=None, description="Multiple-choice options for BRIDGE questions"
    )
    difficulty: int | None = Field(
        default=None, ge=1, le=5, description="Difficulty rating 1-5"
    )
    domains: list[str] | None = Field(
        default=None, description="Domain tags (e.g. ['computing', 'hardware'])"
    )
    source: str | None = Field(
        default=None, description="Source citation for the question"
    )


class Result(BaseModel):
    """Result from evaluating a single question."""

    question_id: str
    question_type: QuestionType
    variant: Literal["factual", "counterfactual"]
    pair_id: str
    model_response: str
    parsed_answer: str | list[str]
    expected_answer: str | list[str]
    score: float = Field(ge=0.0, le=1.0)
    latency_ms: float | None = Field(default=None, description="API call latency in ms")
    error: str | None = Field(default=None, description="Error message if evaluation failed")


class TypeScore(BaseModel):
    """Aggregated score for one question type."""

    question_type: QuestionType
    factual_score: float
    counterfactual_score: float
    reasoning_gap: float = Field(description="factual - counterfactual")
    epoch_score: float = Field(description="0.4 * factual + 0.6 * counterfactual")
    n_pairs: int
    std_dev: float | None = Field(default=None, description="Standard deviation of scores")
    median_score: float | None = Field(default=None, description="Median EPOCH score")
    ci_lower: float | None = Field(default=None, description="95% CI lower bound")
    ci_upper: float | None = Field(default=None, description="95% CI upper bound")


class BenchmarkResult(BaseModel):
    """Full benchmark run result."""

    model: str
    provider: str
    results: list[Result]
    type_scores: list[TypeScore]
    overall_factual: float
    overall_counterfactual: float
    overall_reasoning_gap: float
    overall_epoch_score: float
    evaluation_duration_seconds: float | None = Field(
        default=None, description="Total wall-clock evaluation time"
    )
    total_questions: int | None = Field(default=None, description="Total questions evaluated")
    failed_questions: int | None = Field(default=None, description="Questions that errored")
