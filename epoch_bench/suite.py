"""Suite runner: run benchmark across multiple models from YAML config."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from epoch_bench.models import get_provider
from epoch_bench.runner import run_benchmark
from epoch_bench.schema import BenchmarkResult, QuestionType


@dataclass
class SuiteModelConfig:
    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    concurrency: int = 5


@dataclass
class SuiteConfig:
    models: list[SuiteModelConfig]
    types: list[str] | None = None
    output_dir: str = "results"
    verbose: bool = False


def load_suite_config(path: str | Path) -> SuiteConfig:
    """Load and validate suite configuration from YAML."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Suite config must be a YAML mapping")

    if "models" not in raw or not raw["models"]:
        raise ValueError("Suite config must have a non-empty 'models' list")

    models = []
    for m in raw["models"]:
        if not isinstance(m, dict):
            raise ValueError(f"Each model entry must be a mapping, got {type(m).__name__}")
        if "provider" not in m or "model" not in m:
            raise ValueError("Each model entry must have 'provider' and 'model' keys")
        models.append(
            SuiteModelConfig(
                provider=m["provider"],
                model=m["model"],
                temperature=m.get("temperature"),
                max_tokens=m.get("max_tokens"),
                concurrency=m.get("concurrency", 5),
            )
        )

    types = raw.get("types")
    if types is not None:
        for t in types:
            QuestionType(t)  # validate

    return SuiteConfig(
        models=models,
        types=types,
        output_dir=raw.get("output_dir", "results"),
        verbose=raw.get("verbose", False),
    )


async def run_suite(config: SuiteConfig) -> list[BenchmarkResult]:
    """Run benchmark sequentially for each model in the suite config."""
    from rich.console import Console

    console = Console()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qt_filter = [QuestionType(t) for t in config.types] if config.types else None
    all_results: list[BenchmarkResult] = []

    for i, mc in enumerate(config.models, start=1):
        console.print(f"\n[bold]Running model {i}/{len(config.models)}: {mc.model}[/bold]")
        console.print(f"  Provider: {mc.provider}")

        provider = get_provider(mc.provider, mc.model)
        result = await run_benchmark(
            provider,
            qt_filter,
            mc.concurrency,
            verbose=config.verbose,
            temperature=mc.temperature,
            max_tokens=mc.max_tokens,
        )

        # Save per-model results
        safe_name = mc.model.replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"results_{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        console.print(f"  Saved to {out_path}")

        all_results.append(result)

    return all_results
