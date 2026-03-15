# EPOCH Benchmark

**Evaluating Progress Origins in Causal History**

A benchmark for testing LLM causal reasoning about technology dependencies. Every question has a factual + counterfactual twin — the delta between them measures how much a model relies on memorization vs. genuine causal reasoning.

## Leaderboard

320 questions, 12 models. Ranked by EPOCH Score (0.4 × Factual + 0.6 × Counterfactual).

| # | Model | EPOCH | Factual | CF | Gap |
|---|-------|------:|--------:|---:|----:|
| 1 | **claude-sonnet-4-6** | **76.9%** | 87.0% | **70.1%** | 16.8% |
| 2 | claude-opus-4-6 | 76.6% | **90.7%** | 67.2% | 23.5% |
| 3 | o3-mini | 74.4% | 88.5% | 65.0% | 23.5% |
| 4 | gpt-4.1-mini | 74.1% | 83.4% | 68.0% | **15.5%** |
| 5 | o4-mini | 73.9% | 86.2% | 65.7% | 20.5% |
| 6 | gpt-4.1 | 73.5% | 89.1% | 63.1% | 26.0% |
| 7 | gpt-4o | 72.8% | 90.3% | 61.0% | 29.3% |
| 8 | gpt-4o-mini | 72.6% | 82.3% | 66.1% | 16.1% |
| 9 | gpt-4-turbo | 71.5% | 88.5% | 60.1% | 28.3% |
| 10 | claude-haiku-4.5 | 69.7% | 87.4% | 58.0% | 29.4% |
| 11 | gpt-4.1-nano | 67.4% | 81.5% | 58.1% | 23.4% |
| 12 | gpt-3.5-turbo | 61.1% | 79.4% | 49.0% | 30.4% |

### Key Findings

- **Factual score != reasoning ability.** GPT-4o has the highest factual score (90.3%) but ranks 7th overall — its counterfactual reasoning collapses.
- **Scaling does not significantly close the gap** (slope=-0.004, p=0.44). Bigger models memorize more but don't proportionally reason better.
- **GATE and BRIDGE show genuine counterfactual reasoning.** Models massively outperform a "copy factual answer" baseline on these types (+72% and +62% margins). But on CHAIN and RIPPLE, models fail to beat the baseline — their CF answers are no better than just repeating factual knowledge.
- **Mini models punch above their weight.** gpt-4.1-mini and gpt-4o-mini have the smallest gaps (~16%) — less memorization means more genuine reasoning.
- **Difficulty-adjusted contamination** separates memorization from inherent difficulty. After adjustment, gpt-3.5-turbo shows the strongest contamination signal (0.173), while gpt-4.1-mini is cleanest (0.035).

![Reasoning Gap](figures/reasoning_gap.png)

## Question Types

| Type | Task | Scoring |
|------|------|---------|
| **CHAIN** | Order N technologies by dependency | Kendall's tau |
| **GATE** | Is X achievable given constraints? (Yes/No) | Accuracy |
| **RIPPLE** | If X never existed, what breaks? | F1 |
| **BRIDGE** | Missing step A → ? → C (multiple choice) | Accuracy |

## Scoring

- **EPOCH Score** = 0.4 × Factual + 0.6 × Counterfactual (0–100)
- **Reasoning Gap** = Factual − Counterfactual

A high Reasoning Gap means the model scores well on real-world facts but poorly on counterfactuals — suggesting memorization over causal understanding. The EPOCH Score weights counterfactual performance higher because it better reflects genuine reasoning ability.

### Scoring Notes and Known Limitations

**Copy-factual baseline:** We compare model CF scores against a naive strategy that copies the factual answer to CF questions. Models genuinely reason on GATE (+72% margin) and BRIDGE (+62%), but CHAIN and RIPPLE CF scores don't beat this baseline — indicating those CF components primarily reflect factual knowledge leaking through rather than counterfactual reasoning. This is an active area of improvement.

**RIPPLE F1 asymmetry:** CF RIPPLE answers average 1.4 items vs 3.5 for factual. Models over-predict (listing factual-world answers instead of reasoning about the alternative), resulting in low precision.

**GATE flip pattern:** 90% of GATE pairs flip the answer between variants. A model that memorizes factual answers and flips would score 90% on CF. Despite this, models show nuanced patterns — 7 of 12 score better on CF GATE, suggesting the alternative-world framing genuinely aids prerequisite reasoning.

## Setup

```bash
pip install -e .
```

Requires Python 3.10+. Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-...
# or
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Run full benchmark
epoch-bench run --provider anthropic --model claude-sonnet-4-6

# Run specific question types
epoch-bench run --provider openai --model gpt-4o --types CHAIN --types GATE

# Save results to file
epoch-bench run --provider anthropic --model claude-sonnet-4-6 --output results.json

# Compare two models
epoch-bench compare results_a.json results_b.json

# Statistical analysis
epoch-bench analyze results_a.json results_b.json --output analysis.json

# Generate leaderboard
epoch-bench leaderboard results_a.json results_b.json

# Generate figures
epoch-bench figures results/*.json --output-dir figures/
```

## Dataset

320 hand-crafted questions across 4 JSONL files in `epoch_bench/data/`:

- `chain.jsonl` — 40 factual/counterfactual pairs (80 questions)
- `gate.jsonl` — 40 pairs (80 questions)
- `ripple.jsonl` — 40 pairs (80 questions)
- `bridge.jsonl` — 40 pairs (80 questions)

## Novel Features

### Technology Dependency Graph

Extracts a directed acyclic graph (~330 nodes, ~287 edges) from the 320 hand-crafted questions, then procedurally generates unlimited new factual/counterfactual question pairs from it.

```bash
# Build the graph
epoch-bench graph --rebuild --stats

# Generate new question pairs
epoch-bench generate --n-per-type 10 --pairs --output generated.jsonl --seed 42
```

### Contamination Analysis

Reframes the factual/counterfactual gap as a training data contamination detector. Counterfactual questions are structurally immune to contamination — if a model scores much higher on factual than counterfactual variants, it's likely memorizing training data rather than reasoning.

```bash
epoch-bench contamination results/*.json --threshold 0.3
```

![Contamination Heatmap](figures/contamination_heatmap.png)

### Scaling Analysis

Analyzes whether increased model capability closes the reasoning gap. Groups models by family, regresses gap vs. capability rank, and classifies per-family trends.

```bash
epoch-bench scaling results/*.json
```

![Scaling](figures/scaling_gap.png)

## Project Structure

```
epoch_bench/
├── cli.py              # CLI entry point (13 commands)
├── schema.py           # Pydantic data models
├── evaluate.py         # Scoring logic (Kendall's tau, accuracy, F1)
├── analysis.py         # Statistical analysis (gap significance, correlations)
├── graph.py            # Dependency graph + procedural question generator
├── contamination.py    # Contamination detection via F/CF gap
├── scaling.py          # Scaling analysis (gap vs capability)
├── figures.py          # Publication-quality matplotlib figures
├── runner.py           # Benchmark orchestration
├── report.py           # Rich console output
├── prompts.py          # Structured prompt templates
├── suite.py            # Multi-model suite runner
├── leaderboard.py      # Leaderboard generation
├── validation.py       # Expert review export/import
├── human_baseline.py   # Interactive human baseline
├── models/
│   ├── base.py               # Abstract ModelProvider
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   ├── gemini_provider.py
│   └── deepseek_provider.py
└── data/
    ├── chain.jsonl
    ├── gate.jsonl
    ├── ripple.jsonl
    ├── bridge.jsonl
    └── tech_aliases.json  # Technology name normalization
```

## License

MIT
