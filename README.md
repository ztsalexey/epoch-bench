# EPOCH Benchmark

**Evaluating Progress Origins in Causal History**

A benchmark for testing LLM causal reasoning about technology dependencies. Every question has a factual + counterfactual twin — the delta between them measures how much a model relies on memorization vs. genuine causal reasoning.

## Leaderboard

320 questions, 12 models. Ranked by EPOCH Score (0.4 × Factual + 0.6 × Counterfactual).

| # | Model | EPOCH | Factual | CF | Gap |
|---|-------|------:|--------:|---:|----:|
| 1 | **claude-sonnet-4-6** | **73.5%** | 85.1% | **65.8%** | 19.3% |
| 2 | claude-opus-4-6 | 73.2% | 88.8% | 62.8% | 26.0% |
| 3 | gpt-4.1-mini | 71.0% | 82.2% | 63.6% | 18.6% |
| 4 | o3-mini | 71.0% | 86.6% | 60.6% | 26.0% |
| 5 | o4-mini | 70.2% | 84.3% | 60.7% | 23.6% |
| 6 | gpt-4.1 | 70.1% | 87.2% | 58.7% | 28.5% |
| 7 | gpt-4o | 69.9% | **89.7%** | 56.7% | 33.0% |
| 8 | gpt-4o-mini | 69.6% | 80.4% | 62.4% | **18.0%** |
| 9 | gpt-4-turbo | 68.5% | 86.6% | 56.4% | 30.2% |
| 10 | claude-haiku-4.5 | 66.3% | 85.5% | 53.6% | 31.9% |
| 11 | gpt-4.1-nano | 64.7% | 80.2% | 54.3% | 25.9% |
| 12 | gpt-3.5-turbo | 58.0% | 78.2% | 44.6% | 33.5% |

### Key Findings

- **Factual score != reasoning ability.** GPT-4o has the highest factual score (89.7%) but ranks 7th overall — its counterfactual reasoning collapses.
- **Scaling barely closes the gap** (slope=-0.005, p=0.40). Bigger models memorize more but don't proportionally reason better.
- **RIPPLE is the universal weakness.** Every model has a 46-60% gap on ripple-effect questions. No model can reason about counterfactual cascading effects.
- **Counterfactual framing helps GATE reasoning.** 7 of 12 models score *better* on counterfactual GATE questions than factual ones — alternative-world framing may actually aid prerequisite reasoning.
- **Mini models punch above their weight.** gpt-4.1-mini and gpt-4o-mini have the smallest gaps (~18%) — less memorization means more genuine reasoning.

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

### Scoring Notes

**RIPPLE caveat:** Counterfactual RIPPLE answers are shorter on average (1.4 items vs 3.5 for factual). F1 scoring on shorter lists is inherently harder — models tend to over-predict, tanking precision. The 46-60% gap is partly a genuine reasoning failure and partly a scoring artifact from asymmetric list lengths.

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
