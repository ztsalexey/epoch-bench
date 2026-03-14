# Contributing to EPOCH Benchmark

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/ztsalexey/epoch-bench.git
cd epoch-bench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,paper]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Adding Questions

Questions live in `epoch_bench/data/*.jsonl`. Every question must have a factual/counterfactual twin sharing the same `pair_id`. See existing questions for the schema.

You can also generate questions procedurally:

```bash
epoch-bench generate --n-per-type 10 --pairs --output new_questions.jsonl
```

## Adding Model Providers

1. Create a new provider in `epoch_bench/models/` inheriting from `ModelProvider`
2. Register it in `epoch_bench/models/__init__.py`
3. Add the dependency to `pyproject.toml`

## Code Style

- Type hints on all function signatures
- Docstrings on public functions
- Tests for new functionality
