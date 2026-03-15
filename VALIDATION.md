# Question Validation Methodology

## Hand-Crafted Questions (340)

All 340 questions were hand-crafted following this process:

1. **Domain expert authoring**: Questions written by researchers with domain knowledge in computing, electronics, energy, medicine, and related fields.

2. **Structural validation**: Every question verified to have:
   - A valid factual/counterfactual pair sharing the same `pair_id`
   - Correct schema fields (type, variant, answer format, difficulty 1-5, domain tags)
   - Counterfactual premises that describe plausible alternative histories

3. **12-model consensus validation**: All 340 questions evaluated against 12 diverse LLMs spanning 3 generations (GPT-3.5 through Claude Opus 4.6). Questions where all 12 models unanimously disagreed with the gold answer were flagged and corrected. This identified and fixed 11 incorrect gold answers across multiple review rounds.

4. **Copy-factual baseline check**: CHAIN counterfactual questions verified to use different item sets (20/40 pairs) so that copying the factual answer does not score well. Baseline drops from ~80% (same items) to ~40% (different items).

5. **Non-flipping GATE pairs**: 16 of 50 GATE pairs intentionally have the same answer for both factual and counterfactual variants, preventing the "always flip" exploit.

## Procedurally Generated Questions

Generated questions (`epoch_bench/data/generated.jsonl`) are produced from the technology dependency graph:

1. **Graph extraction**: ~330 nodes and ~290 edges extracted from factual questions
2. **Structural correctness**: CHAIN answers verified in topological order, GATE answers verified against graph ancestry, RIPPLE affected sets verified as descendants
3. **Deduplication**: No duplicate prompts within a batch
4. **Seed reproducibility**: Same seed produces identical questions

## Known Limitations

- **No formal inter-annotator agreement metric**: Questions were authored by a small team, not independently annotated by multiple experts. The 12-model consensus serves as a proxy for answer quality.
- **RIPPLE CF precision issue**: Models consistently over-predict on counterfactual RIPPLE questions (high recall ~70%, low precision ~30%), suggesting they apply factual knowledge instead of reasoning about the alternative timeline.
- **Domain imbalance**: "computing" covers 26% of questions. Some domains (nuclear, quantum mechanics) have only 1-2 questions.
- **English-only**: All questions concern primarily Western technology history.
