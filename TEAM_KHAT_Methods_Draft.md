# Methods

This study tests whether mathematical reasoning and financial interpretation rely on distinguishable parameters inside an open-weight language model. We adapt the MathNeuro localization procedure: compute weight–activation importance scores on matched problem sets, isolate parameters that are important for one condition but not another, and use targeted pruning as a causal check. All runs log configuration, model revision, dataset fingerprints, and metrics.

## Overview of the experimental pipeline

1. Construct matched problem quadruplets that share numbers, operation, and answer, and differ only in domain framing.
2. Measure baseline accuracy on each condition with the unmodified model.
3. Compute importance scores from forward passes and form isolated parameter masks.
4. Prune those masks (and size-matched random masks) and re-evaluate accuracy.

A model is used for mechanistic analysis only if baseline accuracy on the simplified finance subset is high enough that pruning drops can be interpreted.

## Matched dataset

### Source and scope

Financial items are adapted from FinanceMath (and related formula banks) into a small controlled set. The pilot uses only simple operations: compound growth, percentages and profit margins, present value, and ratios. Table-heavy or multi-knowledge FinanceMath items are deferred. Pure-math items follow the same numerical skeleton.

### Four matched conditions

| Condition | Held constant | What changes |
|-----------|---------------|--------------|
| Pure / symbolic math | Numbers, operation, answer | No narrative; equation or bare calculation |
| Financial word problem | Numbers, operation, answer | Must recognize a finance concept or formula |
| Non-financial word problem | Numbers, operation, answer | Everyday narrative, same calculation |
| Nonsense / control | Numbers, token-length band, numeric format | Meaningful finance or everyday context removed |

**Control construction (working rule):** replace interpretable quantity names with nonsense labels while keeping the same numerals (mentor constraint: nonsense labels instead of worded financial quantities). Do not introduce a second coherent domain. Apply the same token-count and number-format rules as the other word-problem conditions.

### Token-count and number-format controls

- Match conditions using the **model tokenizer**, not word count.
- Keep finance, non-finance, and control prompts in a tight token band; log leftover gaps.
- Symbolic math may be shorter; report that gap.
- Do not mix `5000`, `5,000`, and `five thousand` inside one quadruplet.
- Each item record stores: condition, token count, numerical format, operation/formula, correct answer.

### Item log and paraphrases

Pilot scale is tens of hand-checked quadruplets. Optional paraphrases of finance and non-finance stems (same numbers and answer) are reserved for robustness checks.

## Model and inference setup

- **Pilot model:** `Qwen/Qwen2.5-0.5B-Instruct`, float16, `device_map="auto"`.
- **Upgrade rule:** if finance baseline on the simplified subset is too low, rerun the same pipeline on a larger open-weight Instruct model in the same family.
- Closed models are not used (weights must be hookable and writable).
- Evaluation decoding: greedy, `max_new_tokens=200`, left padding; pad token = EOS if missing.
- One seed for Python, PyTorch, CUDA, and dataset shuffle; cuDNN deterministic mode on for the pilot.
- Record Hugging Face model id and Hub commit SHA.

## Baseline evaluation

Unmodified model, separately per condition. Prompt template: `Question: …\nAnswer:`. Parse the last numeric span in the generation; compare to gold after comma stripping (GSM8K gold = number after `####`). Accuracy = match rate.

Gate: if almost no finance items are solved, simplify the subset before any pruning study. Keep the slice where pure math is correct and finance is incorrect.

## Parameter localization (MathNeuro)

**Target modules:** names ending in `mlp.down_proj`.

**Importance:** on each forward pass, accumulate attention-masked sum of squared activations over batch and sequence. Importance = `|W|` × √(accumulated squared activations). Default calibration: up to 200 prompts/condition, `max_length=512`, batch size 8.

**Isolation:** top-k mask with `k = max(1, floor(0.01 × numel))`. Isolated A-not-B mask = top-k(A) AND NOT top-k(B).

Primary masks:

- Math-shared candidates: important for pure math, not for general language (WikiText-style).
- Finance-associated candidates: important for finance, not for matched non-finance and/or nonsense control.
- Shared calculation candidates: in the top-k set for pure math, finance, and non-finance.

Overlap is compared with a null distribution of random sets of the same size (and similar layer allocation when possible); target 1,000 draws.

## Causal check by pruning

Zero the masked weights (factor 0), evaluate, restore from cache. Random-mask control uses the same parameter count (pilot: 3 trials).

- Math-isolated prune hurts all matched conditions → shared calculation.
- Finance-isolated prune hurts finance much more than pure math / non-finance → finance interpretation.
- Random prune of the same count hurts about as much → do not treat the mask as specific.

## Additional analyses

Layer location of finance vs math masks; paraphrase robustness; tabulated token-length gaps.

## Metrics

- Per-condition baseline accuracy
- Accuracy after targeted prune
- Accuracy after random prune
- Targeted-minus-random drop
- Finance vs non-finance prune drop
- Parameter-set overlap vs null
- Token-count gaps across matched conditions

## Reproducibility

Each run writes config, environment, model SHA, dataset fingerprints, prompt-list hashes, token-length summaries, results, and isolated masks (`.pt`). Optional W&B logging records the same fields.

## Method limitations

Pruning shows necessity for a drop, not a one-weight-one-concept map. Residual wording differences can still move scores. A 0.5B pilot may not generalize. Exact token equality between equations and word problems is not always possible; gaps are minimized and reported.

---

*Draft status: follows the TEAM KHAT proposal and the tracked eval pipeline. Replace the control template with the exact wording locked at the mentor checkpoint before calling the dataset final in a submission.*
