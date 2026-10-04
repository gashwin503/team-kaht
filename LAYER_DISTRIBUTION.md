# Layer-distribution summary

Source: attached top-k masks for pure symbolic, financial, and non-financial conditions. 24 modules, all `model.layers.{i}.mlp.down_proj`, shape 896 × 4864 (4,358,144 parameters each). These are the saved top-k sets, not a new forward pass. The nonsense control is not in these files, so finance-specific here is finance top-k minus non-financial top-k only.

Definitions used:

- Math-specific = pure-symbolic top-k and not non-financial top-k.
- Finance-specific = financial top-k and not non-financial top-k.
- Shared-calc = intersection of all three top-k sets.
- Math ∩ finance = pure-symbolic top-k and financial top-k (non-financial not required).

Top-k size is not uniform across layers (about 15k–36k per layer; totals 676,452 math, 624,103 finance, 605,929 non-financial). Do not treat each layer as exactly 1% of that layer.

## Depth bands

| Band | Layers | Math top-k | Finance top-k | Math-specific | Finance-specific | Shared-calc | Finance-specific / math-specific |
|------|--------|------------|---------------|---------------|------------------|-------------|------------------------------|
| Early | 0–7 | 243,832 | 228,642 | 127,265 | 88,603 | 99,244 | 0.70 |
| Middle | 8–15 | 201,577 | 180,246 | 122,545 | 65,589 | 65,524 | 0.54 |
| Late | 16–23 | 231,043 | 215,215 | 134,917 | 102,143 | 76,312 | 0.76 |
| Total | 0–23 | 676,452 | 624,103 | 384,727 | 256,335 | 241,080 | 0.67 |

## How to read it

- Math-specific weights are spread across depth, not one block. Early 127k, middle 123k, late 135k. That matches the shape MathNeuro reported for math-vs-language (distributed), with the caveat that the control here is non-financial word problems, not WikiText.
- Finance-specific weights are also distributed. They are smallest in the middle (66k) and largest late (102k). This is not a clean “finance only in the last layers” result. Late layers hold the most finance-not-nonfinance parameters, but early layers are close.
- About 41% of the finance top-k is outside the non-financial top-k (256,335 / 624,103). About 57% of the math top-k is outside the non-financial top-k (384,727 / 676,452). So both conditions have a large condition-private set, and math’s private set is bigger.
- Shared-calc is large: 241,080 parameters sit in all three top-k sets. That is the best current evidence for a shared calculation pool. It is densest early (99k) and still present late (76k).
- Math ∩ finance is 292,058. Of the finance top-k, 332,045 parameters are not in the math top-k. Overlap is real, but most of each top-k is not shared with the other.

## Per-layer finance-specific counts (high / low)

Highest finance-specific layers: 21 (18,074), 5 (17,325), 18 (16,399), 6 (14,354), 20 (13,728). Lowest: 23 (5,348), 2 (5,392), 9 (6,234), 13 (6,251), 12 (6,758). Layer 23 is also the smallest top-k overall (math 14,738, finance 13,212).

## What this does not show

- No nonsense-number control, so some “finance-specific” weights may be word-problem or number-context weights rather than finance interpretation.
- No null overlap distribution yet. Shared-calc of 241k should be compared with random sets of the same size and layer mix before calling it above chance.
- No pruning. Location is not a causal result. Baseline still stands: pure 65.6%, non-financial 68.8%, financial 53.1%, n = 32.

Per-layer counts are in `layer_distribution.csv`.
