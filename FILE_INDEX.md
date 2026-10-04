# TEAM KHAT files from this chat

All of these are in the same folder. Download `TEAM_KHAT_files.zip` if individual files do not show up.

| File | What it is |
|------|------------|
| EXPERIMENT_TRACKING.md | How runs log config, model SHA, dataset fingerprints |
| eval_tracked.py | Tracked MathNeuro pipeline (pilot script) |
| TEAM_KHAT_Methods_Draft.md | Methods section, plain text |
| TEAM_KHAT_Methods_Draft.docx | Methods section, Word |
| LAYER_DISTRIBUTION.md | Layer summary from the attached top-k masks |
| layer_distribution.csv | Per-layer counts |
| layer_distribution_meta.json | Band totals |
| layer_distribution_summary.xlsx | Per-layer sheet, band formulas, baseline sheet |

Baseline used in the spreadsheet: pure symbolic 65.6%, non-financial 68.8%, financial 53.1%, n = 32, Qwen. Nonsense control not built.

Layer definitions: math-specific = pure top-k and not non-financial top-k. Finance-specific = finance top-k and not non-financial top-k. Shared-calc = all three. Totals: math-specific 384,727; finance-specific 256,335; shared-calc 241,080.
