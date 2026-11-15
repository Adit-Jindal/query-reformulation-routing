# D_stats: confirmatory statistics

Commit: d0fef27a8e4229897380eea6d122b03dc9f6a6dd | seed 0 | bootstrap B=10000

## Hypotheses (fixed before running; see PREREG.md)
- H1 (primary): iteration-3 router > Const on nDCG@10. H2 (primary): router > best-fixed. H3 (descriptive): action - a0 and oracle - a0.
- Primary router: {'retriever': 'dense', 'model': 'gbr', 'protocol': 'within'}. Others (BM25, ridge, lodo) are confirmation.
- Secondary: same contrasts on MAP@100 and Recall@100 (primary router only; exploratory, Holm and BH columns labelled separately).

## Families / corrections
- H1, H2: Holm within (model x protocol x contrast) over 2 retrievers x 5 datasets (10 tests). H3: Holm within contrast over 10 tests.
- `ALL` rows (query-weighted) are descriptive and never in a family. BH (`q_bh_exploratory`) appears only in D2.
- Pooling (D3): macro-average over datasets is PRIMARY; query-weighted ALL and DerSimonian-Laird random effects (Hartung-Knapp) reported too.

## Conventions
- Paired bootstrap over queries, percentile 95% CI, two-sided p, floored at 1/B (a p of 1/B means "<= 1/B"; Holm-adjusted values inherit the floor).
- Wilcoxon signed-rank with Pratt zero handling (all-zero -> p=1). `boot_wilcoxon_disagree` marks p<0.05 disagreements.
- Win/tie/loss: |delta| <= 1e-9 is a tie. Cohen's d = mean(delta)/SD(delta).
- Router values are out-of-fold (nested CV); same seed/folds/repeats as run_router.py (see check_vs_iteration_results.csv).

## Caveats
- Unit of inference is the query: results generalise to new queries from these collections, not to new collections (5 datasets only).
- Pilot results were seen before fixing hypotheses; the held-out sets (G) are the clean test.
- Seed variability uses only 3 LLM seeds; seed-resampling CIs are indicative.
