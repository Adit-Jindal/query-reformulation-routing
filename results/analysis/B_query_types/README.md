# B_query_types - README

Exploratory query-type analysis (Workstream B). Results are HYPOTHESES, not confirmed effects.

## Settings (also in B_run_manifest.json)
{"seed": 0, "B": 5000, "eps": 0.05, "min_n": 15, "min_ds_n": 100, "depth": 3, "leaf": 20, "n_perm": 50, "auc_boot": 500}

## Definitions
- gain = seed-mean metric(action) - metric(a0), nDCG@10 (a0/PRF seedless). Actions: prf, query2doc, hyde, decomp.
- oracle_best = argmax of seed-mean metric over all 5 actions (ties -> a0, then action order); oracle_go = oracle_best != a0.
  oracle_value = cross-fitted oracle value (routing.py). 
- helped/harmed: gain > +eps / < -eps (eps = 0.05); too_small: n < 15 (mean shown, no CI/p, best_action = 'too small', never ranked).
- Buckets: terciles within dataset (qcut with duplicate-edge dropping: heavily tied descriptors can give 2 buckets or be skipped).
  Within-dataset B1 reporting only for datasets with n >= 100; smaller datasets contribute to POOLED only.
  'low' for a descriptor means low relative to its own dataset.
- Descriptors: s0_* known before retrieval; s1_* after a0 retrieval (retriever-specific: s1_top1, s1_gap1_10, s1_wig, s1_nqc, s1_entropy100; s1_overlap10 = BM25/dense top-10 overlap). form = rule-based claim/question/keyword (see below); has_digit; starts_qword.
- a0_band (POST-HOC, not a router feature): zero (<=0.01), low (<=0.30), mid (<=0.70), high (>0.70) of a0 nDCG@10. Mechanical effect: when a0 is near 1 gain can only be <= 0, and low-a0 queries regress upward; read B5 with that in mind.
- Multiple testing: Benjamini-Hochberg FDR over all non-too-small cells of B1 (1056 tests), column q_bh. B2 coefficients: BH over all terms x actions per retriever (q_bh). Interaction tests: BH over all (q_bh).

## Files
- B0_query_descriptors.csv: one row per (retriever, query): raw descriptors, b_<descriptor> bucket labels, g_<action> gains, a0_ndcg, a0_band, oracle_best, oracle_go, oracle_value.
- B0_form_check_sample.csv: 30 queries/dataset for hand-checking the form rule (fill hand_label, re-run).
- B1_bucket_gains.csv: retriever, dataset (POOLED = all datasets), descriptor, bucket, action, n, mean_gain, ci_lo/ci_hi (95% bootstrap), p_boot, q_bh, cohens_d, helped, harmed, best_action (in bucket), too_small.
- B2_regression_coefs.csv: per retriever x action: gain ~ z-scored descriptors (+ dataset fixed effects). coef_per_1sd (nDCG per SD), std_beta (= coef / SD of gain), bootstrap CI (resampling queries), p_boot, q_bh; term=_R2 rows give R^2. B2_interaction_tests.csv: F-test of descriptor x dataset interactions. B2_vif.csv: collinearity.
- B3_tree_rules.txt, B3_tree_accuracy.csv: depth<=3 decision tree on oracle_best (s0 + s1 features, z-scored per dataset). protocols: kfold (5-fold x3, each query its own group) and lodo (leave one dataset out); heldout rows per dataset for lodo. majority_acc uses the train-fold majority; null = labels shuffled in train folds (50 perms); p_vs_null empirical.
- B4_go_nogo.csv: same for oracle_go (tree with AUC) + univariate AUC of each feature for oracle_go (kind column).
- B5_a0_strength.csv: gain by a0 band. B6_length_vs_action.csv: query2doc vs hyde (and their difference) by length tercile, absolute length bin, and Spearman(length, gain) (rows kind=spearman_len_vs_gain: rho in mean_gain, p in p_boot).
- B7_consistency.csv: per (descriptor, bucket, action) sign agreement across retrievers and datasets (non-too-small dataset cells). consistent = same sign in both retrievers in >=2 datasets; dataset_specific = agreement in 1 dataset; else inconsistent.
- B_summary.json, B_checks (inside summary), B_run_manifest.json (md5 of outputs; compare two runs for identical output).
- Figures B_*.png/.pdf, LaTeX tables B_tab_*.tex.

## Caveats
Exploratory, many comparisons; descriptors correlate (see VIF) so single-feature effects are confounded; a0 band is post-hoc; buckets are relative within dataset; gains are noisy because of LLM seeds (noise SDs in B_summary.json); TREC-COVID and Touche (~50 queries) only enter POOLED; form rule is heuristic (validation in B_summary.json); git commit: d0fef27a8e4229897380eea6d122b03dc9f6a6dd.
