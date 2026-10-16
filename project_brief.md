# Project 1 — Query-Adaptive Reformulation for Retrieval

**Course:** COL764 Information Retrieval and Web Search  
**Status:** Tentative research dossier; final scope follows a two-week feasibility pilot.  
**Required background:** Solid classical IR and evaluation, basic learning-to-rank/classification, and enough Python/Pyserini or equivalent experience to run BM25 and dense-retrieval experiments; prior RL is helpful but not required.

## Research question

Given a query (q), choose one action from a small menu—no reformulation, pseudo-relevance feedback (PRF), an LLM expansion, a hypothetical document, or decomposition—so as to improve retrieval while paying for latency and avoiding query drift. The publishable version is **not** “predict query difficulty and reformulate hard queries.” It is:

> Can a cost-sensitive policy estimate the *query-level treatment effect* of alternative reformulations, select an action that is better than the unreformulated query, and transfer across collections and retrievers?

For action a, define

$$
\tau_a(q)=M(q,a)-M(q,a_0),
$$

where M is a retrieval metric and $a_0$ is “do nothing.” A policy selects

$$
\pi(q)=\arg\max_a \mathbb{E}[\tau_a(q)\mid x(q)]-\lambda C(a),
$$

with features x(q) available before or after one inexpensive retrieval and cost C(a) measured in latency, calls, or generated tokens.

## Why this is interesting

Query-performance prediction (QPP) asks whether a query is likely to perform poorly. That is not the same as asking whether a particular intervention will help: a hard query may be harmed by expansion, while an apparently easy query may receive a reliable recall gain. The project connects three established IR problems—QPP, selective query expansion, and retrieval configuration selection—but makes the decision counterfactual and cost-aware.

The most compelling research contribution would establish at least one of the following:

- treatment-effect features predict *gain over no reformulation* better than ordinary QPP predicts absolute effectiveness;
- a safe policy reduces harmful reformulations and improves a quality–cost Pareto frontier;
- policies trained on one collection or retriever transfer after lightweight calibration;
- uncertainty-aware abstention is more reliable than forced multi-way routing.

Simply training a classifier on the identity of the best arm is weak: per-query labels are noisy, action margins differ, and ties are common. Prefer regression over gains, pairwise action preferences, or distributional prediction with an explicit abstain/no-op option.

## Recommended experimental design

### Actions

Start with five reproducible actions:

1. original query;
2. RM3 or another classical PRF method;
3. Query2doc-style generated expansion;
4. HyDE-style hypothetical-document retrieval;
5. LLM decomposition followed by reciprocal-rank fusion.

Cache all generated text. Use a frozen BM25 and one frozen dense retriever. Train only a small MLP, gradient-boosted tree, or BERT-sized router. A contextual-bandit extension is appropriate only after an offline supervised policy is working.

### Benchmarks

- **Primary:** TREC Deep Learning 2019/2020 and three heterogeneous BEIR collections such as SciFact, FiQA, and TREC-COVID.
- **Transfer:** train/calibrate on one set and test on held-out domains.
- **Optional stress test:** BRIGHT, if decomposition is included and compute remains manageable.

Use nDCG@10 as the principal metric and Recall@100 as a diagnostic. Report mean/p95 latency, generated tokens, no-op rate, harmful-action rate, and policy regret against the per-query oracle. Use paired randomization/bootstrap tests and confidence intervals over queries.

### Baselines and ablations

- BM25/dense retrieval without reformulation;
- each action applied to every query;
- best global action selected on validation data;
- a QPP-threshold rule (“reformulate hard queries”);
- RRF or score fusion over all actions;
- oracle action, with both single-run and seed-stable definitions.

Ablate pre-retrieval versus post-retrieval features, cost penalty, action set, gain versus best-arm labels, uncertainty threshold, and generator seed. Test whether collection identifiers or score scales leak the domain into the router.

## Closely related IR literature

- Carmel and Yom-Tov, [Estimating the Query Difficulty for Information Retrieval](https://www.yom-tov.info/Papers/SIGIR2005.pdf), SIGIR 2005 — classical foundation for QPP.
- Cronen-Townsend et al., [Predicting Query Performance](https://ciir-publications.cs.umass.edu/getpdf.php?id=521), SIGIR 2002, and the later selective-QE literature — establishes that difficult queries and risky feedback can be detected imperfectly.
- [On the Limitations of Query Performance Prediction for Selective Query Processing](https://dl.acm.org/doi/10.1145/3774427), TOIS 2026 — especially important negative evidence: conventional QPP can yield only marginal selective-processing gains.
- Culpepper et al., [Adaptive Query Processing in Information Retrieval](https://dl.acm.org/doi/10.1145/3231937), TOIS 2018 — selecting retrieval configurations per query.
- [Selective Query Processing for Information Retrieval](https://dl.acm.org/doi/10.1145/3608474), TOIS 2024 — a direct modern comparator for selective execution.
- Gao et al., [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://aclanthology.org/2023.acl-long.99/), ACL 2023 (HyDE).
- Wang et al., [Query2doc: Query Expansion with Large Language Models](https://aclanthology.org/2023.emnlp-main.585/), EMNLP 2023.
- Feng et al., [Synergistic Interplay between Search and Large Language Models for Information Retrieval](https://aclanthology.org/2024.acl-long.517/), ACL 2024.
- [MAAQR: Multi-Agent Adaptive Query Reformulation](https://dl.acm.org/doi/10.1145/3726302.3731950), SIGIR 2025 — makes it essential to emphasize controlled action selection, treatment effects, and cost rather than generic multi-agent reformulation.

## Strongest likely criticisms and responses

| Criticism | Why it is serious | How to address it |
|---|---|---|
| “This is classical selective QE with an LLM action.” | Routing reformulations is not new. | Make treatment-effect estimation, safety/no-op behavior, cost, and cross-domain transfer the stated contribution; compare directly with QPP and adaptive-configuration baselines. |
| “Difficulty does not identify who benefits.” | Absolute QPP may correlate poorly with intervention gain. | Predict per-action gain or preference relative to no-op; report that correlation against standard QPP and include hard/easy × help/harm quadrants. |
| “The oracle gap is an artifact of noisy qrels or random generation.” | A per-query maximum exaggerates headroom. | Repeat stochastic actions, use confidence-aware or seed-stable oracle labels, quantify margins, and evaluate densely judged topics separately. |
| “A global fusion is simpler and stronger.” | RRF may collect most benefits without routing. | Include cost-matched fusion and measure duplicate retrieval/generation cost; the policy must beat or dominate it on a Pareto curve. |
| “The router recognizes datasets, not query needs.” | BEIR domains have obvious lexical signatures. | Leave-one-domain-out evaluation, score normalization, adversarial removal of domain features, and within-domain analysis. |
| “Modern dense retrieval leaves no reformulation headroom.” | A stronger first-stage system could erase gains. | Use one competitive frozen dense retriever, stratify by baseline strength, and frame negative findings as conditions under which reformulation is unnecessary. |
| “Training requires the qrels it is supposed to avoid.” | Offline best-action labels depend on judgments. | Be explicit that qrels train a reusable policy; evaluate transfer to unseen domains and test weak/self-supervised proxies separately. |
| “Latency savings are irrelevant if the LLM call dominates.” | A router only matters if it can avoid expensive calls. | Measure end-to-end latency and tokens, cache only for reproducibility, and include a no-LLM PRF arm and a strict budget analysis. |

## Principal risks and mitigation

| Risk | Early warning | Mitigation |
|---|---|---|
| One action wins almost everywhere. | Best global arm is oracle-best on more than about 70–75% of queries. | Reduce the claim to safe reformulation/no-op; add a genuinely different action or shift to adaptive budget selection. |
| Oracle gain is tiny. | Seed-stable oracle improves less than roughly 0.02 nDCG@10. | Study harm prediction and abstention; publish a careful negative benchmark of reformulation need. |
| Labels are unstable. | Best action changes across seeds or qrel samples. | Predict gain distributions, merge statistically tied actions, and use conformal/uncertainty-based abstention. |
| Router overfits. | Validation gains disappear on held-out collections. | Simplify features/model, perform leave-one-collection-out evaluation, and calibrate using a very small target set. |
| Generation cost exceeds benefit. | Quality–latency frontier is dominated by PRF/no-op. | Limit generation length, distill reformulations, or pivot to choosing PRF depth and fusion budget. |

## Go/no-go pilot

Run 200–500 queries, all five actions, and three generator seeds. Before training a router, inspect: (i) the seed-stable oracle gap; (ii) the distribution of best actions; (iii) action margins; (iv) the fraction of queries harmed by each action; and (v) cross-validated regret using simple features. Continue with multi-action routing only if there is heterogeneous, stable headroom. These numerical checks are design heuristics, not universal thresholds.

## Backoff formulations

1. **Binary safe reformulation:** predict reformulate versus no-op, optimizing the probability of non-negative gain.
2. **Adaptive generation budget:** choose one, three, or five generations for a single method.
3. **Reformulation-risk predictor:** estimate drift or seed instability rather than the best action.
4. **Empirical benchmark paper:** characterize which query families need reformulation and when strong retrievers eliminate its value.
5. **Cost-aware fusion:** select how many cached retrieval runs to fuse under a latency budget.

## What would count as a publishable result?

A credible ECIR/SIGIR-style paper would need more than a small average gain. It should show a statistically reliable quality–cost improvement over the best global action and cost-matched fusion, substantially lower harm than always-reformulate policies, and transfer across at least two domains. A rigorous negative result—showing that apparent oracle headroom is largely label/generation noise and defining a reliable protocol for measuring it—could also be publishable.

## Deliverables

- action-output and run cache with fixed versions/seeds;
- policy code and reproducible index configurations;
- oracle/noise feasibility report;
- effectiveness, cost, risk–coverage, and transfer analyses;
- at least 50 manually categorized routing failures;
- six-to-eight-page paper-style report with limitations.

