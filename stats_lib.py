#!/usr/bin/env python3
"""
stats_lib.py -- confirmatory statistics layer (workstream D). Reused by A, C, D and the held-out test in G.

Conventions (also returned by settings(), and written into README/D_summary.json):
  * unit of inference = query; all contrasts are PAIRED per-query differences
  * bootstrap: B=10,000 resamples of queries; percentile 95% CI; two-sided p = 2*min(P(mean<=0), P(mean>=0)),
    floored at 1/B (so a reported p of 1e-4 means "<= 1e-4", not a precise value)
  * Wilcoxon signed-rank: zero differences handled with the Pratt convention (zeros ranked, then dropped);
    all-zero input -> p = 1.0
  * ties for win/tie/loss: |delta| <= TIE_TOL counts as a tie
  * Holm = primary families (confirmatory); Benjamini-Hochberg = exploratory only, always labelled separately
  * random-effects pooling = DerSimonian-Laird tau^2 with Hartung-Knapp (t, k-1 df) CI; with k=5 datasets
    this is wide and should be read as "weak evidence about between-dataset generalisation"
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_B = 10000
ZERO_METHOD = "pratt"
TIE_TOL = 1e-9
ALPHA = 0.05


def settings() -> dict:
    return {"B": DEFAULT_B, "zero_method": ZERO_METHOD, "tie_tol": TIE_TOL, "alpha": ALPHA,
            "ci": "percentile 95%", "p_boot": "2*min(P(mean<=0),P(mean>=0)), floor 1/B",
            "multiplicity": "Holm (primary families); BH (exploratory, separate columns)",
            "random_effects": "DerSimonian-Laird tau2, Hartung-Knapp t CI"}


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


# ----------------------------------------------------------------------------
# Paired bootstrap
# ----------------------------------------------------------------------------
def _boot_means(x: np.ndarray, B: int, rng, chunk_elems: int = 20_000_000) -> np.ndarray:
    n = len(x)
    out = np.empty(B)
    step = max(1, chunk_elems // max(n, 1))
    for s in range(0, B, step):
        e = min(B, s + step)
        out[s:e] = x[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    return out


def _boot_p(means: np.ndarray, B: int) -> float:
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(min(max(p, 1.0 / B), 1.0))


def paired_bootstrap(diff, B: int = DEFAULT_B, seed: int = 0):
    """(mean, ci_lo, ci_hi, p) for the mean of paired differences."""
    d = _clean(diff)
    if len(d) < 2:
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    m = _boot_means(d, B, rng)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), _boot_p(m, B)


# ----------------------------------------------------------------------------
# Wilcoxon, effect sizes
# ----------------------------------------------------------------------------
def wilcoxon_p(diff, zero_method: str = ZERO_METHOD, tol: float = TIE_TOL) -> float:
    d = _clean(diff).copy()
    d[np.abs(d) <= tol] = 0.0
    if not np.any(d != 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method=zero_method).pvalue)
    except ValueError:
        return float("nan")


def cohens_d(diff) -> float:
    d = _clean(diff)
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def win_tie_loss(diff, tol: float = TIE_TOL):
    d = _clean(diff)
    return int((d > tol).sum()), int((np.abs(d) <= tol).sum()), int((d < -tol).sum())


def helped_harmed(diff, eps: float = 0.05):
    d = _clean(diff)
    return float((d > eps).mean()), float((d < -eps).mean())


def contrast(diff, B: int = DEFAULT_B, seed: int = 0, zero_method: str = ZERO_METHOD, tol: float = TIE_TOL) -> dict:
    """One row of a D1-style table (everything except labels and multiplicity columns)."""
    d = _clean(diff)
    m, lo, hi, pb = paired_bootstrap(d, B, seed)
    pw = wilcoxon_p(d, zero_method, tol)
    w, t, l = win_tie_loss(d, tol)
    disagree = bool((pb < ALPHA) != (pw < ALPHA)) if np.isfinite(pw) else False
    return {"n": len(d), "mean_delta": m, "ci_lo": lo, "ci_hi": hi, "cohens_d": cohens_d(d),
            "win": w, "tie": t, "loss": l, "p_boot": pb, "p_wilcoxon": pw, "boot_wilcoxon_disagree": disagree}


# ----------------------------------------------------------------------------
# Multiplicity
# ----------------------------------------------------------------------------
def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are ignored and stay NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.empty_like(q)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(q) - rank) * q[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def bh(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs ignored."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = len(q)
    order = np.argsort(q)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, q[i] * m / (rank + 1))
        adj[i] = prev
    out[ok] = np.minimum(adj, 1.0)
    return out


def add_holm(df: pd.DataFrame, family_cols, p_col: str = "p_boot", out: str = "p_holm", size_col: str = "family_size"):
    """Holm within each family (rows sharing the values of family_cols). Primary / confirmatory."""
    df = df.copy()
    df[out], df[size_col] = np.nan, 0
    groups = [df.index] if not family_cols else [g.index for _, g in df.groupby(list(family_cols))]
    for idx in groups:
        df.loc[idx, out] = holm(df.loc[idx, p_col].to_numpy())
        df.loc[idx, size_col] = int(df.loc[idx, p_col].notna().sum())
    return df


def add_bh(df: pd.DataFrame, family_cols, p_col: str = "p_boot", out: str = "q_bh"):
    """BH within each family. Exploratory only -- never mix with Holm columns."""
    df = df.copy()
    df[out] = np.nan
    groups = [df.index] if not family_cols else [g.index for _, g in df.groupby(list(family_cols))]
    for idx in groups:
        df.loc[idx, out] = bh(df.loc[idx, p_col].to_numpy())
    return df


# ----------------------------------------------------------------------------
# Pooling across datasets
# ----------------------------------------------------------------------------
def random_effects(y, v, hksj: bool = True) -> dict:
    """DerSimonian-Laird random-effects mean of per-dataset effects y with variances v (of the mean)."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    v = np.where(v <= 0, 1e-12, v)
    w = 1 / v
    mu_f = (w * y).sum() / w.sum()
    Q = (w * (y - mu_f) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1 / (v + tau2)
    mu = (ws * y).sum() / ws.sum()
    if hksj and k > 1:
        se = math.sqrt((ws * (y - mu) ** 2).sum() / ((k - 1) * ws.sum()))
        crit = stats.t.ppf(0.975, k - 1)
        p = float(2 * stats.t.sf(abs(mu / se), k - 1)) if se > 0 else float("nan")
    else:
        se = math.sqrt(1 / ws.sum())
        crit = stats.norm.ppf(0.975)
        p = float(2 * stats.norm.sf(abs(mu / se)))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return {"mu": float(mu), "se": float(se), "ci_lo": float(mu - crit * se), "ci_hi": float(mu + crit * se),
            "p": p, "tau2": float(tau2), "I2": float(I2), "k": k}


def pooled_estimates(diffs: dict, B: int = DEFAULT_B, seed: int = 0, hksj: bool = True) -> pd.DataFrame:
    """diffs: {dataset: per-query deltas}. Returns 3 rows: macro_average (PRIMARY), all_query_weighted, random_effects."""
    names = list(diffs)
    arrs = [_clean(diffs[k]) for k in names]
    rows = []
    rng = np.random.default_rng(seed)
    boots = np.mean([_boot_means(a, B, rng) for a in arrs], axis=0)  # stratified: resample queries within dataset
    rows.append({"estimator": "macro_average", "primary": True, "n_datasets": len(arrs), "n_queries": int(sum(map(len, arrs))),
                 "mean_delta": float(np.mean([a.mean() for a in arrs])), "ci_lo": float(np.percentile(boots, 2.5)),
                 "ci_hi": float(np.percentile(boots, 97.5)), "p": _boot_p(boots, B), "tau2": np.nan, "I2": np.nan})
    allq = np.concatenate(arrs)
    m, lo, hi, p = paired_bootstrap(allq, B, seed)
    rows.append({"estimator": "all_query_weighted", "primary": False, "n_datasets": len(arrs), "n_queries": len(allq),
                 "mean_delta": m, "ci_lo": lo, "ci_hi": hi, "p": p, "tau2": np.nan, "I2": np.nan})
    y = np.array([a.mean() for a in arrs])
    v = np.array([a.var(ddof=1) / len(a) for a in arrs])
    re = random_effects(y, v, hksj)
    rows.append({"estimator": "random_effects" + ("_HK" if hksj else ""), "primary": False, "n_datasets": len(arrs),
                 "n_queries": len(allq), "mean_delta": re["mu"], "ci_lo": re["ci_lo"], "ci_hi": re["ci_hi"], "p": re["p"],
                 "tau2": re["tau2"], "I2": re["I2"]})
    return pd.DataFrame(rows)


def leave_one_dataset_out(diffs: dict, B: int = DEFAULT_B, seed: int = 0) -> pd.DataFrame:
    """Pooled estimates recomputed with each dataset removed in turn (is one dataset driving the result?)."""
    out = []
    for left in diffs:
        sub = {k: v for k, v in diffs.items() if k != left}
        if len(sub) < 2:
            continue
        df = pooled_estimates(sub, B, seed)
        df.insert(0, "left_out", left)
        out.append(df)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# Seed variability
# ----------------------------------------------------------------------------
def seed_variability(V: dict, base=None, B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """V: {action: (n_queries, n_seeds) metric array}; base: optional (n,) array subtracted (e.g. a0 -> gains).
    Per action: grand mean; SD across seeds of the per-seed mean; mean per-query SD across seeds; and two CIs:
    queries-only bootstrap vs queries+seeds bootstrap (the latter reflects LLM randomness; with only 3 seeds the
    seed resampling is crude, so treat its width as indicative)."""
    rng = np.random.default_rng(seed)
    rows = []
    for a, M in V.items():
        M = np.asarray(M, float)
        M = M[np.all(np.isfinite(M), axis=1)] if base is None else M
        if base is not None:
            b = np.asarray(base, float)
            ok = np.all(np.isfinite(M), axis=1) & np.isfinite(b)
            M, b = M[ok], b[ok]
            M = M - b[:, None]
        n, S = M.shape
        seed_means = M.mean(axis=0)
        qonly = _boot_means(M.mean(axis=1), B, rng)
        qs = np.empty(B)
        for i in range(B):
            qs[i] = M[rng.integers(0, n, n)][:, rng.integers(0, S, S)].mean()
        rows.append({"action": a, "n": n, "n_seeds": S, "mean": float(M.mean()),
                     "sd_of_seed_means": float(seed_means.std(ddof=1)) if S > 1 else float("nan"),
                     "mean_per_query_sd": float(M.std(axis=1, ddof=1).mean()) if S > 1 else float("nan"),
                     "ci_q_lo": float(np.percentile(qonly, 2.5)), "ci_q_hi": float(np.percentile(qonly, 97.5)),
                     "ci_qs_lo": float(np.percentile(qs, 2.5)), "ci_qs_hi": float(np.percentile(qs, 97.5))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Judged-only nDCG (metric validity check)
# ----------------------------------------------------------------------------
def _dcg(rels) -> float:
    r = np.asarray(rels, dtype=float)
    return float((r / np.log2(np.arange(2, len(r) + 2))).sum()) if len(r) else 0.0


def ndcg_at_k(ranked_docs, qrels_q: dict, k: int = 10, judged_only: bool = False) -> float:
    """Linear-gain nDCG@k (as trec_eval). judged_only=True drops unjudged docs BEFORE taking the top k,
    i.e. the 'condensed list' sensitivity analysis. qrels_q: {doc_id: rel} for one query."""
    docs = list(ranked_docs)
    if judged_only:
        docs = [d for d in docs if d in qrels_q]
    gains = [max(qrels_q.get(d, 0), 0) for d in docs[:k]]
    ideal = sorted((max(v, 0) for v in qrels_q.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def judged_at_k(ranked_docs, qrels_q: dict, k: int = 10) -> float:
    docs = list(ranked_docs)[:k]
    return sum(d in qrels_q for d in docs) / float(k)


# ----------------------------------------------------------------------------
# Power, null calibration
# ----------------------------------------------------------------------------
def mde(sd: float, n: int, alpha: float = ALPHA, power: float = 0.8) -> float:
    """Minimum detectable mean difference (two-sided, paired, normal approximation)."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sd / math.sqrt(n))


def power_row(diff, alpha: float = ALPHA, power: float = 0.8) -> dict:
    d = _clean(diff)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    m = mde(sd, len(d), alpha, power)
    se = sd / math.sqrt(len(d))
    z = stats.norm.ppf(1 - alpha / 2)
    obs_power = float(stats.norm.sf(z - abs(d.mean()) / se) + stats.norm.cdf(-z - abs(d.mean()) / se)) if se > 0 else float("nan")
    return {"n": len(d), "sd_delta": sd, "mean_delta": float(d.mean()), "mde_80": m, "mde_d": m / sd if sd > 0 else float("nan"),
            "observed_power": obs_power}


def sign_flip_null(diff, reps: int = 200, B: int = 2000, seed: int = 0, alpha: float = ALPHA) -> dict:
    """Null calibration: randomly flip the sign of each observed difference (a symmetric null by construction),
    re-test, and report the rejection rate at alpha. Should be close to alpha for a well-calibrated test."""
    d = _clean(diff)
    rng = np.random.default_rng(seed)
    rb = rw = 0
    for r in range(reps):
        f = d * rng.choice([-1.0, 1.0], size=len(d))
        rb += paired_bootstrap(f, B, seed + r)[3] < alpha
        pw = wilcoxon_p(f)
        rw += (pw < alpha) if np.isfinite(pw) else 0
    return {"reps": reps, "alpha": alpha, "reject_rate_boot": rb / reps, "reject_rate_wilcoxon": rw / reps}