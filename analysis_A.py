#!/usr/bin/env python3
"""
analysis_A.py -- Workflow A: per-dataset, descriptive analysis of the five reformulation actions.

    python analysis_A.py --stores store/scifact store/trec-covid store/nfcorpus store/fiqa store/webis-touche2020 \
                         --out results/analysis/A_per_dataset

Question A answers: which dataset benefits from which action, how much headroom each has,
how heterogeneous the queries are, and whether this is consistent across BM25 and dense.
It is DESCRIPTIVE and dataset-level: it does not look inside datasets (B) or evaluate routers (C/D).

It re-uses routing.build_pool, so every number is computed by exactly the same code path (same query
filtering, same cross-fitted oracle, same seeds) as results/router_*/iteration_log.md.
That is what makes the acceptance check (bottom of this file) meaningful.

Outputs (all in --out):
  A0_per_query.csv  A1_action_gains.csv  A2_helped_harmed.csv  A3_win_share.csv  A4_headroom.csv
  A5_heterogeneity.csv  A6_best_action_prob.csv  A6b_pairwise.csv  A7_cost.csv  A8_cross_retriever.csv
  A9_dataset_descriptors.csv  A_acceptance.csv  A_summary.json  A_hashes.json  README.md
  tab_*.tex   fig_*.png / fig_*.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing as R  # noqa: E402
from run_dataset import QUESTION_WORDS, Store  # noqa: E402

ACTIONS = list(R.ACTIONS)            # a0, prf, query2doc, hyde, decomp
NON_A0 = ACTIONS[1:]
RETRS = ["bm25", "dense"]
METRICS = ["ndcg10", "map100", "recall100"]
EPS_LIST = (0.05, 0.01)
TIE_TOL = 1e-9                       # |x - max| <= TIE_TOL counts as a tie (float-noise safe)
PRETTY = {"a0": "a0", "prf": "PRF", "query2doc": "Query2doc", "hyde": "HyDE", "decomp": "Decomp"}
COLORS = {"a0": "#8a8a8a", "prf": "#E69F00", "query2doc": "#0072B2", "hyde": "#009E73", "decomp": "#CC79A7"}

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def seed_for(base, *parts):
    """Deterministic across runs (Python's hash() is randomised, so we use sha1)."""
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return (int(h[:8], 16) + int(base)) % (2 ** 32)


def blocks(pool):
    """[(name, mask)] for every dataset, then ALL (all queries pooled)."""
    out = [(n, pool.ds_id == i) for i, n in enumerate(pool.ds_names)]
    out.append(("ALL", np.ones(len(pool.qid), bool)))
    return out


def seedmean(V):
    return np.nanmean(V, axis=2)


def holm_nan(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = R.holm(p[ok])
    return out


def wilcoxon_p(x):
    x = np.asarray(x, float)
    if not np.any(np.abs(x) > 1e-12):
        return np.nan
    try:
        return float(stats.wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def cohens_d(x):
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / s) if s > 1e-12 else np.nan


class Boot:
    """Paired bootstrap over queries. The SAME resample indices are reused for a given (retriever, block),
    so A1, A4, A6, A6b are mutually consistent and everything is reproducible from --seed."""

    def __init__(self, B, seed):
        self.B, self.seed, self.cache = B, seed, {}

    def idx(self, retr, block, n):
        k = (retr, block, n)
        if k not in self.cache:
            rng = np.random.default_rng(seed_for(self.seed, "boot", retr, block))
            self.cache[k] = rng.integers(0, n, size=(self.B, n), dtype=np.int32)
        return self.cache[k]

    def mean_ci_p(self, x, retr, block):
        """mean, 2.5%, 97.5%, two-sided bootstrap p (same formula as routing.paired_bootstrap), replicate means."""
        x = np.asarray(x, float)
        means = x[self.idx(retr, block, len(x))].mean(axis=1)
        p = 2 * min((means <= 0).mean(), (means >= 0).mean())
        return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), \
            float(min(max(p, 1.0 / self.B), 1.0)), means


def crossfit_choice(V):
    """(n, S) index of the action picked for held-out seed e using the OTHER seeds (ties -> first = a0)."""
    n, A, S = V.shape
    out = np.zeros((n, S), int)
    for e in range(S):
        others = [s for s in range(S) if s != e] or [0]
        sel = np.nanmean(V[:, :, others], axis=2)
        out[:, e] = np.argmax(np.where(np.isnan(sel), -np.inf, sel), axis=1)
    return out


def query_form(q):
    """HEURISTIC (validate by hand on ~50 queries before quoting):
    question = ends with '?' or starts with a question word; keyword = <=4 words; claim = other >=5-word statements."""
    w = str(q).strip().split()
    if not w:
        return "keyword"
    if str(q).strip().endswith("?") or w[0].lower() in QUESTION_WORDS:
        return "question"
    return "keyword" if len(w) <= 4 else "claim"


# ----------------------------------------------------------------------------
# Pools
# ----------------------------------------------------------------------------
def build_pools(stores, retrievers):
    P = {}
    for retr in retrievers:
        P[retr] = {m: R.build_pool(stores, retr, m, 1, "dataset", "crossfit") for m in METRICS}
        base = P[retr]["ndcg10"]
        for m in METRICS[1:]:
            if not np.array_equal(base.qid, P[retr][m].qid):
                raise SystemExit(f"{retr}: queries kept for {m} differ from ndcg10 (missing values?). Investigate before continuing.")
    if len(retrievers) == 2 and not np.array_equal(P[retrievers[0]]["ndcg10"].qid, P[retrievers[1]]["ndcg10"].qid):
        raise SystemExit("bm25 and dense pools contain different queries; cannot align per-query results.")
    return P


# ----------------------------------------------------------------------------
# A0 per-query table (feeds B and C)
# ----------------------------------------------------------------------------
def table_A0(P):
    frames = []
    for retr, pm in P.items():
        pool = pm["ndcg10"]
        d = pd.DataFrame({"retriever": retr, "dataset": [pool.ds_names[i] for i in pool.ds_id], "qid": pool.raw_qid,
                          "query": pool.query})
        d["n_words"] = [len(str(q).split()) for q in pool.query]
        d["form"] = [query_form(q) for q in pool.query]
        Vm = seedmean(pool.V)
        for j, a in enumerate(ACTIONS):
            d[f"ndcg10_{a}"] = Vm[:, j]
        for j, a in enumerate(ACTIONS[1:], 1):
            d[f"gain_{a}"] = Vm[:, j] - Vm[:, 0]
        for a in ("query2doc", "hyde", "decomp"):
            j = ACTIONS.index(a)
            for s_i, s in enumerate(pool.seeds):
                d[f"ndcg10_{a}_seed{s}"] = pool.V[:, j, s_i]
        for m in METRICS[1:]:
            Vo = seedmean(pm[m].V)
            for j, a in enumerate(ACTIONS):
                d[f"{m}_{a}"] = Vo[:, j]
        d["oracle_crossfit"] = pool.T["all"]
        d["oracle_naive"] = pool.naive["all"]
        d["oracle_choice"] = pool.choice["all"]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# A1 fixed-action performance
# ----------------------------------------------------------------------------
def table_A1(P, boot):
    rows = []
    for retr, pm in P.items():
        for metric in METRICS:
            pool = pm[metric]
            Vm = seedmean(pool.V)
            for bname, m in blocks(pool):
                X = Vm[m]
                for j, a in enumerate(ACTIONS):
                    r = {"retriever": retr, "metric": metric, "dataset": bname, "n": int(m.sum()), "action": a,
                         "mean_score": X[:, j].mean()}
                    if j > 0:
                        g = X[:, j] - X[:, 0]
                        mean, lo, hi, p, _ = boot.mean_ci_p(g, retr, bname)
                        r.update({"mean_gain": mean, "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                                  "p_wilcoxon": wilcoxon_p(g), "cohens_d": cohens_d(g)})
                    rows.append(r)
    df = pd.DataFrame(rows)
    # Holm across the datasets (ALL excluded: it pools the same queries again), per retriever x metric x action
    for c in ("p_boot", "p_wilcoxon"):
        df[c + "_holm"] = np.nan
    for _, g in df[(df.dataset != "ALL") & (df.action != "a0")].groupby(["retriever", "metric", "action"]):
        for c in ("p_boot", "p_wilcoxon"):
            df.loc[g.index, c + "_holm"] = holm_nan(g[c].to_numpy())
    return df


# ----------------------------------------------------------------------------
# A2 helped / harmed / neutral (ndcg@10, on the seed-mean gain)
# ----------------------------------------------------------------------------
def table_A2(P):
    rows = []
    for retr, pm in P.items():
        pool = pm["ndcg10"]
        Vm = seedmean(pool.V)
        for bname, m in blocks(pool):
            X = Vm[m]
            for eps in EPS_LIST:
                for j, a in enumerate(NON_A0, 1):
                    g = X[:, j] - X[:, 0]
                    h, hm = g > eps, g < -eps
                    rows.append({"retriever": retr, "dataset": bname, "n": int(m.sum()), "eps": eps, "action": a,
                                 "helped_rate": h.mean(), "harmed_rate": hm.mean(), "neutral_rate": (~h & ~hm).mean(),
                                 "net_rate": h.mean() - hm.mean(),
                                 "identical_rate": (np.abs(g) < 1e-12).mean(),
                                 "mean_gain_when_helped": g[h].mean() if h.any() else np.nan,
                                 "mean_loss_when_harmed": g[hm].mean() if hm.any() else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# A3 win share
# ----------------------------------------------------------------------------
def table_A3(P):
    rows = []
    for retr, pm in P.items():
        pool = pm["ndcg10"]
        Vm = seedmean(pool.V)
        mx = Vm.max(axis=1, keepdims=True)
        tied = Vm >= mx - TIE_TOL                                  # (n, A) actions within tolerance of the best
        w_split = tied / tied.sum(axis=1, keepdims=True)
        w_a0 = np.zeros_like(w_split)
        first = np.where(tied[:, 0], 0, np.argmax(tied, axis=1))   # a0 wins ties, else first in action order
        w_a0[np.arange(len(first)), first] = 1.0
        cf = crossfit_choice(pool.V)
        for bname, m in blocks(pool):
            for j, a in enumerate(ACTIONS):
                rows.append({"retriever": retr, "dataset": bname, "n": int(m.sum()), "action": a,
                             "share_exact_a0first": float((pool.choice["all"][m] == a).mean()),
                             "share_a0first_tol": float(w_a0[m, j].mean()),
                             "share_split_tol": float(w_split[m, j].mean()),
                             "share_crossfit": float((cf[m] == j).mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# A4 headroom
# ----------------------------------------------------------------------------
def table_A4(P, boot):
    rows = []
    for retr, pm in P.items():
        for metric in METRICS:
            pool = pm[metric]
            Vm = seedmean(pool.V)
            tied_best = ((Vm >= Vm.max(axis=1, keepdims=True) - TIE_TOL).sum(axis=1) >= 2)
            for bname, m in blocks(pool):
                X = Vm[m]
                bf = int(np.argmax(X.mean(axis=0)))
                a0v, bfv = X[:, 0].mean(), X[:, bf].mean()
                ocf, onv = pool.T["all"][m].mean(), pool.naive["all"][m].mean()
                d_of = pool.T["all"][m] - X[:, bf]
                mean, lo, hi, _, _ = boot.mean_ci_p(d_of, retr, bname)
                first_tol = np.where((X >= X.max(axis=1, keepdims=True) - TIE_TOL)[:, 0], 0,
                                     np.argmax(X >= X.max(axis=1, keepdims=True) - TIE_TOL, axis=1))
                rows.append({"retriever": retr, "metric": metric, "dataset": bname, "n": int(m.sum()),
                             "a0": a0v, "best_fixed_action": ACTIONS[bf], "best_fixed": bfv,
                             "oracle_crossfit": ocf, "oracle_naive": onv,
                             "headroom_vs_a0": ocf - a0v, "headroom_vs_best_fixed": ocf - bfv,
                             "headroom_vs_best_fixed_ci_lo": lo, "headroom_vs_best_fixed_ci_hi": hi,
                             "best_fixed_gain_over_a0": bfv - a0v,
                             "best_fixed_share_of_oracle_gain": (bfv - a0v) / max(ocf - a0v, 1e-9),
                             "optimism_gap": onv - ocf,
                             "oracle_go_rate": float((pool.choice["all"][m] != "a0").mean()),
                             "oracle_go_rate_tol": float((first_tol != 0).mean()),
                             "tied_best_frac": float(tied_best[m].mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# A5 heterogeneity (noise-corrected)
# ----------------------------------------------------------------------------
def table_A5(P):
    rows = []
    for retr, pm in P.items():
        pool = pm["ndcg10"]
        V = pool.V
        S = V.shape[2]
        within = np.nanvar(V, axis=2, ddof=1) if S > 1 else np.zeros(V.shape[:2])
        for bname, m in blocks(pool):
            for j, a in enumerate(NON_A0, 1):
                Gs = V[m][:, j, :] - V[m][:, 0, :]                  # per-seed gain (n, S)
                var_single = float(np.nanvar(Gs, axis=0, ddof=1).mean())
                gm = np.nanmean(Gs, axis=1)
                var_mean = float(gm.var(ddof=1))
                var_within = float(within[m, j].mean())
                var_true = max(0.0, var_single - var_within)
                var_true_mean = max(0.0, var_mean - var_within / S)
                mg = float(gm.mean())
                rows.append({"retriever": retr, "dataset": bname, "n": int(m.sum()), "action": a,
                             "seedless": bool(var_within == 0.0), "n_seeds": S, "mean_gain": mg,
                             "sd_gain_single_seed": np.sqrt(var_single), "sd_gain_seedmean": np.sqrt(var_mean),
                             "within_seed_var": var_within, "noise_share_single": var_within / var_single if var_single > 0 else np.nan,
                             "var_true_single": var_true, "sd_true_single": np.sqrt(var_true),
                             "var_true_seedmean": var_true_mean, "sd_true_seedmean": np.sqrt(var_true_mean),
                             "sd_true_over_abs_mean_gain": np.sqrt(var_true) / abs(mg) if abs(mg) > 1e-9 else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# A6 / A6b which action is best, and how sure
# ----------------------------------------------------------------------------
def tables_A6(P, boot, ref_name):
    a6, b6 = [], []
    ref = ACTIONS.index(ref_name)
    for retr, pm in P.items():
        pool = pm["ndcg10"]
        Vm = seedmean(pool.V)
        for bname, m in blocks(pool):
            X, n = Vm[m], int(m.sum())
            idx = boot.idx(retr, bname, n)
            means = np.column_stack([X[:, j][idx].mean(axis=1) for j in range(len(ACTIONS))])   # (B, A)
            prob = np.bincount(means.argmax(axis=1), minlength=len(ACTIONS)) / boot.B
            ranks = (-X.mean(axis=0)).argsort().argsort() + 1
            for j, a in enumerate(ACTIONS):
                a6.append({"retriever": retr, "dataset": bname, "n": n, "action": a, "mean_score": X[:, j].mean(),
                           "rank_by_mean": int(ranks[j]), "prob_best_fixed": float(prob[j])})
            others = [j for j in range(len(ACTIONS)) if j != ref]
            strongest = max(others, key=lambda j: X[:, j].mean())
            comps = [(f"{ref_name}_vs_{a}", ACTIONS.index(a), "main" if bname != "ALL" else "pooled_ALL")
                     for a in ("hyde", "prf", "decomp")]
            comps.append((f"{ref_name}_vs_strongest_competitor({ACTIONS[strongest]})", strongest, "extra"))
            for label, j, fam in comps:
                d = X[:, ref] - X[:, j]
                mean, lo, hi, p, mb = boot.mean_ci_p(d, retr, bname)
                b6.append({"retriever": retr, "dataset": bname, "n": n, "comparison": label, "family": fam,
                           "mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "p_boot": p, "p_wilcoxon": wilcoxon_p(d),
                           "prob_ref_better": float((mb > 0).mean())})
    b6 = pd.DataFrame(b6)
    b6["p_boot_holm"] = np.nan
    b6["p_wilcoxon_holm"] = np.nan
    for _, g in b6[b6.family == "main"].groupby("retriever"):      # family: 5 datasets x 3 comparisons per retriever
        b6.loc[g.index, "p_boot_holm"] = holm_nan(g.p_boot.to_numpy())
        b6.loc[g.index, "p_wilcoxon_holm"] = holm_nan(g.p_wilcoxon.to_numpy())
    return pd.DataFrame(a6), b6


# ----------------------------------------------------------------------------
# A7 cost
# ----------------------------------------------------------------------------
def table_A7(P):
    rows = []
    for retr, pm in P.items():
        pool = pm["ndcg10"]
        Vm = seedmean(pool.V)
        for bname, m in blocks(pool):
            c0 = pool.cost[m, 0]
            for j, a in enumerate(ACTIONS):
                c = pool.cost[m, j]
                g = (Vm[m, j] - Vm[m, 0]).mean()
                extra = float((c - c0).mean())
                rows.append({"retriever": retr, "dataset": bname, "n": int(m.sum()), "action": a,
                             "mean_cost_s": float(c.mean()), "p95_cost_s": float(np.percentile(c, 95)),
                             "extra_cost_over_a0_s": extra, "mean_gain": g,
                             "gain_per_s_total": g / c.mean() if j > 0 and c.mean() > 1e-9 else np.nan,
                             "gain_per_extra_s": g / extra if j > 0 and extra > 1e-9 else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# A8 cross-retriever consistency
# ----------------------------------------------------------------------------
def table_A8(P):
    if len(P) < 2:
        return pd.DataFrame(columns=["dataset", "n", "measure", "action", "value", "p_value", "note"])
    pb, pd_ = P["bm25"]["ndcg10"], P["dense"]["ndcg10"]
    Vb, Vd = seedmean(pb.V), seedmean(pd_.V)
    rows = []
    cell_b, cell_d = [], []
    for bname, m in blocks(pb):
        n = int(m.sum())
        gb = Vb[m][:, 1:] - Vb[m][:, [0]]
        gd = Vd[m][:, 1:] - Vd[m][:, [0]]
        mb, md = gb.mean(0), gd.mean(0)
        if bname != "ALL":
            cell_b += list(mb)
            cell_d += list(md)
        rho, p = stats.spearmanr(mb, md)
        rows.append({"dataset": bname, "n": n, "measure": "action_mean_gain_spearman", "action": "all4",
                     "value": rho, "p_value": p, "note": "4 actions only: very coarse"})
        for j, a in enumerate(NON_A0):
            r_p = stats.pearsonr(gb[:, j], gd[:, j])
            r_s = stats.spearmanr(gb[:, j], gd[:, j])
            rows.append({"dataset": bname, "n": n, "measure": "per_query_gain_pearson", "action": a, "value": r_p[0], "p_value": r_p[1], "note": ""})
            rows.append({"dataset": bname, "n": n, "measure": "per_query_gain_spearman", "action": a, "value": r_s[0], "p_value": r_s[1], "note": ""})
        cb, cd = pb.choice["all"][m], pd_.choice["all"][m]
        for label, xb, xd in (("oracle_best_action_agreement", cb, cd),
                              ("oracle_go_nogo_agreement", np.where(cb == "a0", "a0", "go"), np.where(cd == "a0", "a0", "go"))):
            po = float((xb == xd).mean())
            cats = np.unique(np.concatenate([xb, xd]))
            pe = float(sum((xb == c).mean() * (xd == c).mean() for c in cats))
            rows.append({"dataset": bname, "n": n, "measure": label, "action": "all", "value": po, "p_value": np.nan,
                         "note": f"chance_agreement={pe:.3f}"})
            rows.append({"dataset": bname, "n": n, "measure": label + "_kappa", "action": "all",
                         "value": (po - pe) / (1 - pe) if pe < 1 else np.nan, "p_value": np.nan, "note": "Cohen kappa"})
    rho, p = stats.spearmanr(cell_b, cell_d)
    tau, pt = stats.kendalltau(cell_b, cell_d)
    rows.append({"dataset": "POOLED_CELLS", "n": len(cell_b), "measure": "action_mean_gain_spearman", "action": "dataset x action cells",
                 "value": rho, "p_value": p, "note": "cells are not independent (same queries)"})
    rows.append({"dataset": "POOLED_CELLS", "n": len(cell_b), "measure": "action_mean_gain_kendall", "action": "dataset x action cells",
                 "value": tau, "p_value": pt, "note": ""})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# A9 dataset descriptors
# ----------------------------------------------------------------------------
def table_A9(P):
    pool = P[next(iter(P))]["ndcg10"]
    rows, jf = [], []
    for i, name in enumerate(pool.ds_names):
        m = pool.ds_id == i
        qids = [str(q) for q in pool.raw_qid[m]]
        s = Store(pool.store_paths[name])
        qr = s.qrels.assign(qid=lambda d: d.qid.astype(str))
        qr = qr[qr.qid.isin(set(qids))]
        rel = qr[qr.rel > 0].groupby("qid").size().reindex(qids).fillna(0)
        judged = qr.groupby("qid").size().reindex(qids).fillna(0)
        forms = pd.Series([query_form(q) for q in pool.query[m]])
        met = s.metrics
        met = met[met.qid.astype(str).isin(set(qids)) & met.action.isin(ACTIONS)]
        j = met.groupby(["retriever", "action", "qid"]).judged10.mean().reset_index()
        j["dataset"] = name
        jf.append(j)
        rows.append({"dataset": name, "n_queries": int(m.sum()), "n_docs": s.manifest.get("n_docs"),
                     "mean_query_words": float(np.mean([len(str(q).split()) for q in pool.query[m]])),
                     "mean_query_chars": float(np.mean([len(str(q)) for q in pool.query[m]])),
                     "mean_rel_docs": float(rel.mean()), "median_rel_docs": float(rel.median()),
                     "mean_judged_docs": float(judged.mean()),
                     "share_claim": float((forms == "claim").mean()), "share_question": float((forms == "question").mean()),
                     "share_keyword": float((forms == "keyword").mean())})
    allj = pd.concat(jf, ignore_index=True)
    allq = np.concatenate([[query_form(q) for q in pool.query]])
    rows.append({"dataset": "ALL", "n_queries": len(pool.qid), "n_docs": np.nan,
                 "mean_query_words": float(np.mean([len(str(q).split()) for q in pool.query])),
                 "mean_query_chars": float(np.mean([len(str(q)) for q in pool.query])),
                 "mean_rel_docs": np.nan, "median_rel_docs": np.nan, "mean_judged_docs": np.nan,
                 "share_claim": float((allq == "claim").mean()), "share_question": float((allq == "question").mean()),
                 "share_keyword": float((allq == "keyword").mean())})
    df = pd.DataFrame(rows)
    # weighted ALL means for rel docs / judged docs
    w = df[df.dataset != "ALL"]
    for c in ("mean_rel_docs", "mean_judged_docs"):
        df.loc[df.dataset == "ALL", c] = float((w[c] * w.n_queries).sum() / w.n_queries.sum())
    for retr in P:
        Vm = seedmean(P[retr]["ndcg10"].V)
        vals = [Vm[mm, 0].mean() for _, mm in blocks(P[retr]["ndcg10"])]
        df[f"a0_ndcg10_{retr}"] = vals
        for a in ACTIONS:
            sub = allj[(allj.retriever == retr) & (allj.action == a)]
            by = sub.groupby("dataset").judged10.mean()
            df[f"judged10_{retr}_{a}"] = [by.get(n, np.nan) for n in df.dataset[:-1]] + [sub.judged10.mean()]
    return df


# ----------------------------------------------------------------------------
# Acceptance check against iteration_log.md baselines
# ----------------------------------------------------------------------------
def _e(n, a0, prf, q2d, hyde, dec, orc, nv, head, go):
    return dict(n=n, a0=a0, prf=prf, query2doc=q2d, hyde=hyde, decomp=dec, oracle=orc, naive=nv, headroom=head, go_rate=go)


EXPECTED = {  # copied from results/router_{bm25,dense}/iteration_log.md, "Baselines (nDCG@10)" block
    "bm25": {"scifact": _e(300, .6617, .5692, .6982, .6662, .6192, .7618, .7739, .1001, .3367),
             "trec-covid": _e(50, .5945, .5977, .7071, .5982, .6247, .7482, .7655, .1537, .8200),
             "nfcorpus": _e(323, .3064, .3044, .3427, .2973, .3085, .3915, .4012, .0851, .6006),
             "fiqa": _e(648, .2326, .1848, .2735, .2125, .2033, .3409, .3556, .1083, .5077),
             "webis-touche2020": _e(49, .3579, .3749, .4414, .3006, .3386, .4710, .4926, .1131, .7959),
             "ALL": _e(1370, .3616, .3190, .4047, .3491, .3394, .4645, .4778, .1029, .5139)},
    "dense": {"scifact": _e(300, .6451, .6062, .6910, .6839, .5887, .7702, .7747, .1251, .3800),
              "trec-covid": _e(50, .4725, .5813, .6535, .6055, .4573, .7170, .7240, .2446, .9600),
              "nfcorpus": _e(323, .3159, .3216, .3352, .3323, .3063, .4251, .4275, .1091, .5975),
              "fiqa": _e(648, .3687, .3242, .3751, .3384, .2986, .4561, .4675, .0874, .4583),
              "webis-touche2020": _e(49, .1690, .1917, .2606, .2403, .1757, .2967, .3086, .1276, .8571),
              "ALL": _e(1370, .4134, .3900, .4409, .4189, .3653, .5214, .5290, .1080, .5066)},
}


def acceptance(P, tol=6e-5):
    rows = []
    for retr, pm in P.items():
        pool = pm["ndcg10"]
        Vm = seedmean(pool.V)
        for bname, m in blocks(pool):
            exp = EXPECTED.get(retr, {}).get(bname)
            if exp is None:
                continue
            if bname == "ALL" and len(pool.ds_names) != 5:
                continue
            obs = {"n": int(m.sum()), **{a: Vm[m, j].mean() for j, a in enumerate(ACTIONS)},
                   "oracle": pool.T["all"][m].mean(), "naive": pool.naive["all"][m].mean(),
                   "go_rate": float((pool.choice["all"][m] != "a0").mean())}
            obs["headroom"] = obs["oracle"] - obs["a0"]
            for k, ev in exp.items():
                ov = obs[k]
                ok = (ov == ev) if k == "n" else abs(ov - ev) <= tol
                rows.append({"retriever": retr, "dataset": bname, "quantity": k, "expected": ev, "observed": ov,
                             "abs_diff": abs(ov - ev), "status": "PASS" if ok else "FAIL"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Summary JSON
# ----------------------------------------------------------------------------
def build_summary(P, T, ref_name, acc):
    A1 = T["A1"][T["A1"].metric == "ndcg10"]
    A4 = T["A4"][T["A4"].metric == "ndcg10"]
    A5, A6, A6b, A3 = T["A5"], T["A6"], T["A6b"], T["A3"]
    S = {"settings": T["settings"], "n_queries": {r: int(len(P[r]["ndcg10"].qid)) for r in P}, "headline": {}, "expected_findings": {}}
    for retr in P:
        row = A4[(A4.retriever == retr) & (A4.dataset == "ALL")].iloc[0]
        S["headline"][retr] = {k: (float(row[k]) if isinstance(row[k], (float, np.floating)) else row[k])
                               for k in ("a0", "best_fixed_action", "best_fixed", "oracle_crossfit", "oracle_naive", "headroom_vs_a0",
                                         "headroom_vs_best_fixed", "optimism_gap", "oracle_go_rate", "tied_best_frac")}
    ef = S["expected_findings"]
    # 1. ref action best fixed everywhere?
    bf = A4[["retriever", "dataset", "best_fixed_action"]]
    ef["ref_action_best_fixed_everywhere"] = {
        "ref": ref_name, "holds": bool((bf.best_fixed_action == ref_name).all()),
        "exceptions": bf[bf.best_fixed_action != ref_name].to_dict("records")}
    strong = A6b[A6b.comparison.str.contains("strongest")]
    ef["ref_vs_strongest_competitor"] = strong[["retriever", "dataset", "n", "comparison", "mean_diff", "ci_lo", "ci_hi", "prob_ref_better"]].round(4).to_dict("records")
    ef["lowest_prob_ref_beats_competitor"] = strong.sort_values("prob_ref_better").head(3)[["retriever", "dataset", "comparison", "prob_ref_better"]].round(4).to_dict("records")
    # 2. PRF sign flips
    prf = A1[(A1.action == "prf") & (A1.dataset != "ALL")]
    ef["prf_sign_flips"] = {r: {"gains": prf[prf.retriever == r].set_index("dataset").mean_gain.round(4).to_dict(),
                                "flips": bool((prf[prf.retriever == r].mean_gain > 0).any() and (prf[prf.retriever == r].mean_gain < 0).any())}
                            for r in P}
    # 3. weakest action per cell
    weak = []
    for (r, d), g in A1[A1.action != "a0"].groupby(["retriever", "dataset"]):
        weak.append({"retriever": r, "dataset": d, "weakest": g.sort_values("mean_gain").iloc[0].action})
    ef["weakest_action_per_cell"] = {"decomp_is_weakest_in": int(sum(w["weakest"] == "decomp" for w in weak)), "of": len(weak),
                                     "cells": weak}
    # 4. heterogeneity
    small = A5[A5.mean_gain.abs() < 0.02]
    ef["heterogeneity"] = {"median_sd_true_all_cells": float(A5.sd_true_single.median()),
                           "n_cells_with_abs_mean_gain_lt_0.02": int(len(small)),
                           "median_sd_true_in_those_cells": float(small.sd_true_single.median()) if len(small) else None,
                           "min_sd_true_in_those_cells": float(small.sd_true_single.min()) if len(small) else None,
                           "note": "no threshold is applied; judge by comparing to mean gains"}
    if "A8" in T and len(T["A8"]):
        a8 = T["A8"]
        S["cross_retriever"] = {
            "pooled_cells_spearman": a8[(a8.dataset == "POOLED_CELLS") & (a8.measure == "action_mean_gain_spearman")].value.round(4).tolist(),
            "oracle_agreement_ALL": a8[(a8.dataset == "ALL") & (a8.measure == "oracle_best_action_agreement")][["value", "note"]].to_dict("records"),
            "oracle_kappa_ALL": a8[(a8.dataset == "ALL") & (a8.measure == "oracle_best_action_agreement_kappa")].value.round(4).tolist()}
    S["acceptance"] = {"n_checks": int(len(acc)), "n_fail": int((acc.status == "FAIL").sum()) if len(acc) else None}
    S["caveats"] = [
        "TREC-COVID and Touche have ~50 queries: CIs are wide; n is reported in every table.",
        "nDCG@10 on TREC-COVID depends on incomplete judgments: see judged10_* in A9.",
        "Cross-fit oracle needs >= 2 seeds (here %d)." % T["settings"]["n_seeds"],
        "a0 and PRF are seedless (zero seed variance); the noise correction only removes noise from LLM actions.",
        "Best fixed action is chosen in-sample on each block (it is NOT the train-selected one used in iteration_log).",
        "Cross-dataset results are descriptive, not a hypothesis test; ALL pools the same queries again, so ALL is excluded from Holm.",
        "Query-form shares use a heuristic classifier; validate by hand before quoting.",
        "Cost uses cost_table; latency measured under --workers>1 is inflated.",
    ]
    return S


# ----------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------
def esc(s):
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write_tex(path, caption, label, header, rows, colspec=None):
    nc = len(header)
    colspec = colspec or "l" + "r" * (nc - 1)
    L = ["% auto-generated by analysis_A.py (needs booktabs + graphicx)", "\\begin{table}[t]", "\\centering", "\\small",
         "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MID":
            L.append("\\midrule")
        elif isinstance(r, tuple):
            L.append(f"\\multicolumn{{{nc}}}{{l}}{{\\textit{{{esc(r[1])}}}}} \\\\")
        else:
            L.append(" & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


def write_all_tex(out, P, T):
    dagger = "$^{\\dagger}$"
    names = {r: P[r]["ndcg10"].ds_names + ["ALL"] for r in P}
    RN = {"bm25": "BM25", "dense": "Dense"}
    A1, A2, A3, A4, A5, A6, A6b, A7, A9 = (T[k] for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A6b", "A7", "A9"))
    hdr_a = [PRETTY[a] for a in NON_A0]

    rows = []
    for r in P:
        rows.append(("SEC", RN[r]))
        for d in names[r]:
            g = A1[(A1.retriever == r) & (A1.metric == "ndcg10") & (A1.dataset == d)].set_index("action")
            cells = [f"{g.loc[a].mean_gain:+.3f} [{g.loc[a].ci_lo:+.3f}, {g.loc[a].ci_hi:+.3f}]"
                     + (dagger if (g.loc[a].p_boot_holm < 0.05) else "") for a in NON_A0]
            rows.append([esc(d), str(int(g.iloc[0].n)), f"{g.loc['a0'].mean_score:.3f}"] + cells)
    write_tex(out / "tab_A1_action_gains.tex", "nDCG@10 of a0 and mean gain over a0 (95\\% paired-bootstrap CI over queries). "
              "$\\dagger$: Holm-adjusted bootstrap $p<0.05$ across datasets (ALL not tested).", "tab:A1",
              ["Dataset", "$n$", "a0"] + hdr_a, rows)

    rows = []
    for r in P:
        rows.append(("SEC", RN[r]))
        for d in names[r]:
            cells = []
            for met in ("map100", "recall100"):
                g = A1[(A1.retriever == r) & (A1.metric == met) & (A1.dataset == d)].set_index("action")
                cells += [f"{g.loc[a].mean_gain:+.3f}" + (dagger if g.loc[a].p_boot_holm < 0.05 else "") for a in NON_A0]
            n = int(A1[(A1.retriever == r) & (A1.dataset == d)].n.iloc[0])
            rows.append([esc(d), str(n)] + cells)
    write_tex(out / "tab_A1b_other_metrics.tex", "Mean gain over a0 in MAP@100 (left) and Recall@100 (right).", "tab:A1b",
              ["Dataset", "$n$"] + [f"MAP {h}" for h in hdr_a] + [f"R@100 {h}" for h in hdr_a], rows)

    rows = []
    for r in P:
        rows.append(("SEC", RN[r]))
        for d in names[r]:
            g = A2[(A2.retriever == r) & (A2.dataset == d) & (A2.eps == 0.05)].set_index("action")
            rows.append([esc(d), str(int(g.iloc[0].n))] + [f"{g.loc[a].helped_rate*100:.0f} / {g.loc[a].harmed_rate*100:.0f}" for a in NON_A0])
    write_tex(out / "tab_A2_helped_harmed.tex", "Percentage of queries helped / harmed ($|\\Delta\\mathrm{nDCG@10}|>0.05$) relative to a0.", "tab:A2",
              ["Dataset", "$n$"] + hdr_a, rows)

    rows = []
    for r in P:
        rows.append(("SEC", RN[r]))
        for d in names[r]:
            g = A3[(A3.retriever == r) & (A3.dataset == d)].set_index("action")
            tf = A4[(A4.retriever == r) & (A4.metric == "ndcg10") & (A4.dataset == d)].iloc[0].tied_best_frac
            rows.append([esc(d), str(int(g.iloc[0].n))] + [f"{g.loc[a].share_exact_a0first*100:.0f}" for a in ACTIONS] + [f"{tf*100:.0f}"])
    write_tex(out / "tab_A3_win_share.tex", "Share (\\%) of queries on which each action is best (seed-mean nDCG@10; a0 wins ties). Last column: queries where $\\ge 2$ actions tie for best.",
              "tab:A3", ["Dataset", "$n$"] + [PRETTY[a] for a in ACTIONS] + ["tied"], rows)

    rows = []
    for r in P:
        rows.append(("SEC", RN[r]))
        for d in names[r]:
            g = A4[(A4.retriever == r) & (A4.metric == "ndcg10") & (A4.dataset == d)].iloc[0]
            rows.append([esc(d), str(int(g.n)), f"{g.a0:.3f}", f"{g.best_fixed:.3f} ({PRETTY[g.best_fixed_action]})", f"{g.oracle_crossfit:.3f}",
                         f"{g.oracle_naive:.3f}", f"{g.headroom_vs_a0:.3f}", f"{g.headroom_vs_best_fixed:.3f}", f"{g.optimism_gap:.3f}",
                         f"{g.oracle_go_rate*100:.0f}"])
    write_tex(out / "tab_A4_headroom.tex", "Headroom in nDCG@10. Best fixed is chosen in-sample per dataset; optimism gap = naive minus cross-fit oracle.",
              "tab:A4", ["Dataset", "$n$", "a0", "Best fixed", "Oracle (cf)", "Oracle (naive)", "Oracle$-$a0", "Oracle$-$best", "Optimism", "go \\%"], rows)

    rows = []
    for r in P:
        rows.append(("SEC", RN[r]))
        for d in names[r]:
            g = A5[(A5.retriever == r) & (A5.dataset == d)].set_index("action")
            rows.append([esc(d), str(int(g.iloc[0].n))] + [f"{g.loc[a].sd_true_single:.3f} ({g.loc[a].sd_gain_single_seed:.3f})" for a in NON_A0])
    write_tex(out / "tab_A5_heterogeneity.tex", "SD across queries of the gain over a0: noise-corrected (raw single-seed in parentheses). PRF is seedless, so no correction.",
              "tab:A5", ["Dataset", "$n$"] + hdr_a, rows)

    rows = []
    for r in P:
        rows.append(("SEC", RN[r]))
        for d in names[r]:
            g = A6[(A6.retriever == r) & (A6.dataset == d)].set_index("action")
            s = A6b[(A6b.retriever == r) & (A6b.dataset == d) & A6b.comparison.str.contains("strongest")].iloc[0]
            rows.append([esc(d), str(int(g.iloc[0].n))] + [f"{g.loc[a].prob_best_fixed:.2f}" for a in ACTIONS] + [f"{s.prob_ref_better:.2f}"])
    write_tex(out / "tab_A6_best_action_prob.tex", "Bootstrap probability that each action has the highest mean nDCG@10 (best fixed action); last column: probability the reference action beats its strongest competitor.",
              "tab:A6", ["Dataset", "$n$"] + [PRETTY[a] for a in ACTIONS] + ["P(ref $>$ best other)"], rows)

    rows = []
    for r in P:
        rows.append(("SEC", RN[r]))
        for d in names[r]:
            g = A7[(A7.retriever == r) & (A7.dataset == d)].set_index("action")
            cells = [f"{g.loc['a0'].mean_cost_s:.3f}"]
            for a in NON_A0:
                gp = g.loc[a].gain_per_extra_s
                cells.append(f"{g.loc[a].mean_cost_s:.2f} ({'--' if pd.isna(gp) else f'{gp:+.3f}'})")
            rows.append([esc(d), str(int(g.iloc[0].n))] + cells)
    write_tex(out / "tab_A7_cost.tex", "Mean cost in seconds per query (gain in nDCG@10 per extra second over a0 in parentheses).", "tab:A7",
              ["Dataset", "$n$", "a0"] + hdr_a, rows)

    rows = []
    for _, g in A9.iterrows():
        rows.append([esc(g.dataset), str(int(g.n_queries)), "--" if pd.isna(g.n_docs) else f"{int(g.n_docs):,}".replace(",", "\\,"),
                     f"{g.mean_query_words:.1f}", f"{g.mean_rel_docs:.1f}", f"{g.a0_ndcg10_bm25:.3f}"] +
                    ([f"{g.a0_ndcg10_dense:.3f}"] if "a0_ndcg10_dense" in g else []) +
                    [f"{g.judged10_bm25_a0:.2f}"] + ([f"{g.judged10_dense_a0:.2f}"] if "judged10_dense_a0" in g else []) +
                    [f"{g.share_claim*100:.0f} / {g.share_question*100:.0f} / {g.share_keyword*100:.0f}"])
    hdr = ["Dataset", "$n$", "Docs", "Words/query", "Rel/query", "a0 BM25"] + (["a0 dense"] if "a0_ndcg10_dense" in A9 else []) + \
          ["judged@10 BM25"] + (["judged@10 dense"] if "judged10_dense_a0" in A9 else []) + ["claim/question/keyword \\%"]
    write_tex(out / "tab_A9_dataset_descriptors.tex", "Dataset descriptors. Query-form shares are heuristic.", "tab:A9", hdr, rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def save_fig(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_figures(out, P, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                                "axes.titleweight": "normal"})
    A1 = T["A1"][T["A1"].metric == "ndcg10"]
    A3, A4 = T["A3"], T["A4"][T["A4"].metric == "ndcg10"]
    RN = {"bm25": "BM25", "dense": "Dense"}

    for r in P:
        pool = P[r]["ndcg10"]
        names = pool.ds_names + ["ALL"]
        g = A1[A1.retriever == r]

        # 1. heatmap
        M = np.array([[g[(g.action == a) & (g.dataset == d)].mean_gain.iloc[0] for d in names] for a in NON_A0])
        S = np.array([[g[(g.action == a) & (g.dataset == d)].p_boot_holm.iloc[0] for d in names] for a in NON_A0])
        fig, ax = plt.subplots(figsize=(1.2 * len(names) + 1.5, 3))
        v = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(names)), [f"{n}\n(n={int(g[g.dataset == n].n.iloc[0])})" for n in names])
        ax.set_yticks(range(len(NON_A0)), [PRETTY[a] for a in NON_A0])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.3f}" + ("*" if S[i, j] < 0.05 else ""), ha="center", va="center", color="#1a1a1a", fontsize=8)
        fig.colorbar(im, ax=ax, label="mean gain in nDCG@10 over a0")
        ax.set_title(f"{RN[r]}: mean gain over a0 (* = Holm p<0.05 across datasets)")
        save_fig(fig, out, f"fig_A1_heatmap_{r}")

        # 2. bars with CI
        fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 3.6))
        w = 0.8 / len(NON_A0)
        for k, a in enumerate(NON_A0):
            ga = g[g.action == a].set_index("dataset").loc[names]
            x = np.arange(len(names)) + (k - (len(NON_A0) - 1) / 2) * w
            ax.bar(x, ga.mean_gain, w * 0.92, color=COLORS[a], label=PRETTY[a],
                   yerr=[ga.mean_gain - ga.ci_lo, ga.ci_hi - ga.mean_gain], error_kw={"lw": 1, "capsize": 2, "ecolor": "#333"})
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(range(len(names)), [f"{n}\n(n={int(g[g.dataset == n].n.iloc[0])})" for n in names])
        ax.set_ylabel("mean gain in nDCG@10 over a0")
        ax.set_title(f"{RN[r]}: mean gain with 95% paired-bootstrap CI")
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        save_fig(fig, out, f"fig_A2_gain_bars_{r}")

        # 5. gain distributions
        Vm = seedmean(pool.V)
        fig, axes = plt.subplots(1, len(names), figsize=(2.2 * len(names), 3.2), sharey=True)
        for ax, (bname, m) in zip(np.atleast_1d(axes), blocks(pool)):
            data = [Vm[m, j] - Vm[m, 0] for j in range(1, len(ACTIONS))]
            parts = ax.violinplot(data, showmeans=True, showextrema=False)
            for pc, a in zip(parts["bodies"], NON_A0):
                pc.set_facecolor(COLORS[a]); pc.set_alpha(0.75); pc.set_edgecolor("none")
            parts["cmeans"].set_color("#222")
            ax.axhline(0, color="#555", lw=0.7)
            ax.set_xticks(range(1, len(NON_A0) + 1), [PRETTY[a][:5] for a in NON_A0], rotation=45)
            ax.set_title(f"{bname}\n(n={int(m.sum())})", fontsize=8)
        np.atleast_1d(axes)[0].set_ylabel("per-query gain in nDCG@10 (line = mean)")
        fig.suptitle(f"{RN[r]}: distribution of per-query gain over a0", y=1.04)
        save_fig(fig, out, f"fig_A5_gain_distributions_{r}")

    # 3. stacked win share
    fig, axes = plt.subplots(1, len(P), figsize=(5.2 * len(P), 3.6), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), P):
        names = P[r]["ndcg10"].ds_names + ["ALL"]
        left = np.zeros(len(names))
        for a in ACTIONS:
            s = A3[(A3.retriever == r) & (A3.action == a)].set_index("dataset").loc[names].share_exact_a0first.to_numpy()
            ax.barh(range(len(names)), s, left=left, color=COLORS[a], label=PRETTY[a], edgecolor="white", linewidth=1.2)
            for i, (l_, v_) in enumerate(zip(left, s)):
                if v_ >= 0.07:
                    ax.text(l_ + v_ / 2, i, f"{v_*100:.0f}", ha="center", va="center", fontsize=7, color="#111")
            left += s
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlabel("share of queries where action is best (a0 wins ties)")
        ax.set_title(RN[r])
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(1.0, -0.2))
    save_fig(fig, out, "fig_A3_win_share")

    # 4. headroom waterfall
    fig, axes = plt.subplots(len(P), 6, figsize=(13, 2.8 * len(P)), squeeze=False)
    for ri, r in enumerate(P):
        names = P[r]["ndcg10"].ds_names + ["ALL"]
        for ci, d in enumerate(names[:6]):
            ax = axes[ri, ci]
            g = A4[(A4.retriever == r) & (A4.dataset == d)].iloc[0]
            steps = [("a0", 0, g.a0, "#8a8a8a"), ("best\nfixed", g.a0, g.best_fixed, "#0072B2"),
                     ("oracle\n(cf)", g.best_fixed, g.oracle_crossfit, "#E69F00"), ("naive\nextra", g.oracle_crossfit, g.oracle_naive, "#ffffff")]
            for i, (lab, b0, b1, c) in enumerate(steps):
                ax.bar(i, b1 - b0, bottom=b0, color=c, edgecolor="#555" if lab.startswith("naive") else "none", hatch="///" if lab.startswith("naive") else None)
                ax.text(i, max(b0, b1) + 0.005, f"{b1:.3f}", ha="center", fontsize=7)
            ax.set_xticks(range(4), [s[0] for s in steps], fontsize=7)
            ax.set_title(f"{RN[r]} / {d}\n(n={int(g.n)}, best={PRETTY[g.best_fixed_action]})", fontsize=8)
            ax.set_ylim(0, max(g.oracle_naive * 1.12, 0.05))
    fig.suptitle("Headroom in nDCG@10: a0 -> best fixed -> cross-fit oracle (hatched = naive-oracle optimism)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out, "fig_A4_headroom_waterfall")


# ----------------------------------------------------------------------------
# README
# ----------------------------------------------------------------------------
DESC = {
    "retriever": "bm25 or dense", "metric": "ndcg10 / map100 / recall100", "dataset": "dataset name; ALL = every query pooled",
    "n": "number of queries in the block", "action": "a0, prf, query2doc, hyde, decomp", "mean_score": "mean metric (seed-mean per query, then mean over queries)",
    "mean_gain": "mean of (action - a0) per query", "ci_lo": "2.5% paired-bootstrap percentile", "ci_hi": "97.5% paired-bootstrap percentile",
    "p_boot": "two-sided bootstrap p (same formula as routing.paired_bootstrap)", "p_wilcoxon": "Wilcoxon signed-rank p on per-query gains (NaN if all zero)",
    "cohens_d": "paired Cohen d = mean gain / SD of gain", "p_boot_holm": "Holm across datasets (ALL excluded) per retriever x metric x action",
    "p_wilcoxon_holm": "same, for the Wilcoxon p", "eps": "threshold for helped/harmed", "helped_rate": "fraction with gain > eps",
    "harmed_rate": "fraction with gain < -eps", "neutral_rate": "|gain| <= eps", "net_rate": "helped - harmed", "identical_rate": "|gain| < 1e-12",
    "mean_gain_when_helped": "mean gain over helped queries", "mean_loss_when_harmed": "mean gain over harmed queries (negative)",
    "share_exact_a0first": "win share, exact seed-mean argmax, ties go to the earliest action (a0); matches routing.py oracle_go_rate",
    "share_a0first_tol": "same but ties = within 1e-9 of the max", "share_split_tol": "ties within 1e-9 split evenly among tied actions",
    "share_crossfit": "share of (query, seed) cross-fit picks (pick from other seeds)", "a0": "mean a0 score",
    "best_fixed_action": "action with the highest mean (in-sample, over all 5 incl. a0)", "best_fixed": "its mean score",
    "oracle_crossfit": "per-query best action chosen on other seeds, scored on held-out seed", "oracle_naive": "per-query max of seed-mean (optimistic)",
    "headroom_vs_a0": "oracle_crossfit - a0", "headroom_vs_best_fixed": "oracle_crossfit - best_fixed",
    "headroom_vs_best_fixed_ci_lo": "bootstrap CI of the per-query (oracle - best fixed)", "headroom_vs_best_fixed_ci_hi": "",
    "best_fixed_gain_over_a0": "best_fixed - a0", "best_fixed_share_of_oracle_gain": "(best_fixed - a0)/(oracle - a0)",
    "optimism_gap": "oracle_naive - oracle_crossfit", "oracle_go_rate": "share of queries where oracle best != a0 (exact tie rule)",
    "oracle_go_rate_tol": "same with 1e-9 tolerance", "tied_best_frac": "share of queries where >=2 actions tie for best",
    "seedless": "True if the action has no seed variance (a0, prf)", "n_seeds": "number of LLM seeds",
    "sd_gain_single_seed": "SD over queries of single-seed gain (avg over seeds)", "sd_gain_seedmean": "SD over queries of seed-mean gain",
    "within_seed_var": "mean over queries of the variance across seeds (ddof=1)", "noise_share_single": "within_seed_var / total single-seed variance",
    "var_true_single": "max(0, total single-seed var - within-seed var)", "sd_true_single": "sqrt(var_true_single)  <- headline heterogeneity",
    "var_true_seedmean": "max(0, var of seed-mean gain - within-seed var / n_seeds)", "sd_true_seedmean": "sqrt of that",
    "sd_true_over_abs_mean_gain": "sd_true_single / |mean gain|", "rank_by_mean": "rank of the action by mean score (1 = best)",
    "prob_best_fixed": "bootstrap probability this action has the highest mean", "comparison": "reference action vs another action",
    "family": "main = Holm family; pooled_ALL / extra = not corrected", "mean_diff": "mean per-query difference (reference - other)",
    "prob_ref_better": "fraction of bootstrap replicates where reference > other", "mean_cost_s": "mean seconds per query (cost_table, LLM included)",
    "p95_cost_s": "95th percentile seconds", "extra_cost_over_a0_s": "mean (cost - a0 cost)", "gain_per_s_total": "mean gain / mean cost",
    "gain_per_extra_s": "mean gain / extra cost over a0", "measure": "name of the cross-retriever statistic", "value": "statistic", "p_value": "p of the statistic", "note": "free text",
    "n_queries": "queries", "n_docs": "corpus size", "mean_query_words": "mean words per query", "mean_query_chars": "mean characters per query",
    "mean_rel_docs": "mean relevant docs (rel>0) per query", "median_rel_docs": "median", "mean_judged_docs": "mean judged docs per query",
    "share_claim": "heuristic: non-question, >=5 words", "share_question": "heuristic: '?' or question word first", "share_keyword": "heuristic: <=4 words",
    "status": "PASS/FAIL", "quantity": "checked quantity", "expected": "value in iteration_log.md", "observed": "value recomputed here", "abs_diff": "|observed-expected|",
    "qid": "query id", "query": "query text", "n_words": "words in query", "form": "heuristic query form", "oracle_choice": "oracle best action (exact tie rule)",
}


def readme(out, T, args, commit, tables):
    L = ["# A: per-dataset analysis", "",
         "Generated by `analysis_A.py`. Re-running with the same stores and settings gives byte-identical CSV/TeX/JSON (see `A_hashes.json`).", "",
         "## Settings", "",
         f"- stores: `{' '.join(args.stores)}`", f"- retrievers: {args.retrievers}; primary metric ndcg10 (also map100, recall100 in A1/A4)",
         f"- bootstrap B={args.B}, bootstrap seed={args.seed} (paired over queries, resample indices shared per retriever x block)",
         f"- eps for helped/harmed: {list(EPS_LIST)} (helped: gain > eps; harmed: gain < -eps)",
         f"- LLM seeds used: {T['settings']['seeds']} (cross-fit oracle needs >= 2)",
         f"- tie rule: **a0 wins ties** (exact seed-mean argmax, earliest action in order a0,prf,query2doc,hyde,decomp; identical to routing.py). "
         f"Tolerance variants use |x-max| <= {TIE_TOL}; `share_split_tol` splits ties evenly.",
         f"- reference action for pairwise tests: {args.ref_action}", f"- git commit (HEAD of repo, uncommitted edits not captured): {commit}",
         f"- python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}", "",
         "## Caveats", ""] + [f"- {c}" for c in T["summary"]["caveats"]] + ["", "## Files and columns", ""]
    for name, df in tables.items():
        L += [f"### `{name}.csv` ({len(df)} rows)", ""]
        L += [f"- `{c}`: {DESC.get(c, '(see code)')}" for c in df.columns if not (name == "A0_per_query" and c.startswith(("ndcg10_", "gain_", "map100_", "recall100_")))]
        if name == "A0_per_query":
            L += ["- `ndcg10_<action>`, `gain_<action>`, `map100_<action>`, `recall100_<action>`: seed-mean per-query values; `ndcg10_<llm action>_seed<k>`: single seed"]
        L.append("")
    L += ["## Figures / tables", "", "`fig_A1_heatmap_*`, `fig_A2_gain_bars_*`, `fig_A3_win_share`, `fig_A4_headroom_waterfall`, `fig_A5_gain_distributions_*` (PNG + PDF); `tab_A*.tex` (booktabs, need `graphicx`).", ""]
    (out / "README.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--retrievers", nargs="+", default=RETRS, choices=RETRS)
    ap.add_argument("--out", default="results/analysis/A_per_dataset")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-action", default="query2doc", choices=NON_A0)
    ap.add_argument("--skip-checks", action="store_true", help="do not fail (exit 2) if the baselines differ from iteration_log")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    boot = Boot(args.B, args.seed)

    print("loading pools ...", flush=True)
    P = build_pools(args.stores, args.retrievers)
    first = P[args.retrievers[0]]["ndcg10"]
    if len(first.seeds) < 2:
        print("WARNING: <2 LLM seeds -> cross-fit oracle degrades to the naive oracle", file=sys.stderr)
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    T = {"settings": {"stores": args.stores, "retrievers": args.retrievers, "B": args.B, "seed": args.seed, "eps": list(EPS_LIST),
                      "tie_rule": "a0 wins ties (exact); tol variants %g" % TIE_TOL, "seeds": first.seeds, "n_seeds": len(first.seeds),
                      "ref_action": args.ref_action, "git_commit": commit}}
    print("computing tables ...", flush=True)
    T["A0_per_query"] = table_A0(P)
    T["A1"] = table_A1(P, boot)
    T["A2"] = table_A2(P)
    T["A3"] = table_A3(P)
    T["A4"] = table_A4(P, boot)
    T["A5"] = table_A5(P)
    T["A6"], T["A6b"] = tables_A6(P, boot, args.ref_action)
    T["A7"] = table_A7(P)
    T["A8"] = table_A8(P)
    T["A9"] = table_A9(P)
    acc = acceptance(P)
    files = {"A0_per_query": T["A0_per_query"], "A1_action_gains": T["A1"], "A2_helped_harmed": T["A2"], "A3_win_share": T["A3"],
             "A4_headroom": T["A4"], "A5_heterogeneity": T["A5"], "A6_best_action_prob": T["A6"], "A6b_pairwise": T["A6b"],
             "A7_cost": T["A7"], "A8_cross_retriever": T["A8"], "A9_dataset_descriptors": T["A9"], "A_acceptance": acc}
    for name, df in files.items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.8g")
    summary = build_summary(P, T, args.ref_action, acc)
    T["summary"] = summary
    (out / "A_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_all_tex(out, P, T)
    if not args.no_figures:
        make_figures(out, P, T)
    readme(out, T, args, commit, files)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
              if p.suffix in (".csv", ".json", ".tex", ".md") and p.name != "A_hashes.json"}
    (out / "A_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    print(f"\nwrote {out}/")
    if len(acc):
        bad = acc[acc.status == "FAIL"]
        print(f"acceptance vs iteration_log.md: {len(acc) - len(bad)}/{len(acc)} PASS")
        if len(bad):
            print(bad.round(5).to_string(index=False))
            if not args.skip_checks:
                print("\nACCEPTANCE FAILED: do not use these numbers until the mismatch is explained.", file=sys.stderr)
                sys.exit(2)
    else:
        print("acceptance: no matching dataset names, nothing checked")


if __name__ == "__main__":
    main()