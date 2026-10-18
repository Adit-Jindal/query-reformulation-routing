#!/usr/bin/env python3
"""
routing.py -- hierarchical query-reformulation router, built in iterations.

Decision tree (each node chooses between "children"; every child is a group of actions):

    node 1  go_nogo     : a0            | act = {prf, query2doc, hyde, decomp}
    node 2  llm_vs_prf  : prf           | llm = {query2doc, hyde, decomp}        (only for queries routed to act)
    node 3  llm_choice  : query2doc | hyde | decomp                              (only for queries routed to llm)

Iteration L learns nodes 1..L; everything BELOW the last learned node is an ORACLE
(the best action inside the chosen group is assumed known). So iteration 1 measures
"is go/no-go learnable?", iteration 2 adds "LLM vs PRF", iteration 3 removes the last oracle.
The achieved nDCG should fall monotonically from iteration 1 to 3 as assumptions are removed.

Key design choices (all documented in the iteration log the CLI writes):
  * VALUES: V[q, action, seed] = per-query metric (default nDCG@10). a0/prf have no seed noise.
  * ORACLE VALUE of a group = "cross-fitted": pick the best action in the group using the OTHER
    seeds, score it on the held-out seed, average. This removes the winner's-curse optimism of
    "max over noisy actions". The naive (optimistic) oracle is also stored for comparison.
  * TARGETS: each node regresses the child values, CENTERED per query (so query difficulty cancels
    and the model learns the *gain*, not the difficulty). Decision = argmax(predicted gain - lambda * cost).
  * FEATURES: only stage<=1 by default (known before any LLM call).
  * EVALUATION: grouped-by-query K-fold, repeated; protocols `within` (per dataset), `lodo`
    (leave-one-dataset-out, features z-scored per dataset) and `pooled`.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from run_dataset import PRIMARY_ACTIONS, SEEDLESS_ACTIONS, Store

ACTIONS = list(PRIMARY_ACTIONS)  # a0, prf, query2doc, hyde, decomp
LLM_ACTIONS = ["query2doc", "hyde", "decomp"]
GROUPS = {
    "a0": ["a0"], "prf": ["prf"], "query2doc": ["query2doc"], "hyde": ["hyde"], "decomp": ["decomp"],
    "act": ["prf", "query2doc", "hyde", "decomp"], "llm": LLM_ACTIONS, "all": ACTIONS,
}
NODES = [
    {"id": 1, "name": "go_nogo", "children": ["a0", "act"]},
    {"id": 2, "name": "llm_vs_prf", "children": ["prf", "llm"]},
    {"id": 3, "name": "llm_choice", "children": LLM_ACTIONS},
]
GRID = {
    "ridge": [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0)],
    "gbr": [{"max_depth": d, "n_estimators": n} for d in (1, 2) for n in (50, 150)],
}
MODEL_KINDS = ["const", "stump", "ridge", "gbr"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@dataclass
class DS:
    name: str
    retriever: str
    store_path: Path
    qids: list
    queries: list
    seeds: list
    V: np.ndarray          # (n, A, S)
    cost: np.ndarray       # (n, A) seconds
    X: pd.DataFrame        # (n, F) raw features


def load_ds(store_dir, retriever: str, metric: str = "ndcg10", max_stage: int = 1) -> DS:
    s = Store(store_dir)
    m = s.metrics
    m = m[(m.retriever == retriever) & m.action.isin(ACTIONS)]
    qids = s.queries.qid.astype(str).tolist()
    seeds = sorted(int(x) for x in m[m.seed >= 0].seed.unique()) or [0]
    V = np.full((len(qids), len(ACTIONS), len(seeds)), np.nan)
    for i, a in enumerate(ACTIONS):
        sub = m[m.action == a]
        if a in SEEDLESS_ACTIONS:
            vec = sub.groupby("qid")[metric].mean().reindex(qids).to_numpy()
            V[:, i, :] = vec[:, None]
        else:
            piv = sub.pivot_table(index="qid", columns="seed", values=metric).reindex(index=qids, columns=seeds)
            V[:, i, :] = piv.to_numpy()
    try:
        ct = s.cost_table()
        ct = ct[(ct.retriever == retriever) & ct.action.isin(ACTIONS)]
        cm = ct.groupby(["qid", "action"])["total_ms"].mean().unstack("action").reindex(index=qids, columns=ACTIONS) / 1000.0
        cm = cm.fillna(cm.mean()).fillna(0.0)
    except Exception:  # pragma: no cover
        cm = pd.DataFrame(0.0, index=qids, columns=ACTIONS)
    cols = s.feature_columns(max_stage)
    F = s.features.set_index("qid").reindex(qids)[cols]
    if max_stage >= 2:
        fs = s.features_seed
        num = [c for c in fs.columns if c.startswith("s2_")]
        F = F.join(fs.groupby("qid")[num].mean(), how="left")
    return DS(s.manifest["dataset"], retriever, Path(store_dir), qids, s.queries.text.tolist(), seeds, V,
              cm.to_numpy(dtype=float), F)


# ----------------------------------------------------------------------------
# Oracle values
# ----------------------------------------------------------------------------
def group_value(V: np.ndarray, g: list, crossfit: bool = True):
    """Value of the best action inside group g (indices into axis 1).
    Returns (value (n,), choice (n,) = argmax of the all-seed mean, i.e. the 'naive' choice)."""
    n, _, S = V.shape
    Vg = V[:, g, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_all = np.nanmean(Vg, axis=2)
        choice = np.argmax(np.where(np.isnan(mean_all), -np.inf, mean_all), axis=1)
        if len(g) == 1 or not crossfit or S < 2:
            return mean_all[np.arange(n), choice], choice
        vals = np.full((n, S), np.nan)
        for e in range(S):
            sel = np.nanmean(Vg[:, :, [s for s in range(S) if s != e]], axis=2)
            a = np.argmax(np.where(np.isnan(sel), -np.inf, sel), axis=1)
            vals[:, e] = Vg[np.arange(n), a, e]
        return np.nanmean(vals, axis=1), choice


def build_targets(V, cost, oracle="crossfit"):
    idx = {a: i for i, a in enumerate(ACTIONS)}
    T, choice, G, naive = {}, {}, {}, {}
    for name, acts in GROUPS.items():
        g = [idx[a] for a in acts]
        val, ch = group_value(V, g, crossfit=(oracle == "crossfit"))
        T[name] = val
        naive[name] = group_value(V, g, crossfit=False)[0]
        choice[name] = np.array(acts, dtype=object)[ch]
        G[name] = cost[np.arange(len(val)), np.array(g)[ch]]
    return T, choice, G, naive


# ----------------------------------------------------------------------------
# Pool (one or more datasets concatenated)
# ----------------------------------------------------------------------------
@dataclass
class Pool:
    retriever: str
    ds_names: list
    ds_id: np.ndarray
    qid: np.ndarray           # "dataset:qid"
    raw_qid: np.ndarray
    query: np.ndarray
    seeds: list
    V: np.ndarray
    cost: np.ndarray
    X: np.ndarray
    feat_names: list
    T: dict = field(default_factory=dict)
    choice: dict = field(default_factory=dict)
    G: dict = field(default_factory=dict)
    naive: dict = field(default_factory=dict)
    store_paths: dict = field(default_factory=dict)
    oracle: str = "crossfit"
    n_dropped: int = 0


def build_pool(store_dirs, retriever, metric="ndcg10", max_stage=1, normalize="dataset", oracle="crossfit") -> Pool:
    dss = [load_ds(d, retriever, metric, max_stage) for d in store_dirs]
    seeds = sorted(set.intersection(*[set(d.seeds) for d in dss]))
    if not seeds:
        raise ValueError("datasets share no common LLM seeds")
    cols = [c for c in dss[0].X.columns if all(c in d.X.columns for d in dss)]
    Vs, Xs, ids, qids, rawq, queries = [], [], [], [], [], []
    for i, d in enumerate(dss):
        Vs.append(d.V[:, :, [d.seeds.index(s) for s in seeds]])
        X = d.X[cols].to_numpy(dtype=float)
        if normalize == "dataset":
            mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
            X = (X - mu) / np.where(sd < 1e-9, 1.0, sd)
        Xs.append(X)
        ids += [i] * len(d.qids)
        qids += [f"{d.name}:{q}" for q in d.qids]
        rawq += d.qids
        queries += d.queries
    V, cost, X = np.concatenate(Vs), np.concatenate([d.cost for d in dss]), np.concatenate(Xs)
    keep_cols = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).any() and np.nanstd(X[:, j]) > 1e-9]
    X, cols = X[:, keep_cols], [cols[j] for j in keep_cols]
    X = np.nan_to_num(X, nan=0.0) if normalize == "dataset" else np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    T, choice, G, naive = build_targets(V, cost, oracle)
    ok = np.ones(len(V), dtype=bool)
    for arr in list(T.values()) + list(G.values()):
        ok &= np.isfinite(arr)
    sel = np.flatnonzero(ok)
    f = lambda a: a[sel]
    return Pool(retriever, [d.name for d in dss], f(np.array(ids)), f(np.array(qids, dtype=object)), f(np.array(rawq, dtype=object)),
                f(np.array(queries, dtype=object)), seeds, f(V), f(cost), f(X), cols,
                {k: f(v) for k, v in T.items()}, {k: f(v) for k, v in choice.items()}, {k: f(v) for k, v in G.items()},
                {k: f(v) for k, v in naive.items()}, {d.name: d.store_path for d in dss}, oracle, int((~ok).sum()))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def _mk(kind, p, seed):
    if kind == "ridge":
        return Ridge(alpha=p["alpha"])
    return GradientBoostingRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=10, random_state=seed)


def _fit_targets(Tn):
    """Binary node: one target = child1 - child0 (the gain). Multi-way: per-child values centered per query."""
    if Tn.shape[1] == 2:
        return (Tn[:, 1] - Tn[:, 0])[:, None]
    return Tn - Tn.mean(axis=1, keepdims=True)


def tune_params(kind, X, Tn, seed):
    if kind not in GRID or len(X) < 40:
        return GRID.get(kind, [None])[0]
    Y = _fit_targets(Tn)
    best, best_err = None, np.inf
    for p in GRID[kind]:
        err = 0.0
        for tr, va in KFold(3, shuffle=True, random_state=seed).split(X):
            for j in range(Y.shape[1]):
                err += float(((_mk(kind, p, seed).fit(X[tr], Y[tr, j]).predict(X[va]) - Y[va, j]) ** 2).sum())
        if err < best_err:
            best, best_err = p, err
    return best


class NodeModel:
    def __init__(self, kind, params=None, seed=0):
        self.kind, self.params, self.seed = kind, params, seed

    def fit(self, X, Tn):
        self.C = Tn.shape[1]
        Y = _fit_targets(Tn)
        if self.kind == "const":
            self.mu = Y.mean(axis=0)
        elif self.kind == "stump":  # QPP-style single-feature threshold rule (binary node only)
            d, best = Y[:, 0], (-np.inf, 0, 0.0, 1)
            for j in range(X.shape[1]):
                for thr in np.quantile(X[:, j], np.linspace(0.1, 0.9, 9)):
                    for sign in (1, -1):
                        gain = d[(sign * (X[:, j] - thr)) > 0].sum()
                        if gain > best[0]:
                            best = (gain, j, thr, sign)
            _, self.j, self.thr, self.sign = best
        else:
            self.models = [_mk(self.kind, self.params, self.seed).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict(self, X):
        n = len(X)
        if self.kind == "const":
            Y = np.tile(self.mu, (n, 1))
        elif self.kind == "stump":
            Y = np.where(self.sign * (X[:, self.j] - self.thr) > 0, 1.0, -1.0)[:, None]
        else:
            Y = np.column_stack([m.predict(X) for m in self.models])
        if self.C == 2:
            return np.column_stack([-Y[:, 0] / 2, Y[:, 0] / 2])
        return Y


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
def make_folds(pool: Pool, protocol: str, k: int, rng):
    ids, folds = pool.ds_id, []
    if protocol == "within":
        for d in np.unique(ids):
            idx = np.flatnonzero(ids == d)
            if len(idx) < 4:
                continue
            for tr, te in KFold(min(k, len(idx)), shuffle=True, random_state=int(rng.integers(1 << 30))).split(idx):
                folds.append((idx[tr], idx[te]))
    elif protocol == "lodo":
        if len(np.unique(ids)) < 2:
            raise ValueError("lodo needs at least 2 datasets")
        for d in np.unique(ids):
            folds.append((np.flatnonzero(ids != d), np.flatnonzero(ids == d)))
    elif protocol == "pooled":
        idx = np.arange(len(ids))
        for tr, te in KFold(k, shuffle=True, random_state=int(rng.integers(1 << 30))).split(idx):
            folds.append((idx[tr], idx[te]))
    else:
        raise ValueError(protocol)
    return folds


@dataclass
class CVResult:
    protocol: str
    kind: str
    permuted: bool
    P: list          # per repeat: {node_id: (n, C) predicted centered child values}
    CC: list         # per repeat: {node_id: (n, C) train-mean child cost (s) used for the lambda term}
    best_fixed: np.ndarray  # (R, n) value of the train-selected best single action
    best_fixed_name: np.ndarray
    params: list


def run_cv(pool: Pool, kind: str, protocol: str, folds: int = 5, repeats: int = 3, seed: int = 0,
           tune: bool = True, permute: bool = False) -> CVResult:
    rng = np.random.default_rng(seed)
    n = len(pool.qid)
    X = pool.X.copy()
    if permute:  # null control: break the feature<->target link, within each dataset
        for d in np.unique(pool.ds_id):
            idx = np.flatnonzero(pool.ds_id == d)
            X[idx] = X[rng.permutation(idx)]
    Vmean = np.nanmean(pool.V, axis=2)
    P_reps, CC_reps, BF, BFN, params_log = [], [], [], [], []
    for r in range(repeats if protocol != "lodo" else 1):
        P = {nd["id"]: np.zeros((n, len(nd["children"]))) for nd in NODES}
        CC = {nd["id"]: np.zeros((n, len(nd["children"]))) for nd in NODES}
        bf, bfn = np.zeros(n), np.empty(n, dtype=object)
        for tr, te in make_folds(pool, protocol, folds, rng):
            best_a = int(np.argmax(np.nanmean(Vmean[tr], axis=0)))
            bf[te] = Vmean[te, best_a]
            bfn[te] = ACTIONS[best_a]
            for nd in NODES:
                Tn = np.column_stack([pool.T[c][tr] for c in nd["children"]])
                nk = kind if not (kind == "stump" and nd["id"] != 1) else "const"
                p = tune_params(nk, X[tr], Tn, seed + r) if tune else GRID.get(nk, [None])[0]
                params_log.append({"repeat": r, "node": nd["id"], "params": p})
                P[nd["id"]][te] = NodeModel(nk, p, seed + r).fit(X[tr], Tn).predict(X[te])
                CC[nd["id"]][te] = np.column_stack([pool.G[c][tr].mean() for c in nd["children"]])
        P_reps.append(P)
        CC_reps.append(CC)
        BF.append(bf)
        BFN.append(bfn)
    return CVResult(protocol, kind, permute, P_reps, CC_reps, np.array(BF), np.array(BFN), params_log)


# ----------------------------------------------------------------------------
# Policy evaluation (hierarchical chain, oracle below the last learned node)
# ----------------------------------------------------------------------------
def decisions(P, CC, lam):
    return {k: np.argmax(P[k] - lam * CC[k], axis=1) for k in P}


def chain_eval(pool: Pool, level: int, dec: dict):
    n = len(pool.qid)
    val, cost, lab = np.full(n, np.nan), np.full(n, np.nan), np.empty(n, dtype=object)

    def setv(mask, grp):
        if not mask.any():
            return
        val[mask], cost[mask] = pool.T[grp][mask], pool.G[grp][mask]
        lab[mask] = grp if grp in ACTIONS else np.array([f"{grp}:{c}" for c in pool.choice[grp][mask]], dtype=object)

    setv(dec[1] == 0, "a0")
    act = dec[1] == 1
    if level == 1:
        setv(act, "act")
        return val, cost, lab
    setv(act & (dec[2] == 0), "prf")
    llm = act & (dec[2] == 1)
    if level == 2:
        setv(llm, "llm")
        return val, cost, lab
    for j, a in enumerate(LLM_ACTIONS):
        setv(llm & (dec[3] == j), a)
    return val, cost, lab


def evaluate(pool: Pool, res: CVResult, level: int, lambdas) -> dict:
    """Per-query values/costs/go averaged over repeats, for each lambda. Returns arrays shaped (L, n)."""
    R, n, L = len(res.P), len(pool.qid), len(lambdas)
    val, cost, go = np.zeros((L, n)), np.zeros((L, n)), np.zeros((L, n))
    labels0 = None
    for r in range(R):
        for li, lam in enumerate(lambdas):
            v, c, lab = chain_eval(pool, level, decisions(res.P[r], res.CC[r], lam))
            val[li] += v / R
            cost[li] += c / R
            go[li] += (lab != "a0").astype(float) / R
            if r == 0 and li == 0:
                labels0 = lab
    return {"value": val, "cost": cost, "go": go, "labels0": labels0}


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def paired_bootstrap(diff, B=10000, seed=0):
    """Mean of paired differences with percentile CI and two-sided bootstrap p-value."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(B, len(diff)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(min(max(p, 1.0 / B), 1.0))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(pool: Pool, res: CVResult, level: int, lambdas, B: int = 5000) -> pd.DataFrame:
    ev = evaluate(pool, res, level, lambdas)
    bf = res.best_fixed.mean(axis=0)
    oracle, a0 = pool.T["all"], pool.T["a0"]
    rows = []
    blocks = [(name, pool.ds_id == i) for i, name in enumerate(pool.ds_names)] + [("ALL", np.ones(len(pool.qid), bool))]
    for li, lam in enumerate(lambdas):
        for name, m in blocks:
            v = ev["value"][li][m]
            row = {"dataset": name, "level": level, "lam": lam, "n": int(m.sum()), "a0": a0[m].mean(),
                   "best_fixed": bf[m].mean(), "policy": v.mean(), "oracle": oracle[m].mean(),
                   "recovered": (v.mean() - a0[m].mean()) / max(oracle[m].mean() - a0[m].mean(), 1e-9),
                   "go_rate": ev["go"][li][m].mean(), "cost_s": ev["cost"][li][m].mean(), "cost_a0_s": pool.G["a0"][m].mean()}
            if li == 0:
                d0 = paired_bootstrap(v - a0[m], B)
                d1 = paired_bootstrap(v - bf[m], B)
                row.update({"d_a0": d0[0], "d_a0_lo": d0[1], "d_a0_hi": d0[2], "p_a0": d0[3],
                            "d_bf": d1[0], "d_bf_lo": d1[1], "d_bf_hi": d1[2], "p_bf": d1[3]})
            rows.append(row)
    df = pd.DataFrame(rows)
    m0 = df.lam == lambdas[0]
    ps = df.loc[m0, ["p_a0", "p_bf"]].to_numpy().ravel()
    adj = holm(ps).reshape(-1, 2)
    df.loc[m0, "p_a0_holm"], df.loc[m0, "p_bf_holm"] = adj[:, 0], adj[:, 1]
    return df


def node_diagnostics(pool: Pool, res: CVResult, levels=(1, 2, 3)) -> pd.DataFrame:
    """How well does each node's predicted gain match the true (cross-fitted) child values?"""
    rows = []
    P = {k: np.mean([p[k] for p in res.P], axis=0) for k in res.P[0]}
    oracle_child = {nd["id"]: np.argmax(np.column_stack([pool.T[c] for c in nd["children"]]), axis=1) for nd in NODES}
    for nd in NODES:
        k, ch = nd["id"], nd["children"]
        Tn = np.column_stack([pool.T[c] for c in ch])
        routed = np.ones(len(Tn), bool)
        if k == 2:
            routed = oracle_child[1] == 1
        if k == 3:
            routed = (oracle_child[1] == 1) & (oracle_child[2] == 1)
        for scope, m in (("all", np.ones(len(Tn), bool)), ("oracle_routed", routed)):
            if m.sum() < 8:
                continue
            row = {"node": k, "name": nd["name"], "scope": scope, "n": int(m.sum()),
                   "acc_vs_oracle": float((np.argmax(P[k][m], axis=1) == oracle_child[k][m]).mean()),
                   "acc_majority": float(np.bincount(oracle_child[k][m], minlength=len(ch)).max() / m.sum())}
            if len(ch) == 2:
                gain = Tn[m, 1] - Tn[m, 0]
                score = P[k][m, 1] - P[k][m, 0]
                row["spearman"] = float(spearmanr(score, gain)[0])
                if len(np.unique(gain > 0)) == 2:
                    row["auc"] = float(roc_auc_score(gain > 0, score))
            else:
                cen = Tn[m] - Tn[m].mean(axis=1, keepdims=True)
                row["spearman"] = float(np.nanmean([spearmanr(P[k][m, j], cen[:, j])[0] for j in range(len(ch))]))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Failure sheet
# ----------------------------------------------------------------------------
def _context(pool: Pool, ds_name: str, qids, labels):
    """Best-effort extra text for manual analysis: passage/sub-questions, PRF terms, top-3 titles."""
    out = {q: {} for q in qids}
    try:
        sp = pool.store_paths[ds_name]
        s = Store(sp)
        g = s.generations
        for q in qids:
            gq = g[(g.qid == q) & g.ok]
            for kind, col, key in (("passage", "text", "passage"), ("decomp", "subqs", "subqs")):
                r = gq[(gq.kind == kind)].sort_values("seed")
                if len(r):
                    v = r.iloc[0][col]
                    out[q][key] = v if isinstance(v, str) else " | ".join(v)
        st = s.search_texts
        st = st[(st.action == "prf") & (st.retriever == "bm25") & st.qid.isin(qids)]
        for _, r in st.iterrows():
            try:
                out[r.qid]["prf_terms"] = ", ".join(t for t, _ in json.loads(r.aux)["terms"])
            except Exception:
                pass
        runs = s.runs
        runs = runs[runs.qid.isin(qids) & (runs["rank"] <= 3) & (runs.subq_idx == -1) & (runs.retriever == pool.retriever)]
        titles = {}
        corpus = Path(sp).parent.parent / "data" / ds_name / "corpus.jsonl"
        ids = s.corpus_ids
        need = {ids[int(i)] for i in runs.doc_idx.unique()}
        if corpus.exists():
            with open(corpus, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if str(d["_id"]) in need:
                        titles[str(d["_id"])] = (d.get("title") or d.get("text", ""))[:80]
        def top3(q, action, seed):
            r = runs[(runs.qid == q) & (runs.action == action) & (runs.seed == seed)].sort_values("rank")
            return " || ".join(titles.get(ids[int(i)], ids[int(i)]) for i in r.doc_idx)
        for q, lab in zip(qids, labels):
            out[q]["top3_a0"] = top3(q, "a0", -1)
            act = lab.split(":")[-1] if lab != "a0" else None
            if act in ACTIONS and act != "a0":
                out[q]["top3_chosen"] = top3(q, act, -1 if act in SEEDLESS_ACTIONS else 0)
    except Exception as e:  # context is a convenience, never fatal
        out["_error"] = str(e)
    return out


def failure_sheet(pool: Pool, res: CVResult, level: int, per_type: int = 15, n_success: int = 12,
                  eps: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Sample cases to label by hand. Failure types follow the decision tree; half the sample per
    type is the largest-regret cases, half is random (so the analysis is not only worst cases)."""
    rng = np.random.default_rng(seed)
    dec = decisions(res.P[0], res.CC[0], 0.0)
    val, cost, lab = chain_eval(pool, level, dec)
    a0, best = pool.T["a0"], pool.T["all"]
    regret = best - val
    fam_best = np.where(pool.T["prf"] >= pool.T["llm"], "prf", "llm")
    typ = np.empty(len(val), dtype=object)
    for i in range(len(val)):
        is_go = lab[i] != "a0"
        if is_go and val[i] < a0[i] - eps:
            typ[i] = "harmful_go"
        elif not is_go and best[i] - a0[i] > eps:
            typ[i] = "missed_go"
        elif is_go and level >= 2 and regret[i] > eps and not str(lab[i]).startswith(fam_best[i]) and lab[i] != fam_best[i]:
            typ[i] = "wrong_family"
        elif is_go and level >= 3 and regret[i] > eps:
            typ[i] = "wrong_llm_action"
        elif (is_go and val[i] > a0[i] + eps) or (not is_go and best[i] - a0[i] <= eps):
            typ[i] = "success"
        else:
            typ[i] = "neutral"
    picks = []
    for t, k in [("harmful_go", per_type), ("missed_go", per_type), ("wrong_family", per_type),
                 ("wrong_llm_action", per_type), ("success", n_success)]:
        idx = np.flatnonzero(typ == t)
        if len(idx) == 0:
            continue
        order = idx[np.argsort(-regret[idx])] if t != "success" else rng.permutation(idx)
        top = list(order[: (k + 1) // 2])
        rest = [i for i in idx if i not in top]
        picks += top + list(rng.permutation(rest)[: k - len(top)])
    rows = []
    Vm = np.nanmean(pool.V, axis=2)
    for i in picks:
        r = {"dataset": pool.ds_names[pool.ds_id[i]], "qid": pool.raw_qid[i], "retriever": pool.retriever,
             "outcome": typ[i], "chosen": lab[i], "oracle_best_action": pool.choice["all"][i],
             "policy_value": round(float(val[i]), 4), "oracle_value": round(float(best[i]), 4),
             "regret": round(float(regret[i]), 4), "query": pool.query[i]}
        for j, a in enumerate(ACTIONS):
            r[f"ndcg_{a}"] = round(float(Vm[i, j]), 4)
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        ctx = {}
        for name in df.dataset.unique():
            sub = df[df.dataset == name]
            ctx.update({(name, q): v for q, v in _context(pool, name, sub.qid.tolist(), sub.chosen.tolist()).items() if isinstance(v, dict)})
        for col in ("passage", "subqs", "prf_terms", "top3_a0", "top3_chosen"):
            df[col] = [ctx.get((d, q), {}).get(col, "") for d, q in zip(df.dataset, df.qid)]
        df["cause_label"] = ""
        df["notes"] = ""
    return df