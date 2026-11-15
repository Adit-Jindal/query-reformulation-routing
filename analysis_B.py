#!/usr/bin/env python3
"""
analysis_B.py -- Workstream B: query-type analysis (exploratory, hypothesis-generating).

Run (from the project folder that has store/, routing.py, run_dataset.py):

  python analysis_B.py --stores store/scifact store/trec-covid store/nfcorpus store/fiqa store/webis-touche2020 \
        --retrievers bm25 dense --out results/analysis/B_query_types

  add --quick for a fast smoke test (B=200 bootstraps, 10 permutations; do NOT use for reported numbers).

No LLM calls. Reads stores via routing.py (build_pool) and run_dataset.Store.
Everything is seeded: re-running with the same settings gives identical CSVs (see B_run_manifest.json md5s).

Forms check: first run writes B0_form_check_sample.csv (30 queries/dataset). Fill the `hand_label` column
(claim / question / keyword), then re-run (or run --validate-forms <file>) to get the rule's agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing as R  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
ACT = list(R.ACTIONS)            # a0, prf, query2doc, hyde, decomp
NONNULL = ACT[1:]
CONT = ["s0_len_words", "s0_idf_mean", "s0_idf_max", "s0_oov_frac",
        "s1_top1", "s1_gap1_10", "s1_wig", "s1_nqc", "s1_entropy100", "s1_overlap10"]
CATS = ["form", "has_digit", "starts_qword"]
DESCS = CONT + CATS
REG_COLS = CONT + ["has_digit_i"]
BAND_EDGES = [-1e-9, 0.01, 0.30, 0.70, 1.0 + 1e-9]
BAND_NAMES = ["zero", "low", "mid", "high"]
ORDER = {"low": 0, "mid": 1, "high": 2, "zero": 0, "no": 0, "yes": 1, "claim": 0, "question": 1, "keyword": 2}
KNOWN = {"bm25": {"a0": 0.3616, "query2doc": 0.4047, "oracle": 0.4645},
         "dense": {"a0": 0.4134, "query2doc": 0.4409, "oracle": 0.5214}}
QW = {"who", "what", "when", "where", "why", "how", "which", "is", "are", "do", "does", "can", "should",
      "would", "will", "did", "was", "were", "has", "have", "could", "may", "might"}
S = {}  # settings, filled in main()


def bkey(b):
    return (ORDER.get(b, 5), str(b))


def s1_map(retr):
    return {"s1_top1": f"s1_{retr}_top1", "s1_gap1_10": f"s1_{retr}_gap1_10", "s1_wig": f"s1_{retr}_wig",
            "s1_nqc": f"s1_{retr}_nqc", "s1_entropy100": f"s1_{retr}_entropy100", "s1_overlap10": "s1_overlap10"}


def query_form(q: str) -> str:
    """Rule-based form: question (has '?' or starts with a question/aux word), keyword (<=4 words), else claim."""
    w = re.findall(r"[a-z0-9'\-]+", q.lower())
    if "?" in q or (w and w[0] in QW):
        return "question"
    return "keyword" if len(w) <= 4 else "claim"


# ----------------------------------------------------------------------------- stats helpers
def next_rng(counter):
    counter[0] += 1
    return np.random.default_rng([S["seed"], counter[0]])


def boot_means(A, B, rng):
    n, k = A.shape
    out = np.empty((B, k))
    chunk = max(1, int(4e6 // max(n * k, 1)))
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        idx = rng.integers(0, n, size=(e - s, n))
        out[s:e] = A[idx].mean(axis=1)
    return out


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    o = np.argsort(pv, kind="stable")
    m = len(pv)
    adj = np.minimum.accumulate((pv[o] * m / (np.arange(m) + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def cell_rows(G, cols, meta, rng, rank_cols=None):
    """One row per column of G (e.g. per action): n, mean, bootstrap CI/p, d, helped/harmed. n<min_n -> too small."""
    B, min_n, eps = S["B"], S["min_n"], S["eps"]
    A = G[cols].to_numpy(float)
    n = len(A)
    small = n < min_n
    k = len(cols)
    mean = A.mean(0) if n else np.full(k, np.nan)
    lo, hi, p = np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
    best = "too small" if small else ""
    if not small and n >= 2:
        bm = boot_means(A, B, rng)
        lo, hi = np.percentile(bm, [2.5, 97.5], axis=0)
        p = np.clip(2 * np.minimum((bm <= 0).mean(0), (bm >= 0).mean(0)), 1.0 / B, 1.0)
        if rank_cols:
            ii = [cols.index(c) for c in rank_cols]
            j = int(np.argmax(mean[ii]))
            best = rank_cols[j] if mean[ii][j] > 0 else "a0"
    sd = A.std(0, ddof=1) if n > 1 else np.full(k, np.nan)
    rows = []
    for j, c in enumerate(cols):
        rows.append({**meta, "action": c, "n": n, "mean_gain": mean[j], "ci_lo": lo[j], "ci_hi": hi[j], "p_boot": p[j],
                     "cohens_d": mean[j] / sd[j] if sd[j] and sd[j] > 0 else np.nan,
                     "helped": float((A[:, j] > eps).mean()) if n else np.nan,
                     "harmed": float((A[:, j] < -eps).mean()) if n else np.nan,
                     "best_action": best, "too_small": small})
    return rows


# ----------------------------------------------------------------------------- loading / descriptors
def load_retriever(stores, retr, metric):
    pool = R.build_pool(stores, retr, metric, 1, "dataset", "crossfit")
    raw = []
    for sp in stores:
        st = R.Store(sp)
        F = st.features.copy()
        F.index = [f"{st.manifest['dataset']}:{q}" for q in F["qid"].astype(str)]
        raw.append(F)
    raw = pd.concat(raw).reindex(pool.qid.astype(str))
    Vm = np.nanmean(pool.V, axis=2)
    D = pd.DataFrame({"retriever": retr, "dataset": [pool.ds_names[i] for i in pool.ds_id],
                      "qid": pool.raw_qid.astype(str), "query": pool.query})
    D["s0_len_words"] = raw["s0_len_words"].to_numpy()
    D["s0_n_digit_tokens"] = raw["s0_n_digit_tokens"].to_numpy()
    D["s0_has_qmark"] = raw["s0_has_qmark"].to_numpy()
    D["s0_starts_qword_f"] = raw["s0_starts_qword"].to_numpy()
    for c in ("s0_idf_mean", "s0_idf_max", "s0_oov_frac"):
        D[c] = raw[c].to_numpy()
    for g, col in s1_map(retr).items():
        D[g] = raw[col].to_numpy()
    D["form"] = D["query"].map(query_form)
    D["has_digit"] = np.where(D["s0_n_digit_tokens"] > 0, "yes", "no")
    D["has_digit_i"] = (D["s0_n_digit_tokens"] > 0).astype(float)
    D["starts_qword"] = np.where(D["s0_starts_qword_f"] > 0, "yes", "no")
    D["a0_ndcg"] = Vm[:, 0]
    D["a0_band"] = pd.cut(D["a0_ndcg"], BAND_EDGES, labels=BAND_NAMES).astype(object)
    for j, a in enumerate(ACT):
        if j:
            D[f"g_{a}"] = Vm[:, j] - Vm[:, 0]
    D["oracle_best"] = pool.choice["all"]
    D["oracle_go"] = (D["oracle_best"] != "a0").astype(int)
    D["oracle_value"] = pool.T["all"]
    # seed noise of per-query gain (within-query SD across seeds)
    noise = {}
    for j, a in enumerate(NONNULL, start=1):
        gs = pool.V[:, j, :] - pool.V[:, 0, :]
        noise[a] = float(np.sqrt(np.nanmean(np.nanvar(gs, axis=1, ddof=1)))) if gs.shape[1] > 1 else float("nan")
    return pool, D, noise


def assign_buckets(D):
    for c in CONT:
        lab = pd.Series([np.nan] * len(D), index=D.index, dtype=object)
        for ds, idx in D.groupby("dataset").groups.items():
            x = D.loc[idx, c]
            try:
                q = pd.qcut(x, 3, labels=False, duplicates="drop")
            except ValueError:
                continue
            if q.notna().sum() == 0:
                continue
            nb = int(np.nanmax(q)) + 1
            names = {3: ["low", "mid", "high"], 2: ["low", "high"]}.get(nb)
            if names is None:
                continue
            lab.loc[idx] = [names[int(v)] if pd.notna(v) else np.nan for v in q]
        D[f"b_{c}"] = lab
    for c in CATS:
        D[f"b_{c}"] = D[c]
    D["b_a0"] = D["a0_band"]
    return D


def gains(D):
    G = D[[f"g_{a}" for a in NONNULL]].copy()
    G.columns = NONNULL
    return G


# ----------------------------------------------------------------------------- B1 / B5 / B6
def run_b1(D, retr, counter):
    G = gains(D)
    rows = []
    scopes = [(d, (D.dataset == d).to_numpy()) for d in D.dataset.unique() if (D.dataset == d).sum() >= S["min_ds_n"]]
    scopes.append(("POOLED", np.ones(len(D), bool)))
    for scope, m in scopes:
        for desc in DESCS:
            col = f"b_{desc}"
            vals = D.loc[m, col].dropna().unique()
            for b in sorted(vals, key=bkey):
                sel = m & (D[col] == b).to_numpy()
                rows += cell_rows(G[sel], NONNULL, {"retriever": retr, "dataset": scope, "descriptor": desc, "bucket": b},
                                  next_rng(counter), NONNULL)
    return rows


def run_b5(D, retr, counter):
    G = gains(D)
    rows = []
    scopes = [(d, (D.dataset == d).to_numpy()) for d in D.dataset.unique()] + [("POOLED", np.ones(len(D), bool))]
    for scope, m in scopes:
        for b in BAND_NAMES:
            sel = m & (D["a0_band"] == b).to_numpy()
            if sel.sum() == 0:
                continue
            rows += cell_rows(G[sel], NONNULL, {"retriever": retr, "dataset": scope, "descriptor": "a0_band", "bucket": b},
                              next_rng(counter), NONNULL)
    return rows


def run_b6(D, retr, counter):
    H = pd.DataFrame({"query2doc": D["g_query2doc"], "hyde": D["g_hyde"], "q2d_minus_hyde": D["g_query2doc"] - D["g_hyde"]})
    cols = ["query2doc", "hyde", "q2d_minus_hyde"]
    rows = []
    scopes = [(d, (D.dataset == d).to_numpy()) for d in D.dataset.unique()] + [("POOLED", np.ones(len(D), bool))]
    absbin = pd.cut(D["s0_len_words"], [0, 5, 10, 20, 1e9], labels=["<=5w", "6-10w", "11-20w", ">20w"]).astype(object)
    for scope, m in scopes:
        if scope == "POOLED" or m.sum() >= S["min_ds_n"]:
            for b in ["low", "mid", "high"]:
                sel = m & (D["b_s0_len_words"] == b).to_numpy()
                if sel.sum():
                    rows += [{**r, "kind": "tercile"} for r in cell_rows(H[sel], cols, {"retriever": retr, "dataset": scope, "bucket": b}, next_rng(counter))]
        for b in ["<=5w", "6-10w", "11-20w", ">20w"]:
            sel = m & (absbin == b).to_numpy()
            if sel.sum():
                rows += [{**r, "kind": "abs_len_bin"} for r in cell_rows(H[sel], cols, {"retriever": retr, "dataset": scope, "bucket": b}, next_rng(counter))]
        if m.sum() >= 10:
            for c in cols:
                rho, p = spearmanr(D.loc[m, "s0_len_words"], H.loc[m, c])
                rows.append({"retriever": retr, "dataset": scope, "kind": "spearman_len_vs_gain", "bucket": "", "action": c,
                             "n": int(m.sum()), "mean_gain": rho, "p_boot": p, "too_small": False})
    return rows


# ----------------------------------------------------------------------------- B2 regression
def zdf(D, cols):
    Z = pd.DataFrame(index=D.index)
    for c in cols:
        x = D[c].astype(float)
        g = x.groupby(D["dataset"])
        sd = g.transform("std").replace(0, np.nan)
        Z[c] = ((x - g.transform("mean")) / sd).fillna(0.0)
    return Z


def rss(X, Y):
    c = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ c
    return (r ** 2).sum(axis=0)


def run_b2(D, retr, counter):
    Z = zdf(D, REG_COLS)
    dums = pd.get_dummies(D["dataset"], dtype=float).iloc[:, 1:]
    X = np.column_stack([np.ones(len(D)), Z.to_numpy(), dums.to_numpy()])
    Y = gains(D).to_numpy(float)
    n, p = X.shape
    sdy = Y.std(axis=0, ddof=1)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    rng = next_rng(counter)
    boots = np.empty((S["B"], p, Y.shape[1]))
    for b in range(S["B"]):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        Xw = X * w[:, None]
        boots[b] = np.linalg.lstsq(Xw.T @ X, Xw.T @ Y, rcond=None)[0]
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    pv = np.clip(2 * np.minimum((boots <= 0).mean(0), (boots >= 0).mean(0)), 1.0 / S["B"], 1.0)
    rows = []
    r2 = 1 - rss(X, Y) / ((Y - Y.mean(0)) ** 2).sum(0)
    for j, a in enumerate(NONNULL):
        for i, term in enumerate(REG_COLS, start=1):
            rows.append({"retriever": retr, "action": a, "term": term, "coef_per_1sd": coef[i, j], "std_beta": coef[i, j] / sdy[j],
                         "ci_lo": lo[i, j], "ci_hi": hi[i, j], "ci_lo_std": lo[i, j] / sdy[j], "ci_hi_std": hi[i, j] / sdy[j],
                         "p_boot": pv[i, j], "n": n})
        rows.append({"retriever": retr, "action": a, "term": "_R2", "coef_per_1sd": r2[j], "n": n})
    # interaction (descriptor x dataset) F-tests
    inter = []
    if dums.shape[1] >= 1:
        for c in REG_COLS:
            Xi = np.column_stack([X, Z[c].to_numpy()[:, None] * dums.to_numpy()])
            q, df2 = dums.shape[1], n - Xi.shape[1]
            rr, rf = rss(X, Y), rss(Xi, Y)
            Fv = ((rr - rf) / q) / (rf / df2)
            for j, a in enumerate(NONNULL):
                inter.append({"retriever": retr, "action": a, "descriptor": c, "F": Fv[j], "df1": q, "df2": df2,
                              "p": float(f_dist.sf(Fv[j], q, df2))})
    # VIF
    vif = []
    Zm = Z.to_numpy()
    for i, c in enumerate(REG_COLS):
        oth = np.column_stack([np.ones(n), np.delete(Zm, i, axis=1)])
        r = 1 - rss(oth, Zm[:, [i]])[0] / ((Zm[:, i] - Zm[:, i].mean()) ** 2).sum()
        vif.append({"retriever": retr, "descriptor": c, "VIF": 1 / max(1 - r, 1e-9)})
    return rows, inter, vif


# ----------------------------------------------------------------------------- B3 / B4 trees
def tree_feats(pool, retr):
    keep = [j for j, c in enumerate(pool.feat_names)
            if c.startswith("s0_") or c == "s1_overlap10" or c == "s1_overlap100" or c.startswith(f"s1_{retr}_")]
    return pool.X[:, keep], [pool.feat_names[j] for j in keep]


def _clf(seed):
    return DecisionTreeClassifier(max_depth=S["depth"], min_samples_leaf=S["leaf"], random_state=seed)


def cv_eval(X, y, groups, protocol, seed, perm=False, prng=None, auc_pos=None):
    n = len(y)
    if protocol == "kfold":
        reps = [list(KFold(5, shuffle=True, random_state=seed + r).split(np.arange(n))) for r in range(1 if perm else 3)]
    else:
        reps = [[(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]]
    accs, majs, aucs, per_group = [], [], [], {}
    for fold_list in reps:
        correct = maj_correct = 0
        proba = np.full(n, np.nan)
        for tr, te in fold_list:
            ytr = prng.permutation(y[tr]) if perm else y[tr]
            clf = _clf(seed).fit(X[tr], ytr)
            pred = clf.predict(X[te])
            vals, cnt = np.unique(ytr, return_counts=True)
            maj = vals[np.argmax(cnt)]
            c1, c2 = int((pred == y[te]).sum()), int((y[te] == maj).sum())
            correct += c1
            maj_correct += c2
            if auc_pos is not None and auc_pos in clf.classes_:
                proba[te] = clf.predict_proba(X[te])[:, list(clf.classes_).index(auc_pos)]
            if protocol == "lodo" and not perm:
                per_group[int(groups[te][0])] = (c1 / len(te), c2 / len(te), len(te))
        accs.append(correct / n)
        majs.append(maj_correct / n)
        if auc_pos is not None and not perm and np.isfinite(proba).all() and len(np.unique(y)) == 2:
            aucs.append(roc_auc_score(y == auc_pos, proba))
    return float(np.mean(accs)), float(np.mean(majs)), (float(np.mean(aucs)) if aucs else np.nan), per_group


def run_trees(pool, D, retr, rules_f):
    X, feats = tree_feats(pool, retr)
    groups = pool.ds_id
    out = []
    targets = [("oracle_best_action", D["oracle_best"].astype(str).to_numpy(), None),
               ("oracle_go", np.where(D["oracle_go"].to_numpy() == 1, "go", "nogo"), "go")]
    for tname, y, pos in targets:
        for proto in ("kfold", "lodo"):
            acc, maj, auc, per = cv_eval(X, y, groups, proto, S["seed"], auc_pos=pos)
            prng = np.random.default_rng([S["seed"], 77, 0 if tname == "oracle_best_action" else 1, 0 if proto == "kfold" else 1])
            null = np.array([cv_eval(X, y, groups, proto, S["seed"], perm=True, prng=prng)[0] for _ in range(S["n_perm"])])
            out.append({"retriever": retr, "target": tname, "protocol": proto, "heldout": "ALL", "n": len(y),
                        "tree_acc": acc, "majority_acc": maj, "delta_vs_majority": acc - maj, "auc": auc,
                        "null_mean": null.mean(), "null_p95": np.percentile(null, 95),
                        "p_vs_null": (1 + (null >= acc).sum()) / (1 + len(null)), "beats_majority": acc > maj,
                        "n_classes": len(np.unique(y))})
            for g, (a, m_, ng) in per.items():
                out.append({"retriever": retr, "target": tname, "protocol": proto, "heldout": pool.ds_names[g], "n": ng,
                            "tree_acc": a, "majority_acc": m_, "delta_vs_majority": a - m_, "beats_majority": a > m_})
        clf = _clf(S["seed"]).fit(X, y)
        rules_f.write(f"\n=== retriever={retr} target={tname} (fit on all {len(y)} queries; depth<={S['depth']}) ===\n")
        vals, cnt = np.unique(y, return_counts=True)
        rules_f.write("class counts: " + ", ".join(f"{v}={c}" for v, c in zip(vals, cnt)) + "\n")
        rules_f.write(export_text(clf, feature_names=feats, class_names=[str(c) for c in clf.classes_]) + "\n")
    # univariate AUC of each feature for oracle-go (pooled, z-scored per dataset)
    yb = D["oracle_go"].to_numpy()
    uni = []
    rng = np.random.default_rng([S["seed"], 99])
    for j, f in enumerate(feats):
        x = X[:, j]
        if len(np.unique(yb)) < 2 or np.std(x) == 0:
            continue
        a = roc_auc_score(yb, x)
        bs = []
        for _ in range(S["auc_boot"]):
            i = rng.integers(0, len(yb), len(yb))
            if len(np.unique(yb[i])) == 2:
                bs.append(roc_auc_score(yb[i], x[i]))
        uni.append({"retriever": retr, "kind": "univariate_auc_oracle_go", "feature": f, "auc": a,
                    "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "n": len(yb)})
    return out, uni


# ----------------------------------------------------------------------------- consistency
def consistency(B1):
    d = B1[(B1.dataset != "POOLED") & (~B1.too_small)]
    rows = []
    for (desc, b, act), g in d.groupby(["descriptor", "bucket", "action"]):
        piv = g.pivot_table(index="dataset", columns="retriever", values="mean_gain")
        if not {"bm25", "dense"} <= set(piv.columns):
            continue
        piv = piv.dropna()
        if piv.empty:
            continue
        agree = piv[(np.sign(piv.bm25) == np.sign(piv.dense)) & (np.sign(piv.bm25) != 0)]
        npos, nneg = int((agree.bm25 > 0).sum()), int((agree.bm25 < 0).sum())
        dom = max(npos, nneg)
        status = "consistent" if dom >= 2 else ("dataset_specific" if len(agree) >= 1 else "inconsistent")
        rows.append({"descriptor": desc, "bucket": b, "action": act, "n_datasets": len(piv), "n_both_retrievers_agree": len(agree),
                     "n_pos": npos, "n_neg": nneg, "dominant_sign": "+" if npos >= nneg else "-", "status": status,
                     "mean_gain_bm25": piv.bm25.mean(), "mean_gain_dense": piv.dense.mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures / tex
def _save(fig, path):
    for ext in (".png", ".pdf"):
        fig.savefig(path.with_suffix(ext), dpi=150, bbox_inches="tight")


def make_figures(B1, B2, B5, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (retr, scope), d in B1.groupby(["retriever", "dataset"]):
        d = d.copy()
        d["bk"] = d.bucket.map(lambda b: bkey(b)[0])
        d = d.sort_values(["descriptor", "bk", "bucket"])
        d["row"] = d.descriptor + ": " + d.bucket.astype(str)
        rows = list(dict.fromkeys(d.row))
        M = np.full((len(rows), len(NONNULL)), np.nan)
        T = [[""] * len(NONNULL) for _ in rows]
        for _, r in d.iterrows():
            i, j = rows.index(r.row), NONNULL.index(r.action)
            if not r.too_small:
                M[i, j] = r.mean_gain
                T[i][j] = f"{r.mean_gain:+.2f}" + ("*" if r.get("q_bh", 1) < 0.05 else "")
        fig, ax = plt.subplots(figsize=(5.2, 0.2 * len(rows) + 1.2))
        im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu", vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_xticks(range(len(NONNULL)), NONNULL, fontsize=7)
        ax.set_yticks(range(len(rows)), rows, fontsize=6)
        for i in range(len(rows)):
            for j in range(len(NONNULL)):
                if T[i][j]:
                    ax.text(j, i, T[i][j], ha="center", va="center", fontsize=5)
        ax.set_title(f"mean gain over a0 by bucket - {retr} / {scope}  (* BH q<.05; blank = n<{S['min_n']})", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04)
        _save(fig, out / f"B_heatmap_{retr}_{scope}")
        plt.close(fig)
    for retr, d in B2[(B2.term != "_R2")].groupby("retriever"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for k, a in enumerate(NONNULL):
            e = d[d.action == a].set_index("term").reindex(REG_COLS)
            y = np.arange(len(REG_COLS)) + (k - 1.5) * 0.18
            ax.errorbar(e.std_beta, y, xerr=[e.std_beta - e.ci_lo_std, e.ci_hi_std - e.std_beta],
                        fmt="o", ms=3, label=a, capsize=1.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks(range(len(REG_COLS)), REG_COLS, fontsize=7)
        ax.set_xlabel("standardised coefficient (gain SDs per descriptor SD), 95% bootstrap CI")
        ax.legend(fontsize=7)
        ax.set_title(f"pooled regression with dataset fixed effects - {retr}", fontsize=8)
        _save(fig, out / f"B_coefplot_{retr}")
        plt.close(fig)
    for retr, d in B5.groupby("retriever"):
        scopes = list(d.dataset.unique())
        fig, axes = plt.subplots(1, len(scopes), figsize=(2.6 * len(scopes), 3), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, sc in zip(axes, scopes):
            for a in NONNULL:
                e = d[(d.dataset == sc) & (d.action == a)].set_index("bucket").reindex(BAND_NAMES)
                e = e.where(~e.too_small.fillna(True).astype(bool))
                ax.errorbar(range(4), e.mean_gain, yerr=[e.mean_gain - e.ci_lo, e.ci_hi - e.mean_gain], marker="o", ms=3, capsize=1.5, label=a)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(4), BAND_NAMES, fontsize=7)
            ax.set_title(sc, fontsize=8)
        axes[0].set_ylabel("mean gain over a0")
        axes[0].legend(fontsize=6)
        fig.suptitle(f"gain vs a0 strength (post-hoc; not a router feature) - {retr}", fontsize=8)
        _save(fig, out / f"B_a0_strength_{retr}")
        plt.close(fig)


def to_tex(df, path, caption, label):
    try:
        s = df.to_latex(index=False, escape=True, float_format="%.3f", na_rep="--", caption=caption, label=label)
        path.write_text(s, encoding="utf-8")
    except Exception as e:  # pragma: no cover
        print(f"  (tex skipped for {path.name}: {e})")


# ----------------------------------------------------------------------------- checks
def run_checks(Ds, pools, stores, a_dir, expected_total, out):
    res = {}
    sizes = {R.Store(sp).manifest["dataset"]: len(R.Store(sp).queries) for sp in stores}
    for retr, D in Ds.items():
        cnt = D.groupby("dataset").size().to_dict()
        res[f"{retr}_counts_match_store"] = all(cnt.get(k) == v for k, v in sizes.items())
        res[f"{retr}_total"] = int(len(D))
        res[f"{retr}_total_equals_expected_{expected_total}"] = len(D) == expected_total
        if set(sizes) == {"scifact", "trec-covid", "nfcorpus", "fiqa", "webis-touche2020"}:
            a0, q2 = D["a0_ndcg"].mean(), D["a0_ndcg"].mean() + D["g_query2doc"].mean()
            k = KNOWN[retr]
            res[f"{retr}_baseline_match_iteration_log"] = bool(abs(a0 - k["a0"]) < 5e-4 and abs(q2 - k["query2doc"]) < 5e-4
                                                               and abs(D["oracle_value"].mean() - k["oracle"]) < 5e-4)
        res[f"{retr}_a0_mean"] = float(D["a0_ndcg"].mean())
    # descriptors identical to stored features
    ok = True
    for retr, D in Ds.items():
        for sp in stores:
            st = R.Store(sp)
            F = st.features.set_index(st.features["qid"].astype(str))
            sub = D[D.dataset == st.manifest["dataset"]].set_index("qid")
            for g, col in {**s1_map(retr), "s0_len_words": "s0_len_words", "s0_idf_mean": "s0_idf_mean"}.items():
                ok &= bool(np.allclose(sub[g].to_numpy(float), F.reindex(sub.index)[col].to_numpy(float), equal_nan=True))
    res["descriptors_identical_to_store_features"] = ok
    # pooled means vs A
    fa = Path(a_dir) / "A1_action_gains.csv"
    if fa.exists():
        try:
            A = pd.read_csv(fa)
            col = next((c for c in ("mean_gain", "gain", "mean_delta") if c in A.columns), None)
            if col and {"retriever", "dataset", "action"} <= set(A.columns):
                good = True
                for retr, D in Ds.items():
                    for a in NONNULL:
                        v = A[(A.retriever == retr) & (A.dataset == "ALL") & (A.action == a)][col]
                        if len(v):
                            good &= bool(abs(float(v.iloc[0]) - D[f"g_{a}"].mean()) < 5e-4)
                res["pooled_means_match_A"] = good
            else:
                res["pooled_means_match_A"] = "A1 found but columns not recognised (need retriever,dataset,action,mean_gain)"
        except Exception as e:
            res["pooled_means_match_A"] = f"error: {e}"
    else:
        res["pooled_means_match_A"] = f"skipped: {fa} not found"
    return res


def form_sample(D, out):
    f = out / "B0_form_check_sample.csv"
    if f.exists():
        return
    rng = np.random.default_rng([S["seed"], 5])
    rows = []
    for ds, g in D.groupby("dataset"):
        g = g.sort_values("qid")
        idx = rng.choice(len(g), size=min(30, len(g)), replace=False)
        rows.append(g.iloc[np.sort(idx)][["dataset", "qid", "query", "form"]].rename(columns={"form": "rule_label"}))
    s = pd.concat(rows)
    s["hand_label"] = ""
    s.to_csv(f, index=False)


def validate_forms(path):
    d = pd.read_csv(path).dropna(subset=["hand_label"])
    d = d[d.hand_label.astype(str).str.strip() != ""]
    if d.empty:
        return {"status": "not validated (hand_label empty)"}
    d["ok"] = d.rule_label == d.hand_label.str.strip().str.lower()
    return {"status": "validated", "n": int(len(d)), "accuracy": float(d.ok.mean()),
            "per_dataset": d.groupby("dataset").ok.mean().round(3).to_dict()}


README = """# B_query_types - README

Exploratory query-type analysis (Workstream B). Results are HYPOTHESES, not confirmed effects.

## Settings (also in B_run_manifest.json)
{settings}

## Definitions
- gain = seed-mean metric(action) - metric(a0), nDCG@10 (a0/PRF seedless). Actions: prf, query2doc, hyde, decomp.
- oracle_best = argmax of seed-mean metric over all 5 actions (ties -> a0, then action order); oracle_go = oracle_best != a0.
  oracle_value = cross-fitted oracle value (routing.py). 
- helped/harmed: gain > +eps / < -eps (eps = {eps}); too_small: n < {min_n} (mean shown, no CI/p, best_action = 'too small', never ranked).
- Buckets: terciles within dataset (qcut with duplicate-edge dropping: heavily tied descriptors can give 2 buckets or be skipped).
  Within-dataset B1 reporting only for datasets with n >= {min_ds_n}; smaller datasets contribute to POOLED only.
  'low' for a descriptor means low relative to its own dataset.
- Descriptors: s0_* known before retrieval; s1_* after a0 retrieval (retriever-specific: s1_top1, s1_gap1_10, s1_wig, s1_nqc, s1_entropy100; s1_overlap10 = BM25/dense top-10 overlap). form = rule-based claim/question/keyword (see below); has_digit; starts_qword.
- a0_band (POST-HOC, not a router feature): zero (<=0.01), low (<=0.30), mid (<=0.70), high (>0.70) of a0 nDCG@10. Mechanical effect: when a0 is near 1 gain can only be <= 0, and low-a0 queries regress upward; read B5 with that in mind.
- Multiple testing: Benjamini-Hochberg FDR over all non-too-small cells of B1 ({n_tests_b1} tests), column q_bh. B2 coefficients: BH over all terms x actions per retriever (q_bh). Interaction tests: BH over all (q_bh).

## Files
- B0_query_descriptors.csv: one row per (retriever, query): raw descriptors, b_<descriptor> bucket labels, g_<action> gains, a0_ndcg, a0_band, oracle_best, oracle_go, oracle_value.
- B0_form_check_sample.csv: 30 queries/dataset for hand-checking the form rule (fill hand_label, re-run).
- B1_bucket_gains.csv: retriever, dataset (POOLED = all datasets), descriptor, bucket, action, n, mean_gain, ci_lo/ci_hi (95% bootstrap), p_boot, q_bh, cohens_d, helped, harmed, best_action (in bucket), too_small.
- B2_regression_coefs.csv: per retriever x action: gain ~ z-scored descriptors (+ dataset fixed effects). coef_per_1sd (nDCG per SD), std_beta (= coef / SD of gain), bootstrap CI (resampling queries), p_boot, q_bh; term=_R2 rows give R^2. B2_interaction_tests.csv: F-test of descriptor x dataset interactions. B2_vif.csv: collinearity.
- B3_tree_rules.txt, B3_tree_accuracy.csv: depth<={depth} decision tree on oracle_best (s0 + s1 features, z-scored per dataset). protocols: kfold (5-fold x3, each query its own group) and lodo (leave one dataset out); heldout rows per dataset for lodo. majority_acc uses the train-fold majority; null = labels shuffled in train folds ({n_perm} perms); p_vs_null empirical.
- B4_go_nogo.csv: same for oracle_go (tree with AUC) + univariate AUC of each feature for oracle_go (kind column).
- B5_a0_strength.csv: gain by a0 band. B6_length_vs_action.csv: query2doc vs hyde (and their difference) by length tercile, absolute length bin, and Spearman(length, gain) (rows kind=spearman_len_vs_gain: rho in mean_gain, p in p_boot).
- B7_consistency.csv: per (descriptor, bucket, action) sign agreement across retrievers and datasets (non-too-small dataset cells). consistent = same sign in both retrievers in >=2 datasets; dataset_specific = agreement in 1 dataset; else inconsistent.
- B_summary.json, B_checks (inside summary), B_run_manifest.json (md5 of outputs; compare two runs for identical output).
- Figures B_*.png/.pdf, LaTeX tables B_tab_*.tex.

## Caveats
Exploratory, many comparisons; descriptors correlate (see VIF) so single-feature effects are confounded; a0 band is post-hoc; buckets are relative within dataset; gains are noisy because of LLM seeds (noise SDs in B_summary.json); TREC-COVID and Touche (~50 queries) only enter POOLED; form rule is heuristic (validation in B_summary.json); git commit: {commit}.
"""


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+")
    ap.add_argument("--retrievers", nargs="+", default=["bm25", "dense"])
    ap.add_argument("--metric", default="ndcg10")
    ap.add_argument("--out", default="results/analysis/B_query_types")
    ap.add_argument("--a-dir", default="results/analysis/A_per_dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--min-ds-n", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--leaf", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--auc-boot", type=int, default=500)
    ap.add_argument("--expected-total", type=int, default=1370)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--validate-forms")
    a = ap.parse_args()
    if a.validate_forms:
        print(json.dumps(validate_forms(a.validate_forms), indent=2))
        return
    if not a.stores:
        ap.error("--stores required")
    S.update(seed=a.seed, B=200 if a.quick else a.bootstrap, eps=a.eps, min_n=a.min_n, min_ds_n=a.min_ds_n, depth=a.depth,
             leaf=a.leaf, n_perm=10 if a.quick else a.n_perm, auc_boot=50 if a.quick else a.auc_boot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counter = [0]
    Ds, pools, noise = {}, {}, {}
    b1, b2, b2i, vif, b5, b6, b3, b4u = [], [], [], [], [], [], [], []
    rules = open(out / "B3_tree_rules.txt", "w", encoding="utf-8")
    rules.write("Decision-tree rules (exploratory). Features are z-scored per dataset.\n")
    for retr in a.retrievers:
        print(f"== {retr}: loading", flush=True)
        pool, D, nz = load_retriever(a.stores, retr, a.metric)
        D = assign_buckets(D)
        Ds[retr], pools[retr], noise[retr] = D, pool, nz
        print(f"   {len(D)} queries; B1/B5/B6", flush=True)
        b1 += run_b1(D, retr, counter)
        b5 += run_b5(D, retr, counter)
        b6 += run_b6(D, retr, counter)
        print("   B2 regression", flush=True)
        r, i, v = run_b2(D, retr, counter)
        b2 += r
        b2i += i
        vif += v
        print("   B3/B4 trees", flush=True)
        t, u = run_trees(pool, D, retr, rules)
        b3 += t
        b4u += u
    rules.close()

    B1 = pd.DataFrame(b1)
    B1["q_bh"] = np.where(B1.too_small, np.nan, bh(np.where(B1.too_small, np.nan, B1.p_boot)))
    B2 = pd.DataFrame(b2)
    B2["q_bh"] = np.nan
    m = B2.term != "_R2"
    B2.loc[m, "q_bh"] = bh(B2.loc[m, "p_boot"].to_numpy())
    B2I = pd.DataFrame(b2i)
    B2I["q_bh"] = bh(B2I["p"].to_numpy())
    B5 = pd.DataFrame(b5)
    B6 = pd.DataFrame(b6)
    B3 = pd.DataFrame(b3)
    B7 = consistency(B1)
    B0 = pd.concat(Ds.values(), ignore_index=True)

    B0.to_csv(out / "B0_query_descriptors.csv", index=False)
    B1.to_csv(out / "B1_bucket_gains.csv", index=False)
    B2.to_csv(out / "B2_regression_coefs.csv", index=False)
    B2I.to_csv(out / "B2_interaction_tests.csv", index=False)
    pd.DataFrame(vif).to_csv(out / "B2_vif.csv", index=False)
    B3[B3.target == "oracle_best_action"].to_csv(out / "B3_tree_accuracy.csv", index=False)
    pd.concat([B3[B3.target == "oracle_go"].assign(kind="tree_oracle_go"), pd.DataFrame(b4u)], ignore_index=True).to_csv(out / "B4_go_nogo.csv", index=False)
    B5.to_csv(out / "B5_a0_strength.csv", index=False)
    B6.to_csv(out / "B6_length_vs_action.csv", index=False)
    B7.to_csv(out / "B7_consistency.csv", index=False)
    form_sample(Ds[a.retrievers[0]], out)

    print("   figures + tables", flush=True)
    try:
        make_figures(B1, B2, B5, out)
    except Exception as e:  # pragma: no cover
        print("  figures skipped:", e)
    t3 = B3[(B3.target == "oracle_best_action") & (B3.heldout == "ALL")][["retriever", "protocol", "tree_acc", "majority_acc", "null_p95", "p_vs_null", "beats_majority"]]
    to_tex(t3, out / "B_tab_tree_accuracy.tex", "Decision tree vs majority class (oracle best action).", "tab:b-tree")
    t4 = B3[(B3.target == "oracle_go") & (B3.heldout == "ALL")][["retriever", "protocol", "tree_acc", "majority_acc", "auc", "p_vs_null"]]
    to_tex(t4, out / "B_tab_go_nogo.tex", "Predicting whether the oracle goes.", "tab:b-go")
    t5 = B5[(B5.dataset == "POOLED")].assign(cell=lambda d: d.mean_gain.map("{:+.3f}".format) + " (n=" + d.n.astype(str) + ")")
    to_tex(t5.pivot_table(index=["retriever", "bucket"], columns="action", values="cell", aggfunc="first").reset_index(),
           out / "B_tab_a0_strength.tex", "Mean gain over a0 by a0 nDCG band (post-hoc), pooled.", "tab:b-a0")
    cons = B7[B7.status == "consistent"].sort_values(["descriptor", "bucket", "action"])
    to_tex(cons[["descriptor", "bucket", "action", "n_datasets", "dominant_sign", "mean_gain_bm25", "mean_gain_dense"]],
           out / "B_tab_consistent_patterns.tex", "Patterns with the same sign in both retrievers in at least two datasets.", "tab:b-cons")
    co = B2[B2.term != "_R2"].assign(cell=lambda d: d.std_beta.map("{:+.2f}".format) + np.where(d.q_bh < 0.05, "*", ""))
    to_tex(co.pivot_table(index=["retriever", "term"], columns="action", values="cell", aggfunc="first").reset_index(),
           out / "B_tab_regression.tex", "Standardised coefficients (gain SD per descriptor SD); * BH q<.05.", "tab:b-reg")

    checks = run_checks(Ds, pools, a.stores, a.a_dir, a.expected_total, out)
    fv = out / "B0_form_check_sample.csv"
    summ = {"settings": S, "checks": checks, "form_rule_validation": validate_forms(fv) if fv.exists() else {},
            "n_tests_b1": int((~B1.too_small).sum()), "n_b1_sig_bh05": int((B1.q_bh < 0.05).sum()),
            "n_consistent_patterns": int((B7.status == "consistent").sum()) if len(B7) else 0,
            "seed_noise_sd_of_gain": noise, "oracle_go_rate": {r: float(D.oracle_go.mean()) for r, D in Ds.items()},
            "tree_best_action": B3[(B3.target == "oracle_best_action") & (B3.heldout == "ALL")][["retriever", "protocol", "tree_acc", "majority_acc", "p_vs_null"]].to_dict("records"),
            "tree_go": B3[(B3.target == "oracle_go") & (B3.heldout == "ALL")][["retriever", "protocol", "tree_acc", "majority_acc", "auc", "p_vs_null"]].to_dict("records"),
            "pooled_mean_gain": {r: {a_: float(D[f"g_{a_}"].mean()) for a_ in NONNULL} for r, D in Ds.items()},
            "runtime_s": round(time.time() - t0, 1)}
    (out / "B_summary.json").write_text(json.dumps(summ, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "n/a"
    except Exception:
        commit = "n/a"
    (out / "README.md").write_text(README.format(settings=json.dumps(S), eps=S["eps"], min_n=S["min_n"], min_ds_n=S["min_ds_n"],
                                                 n_tests_b1=summ["n_tests_b1"], depth=S["depth"], n_perm=S["n_perm"], commit=commit), encoding="utf-8")
    man = {p.name: md5(p) for p in sorted(out.glob("*.csv")) + [out / "B3_tree_rules.txt"]}
    (out / "B_run_manifest.json").write_text(json.dumps({"settings": S, "md5": man}, indent=2))
    print("\nChecks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nDone in {time.time() - t0:.0f}s -> {out}/  (read B_summary.json first)")


if __name__ == "__main__":
    main()