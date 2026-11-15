#!/usr/bin/env python3
"""
analyze_router_where.py -- Workstream C: where does the router help or hurt?

Evaluates the EXISTING routers (routing.py); builds no new ones. Reproduces the router CV runs
(seed 0, 5 folds x 3 repeats, tuned) so per-query values are available, then writes C1..C9.

  python analyze_router_where.py \
      --stores store/scifact store/trec-covid store/nfcorpus store/fiqa store/webis-touche2020 \
      --retrievers bm25 dense --b0 results/analysis/B0_query_descriptors.csv

Useful flags:  --null-n 5 (quick null)   --skip-null (do C7 later)   --no-cache (recompute CV runs)
Everything expensive is cached in --cache, so re-running is cheap and resumable.

Primary contrast: iteration 3, nDCG@10, router - Const. Iterations 1/2 have an oracle below the
learned nodes, so they are diagnostics only (role column).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing as R  # noqa: E402

warnings.filterwarnings("ignore")
ACT = list(R.ACTIONS)
AI = {a: i for i, a in enumerate(ACT)}
NA = len(ACT)
PRIMARY, SECONDARY = "ndcg10", ["map100", "recall100"]

# Iteration-3 ALL-policy nDCG@10 from the existing iteration_log.md files (acceptance check)
EXPECTED_L3_ALL = {
    ("bm25", "within", "const"): .4047, ("bm25", "within", "stump"): .4021, ("bm25", "within", "ridge"): .4026, ("bm25", "within", "gbr"): .4023,
    ("bm25", "lodo", "const"): .4047, ("bm25", "lodo", "stump"): .4021, ("bm25", "lodo", "ridge"): .4044, ("bm25", "lodo", "gbr"): .4048,
    ("dense", "within", "const"): .4409, ("dense", "within", "stump"): .4387, ("dense", "within", "ridge"): .4439, ("dense", "within", "gbr"): .4451,
    ("dense", "lodo", "const"): .4409, ("dense", "lodo", "stump"): .4404, ("dense", "lodo", "ridge"): .4386, ("dense", "lodo", "gbr"): .4404,
}


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
def blocks(pool):
    out = [(n, pool.ds_id == i) for i, n in enumerate(pool.ds_names)]
    out.append(("ALL", np.ones(len(pool.qid), bool)))
    return out


def pstats(diff, B, seed):
    """Paired stats on per-query differences: mean, bootstrap CI/p, Wilcoxon p, Cohen's dz."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    nan = dict(mean_delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan, p_wilcoxon=np.nan, cohen_dz=np.nan)
    if len(d) < 2:
        return nan
    mean, lo, hi, pb = R.paired_bootstrap(d, B, seed)
    sd = d.std(ddof=1)
    pw = 1.0
    if np.any(d != 0):
        try:
            pw = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            pw = 1.0
    return dict(mean_delta=mean, ci_lo=lo, ci_hi=hi, p_boot=pb, p_wilcoxon=pw, cohen_dz=float(mean / sd) if sd > 0 else np.nan)


def add_holm(df, keys, exclude_all=True):
    df = df.reset_index(drop=True).copy()
    for c in ("p_wilcoxon_holm", "p_boot_holm", "n_tests_holm"):
        df[c] = np.nan
    sub = df[df["dataset"] != "ALL"] if exclude_all and "dataset" in df else df
    for _, g in sub.groupby(keys):
        ix = g.index
        df.loc[ix, "p_wilcoxon_holm"] = R.holm(g["p_wilcoxon"].fillna(1.0).to_numpy())
        df.loc[ix, "p_boot_holm"] = R.holm(g["p_boot"].fillna(1.0).to_numpy())
        df.loc[ix, "n_tests_holm"] = len(ix)
    return df


def chosen_idx(pool, res, lam):
    """(R, n) index into ACTIONS of the iteration-3 policy for each CV repeat."""
    out = []
    for r in range(len(res.P)):
        _, _, lab = R.chain_eval(pool, 3, R.decisions(res.P[r], res.CC[r], lam))
        out.append([AI[x] for x in lab])
    return np.array(out)


def rep_vals(Vm, idx):
    return Vm[np.arange(Vm.shape[0])[None, :], idx]  # (R, n)


def recon_folds(pool, proto, folds, repeats, seed):
    """Re-derive the CV folds run_cv used (its rng is only consumed by make_folds when not permuting)."""
    rng = np.random.default_rng(seed)
    return [R.make_folds(pool, proto, folds, rng) for _ in range(repeats if proto != "lodo" else 1)]


def write_tex(df, path, caption, label, nd=4):
    def f(x):
        if isinstance(x, (float, np.floating)):
            return "--" if np.isnan(x) else f"{x:.{nd}f}"
        return str(x).replace("_", "\\_")
    head = " & ".join(str(c).replace("_", "\\_") for c in df.columns)
    body = "\n".join(" & ".join(f(x) for x in r) + " \\\\" for r in df.itertuples(index=False))
    t = ("\\begin{table}[t]\\centering\\small\n\\caption{" + caption + "}\\label{" + label + "}\n\\begin{tabular}{"
         + "l" * len(df.columns) + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    Path(path).write_text(t, encoding="utf-8")


# ----------------------------------------------------------------------------
# Context: pools, cached CV runs, per-query values
# ----------------------------------------------------------------------------
class Ctx:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(exist_ok=True)
        (self.out / "tables").mkdir(exist_ok=True)
        self.cache = Path(a.cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lam0 = float(a.lambdas[0])
        self.kinds = ["const"] + [m for m in a.models if m != "const"]
        self._pools, self._mats, self._runs, self._idx, self._vals, self._bk = {}, {}, {}, {}, {}, {}
        self.metrics = [PRIMARY]

    def pool(self, retr):
        if retr not in self._pools:
            say(f"building pool: {retr}")
            self._pools[retr] = R.build_pool(self.a.stores, retr, metric=PRIMARY, max_stage=1, normalize="dataset", oracle="crossfit")
        return self._pools[retr]

    def mat(self, retr, metric):
        """(n, A) seed-averaged per-query metric for every action, aligned with the pool."""
        key = (retr, metric)
        if key in self._mats:
            return self._mats[key]
        pool = self.pool(retr)
        if metric == PRIMARY:
            M = np.nanmean(pool.V, axis=2)
        else:
            rows = {}
            for name, path in pool.store_paths.items():
                d = R.load_ds(path, retr, metric, 1)
                sidx = [d.seeds.index(s) for s in pool.seeds]
                Vm = np.nanmean(d.V[:, :, sidx], axis=2)
                for q, v in zip(d.qids, Vm):
                    rows[f"{name}:{q}"] = v
            M = np.vstack([rows[q] for q in pool.qid])
        self._mats[key] = M
        return M

    def run(self, retr, proto, kind):
        key = (retr, proto, kind)
        if key in self._runs:
            return self._runs[key]
        a, pool = self.a, self.pool(retr)
        f = self.cache / f"cv_{retr}_{proto}_{kind}_n{len(pool.qid)}_s{a.seed}_f{a.folds}_r{a.repeats}.pkl"
        if f.exists() and not a.no_cache:
            res = pickle.load(open(f, "rb"))
        else:
            t0 = time.time()
            res = R.run_cv(pool, kind, proto, a.folds, a.repeats, a.seed, tune=True)
            pickle.dump(res, open(f, "wb"))
            say(f"  run_cv {retr}/{proto}/{kind}: {time.time() - t0:.0f}s")
        self._runs[key] = res
        return res

    def idx(self, retr, proto, kind):
        key = (retr, proto, kind)
        if key not in self._idx:
            self._idx[key] = chosen_idx(self.pool(retr), self.run(retr, proto, kind), self.lam0)
        return self._idx[key]

    def vals(self, retr, proto, kind, level=3, metric=PRIMARY):
        """Per-query policy value, averaged over CV repeats (avoids pseudo-replication)."""
        key = (retr, proto, kind, level, metric)
        if key not in self._vals:
            pool = self.pool(retr)
            if level == 3:
                v = rep_vals(self.mat(retr, metric), self.idx(retr, proto, kind)).mean(axis=0)
            else:
                v = R.evaluate(pool, self.run(retr, proto, kind), level, [self.lam0])["value"][0]
            self._vals[key] = v
        return self._vals[key]

    def bf(self, retr, proto, kind, metric=PRIMARY):
        """Value of the train-fold-selected best single action (selected on nDCG@10)."""
        res = self.run(retr, proto, kind)
        idx = np.array([[AI[x] for x in row] for row in res.best_fixed_name])
        return rep_vals(self.mat(retr, metric), idx).mean(axis=0)

    def modal(self, retr, proto, kind):
        idx = self.idx(retr, proto, kind)
        counts = np.stack([(idx == j).sum(axis=0) for j in range(NA)])
        return counts.argmax(axis=0)

    def buckets(self, retr):
        if retr not in self._bk:
            self._bk[retr] = make_buckets(self, retr)
        return self._bk[retr]


# ----------------------------------------------------------------------------
# Query-type buckets (B0 if available + self-computed fallbacks)
# ----------------------------------------------------------------------------
def load_b0(pool, retr, path, cols_arg):
    n = len(pool.qid)
    empty = pd.DataFrame(index=range(n))
    if not path or not Path(path).exists():
        say("B0 file not found -> using fallback buckets only")
        return empty
    b = pd.read_csv(path)
    say(f"B0 columns: {list(b.columns)}")
    low = {c.lower(): c for c in b.columns}
    dcol = next((low[k] for k in ("dataset", "ds", "collection") if k in low), None)
    qcol = next((low[k] for k in ("qid", "query_id", "id") if k in low), None)
    if dcol is None or qcol is None:
        say("B0: no dataset/qid column found -> skipped")
        return empty
    if "retriever" in low:
        bb = b[b[low["retriever"]].astype(str) == retr]
        b = bb if len(bb) else b
    b = b.assign(_key=b[dcol].astype(str) + ":" + b[qcol].astype(str)).drop_duplicates("_key").set_index("_key")
    skip = {dcol, qcol} | {low[k] for k in ("retriever", "query", "text", "query_text") if k in low}
    if cols_arg:
        cols = [c for c in cols_arg if c in b.columns]
    else:
        cols = []
        for c in b.columns:
            if c in skip or c == "_key":
                continue
            nm = c.lower()
            if any(t in nm for t in ("bucket", "band", "tertile", "quantile", "type", "form", "class")) or (b[c].dtype == object and b[c].nunique() <= 12):
                cols.append(c)
    keys = [f"{pool.ds_names[i]}:{q}" for i, q in zip(pool.ds_id, pool.raw_qid)]
    out = {}
    for c in cols:
        s = b[c].reindex(keys)
        out[f"B0:{c}"] = np.array([None if pd.isna(x) else str(x) for x in s], dtype=object)
    out = pd.DataFrame(out) if out else empty
    say(f"B0 bucket columns used: {list(out.columns)} | matched rows: {float(np.mean([x is not None for x in out.iloc[:, 0]])) if len(out.columns) else 0:.2f}")
    return out


def make_buckets(ctx, retr):
    pool = ctx.pool(retr)
    n = len(pool.qid)
    B = {}
    feats = []
    for name in pool.ds_names:
        try:
            feats.append(R.Store(pool.store_paths[name]).features.set_index("qid"))
        except Exception:
            feats.append(None)

    def col(c):
        out = np.full(n, np.nan)
        for i in range(len(pool.ds_names)):
            f = feats[i]
            if f is not None and c in f.columns:
                m = pool.ds_id == i
                out[m] = f[c].reindex(pool.raw_qid[m].astype(str)).to_numpy(float)
        return out

    def tert(x, labels):
        out = np.array([None] * n, dtype=object)
        for i in range(len(pool.ds_names)):
            m = pool.ds_id == i
            xv = x[m]
            if np.isfinite(xv).sum() < 6:
                continue
            q = np.nanquantile(xv, [1 / 3, 2 / 3])
            lab = np.where(xv <= q[0], labels[0], np.where(xv <= q[1], labels[1], labels[2])).astype(object)
            lab[~np.isfinite(xv)] = None
            out[m] = lab
        return out

    for cname, label in (("s0_len_words", "len"), ("s0_idf_mean", "idf"), (f"s1_{retr}_top1", "a0conf")):
        x = col(cname)
        if np.isfinite(x).any():
            B[f"fb:{label}"] = tert(x, (f"{label}:low", f"{label}:mid", f"{label}:high"))
    qw, qm = col("s0_starts_qword"), col("s0_has_qmark")
    if np.isfinite(qw).any() or np.isfinite(qm).any():
        isq = (np.nan_to_num(qw) > 0) | (np.nan_to_num(qm) > 0)
        B["fb:form"] = np.where(isq, "form:question", "form:keyword").astype(object)
    a0 = ctx.mat(retr, PRIMARY)[:, 0]
    B["fb:a0_strength"] = np.array(pd.cut(a0, [-1, .1, .4, .7, 2], labels=["a0<0.1", "a0 0.1-0.4", "a0 0.4-0.7", "a0>=0.7"], right=False).astype(str), dtype=object)
    out = pd.DataFrame(B)
    b0 = load_b0(pool, retr, ctx.a.b0, ctx.a.bucket_cols)
    return pd.concat([out, b0], axis=1)


# ----------------------------------------------------------------------------
# C1: router vs Const / best fixed
# ----------------------------------------------------------------------------
def build_c1(ctx):
    a, rows = ctx.a, []
    for retr in a.retrievers:
        pool = ctx.pool(retr)
        blk = blocks(pool)
        for proto in a.protocols:
            for kind in ctx.kinds:
                for L in (1, 2, 3):
                    for metric in (ctx.metrics if L == 3 else [PRIMARY]):
                        v = ctx.vals(retr, proto, kind, L, metric)
                        c = ctx.vals(retr, proto, "const", L, metric)
                        b = ctx.bf(retr, proto, kind, metric)
                        a0 = ctx.mat(retr, metric)[:, 0]
                        for bname, base in (("const", c), ("best_fixed", b)):
                            if bname == "const" and kind == "const":
                                continue
                            for dn, m in blk:
                                st = pstats(v[m] - base[m], a.boot, a.seed)
                                rows.append(dict(retriever=retr, protocol=proto, model=kind, level=L, role="primary" if L == 3 else "diagnostic",
                                                 metric=metric, baseline=bname, dataset=dn, n=int(m.sum()), policy_mean=v[m].mean(),
                                                 baseline_mean=base[m].mean(), a0_mean=a0[m].mean(),
                                                 oracle_mean=pool.T["all"][m].mean() if metric == PRIMARY else np.nan, **st))
        say(f"C1 done: {retr}")
    df = add_holm(pd.DataFrame(rows), ["protocol", "model", "level", "metric", "baseline"])
    return df[df.baseline == "const"].reset_index(drop=True), df[df.baseline == "best_fixed"].reset_index(drop=True)


# ----------------------------------------------------------------------------
# C2: action choice + confusion
# ----------------------------------------------------------------------------
def build_c2(ctx):
    rows, conf = [], []
    for retr in ctx.a.retrievers:
        pool = ctx.pool(retr)
        ora = np.array([AI[x] for x in pool.choice["all"]])
        for proto in ctx.a.protocols:
            for kind in ctx.kinds:
                idx = ctx.idx(retr, proto, kind)
                for dn, m in blocks(pool):
                    sh = np.array([(idx[:, m] == j).mean() for j in range(NA)])
                    osh = np.array([(ora[m] == j).mean() for j in range(NA)])
                    row = dict(retriever=retr, protocol=proto, model=kind, dataset=dn, n=int(m.sum()))
                    row.update({f"router_{x}": sh[j] for j, x in enumerate(ACT)})
                    row.update({f"oracle_{x}": osh[j] for j, x in enumerate(ACT)})
                    row.update(tv_distance=0.5 * np.abs(sh - osh).sum(), router_go_rate=1 - sh[0], oracle_go_rate=1 - osh[0],
                               router_top_action=ACT[int(sh.argmax())], router_top_share=sh.max(),
                               agree_with_oracle=float((idx[:, m] == ora[m][None, :]).mean()))
                    rows.append(row)
                    for j in range(NA):
                        tot = np.mean([(idx[r][m] == j).sum() for r in range(idx.shape[0])])
                        for k in range(NA):
                            cnt = np.mean([((idx[r][m] == j) & (ora[m] == k)).sum() for r in range(idx.shape[0])])
                            conf.append(dict(retriever=retr, protocol=proto, model=kind, dataset=dn, router_action=ACT[j], oracle_action=ACT[k],
                                             count=cnt, frac_of_router_action=cnt / tot if tot > 0 else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(conf)


# ----------------------------------------------------------------------------
# C3: win/loss
# ----------------------------------------------------------------------------
def wl(d, mask):
    x = d[mask]
    g, l = x[x > 1e-12], x[x < -1e-12]
    return dict(n=len(x), n_differ=len(g) + len(l), frac_differ=(len(g) + len(l)) / max(len(x), 1),
                n_gain=len(g), mean_gain=g.mean() if len(g) else np.nan, n_loss=len(l), mean_loss=l.mean() if len(l) else np.nan,
                sum_gain=g.sum(), sum_loss=l.sum(), net_mean=x.mean() if len(x) else np.nan,
                gain_to_loss_ratio=g.sum() / abs(l.sum()) if len(l) else np.nan)


def build_c3(ctx):
    rows = []
    for retr in ctx.a.retrievers:
        pool = ctx.pool(retr)
        for proto in ctx.a.protocols:
            c = ctx.vals(retr, proto, "const")
            for kind in ctx.kinds:
                if kind == "const":
                    continue
                d = ctx.vals(retr, proto, kind) - c
                modal = ctx.modal(retr, proto, kind)
                for dn, m in blocks(pool):
                    rows.append(dict(retriever=retr, protocol=proto, model=kind, group_type="dataset", group=dn, dataset=dn, **wl(d, m)))
                    for j in range(NA):
                        mm = m & (modal == j)
                        if mm.sum():
                            rows.append(dict(retriever=retr, protocol=proto, model=kind, group_type="router_modal_action", group=ACT[j], dataset=dn, **wl(d, mm)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# C4: by query type
# ----------------------------------------------------------------------------
def build_c4(ctx):
    a, rows, skipped = ctx.a, [], 0
    for retr in a.retrievers:
        pool = ctx.pool(retr)
        B = ctx.buckets(retr)
        for proto in a.protocols:
            c = ctx.vals(retr, proto, "const")
            for kind in ctx.kinds:
                if kind == "const":
                    continue
                v = ctx.vals(retr, proto, kind)
                d = v - c
                for col in B.columns:
                    arr = B[col].to_numpy(object)
                    for lv in sorted({x for x in arr if isinstance(x, str)}):
                        for dn, mb in blocks(pool):
                            m = mb & (arr == lv)
                            if m.sum() < a.min_cell:
                                skipped += 1
                                continue
                            st = pstats(d[m], a.boot2, a.seed)
                            rows.append(dict(retriever=retr, protocol=proto, model=kind, bucket_var=col, bucket=lv, dataset=dn, n=int(m.sum()),
                                             router_mean=v[m].mean(), const_mean=c[m].mean(), exploratory=True, **st))
    df = add_holm(pd.DataFrame(rows), ["retriever", "protocol", "model"], exclude_all=False) if rows else pd.DataFrame(rows)
    say(f"C4: {len(df)} cells reported, {skipped} cells skipped (n < {a.min_cell})")
    return df


# ----------------------------------------------------------------------------
# C5: node calibration
# ----------------------------------------------------------------------------
def build_c5(ctx, ndec=10):
    rows = []
    for retr in ctx.a.retrievers:
        pool = ctx.pool(retr)
        for proto in ctx.a.protocols:
            for kind in ctx.kinds:
                if kind not in ("ridge", "gbr"):
                    continue
                res = ctx.run(retr, proto, kind)
                P = {k: np.mean([p[k] for p in res.P], axis=0) for k in res.P[0]}
                for nd in R.NODES:
                    k, ch = nd["id"], nd["children"]
                    Tn = np.column_stack([pool.T[x] for x in ch])
                    if len(ch) == 2:
                        tg = [(f"gain({ch[1]}-{ch[0]})", P[k][:, 1] - P[k][:, 0], Tn[:, 1] - Tn[:, 0])]
                    else:
                        cen = Tn - Tn.mean(axis=1, keepdims=True)
                        tg = [(f"centered({x})", P[k][:, j], cen[:, j]) for j, x in enumerate(ch)]
                    for label, score, true in tg:
                        base = dict(retriever=retr, protocol=proto, model=kind, node=k, node_name=nd["name"], target=label)
                        if np.ptp(score) < 1e-12:
                            continue
                        q = pd.qcut(pd.Series(score).rank(method="first"), ndec, labels=False).to_numpy()
                        for dcl in range(ndec):
                            m = q == dcl
                            rows.append(dict(**base, scope="decile", decile=dcl + 1, n=int(m.sum()), mean_pred=score[m].mean(), mean_true=true[m].mean(),
                                             spearman=np.nan, slope=np.nan))
                        rows.append(dict(**base, scope="summary", decile="ALL", n=len(score), mean_pred=score.mean(), mean_true=true.mean(),
                                         spearman=stats.spearmanr(score, true)[0], slope=np.polyfit(score, true, 1)[0]))
                        for i, dn in enumerate(pool.ds_names):
                            m = pool.ds_id == i
                            if m.sum() > 10 and np.ptp(score[m]) > 1e-12:
                                rows.append(dict(**base, scope="summary", decile=f"ds:{dn}", n=int(m.sum()), mean_pred=score[m].mean(), mean_true=true[m].mean(),
                                                 spearman=stats.spearmanr(score[m], true[m])[0], slope=np.polyfit(score[m], true[m], 1)[0]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# C6: stability across repeats / folds
# ----------------------------------------------------------------------------
def build_c6(ctx):
    a, rows = ctx.a, []
    for retr in a.retrievers:
        pool = ctx.pool(retr)
        Vm = ctx.mat(retr, PRIMARY)
        for proto in a.protocols:
            vc = rep_vals(Vm, ctx.idx(retr, proto, "const"))
            folds = recon_folds(pool, proto, a.folds, a.repeats, a.seed)
            for kind in ctx.kinds:
                if kind == "const":
                    continue
                vr = rep_vals(Vm, ctx.idx(retr, proto, kind))
                for r in range(vr.shape[0]):
                    dl = vr[r] - vc[r]
                    for dn, m in blocks(pool):
                        rows.append(dict(retriever=retr, protocol=proto, model=kind, unit="repeat", repeat=r, fold=-1, dataset=dn, n=int(m.sum()), delta=dl[m].mean()))
                    for fi, (tr, te) in enumerate(folds[r]):
                        dn = pool.ds_names[pool.ds_id[te[0]]]
                        rows.append(dict(retriever=retr, protocol=proto, model=kind, unit="fold", repeat=r, fold=fi, dataset=dn, n=len(te), delta=(vr[r, te] - vc[r, te]).mean()))
    df = pd.DataFrame(rows)
    g = df.groupby(["retriever", "protocol", "model", "unit", "dataset"])["delta"]
    s = g.agg(delta="mean", sd="std", min="min", max="max", n_units="count").reset_index()
    s["frac_positive"] = g.apply(lambda x: float((x > 0).mean())).to_numpy()
    s["unit"] = s["unit"] + "_summary"
    return pd.concat([df, s], ignore_index=True)


# ----------------------------------------------------------------------------
# C7: null distribution (shuffled features)
# ----------------------------------------------------------------------------
def build_c7(ctx):
    a, rows = ctx.a, []
    for retr in a.retrievers:
        pool = ctx.pool(retr)
        Vm = ctx.mat(retr, PRIMARY)
        ar = np.arange(len(pool.qid))
        for proto in a.protocols:
            cv = ctx.vals(retr, proto, "const")
            for kind in a.null_models:
                real = ctx.vals(retr, proto, kind)
                for dn, m in blocks(pool):
                    rows.append(dict(retriever=retr, protocol=proto, model=kind, shuffle=-1, is_real=True, dataset=dn, delta_vs_const=(real[m] - cv[m]).mean()))
                t0 = time.time()
                for s in range(a.null_n):
                    f = ctx.cache / f"null_{retr}_{proto}_{kind}_n{len(pool.qid)}_s{s}_r{a.null_repeats}_f{a.folds}.npy"
                    if f.exists() and not a.no_cache:
                        v = np.load(f)
                    else:
                        res = R.run_cv(pool, kind, proto, a.folds, a.null_repeats, 1000 + s, tune=True, permute=True)
                        v = rep_vals(Vm, chosen_idx(pool, res, ctx.lam0)).mean(axis=0)
                        np.save(f, v)
                    for dn, m in blocks(pool):
                        rows.append(dict(retriever=retr, protocol=proto, model=kind, shuffle=s, is_real=False, dataset=dn, delta_vs_const=(v[m] - cv[m]).mean()))
                say(f"C7 {retr}/{proto}/{kind}: {a.null_n} shuffles ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    sm = []
    for key, g in df.groupby(["retriever", "protocol", "model", "dataset"]):
        nl, rl = g[~g.is_real].delta_vs_const.to_numpy(), float(g[g.is_real].delta_vs_const.iloc[0])
        if len(nl) == 0:
            continue
        sd = nl.std(ddof=1) if len(nl) > 1 else np.nan
        sm.append(dict(retriever=key[0], protocol=key[1], model=key[2], dataset=key[3], n_null=len(nl), null_mean=nl.mean(), null_sd=sd,
                       null_se=sd / np.sqrt(len(nl)) if len(nl) > 1 else np.nan, null_p05=np.quantile(nl, .05), null_p95=np.quantile(nl, .95),
                       real_delta=rl, z=(rl - nl.mean()) / sd if sd and sd > 0 else np.nan,
                       emp_p_upper=(1 + (nl >= rl).sum()) / (1 + len(nl)), emp_p_two_sided=(1 + (np.abs(nl - nl.mean()) >= abs(rl - nl.mean())).sum()) / (1 + len(nl))))
    return df, pd.DataFrame(sm)


# ----------------------------------------------------------------------------
# C8: lambda sweep
# ----------------------------------------------------------------------------
def build_c8(ctx):
    a, rows = ctx.a, []
    lam = list(a.lambdas)
    for retr in a.retrievers:
        pool = ctx.pool(retr)
        Vm = ctx.mat(retr, PRIMARY)
        for proto in a.protocols:
            for kind in ctx.kinds:
                ev = R.evaluate(pool, ctx.run(retr, proto, kind), 3, lam)
                for li, l in enumerate(lam):
                    for dn, m in blocks(pool):
                        rows.append(dict(retriever=retr, protocol=proto, model=kind, lam=l, dataset=dn, n=int(m.sum()), ndcg10=ev["value"][li][m].mean(),
                                         go_rate=ev["go"][li][m].mean(), cost_s=ev["cost"][li][m].mean()))
        for j, act in enumerate(ACT):
            for dn, m in blocks(pool):
                rows.append(dict(retriever=retr, protocol="fixed", model=f"fixed_{act}", lam=np.nan, dataset=dn, n=int(m.sum()), ndcg10=Vm[m, j].mean(),
                                 go_rate=float(act != "a0"), cost_s=pool.cost[m, j].mean()))
    df = pd.DataFrame(rows)
    df["dominated_by_fixed"], df["dominates_fixed"] = "", ""
    eps = 1e-12
    for retr in a.retrievers:
        fx = df[(df.retriever == retr) & (df.protocol == "fixed") & (df.dataset == "ALL")]
        for i in df[(df.retriever == retr) & (df.protocol != "fixed") & (df.dataset == "ALL")].index:
            q, c = df.at[i, "ndcg10"], df.at[i, "cost_s"]
            dom = [r.model for r in fx.itertuples() if r.cost_s <= c + eps and r.ndcg10 >= q - eps and (r.cost_s < c - eps or r.ndcg10 > q + eps)]
            dm = [r.model for r in fx.itertuples() if c <= r.cost_s + eps and q >= r.ndcg10 - eps and (c < r.cost_s - eps or q > r.ndcg10 + eps)]
            df.at[i, "dominated_by_fixed"], df.at[i, "dominates_fixed"] = ";".join(dom), ";".join(dm)
    return df


# ----------------------------------------------------------------------------
# C9: NFCorpus dense check
# ----------------------------------------------------------------------------
def build_c9(ctx):
    a = ctx.a
    summ, perq = [], []
    rng = np.random.default_rng(a.seed)
    for retr in [r for r in a.nfc_retrievers if r in a.retrievers]:
        pool = ctx.pool(retr)
        if a.nfc_name not in pool.ds_names:
            say(f"C9: dataset {a.nfc_name} not in pool, skipped")
            continue
        m = pool.ds_id == pool.ds_names.index(a.nfc_name)
        Vm = ctx.mat(retr, PRIMARY)
        B = ctx.buckets(retr)
        for proto in a.protocols:
            c = ctx.vals(retr, proto, "const")
            for kind in ctx.kinds:
                if kind == "const":
                    continue
                d = (ctx.vals(retr, proto, kind) - c)[m]
                modal = ctx.modal(retr, proto, kind)[m]

                def add(section, key, value):
                    summ.append(dict(retriever=retr, protocol=proto, model=kind, section=section, key=key, value=value))

                n = len(d)
                for k_, v_ in dict(n=n, mean=d.mean(), median=np.median(d), sd=d.std(ddof=1), frac_pos=(d > 1e-12).mean(), frac_neg=(d < -1e-12).mean(),
                                   frac_zero=(np.abs(d) <= 1e-12).mean(), p05=np.quantile(d, .05), p95=np.quantile(d, .95), skew=stats.skew(d),
                                   trimmed_mean_10=stats.trim_mean(d, .1)).items():
                    add("distribution", k_, v_)
                ds = np.sort(d)[::-1]
                tot = d.sum()
                for k in (5, 10, 20):
                    add("topk", f"share_of_total_gain_from_top{k}_queries", ds[:k].sum() / tot if tot > 0 else np.nan)
                    add("topk", f"mean_without_top{k}_gainers", ds[k:].mean())
                    add("topk", f"mean_without_top{k}_losers", ds[:-k].mean())
                lo = np.array([d[rng.permutation(n)[: int(0.8 * n)]].mean() for _ in range(1000)])
                for k_, v_ in dict(mean=lo.mean(), ci_lo=np.quantile(lo, .025), ci_hi=np.quantile(lo, .975), frac_positive=(lo > 0).mean()).items():
                    add("leave_20pct_out_x1000", k_, v_)
                loo = (d.sum() - d) / (n - 1)
                add("leave_one_out", "min_mean", loo.min())
                add("leave_one_out", "max_mean", loo.max())
                for j in range(NA):
                    mm = modal == j
                    if mm.sum():
                        add("by_modal_action", f"{ACT[j]}:n", mm.sum())
                        add("by_modal_action", f"{ACT[j]}:sum_delta", d[mm].sum())
                        add("by_modal_action", f"{ACT[j]}:mean_delta", d[mm].mean())
                for col in B.columns:
                    arr = B[col].to_numpy(object)[m]
                    for lv in sorted({x for x in arr if isinstance(x, str)}):
                        mm = arr == lv
                        if mm.sum() >= a.min_cell:
                            add("by_bucket", f"{col}={lv}:n", mm.sum())
                            add("by_bucket", f"{col}={lv}:mean_delta", d[mm].mean())
                ora = pool.choice["all"][m]
                for i in range(n):
                    perq.append(dict(retriever=retr, protocol=proto, model=kind, qid=pool.raw_qid[m][i], query=pool.query[m][i], delta=d[i],
                                     router_modal_action=ACT[modal[i]], oracle_action=ora[i], ndcg_a0=Vm[m][i, 0], ndcg_query2doc=Vm[m][i, AI["query2doc"]]))
    return pd.DataFrame(summ), pd.DataFrame(perq)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figures(ctx, c1, c2, c5, c7):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        say("matplotlib missing -> figures skipped")
        return
    fd = ctx.out / "figures"

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fd / f"{name}.png", dpi=200)
        fig.savefig(fd / f"{name}.pdf")
        plt.close(fig)

    # forest
    d = c1[(c1.level == 3) & (c1.metric == PRIMARY)]
    retrs, protos = list(dict.fromkeys(d.retriever)), list(dict.fromkeys(d.protocol))
    models, dsets = list(dict.fromkeys(d.model)), list(dict.fromkeys(d.dataset))
    fig, axes = plt.subplots(len(protos), len(retrs), figsize=(6 * len(retrs), 4.5 * len(protos)), squeeze=False)
    for i, p in enumerate(protos):
        for j, r in enumerate(retrs):
            ax = axes[i][j]
            for k, mod in enumerate(models):
                for y, dn in enumerate(dsets):
                    row = d[(d.protocol == p) & (d.retriever == r) & (d.model == mod) & (d.dataset == dn)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    ax.errorbar(row.mean_delta, y + (k - (len(models) - 1) / 2) * 0.22, xerr=[[row.mean_delta - row.ci_lo], [row.ci_hi - row.mean_delta]],
                                fmt="o", color=f"C{k}", capsize=2, label=mod if y == 0 else None)
            ax.axvline(0, color="k", lw=.8)
            ax.set_yticks(range(len(dsets)))
            ax.set_yticklabels(dsets)
            ax.invert_yaxis()
            ax.set_title(f"{r} / {p}")
            ax.set_xlabel("router - Const, nDCG@10 (95% bootstrap CI)")
            ax.legend(fontsize=7)
    save(fig, "C_forest_router_minus_const")

    # null vs real
    if len(c7):
        sub = c7[c7.dataset == "ALL"]
        combos = list(sub.groupby(["retriever", "protocol", "model"]).groups.keys())
        nc = 4
        nr = int(np.ceil(len(combos) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, key in zip(axes.ravel(), combos):
            g = sub[(sub.retriever == key[0]) & (sub.protocol == key[1]) & (sub.model == key[2])]
            ax.axis("on")
            ax.hist(g[~g.is_real].delta_vs_const, bins=12, color="C0", alpha=.7)
            ax.axvline(float(g[g.is_real].delta_vs_const.iloc[0]), color="C3", lw=2)
            ax.set_title("/".join(key), fontsize=8)
            ax.set_xlabel("router - Const (ALL)", fontsize=7)
        save(fig, "C_null_vs_real")

    # calibration
    if len(c5):
        dec = c5[(c5.scope == "decile") & (c5.node.isin([1, 2]))]
        rp = list(dict.fromkeys(zip(dec.retriever, dec.protocol)))
        fig, axes = plt.subplots(len(rp), 2, figsize=(9, 3.4 * len(rp)), squeeze=False)
        for i, (r, p) in enumerate(rp):
            for j, nd in enumerate([1, 2]):
                ax = axes[i][j]
                for k, mod in enumerate(["ridge", "gbr"]):
                    g = dec[(dec.retriever == r) & (dec.protocol == p) & (dec.node == nd) & (dec.model == mod)]
                    if len(g):
                        ax.plot(g.mean_pred, g.mean_true, "o-", color=f"C{k}", label=mod)
                ax.axhline(0, color="k", lw=.5)
                ax.set_title(f"{r}/{p} node {nd}", fontsize=8)
                ax.set_xlabel("mean predicted gain (decile)")
                ax.set_ylabel("mean true gain")
                ax.legend(fontsize=7)
        save(fig, "C_calibration")

    # action choice bars
    rp = list(dict.fromkeys(zip(c2.retriever, c2.protocol)))
    fig, axes = plt.subplots(len(rp), 1, figsize=(14, 3.6 * len(rp)), squeeze=False)
    for ax, (r, p) in zip(axes.ravel(), rp):
        g = c2[(c2.retriever == r) & (c2.protocol == p)]
        dsn = list(dict.fromkeys(g.dataset))
        variants = ["oracle"] + list(dict.fromkeys(g.model))
        ticks, labs = [], []
        for di, dn in enumerate(dsn):
            for vi, v in enumerate(variants):
                row = g[(g.dataset == dn) & (g.model == (variants[1] if v == "oracle" else v))].iloc[0]
                x, bot = di * (len(variants) + 1) + vi, 0.0
                for j, act in enumerate(ACT):
                    h = row[f"oracle_{act}"] if v == "oracle" else row[f"router_{act}"]
                    ax.bar(x, h, bottom=bot, color=f"C{j}", label=act if (di == 0 and vi == 0) else None)
                    bot += h
                ticks.append(x)
                labs.append(v)
            ax.text(di * (len(variants) + 1) + (len(variants) - 1) / 2, 1.03, dn, ha="center", fontsize=8)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs, rotation=90, fontsize=6)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{r} / {p}", fontsize=9, loc="left")
        ax.legend(fontsize=7, ncol=5, loc="lower right")
    save(fig, "C_action_choice")


# ----------------------------------------------------------------------------
# Acceptance checks, summary, README
# ----------------------------------------------------------------------------
def acceptance(ctx, c7s):
    a, res = ctx.a, []
    for retr in a.retrievers:
        pool = ctx.pool(retr)
        res.append(dict(check=f"{retr}: query count == 1370", ok=len(pool.qid) == 1370, detail=len(pool.qid)))
        for proto in a.protocols:
            for kind in ctx.kinds:
                exp = EXPECTED_L3_ALL.get((retr, proto, kind))
                if exp is not None:
                    got = float(ctx.vals(retr, proto, kind).mean())
                    res.append(dict(check=f"{retr}/{proto}/{kind}: L3 ALL nDCG@10 matches log ({exp})", ok=abs(got - exp) <= 3e-4, detail=round(got, 5)))
            idx = ctx.idx(retr, proto, "const")
            share = float((idx == AI["query2doc"]).mean())
            q2d = float(ctx.mat(retr, PRIMARY)[:, AI["query2doc"]].mean())
            gap = abs(float(ctx.vals(retr, proto, "const").mean()) - q2d)
            res.append(dict(check=f"{retr}/{proto}: Const == query2doc (share, |mean gap|)", ok=share > 0.98 and gap < 1e-3, detail=f"{share:.4f}, {gap:.5f}"))
            for kind in ctx.kinds:
                ev = R.evaluate(ctx.pool(retr), ctx.run(retr, proto, kind), 3, [ctx.lam0])["value"][0]
                diff = float(np.abs(ev - ctx.vals(retr, proto, kind)).max())
                res.append(dict(check=f"{retr}/{proto}/{kind}: custom L3 values == routing.evaluate", ok=diff < 1e-9, detail=diff))
    if c7s is not None and len(c7s):
        for r in c7s[c7s.dataset == "ALL"].itertuples():
            res.append(dict(check=f"null mean ~ 0: {r.retriever}/{r.protocol}/{r.model}", ok=abs(r.null_mean) < 0.005, detail=f"{r.null_mean:.5f} +- {r.null_se:.5f}"))
    return res


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


README = """# Workstream C: where the router helps or hurts

Generated by `analyze_router_where.py`. Git commit: {commit}. Run at {ts}.

## Settings
{settings}

Per-query values are averaged over the CV repeats before any test (query = unit; folds grouped by query).
Primary contrast = iteration 3, nDCG@10, router - Const. Iterations 1/2 (oracle below the learned nodes) are diagnostics (`role` column).
Holm family: 2 retrievers x 5 datasets (ALL excluded) per (protocol, model, level, metric, baseline); `n_tests_holm` gives the family size.
Router hyperparameters were tuned by nested CV on MSE, not on the end metric. lambda>0 results depend on `cost_table` times
(inflated under multi-worker generation). TREC-COVID/Touche have ~50 queries: Holm-corrected tests are rarely significant. C4 cells are exploratory.

## Columns (shared)
retriever, protocol (within|lodo), model (const|stump|ridge|gbr), level (1-3), role, metric, dataset (ALL = pooled queries).
mean_delta = mean per-query difference (router - baseline); ci_lo/ci_hi = paired bootstrap 95% CI; p_boot / p_wilcoxon = two-sided p;
cohen_dz = mean_delta / sd(differences); *_holm = Holm-adjusted; policy_mean/baseline_mean/a0_mean/oracle_mean = metric means.

## Files
- C1_router_vs_const.csv, C1_router_vs_bestfixed.csv: main contrasts (all levels/metrics).
- C2_action_choice.csv: router_<action> / oracle_<action> shares, tv_distance, agree_with_oracle. C2_confusion.csv: router_action x oracle_action (count averaged over repeats).
- C3_win_loss.csv: queries where router != Const; n_gain/mean_gain/n_loss/mean_loss/sums; group_type dataset or router_modal_action.
- C4_by_query_type.csv: router - Const per bucket (bucket_var prefix fb: = computed here, B0: = from B0 file); cells n >= min_cell.
- C5_node_calibration.csv: scope=decile rows (mean_pred vs mean_true per decile) and scope=summary rows (spearman, slope; decile=ALL or ds:<name>).
- C6_stability.csv: unit = repeat | fold | *_summary (sd, min, max, frac_positive across units) of router - Const.
- C7_null_distribution.csv (shuffle=-1 is the real router) and C7_null_summary.csv (null_mean/sd, z, emp_p_upper).
- C8_lambda_sweep.csv: ndcg10, go_rate, cost_s per router and lambda; protocol=fixed rows are fixed actions; dominated_by_fixed / dominates_fixed (ALL rows).
- C9_nfcorpus_dense.csv (section,key,value) and C9_nfcorpus_dense_per_query.csv.
- figures/*.png|pdf, tables/*.tex, C_summary.json, acceptance.json, checksums.json (sha256 of CSVs; compared with the previous run's).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", nargs="+", default=["store/scifact", "store/trec-covid", "store/nfcorpus", "store/fiqa", "store/webis-touche2020"])
    ap.add_argument("--retrievers", nargs="+", default=["bm25", "dense"])
    ap.add_argument("--protocols", nargs="+", default=["within", "lodo"])
    ap.add_argument("--models", nargs="+", default=["stump", "ridge", "gbr"])
    ap.add_argument("--b0", default=None, help="path to B0_query_descriptors.csv (auto-searched under results/ if omitted)")
    ap.add_argument("--bucket-cols", nargs="*", default=None, help="B0 columns to use as buckets (default: auto-detect)")
    ap.add_argument("--min-cell", type=int, default=15)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--boot2", type=int, default=2000, help="bootstrap B for exploratory tables (C4)")
    ap.add_argument("--null-n", type=int, default=20)
    ap.add_argument("--null-repeats", type=int, default=1)
    ap.add_argument("--null-models", nargs="+", default=["ridge", "gbr"])
    ap.add_argument("--skip-null", action="store_true")
    ap.add_argument("--nfc-name", default="nfcorpus")
    ap.add_argument("--nfc-retrievers", nargs="+", default=["dense"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache", default="cache/C_router")
    ap.add_argument("--out", default="results/analysis/C_router_where")
    a = ap.parse_args()
    if a.b0 is None:
        hits = sorted(Path("results").glob("**/B0_query_descriptors.csv"))
        a.b0 = str(hits[0]) if hits else None
    ctx = Ctx(a)
    out = ctx.out
    prev_ck = json.loads((out / "checksums.json").read_text()) if (out / "checksums.json").exists() else None

    # metrics available?
    for m in SECONDARY:
        try:
            for r in a.retrievers:
                ctx.mat(r, m)
            ctx.metrics.append(m)
        except Exception as e:
            say(f"secondary metric {m} unavailable ({e}); skipped")

    # CV runs (cached)
    for retr in a.retrievers:
        say(f"{retr}: {len(ctx.pool(retr).qid)} queries, datasets {ctx.pool(retr).ds_names}")
        for proto in a.protocols:
            for kind in ctx.kinds:
                ctx.run(retr, proto, kind)

    def save(df, name):
        df.to_csv(out / name, index=False, float_format="%.8g")
        say(f"wrote {name} ({len(df)} rows)")

    c1c, c1b = build_c1(ctx)
    save(c1c, "C1_router_vs_const.csv")
    save(c1b, "C1_router_vs_bestfixed.csv")
    c2, c2conf = build_c2(ctx)
    save(c2, "C2_action_choice.csv")
    save(c2conf, "C2_confusion.csv")
    c3 = build_c3(ctx)
    save(c3, "C3_win_loss.csv")
    c4 = build_c4(ctx)
    save(c4, "C4_by_query_type.csv")
    c5 = build_c5(ctx)
    save(c5, "C5_node_calibration.csv")
    c6 = build_c6(ctx)
    save(c6, "C6_stability.csv")
    c8 = build_c8(ctx)
    save(c8, "C8_lambda_sweep.csv")
    c9, c9q = build_c9(ctx)
    save(c9, "C9_nfcorpus_dense.csv")
    save(c9q, "C9_nfcorpus_dense_per_query.csv")
    c7, c7s = (pd.DataFrame(), None)
    if not a.skip_null:
        c7, c7s = build_c7(ctx)
        save(c7, "C7_null_distribution.csv")
        save(c7s, "C7_null_summary.csv")

    figures(ctx, c1c, c2, c5, c7)

    # LaTeX
    t = c1c[(c1c.level == 3) & (c1c.metric == PRIMARY)][["retriever", "protocol", "model", "dataset", "policy_mean", "baseline_mean", "mean_delta", "ci_lo", "ci_hi", "p_wilcoxon_holm", "cohen_dz"]]
    write_tex(t, out / "tables" / "T_C1_router_vs_const.tex", "Iteration 3, nDCG@10: router minus Const (Holm over 2 retrievers x 5 datasets).", "tab:c1")
    write_tex(c2[c2.dataset == "ALL"][["retriever", "protocol", "model", "router_go_rate", "router_top_action", "router_top_share", "tv_distance", "agree_with_oracle"]],
              out / "tables" / "T_C2_action_choice.tex", "Action choice of each router versus the oracle (all queries).", "tab:c2")
    write_tex(c3[c3.group_type == "dataset"][["retriever", "protocol", "model", "dataset", "n_differ", "n_gain", "mean_gain", "n_loss", "mean_loss", "net_mean"]],
              out / "tables" / "T_C3_win_loss.tex", "Gains and losses of the router relative to Const.", "tab:c3")
    if c7s is not None and len(c7s):
        write_tex(c7s[c7s.dataset == "ALL"][["retriever", "protocol", "model", "n_null", "null_mean", "null_sd", "real_delta", "z", "emp_p_upper"]],
                  out / "tables" / "T_C7_null.tex", "Shuffled-feature null versus real router (router minus Const, ALL).", "tab:c7")

    acc = acceptance(ctx, c7s)
    (out / "acceptance.json").write_text(json.dumps(acc, indent=2, default=str))

    # headline summary
    head = []
    for r in c1c[(c1c.level == 3) & (c1c.metric == PRIMARY)].itertuples():
        if r.dataset in ("ALL", a.nfc_name):
            head.append(dict(retriever=r.retriever, protocol=r.protocol, model=r.model, dataset=r.dataset, policy=r.policy_mean, const=r.baseline_mean,
                             delta=r.mean_delta, ci=[r.ci_lo, r.ci_hi], p_wilcoxon_holm=r.p_wilcoxon_holm, cohen_dz=r.cohen_dz))
    summ = dict(settings=vars(a), n_queries={r: int(len(ctx.pool(r).qid)) for r in a.retrievers}, headline_level3_ndcg10_vs_const=head,
                null_summary=None if c7s is None else c7s[c7s.dataset == "ALL"].to_dict("records"),
                best_fixed_rows=c1b[(c1b.level == 3) & (c1b.metric == PRIMARY) & (c1b.dataset == "ALL")][["retriever", "protocol", "model", "policy_mean", "baseline_mean", "mean_delta", "ci_lo", "ci_hi"]].to_dict("records"),
                acceptance_failed=[x for x in acc if not x["ok"]], git_commit=git_commit())
    (out / "C_summary.json").write_text(json.dumps(summ, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    (out / "README.md").write_text(README.format(commit=git_commit(), ts=time.strftime("%Y-%m-%d %H:%M"), settings="\n".join(f"- {k}: {v}" for k, v in vars(a).items())), encoding="utf-8")

    ck = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("C*.csv"))}
    (out / "checksums.json").write_text(json.dumps(ck, indent=2))

    print("\n==== ACCEPTANCE ====")
    for x in acc:
        print(("PASS " if x["ok"] else "FAIL ") + f"{x['check']}  [{x['detail']}]")
    if prev_ck is None:
        print("\nreproducibility: first run (re-run the same command to verify identical CSVs)")
    else:
        diff = [k for k in ck if prev_ck.get(k) != ck[k]]
        print("\nreproducibility: " + ("IDENTICAL to previous run" if not diff else f"DIFFERS from previous run in {diff} (ok only if settings/stores changed)"))
    print("\n==== HEADLINE (iteration 3, nDCG@10, router - Const, ALL) ====")
    h = c1c[(c1c.level == 3) & (c1c.metric == PRIMARY) & (c1c.dataset == "ALL")][["retriever", "protocol", "model", "policy_mean", "baseline_mean", "mean_delta", "ci_lo", "ci_hi", "p_wilcoxon"]]
    print(h.round(4).to_string(index=False))
    if c7s is not None and len(c7s):
        print("\n==== NULL (ALL) ====")
        print(c7s[c7s.dataset == "ALL"][["retriever", "protocol", "model", "null_mean", "null_sd", "real_delta", "z", "emp_p_upper"]].round(4).to_string(index=False))
    print(f"\nwrote {out}/  -- send me: this console output, C_summary.json, acceptance.json, and the 'B0 columns' lines")


if __name__ == "__main__":
    main()