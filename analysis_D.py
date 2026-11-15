#!/usr/bin/env python3
"""
run_D.py -- workstream D: confirmatory statistics + final tables. Run locally from the project folder.

  python run_D.py --smoke                      # fast wiring check (B=300, ridge only, within only, no tuning)
  python run_D.py --stores store/scifact store/trec-covid store/nfcorpus store/fiqa store/webis-touche2020

Pre-registered choices (edit PREREG.md if you change any; do NOT change after seeing results):
  primary router   = dense retriever, GBR, within-dataset protocol, iteration 3, lambda=0, nDCG@10
  confirmation     = BM25, plus ridge and leave-one-dataset-out variants
  primary pooled   = macro-average over datasets (others reported)
  H1 = router - Const ; H2 = router - best-fixed ; H3 = action - a0 (query2doc, prf, hyde, decomp, oracle)
  Holm families    = H1/H2: (router model x protocol x contrast) over 2 retrievers x 5 datasets
                     H3: (contrast) over 2 retrievers x 5 datasets ; ALL rows are never in a family
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing as R  # noqa: E402
import stats_lib as S  # noqa: E402

PRIMARY = {"retriever": "dense", "model": "gbr", "protocol": "within"}
LABELS = {"H1": "router_minus_const", "H2": "router_minus_bestfixed"}
H3_ACTIONS = ["query2doc", "prf", "hyde", "decomp"]
COLS = ["retriever", "model", "protocol", "dataset", "contrast", "metric", "n", "mean_delta", "ci_lo", "ci_hi", "cohens_d",
        "win", "tie", "loss", "p_boot", "p_wilcoxon", "boot_wilcoxon_disagree", "p_holm", "family_size", "primary_router"]


# ----------------------------------------------------------------------------
def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def is_primary(retr, model, proto):
    return bool(retr == PRIMARY["retriever"] and model == PRIMARY["model"] and proto == PRIMARY["protocol"])


def blocks(pool, diff):
    out = [(n, np.asarray(diff)[pool.ds_id == i]) for i, n in enumerate(pool.ds_names)]
    return out + [("ALL", np.asarray(diff))]


def contrast_rows(pool, diff, B, seed, **lab):
    rows = []
    for name, d in blocks(pool, diff):
        r = S.contrast(d, B, seed)
        r.update(lab, dataset=name)
        rows.append(r)
    return rows


def finish(rows, fam_cols):
    """Holm within each family over dataset rows only. ALL rows are shown but get no adjusted p."""
    df = pd.DataFrame(rows)
    ds, al = df[df.dataset != "ALL"].copy(), df[df.dataset == "ALL"].copy()
    ds = S.add_holm(ds, fam_cols)
    al["p_holm"], al["family_size"] = np.nan, 0
    out = pd.concat([ds, al], ignore_index=True)
    out = out.sort_values([c for c in ["retriever", "model", "protocol", "contrast", "metric"] if c in out.columns] + ["dataset"], kind="stable")
    return out.reset_index(drop=True)


def cols(df):
    return df[[c for c in COLS if c in df.columns] + [c for c in df.columns if c not in COLS]]


def router_values(pool, kind, proto, a, cache, retr, metric):
    ck = (retr, metric, proto)
    if ck not in cache:
        rc = R.run_cv(pool, "const", proto, a.folds, a.repeats, a.seed, tune=False)
        cache[ck] = R.evaluate(pool, rc, 3, [0.0])["value"][0]
    res = R.run_cv(pool, kind, proto, a.folds, a.repeats, a.seed, tune=not a.no_tune)
    ev = R.evaluate(pool, res, 3, [0.0])["value"][0]
    return ev, cache[ck], res.best_fixed.mean(axis=0)


def check_vs_csv(retr, proto, kind, pool, ev, a0, a):
    """Compare our mean(policy - a0) with results/router_<retr>/iteration_results.csv (level 3, ALL, lam 0)."""
    p = Path(a.check_dir.format(retr=retr)) / "iteration_results.csv"
    if a.smoke or not p.exists():
        return None
    df = pd.read_csv(p)
    m = df[(df.protocol == proto) & (df.model == kind) & (df.level == 3) & (df.dataset == "ALL") & (df.lam == df.lam.min())]
    if not len(m):
        return None
    return float(abs(m.d_a0.iloc[0] - (ev - a0).mean()))


def judged_table(path, retr, B, seed):
    """Per (dataset, action): judged@10 and nDCG@10 standard vs judged-only (condensed list), with delta vs a0."""
    from run_dataset import Store
    s = Store(path)
    ids = s.corpus_ids
    q = {}
    for qid, d, r in zip(s.qrels.qid.astype(str), s.qrels.doc_id.astype(str), s.qrels.rel):
        q.setdefault(qid, {})[d] = int(r)
    runs = s.runs
    runs = runs[(runs.retriever == retr) & (runs.subq_idx == -1) & runs.action.isin(R.ACTIONS) & (runs["rank"] <= 100)]
    runs = runs.sort_values(["action", "seed", "qid", "rank"], kind="stable")
    rec = []
    for (act, sd, qid), g in runs.groupby(["action", "seed", "qid"], sort=False):
        docs = [ids[i] for i in g.doc_idx.to_numpy()]
        qq = q.get(str(qid), {})
        rec.append((act, sd, str(qid), S.ndcg_at_k(docs, qq, 10), S.ndcg_at_k(docs, qq, 10, True), S.judged_at_k(docs, qq, 10)))
    d = pd.DataFrame(rec, columns=["action", "seed", "qid", "ndcg", "ndcg_cond", "judged10"])
    m = d.groupby(["qid", "action"])[["ndcg", "ndcg_cond", "judged10"]].mean()
    store_mean = s.metric_matrix("ndcg10", retr).mean()
    rows = []
    a0 = m.xs("a0", level="action")
    for act in R.ACTIONS:
        x = m.xs(act, level="action").reindex(a0.index)
        r = {"retriever": retr, "dataset": s.manifest["dataset"], "action": act, "n": len(x),
             "judged10": x.judged10.mean(), "ndcg10": x.ndcg.mean(), "ndcg10_judged_only": x.ndcg_cond.mean(),
             "ndcg10_check_vs_store": abs(x.ndcg.mean() - store_mean.get(act, np.nan))}
        if act != "a0":
            d1, d2 = (x.ndcg - a0.ndcg).to_numpy(), (x.ndcg_cond - a0.ndcg_cond).to_numpy()
            b1, b2 = S.paired_bootstrap(d1, B, seed), S.paired_bootstrap(d2, B, seed)
            r.update({"delta_std": b1[0], "delta_std_lo": b1[1], "delta_std_hi": b1[2], "p_std": b1[3],
                      "delta_judged_only": b2[0], "delta_cond_lo": b2[1], "delta_cond_hi": b2[2], "p_judged_only": b2[3],
                      "sign_changed": bool(np.sign(b1[0]) != np.sign(b2[0])),
                      "significance_changed": bool((b1[3] < 0.05) != (b2[3] < 0.05))})
        rows.append(r)
    return pd.DataFrame(rows)


def tex_num(x, nd=3):
    return "--" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def tex_p(p):
    return "--" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def write_tex(path, h1, h2, d3, retr, model, proto):
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline", r"Dataset & $n$ & $\Delta$ [95\% CI] & $d$ & $p_{\mathrm{Holm}}$ & W/T/L \\", r"\hline"]
    for title, df, key in (("H1: router $-$ Const", h1, "H1"), ("H2: router $-$ best-fixed", h2, "H2")):
        sub = df[(df.retriever == retr) & (df.model == model) & (df.protocol == proto)]
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{title} ({retr}, {model}, {proto})}}}} \\")
        for _, r in sub.iterrows():
            lines.append(f"{r.dataset} & {int(r.n)} & {tex_num(r.mean_delta)} [{tex_num(r.ci_lo)}, {tex_num(r.ci_hi)}] & "
                         f"{tex_num(r.cohens_d, 2)} & {tex_p(r.p_holm)} & {int(r.win)}/{int(r.tie)}/{int(r.loss)} \\\\")
        mac = d3[(d3.retriever == retr) & (d3.model == model) & (d3.protocol == proto) & (d3.contrast == LABELS[key]) & (d3.estimator == "macro_average")]
        for _, r in mac.iterrows():
            lines.append(rf"\textbf{{macro-avg}} & {int(r.n_queries)} & {tex_num(r.mean_delta)} [{tex_num(r.ci_lo)}, {tex_num(r.ci_hi)}] & -- & -- & -- \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", default=["store/scifact", "store/trec-covid", "store/nfcorpus", "store/fiqa", "store/webis-touche2020"])
    ap.add_argument("--retrievers", nargs="+", default=["bm25", "dense"])
    ap.add_argument("--models", nargs="+", default=["ridge", "gbr"])
    ap.add_argument("--protocols", nargs="+", default=["within", "lodo"])
    ap.add_argument("--B", type=int, default=S.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--skip-judged", action="store_true")
    ap.add_argument("--secondary-metrics", nargs="*", default=["map100", "recall100"])
    ap.add_argument("--check-dir", default="results/router_{retr}")
    ap.add_argument("--out", default="results/analysis/D_stats")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.B, a.models, a.protocols, a.repeats, a.no_tune = 300, ["ridge"], ["within"], 1, True
        a.secondary_metrics = a.secondary_metrics[:1]
        PRIMARY["model"] = "ridge"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B, seed = a.B, a.seed

    h1, h2, h3, d4, d6, d7 = [], [], [], [], [], []
    diffs = {}      # (retr, model, proto, contrast) -> {dataset: per-query delta}
    checks, cache = [], {}
    pools = {}

    for retr in a.retrievers:
        print(f"== {retr}: building pool", flush=True)
        pool = R.build_pool(a.stores, retr, "ndcg10", 1, "dataset", "crossfit")
        pools[retr] = pool
        a0 = pool.T["a0"]

        # ---- H3: reformulation vs a0 (+ oracle) ----
        h3d = {f"{act}_minus_a0": pool.T[act] - a0 for act in H3_ACTIONS}
        h3d["oracle_crossfit_minus_a0"] = pool.T["all"] - a0
        h3d["oracle_naive_minus_a0"] = pool.naive["all"] - a0
        for name, d in h3d.items():
            h3 += contrast_rows(pool, d, B, seed, retriever=retr, model="-", protocol="-", contrast=name, metric="ndcg10", primary_router=False)
            diffs[(retr, "-", "-", name)] = {n: d[pool.ds_id == i] for i, n in enumerate(pool.ds_names)}

        # ---- D4: seed variability (LLM actions; gains over a0) ----
        for i, n in enumerate(pool.ds_names + ["ALL"]):
            m = np.ones(len(pool.qid), bool) if n == "ALL" else pool.ds_id == i
            Vd = {act: pool.V[m, R.ACTIONS.index(act), :] for act in R.LLM_ACTIONS}
            t = S.seed_variability(Vd, base=pool.V[m, 0, 0], B=min(B, 2000), seed=seed)
            t.insert(0, "dataset", n), t.insert(0, "retriever", retr)
            d4.append(t)

        # ---- H1 / H2 ----
        for proto in a.protocols:
            for kind in a.models:
                print(f"   {retr}/{proto}/{kind}", flush=True)
                ev, evc, bf = router_values(pool, kind, proto, a, cache, retr, "ndcg10")
                chk = check_vs_csv(retr, proto, kind, pool, ev, a0, a)
                if chk is not None:
                    checks.append({"retriever": retr, "protocol": proto, "model": kind, "abs_diff_vs_iteration_results": chk})
                pr = is_primary(retr, kind, proto)
                for key, diff in (("H1", ev - evc), ("H2", ev - bf)):
                    lab = dict(retriever=retr, model=kind, protocol=proto, contrast=LABELS[key], metric="ndcg10", primary_router=pr)
                    (h1 if key == "H1" else h2).extend(contrast_rows(pool, diff, B, seed, **lab))
                    diffs[(retr, kind, proto, LABELS[key])] = {n: diff[pool.ds_id == i] for i, n in enumerate(pool.ds_names)}
                    for nme, dd in blocks(pool, diff)[:-1]:
                        d7.append({"retriever": retr, "model": kind, "protocol": proto, "contrast": LABELS[key], "dataset": nme, **S.power_row(dd)})
                if pr or a.smoke:
                    for eps in (0.01, 0.05, 0.10):
                        for nme, dd in blocks(pool, ev - a0):
                            hf, hm = S.helped_harmed(dd, eps)
                            d6.append({"analysis": "eps_helped_harmed", "retriever": retr, "model": kind, "protocol": proto, "dataset": nme,
                                       "contrast": "router_minus_a0", "eps": eps, "helped_frac": hf, "harmed_frac": hm})

    # ---- D2: secondary metrics, primary router only (exploratory: Holm AND BH, labelled) ----
    d2 = []
    for metric in a.secondary_metrics:
        for retr in a.retrievers:
            print(f"== secondary {metric}/{retr}", flush=True)
            p2 = R.build_pool(a.stores, retr, metric, 1, "dataset", "crossfit")
            ev, evc, bf = router_values(p2, PRIMARY["model"], PRIMARY["protocol"], a, cache, retr, metric)
            for key, diff in (("H1", ev - evc), ("H2", ev - bf), ("H3", p2.T["query2doc"] - p2.T["a0"])):
                lab = dict(retriever=retr, model=PRIMARY["model"] if key != "H3" else "-", protocol=PRIMARY["protocol"] if key != "H3" else "-",
                           contrast=LABELS.get(key, "query2doc_minus_a0"), metric=metric, primary_router=key != "H3")
                d2.extend(contrast_rows(p2, diff, B, seed, **lab))
    d2 = finish(d2, ["metric", "contrast"]) if d2 else pd.DataFrame()
    if len(d2):
        d2 = S.add_bh(d2[d2.dataset != "ALL"], ["metric", "contrast"], out="q_bh_exploratory").pipe(
            lambda x: pd.concat([x, d2[d2.dataset == "ALL"]], ignore_index=True))

    # ---- finish primary tables ----
    D1_H1, D1_H2 = finish(h1, ["model", "protocol", "contrast"]), finish(h2, ["model", "protocol", "contrast"])
    D1_H3 = finish(h3, ["contrast"])

    # ---- D3 pooled, D6 leave-one-out ----
    d3 = []
    for (retr, model, proto, name), dd in diffs.items():
        t = S.pooled_estimates(dd, B, seed)
        t.insert(0, "contrast", name), t.insert(0, "protocol", proto), t.insert(0, "model", model), t.insert(0, "retriever", retr)
        d3.append(t)
        t = S.leave_one_dataset_out(dd, B, seed)
        t.insert(0, "contrast", name), t.insert(0, "protocol", proto), t.insert(0, "model", model), t.insert(0, "retriever", retr)
        t.insert(0, "analysis", "leave_one_dataset_out")
        d6.append(t)
    D3 = pd.concat(d3, ignore_index=True)
    D6 = pd.concat([pd.DataFrame([x]) if isinstance(x, dict) else x for x in d6], ignore_index=True)
    for retr in a.retrievers:   # naive vs cross-fit oracle headroom (already in H3 as two contrasts; echoed for D6)
        for nme in ("oracle_crossfit_minus_a0", "oracle_naive_minus_a0"):
            for ds, dd in diffs[(retr, "-", "-", nme)].items():
                D6 = pd.concat([D6, pd.DataFrame([{"analysis": "oracle_variant", "retriever": retr, "contrast": nme, "dataset": ds,
                                                   "mean_delta": float(np.mean(dd))}])], ignore_index=True)
    D4 = pd.concat(d4, ignore_index=True)
    D7 = pd.DataFrame(d7)

    # ---- D5 judged sensitivity ----
    D5 = pd.DataFrame()
    if not a.skip_judged:
        parts = []
        for retr in a.retrievers:
            for sp in a.stores:
                print(f"== judged {retr}/{sp}", flush=True)
                parts.append(judged_table(sp, retr, min(B, 2000), seed))
        D5 = pd.concat(parts, ignore_index=True)

    # ---- null calibration + write ----
    base_pr = PRIMARY["retriever"] if PRIMARY["retriever"] in a.retrievers else a.retrievers[0]
    pr_diff = np.concatenate(list(diffs[(base_pr, PRIMARY["model"], PRIMARY["protocol"], LABELS["H1"])].values()))
    null = S.sign_flip_null(pr_diff, reps=100 if a.smoke else 200, B=500, seed=seed)

    cols(D1_H1).to_csv(out / "D1_primary_H1.csv", index=False)
    cols(D1_H2).to_csv(out / "D1_primary_H2.csv", index=False)
    cols(D1_H3).to_csv(out / "D1_H3_reformulation_vs_a0.csv", index=False)
    if len(d2):
        cols(d2).to_csv(out / "D2_secondary_map_recall.csv", index=False)
    D3.to_csv(out / "D3_pooled_estimates.csv", index=False)
    D4.to_csv(out / "D4_seed_variability.csv", index=False)
    if len(D5):
        D5.to_csv(out / "D5_judged_sensitivity.csv", index=False)
    D6.to_csv(out / "D6_leave_one_dataset_out_sensitivity.csv", index=False)
    D7.to_csv(out / "D7_power.csv", index=False)
    if checks:
        pd.DataFrame(checks).to_csv(out / "check_vs_iteration_results.csv", index=False)
    for retr in a.retrievers:
        for model in a.models:
            for proto in a.protocols:
                write_tex(out / f"tab_primary_{retr}_{model}_{proto}.tex", D1_H1, D1_H2, D3, retr, model, proto)

    def macro(retr, model, proto, name):
        r = D3[(D3.retriever == retr) & (D3.model == model) & (D3.protocol == proto) & (D3.contrast == name) & (D3.estimator == "macro_average")]
        return None if r.empty else {k: float(r.iloc[0][k]) for k in ("mean_delta", "ci_lo", "ci_hi", "p")}

    summ = {"settings": S.settings() | {"seed": seed, "folds": a.folds, "repeats": a.repeats, "tuned": not a.no_tune, "smoke": a.smoke},
            "primary_router": PRIMARY, "git_commit": git_commit(), "datasets": pools[a.retrievers[0]].ds_names,
            "n_queries": int(len(pools[a.retrievers[0]].qid)),
            "headline_macro": {f"{r}/{PRIMARY['model']}/{PRIMARY['protocol']}/{k}": macro(r, PRIMARY["model"], PRIMARY["protocol"], LABELS[k])
                               for r in a.retrievers for k in ("H1", "H2")},
            "H3_query2doc_minus_a0_ALL": {r: float(D1_H3[(D1_H3.retriever == r) & (D1_H3.dataset == "ALL") & (D1_H3.contrast == "query2doc_minus_a0")].mean_delta.iloc[0])
                                          for r in a.retrievers},
            "null_calibration_sign_flip": null, "check_vs_iteration_results_max_abs_diff": max([c["abs_diff_vs_iteration_results"] for c in checks], default=None),
            "boot_wilcoxon_disagreements": int(pd.concat([D1_H1, D1_H2, D1_H3]).boot_wilcoxon_disagree.sum())}
    (out / "D_summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    (out / "README.md").write_text(README.format(commit=summ["git_commit"], seed=seed, B=B, primary=PRIMARY), encoding="utf-8")
    print(json.dumps(summ, indent=2, default=str))
    print(f"\nwrote {out}/  (paste D_summary.json + D1_*.csv + D3 here)")


README = """# D_stats: confirmatory statistics

Commit: {commit} | seed {seed} | bootstrap B={B}

## Hypotheses (fixed before running; see PREREG.md)
- H1 (primary): iteration-3 router > Const on nDCG@10. H2 (primary): router > best-fixed. H3 (descriptive): action - a0 and oracle - a0.
- Primary router: {primary}. Others (BM25, ridge, lodo) are confirmation.
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
"""

if __name__ == "__main__":
    main()