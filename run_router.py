#!/usr/bin/env python3
"""
run_router.py -- run the 3-iteration hierarchical router and write everything the report needs.

  python run_router.py --stores store/scifact store/trec-covid store/nfcorpus \
        --retriever bm25 --iteration all --protocol both --model ridge --out results/router_bm25

Outputs in --out:
  iteration_results.csv   per (protocol, model, level, lambda, dataset): policy vs a0 / best-fixed / oracle, CIs, Holm p
  node_diagnostics.csv    per node: accuracy vs oracle, AUC / Spearman (does the router's score track the true gain?)
  null_control.csv        same as results but with features shuffled (should recover ~0 of the gap)
  baselines.csv           a0, best fixed action, each fixed action, naive vs cross-fit oracle
  failures_L{level}_{protocol}.csv   cases to label by hand (blank cause_label / notes)
  iteration_log.md        human-readable log to paste into the report
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing as R  # noqa: E402

pd.set_option("display.width", 220, "display.max_columns", 40)


def baselines(pool):
    rows = []
    Vm = np.nanmean(pool.V, axis=2)
    blocks = [(n, pool.ds_id == i) for i, n in enumerate(pool.ds_names)] + [("ALL", np.ones(len(pool.qid), bool))]
    for name, m in blocks:
        row = {"dataset": name, "n": int(m.sum())}
        for j, a in enumerate(R.ACTIONS):
            row[f"fixed_{a}"] = Vm[m, j].mean()
        row["oracle_crossfit"] = pool.T["all"][m].mean()
        row["oracle_naive"] = pool.naive["all"][m].mean()
        row["go_headroom_crossfit"] = row["oracle_crossfit"] - row["fixed_a0"]
        row["oracle_go_rate"] = float((pool.choice["all"][m] != "a0").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(df):
    c = ["dataset", "level", "lam", "n", "a0", "best_fixed", "policy", "oracle", "recovered", "go_rate", "d_a0", "d_a0_lo", "d_a0_hi", "p_a0_holm"]
    d = df[df.lam == df.lam.min()][[x for x in c if x in df.columns]]
    return d.round(4).to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "dense"])
    ap.add_argument("--metric", default="ndcg10")
    ap.add_argument("--iteration", default="all")
    ap.add_argument("--protocol", default="both", choices=["within", "lodo", "pooled", "both"])
    ap.add_argument("--model", default="ridge", choices=R.MODEL_KINDS + ["all"])
    ap.add_argument("--max-stage", type=int, default=1)
    ap.add_argument("--oracle", default="crossfit", choices=["crossfit", "naive"])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05],
                    help="nDCG points per second of cost; first value is the headline setting")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--out", default="results/router")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = R.build_pool(a.stores, a.retriever, a.metric, a.max_stage, "dataset", a.oracle)
    levels = [1, 2, 3] if a.iteration == "all" else [int(a.iteration)]
    protos = ["within", "lodo"] if a.protocol == "both" else [a.protocol]
    if len(pool.ds_names) < 2:
        protos = [p for p in protos if p != "lodo"]
    kinds = R.MODEL_KINDS if a.model == "all" else [a.model]
    lam = list(a.lambdas)

    base = baselines(pool)
    base.to_csv(out / "baselines.csv", index=False)
    log = [f"# Router iteration log\n",
           f"- datasets: {pool.ds_names}  | retriever: {a.retriever} | metric: {a.metric} | queries: {len(pool.qid)} "
           f"(dropped for missing values: {pool.n_dropped})",
           f"- seeds used for cross-fit oracle: {pool.seeds} | oracle: {a.oracle} | features: stage<={a.max_stage} ({len(pool.feat_names)} usable)",
           f"- lambda grid (nDCG per second): {lam} | folds {a.folds} x repeats {a.repeats} | bootstrap B={a.bootstrap}\n",
           "## Baselines (nDCG@10)\n", "```", base.round(4).to_string(index=False), "```\n"]
    allres, alldiag, allnull = [], [], []
    for proto in protos:
        for kind in kinds:
            print(f"== {proto} / {kind}", flush=True)
            res = R.run_cv(pool, kind, proto, a.folds, a.repeats, a.seed, tune=not a.no_tune)
            null = R.run_cv(pool, kind, proto, a.folds, a.repeats, a.seed, tune=not a.no_tune, permute=True) if kind in ("ridge", "gbr") else None
            diag = R.node_diagnostics(pool, res)
            diag.insert(0, "model", kind), diag.insert(0, "protocol", proto)
            alldiag.append(diag)
            log += [f"## Protocol `{proto}`, model `{kind}`\n", "Node diagnostics (does the predicted gain track the true gain?):", "```",
                    diag.round(3).to_string(index=False), "```\n"]
            for L in levels:
                df = R.summarize(pool, res, L, lam, a.bootstrap)
                df.insert(0, "model", kind), df.insert(0, "protocol", proto)
                allres.append(df)
                log += [f"### Iteration {L} ({'go/no-go; oracle below' if L == 1 else 'adds LLM-vs-PRF; oracle among LLM actions' if L == 2 else 'adds LLM-action choice; no oracle'})\n",
                        "```", fmt(df), "```\n"]
                if null is not None:
                    dn = R.summarize(pool, null, L, lam[:1], max(500, a.bootstrap // 10))
                    dn.insert(0, "model", kind), dn.insert(0, "protocol", proto)
                    allnull.append(dn)
                    log.append(f"Null control (features shuffled), ALL: recovered={dn[dn.dataset == 'ALL'].recovered.iloc[0]:.3f}\n")
                fs = R.failure_sheet(pool, res, L, seed=a.seed)
                fs.to_csv(out / f"failures_L{L}_{proto}_{kind}.csv", index=False)
                if len(fs):
                    log.append(f"Failure sample L{L}: " + ", ".join(f"{k}={v}" for k, v in fs.outcome.value_counts().items()) + "\n")
                print(fmt(df), flush=True)
    pd.concat(allres).to_csv(out / "iteration_results.csv", index=False)
    pd.concat(alldiag).to_csv(out / "node_diagnostics.csv", index=False)
    if allnull:
        pd.concat(allnull).to_csv(out / "null_control.csv", index=False)
    (out / "iteration_log.md").write_text("\n".join(log), encoding="utf-8")
    print(f"\nwrote {out}/ (iteration_log.md is the thing to read first)")


if __name__ == "__main__":
    main()