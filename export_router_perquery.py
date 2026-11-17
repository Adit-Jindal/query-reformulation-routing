#!/usr/bin/env python3
"""Script 3/3: per-query policy values for every router (real + shuffled-feature null), BM25, lambda=0.
Needed for router-vs-const / router-vs-best-fixed paired tests (Wilcoxon, Cohen's d, bootstrap).
Run from the project root (a few minutes of CPU; same CV settings as run_router.py: 5 folds x 3 repeats, seed 0):
    python export_router_perquery.py
Output: results/bm25_router_perquery.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
import routing as R

STORES = ["store/scifact", "store/trec-covid", "store/nfcorpus", "store/fiqa", "store/webis-touche2020"]
pool = R.build_pool(STORES, "bm25", "ndcg10", 1, "dataset", "crossfit")

out = pd.DataFrame({
    "dataset": [pool.ds_names[i] for i in pool.ds_id],
    "qid": pool.raw_qid,
    "a0": pool.T["a0"], "prf": pool.T["prf"], "query2doc": pool.T["query2doc"],
    "hyde": pool.T["hyde"], "decomp": pool.T["decomp"],
    "oracle_all": pool.T["all"], "oracle_choice": pool.choice["all"],
})

for proto in ["within", "lodo"]:
    for kind in ["const", "stump", "ridge", "gbr"]:
        for permute in (False, True):
            if permute and kind not in ("ridge", "gbr"):
                continue
            print(proto, kind, "null" if permute else "real", flush=True)
            res = R.run_cv(pool, kind, proto, folds=5, repeats=3, seed=0, tune=True, permute=permute)
            if kind == "const" and not permute:
                out[f"bestfixed|{proto}"] = res.best_fixed.mean(axis=0)
            for L in (1, 2, 3):
                ev = R.evaluate(pool, res, L, [0.0])
                tag = f"{proto}|{kind}|{'null' if permute else 'real'}|L{L}"
                out[tag + "|value"] = ev["value"][0]
                out[tag + "|go"] = ev["go"][0]

Path("results").mkdir(exist_ok=True)
out.to_csv("results/bm25_router_perquery.csv", index=False)
print("wrote results/bm25_router_perquery.csv", out.shape)