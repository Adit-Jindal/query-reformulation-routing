#!/usr/bin/env python3
"""Script 1/3: dump per-query BM25 metrics + costs for all actions/seeds into ONE small CSV.
Run from the project root (the folder with run_dataset.py and store/):
    python export_perquery.py
Output: results/bm25_perquery.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from run_dataset import Store

DATASETS = ["scifact", "trec-covid", "nfcorpus", "fiqa", "webis-touche2020"]
ACTIONS = ["a0", "prf", "query2doc", "hyde", "decomp"]

parts = []
for d in DATASETS:
    s = Store(f"store/{d}")
    m = s.metrics
    m = m[(m.retriever == "bm25") & m.action.isin(ACTIONS)][
        ["qid", "action", "seed", "ndcg10", "ndcg100", "recall10", "recall100", "map100", "p10", "judged10"]].copy()
    c = s.cost_table()
    c = c[(c.retriever == "bm25") & c.action.isin(ACTIONS)][
        ["qid", "action", "seed", "retrieval_ms", "llm_ms", "llm_tokens_out", "total_ms"]].copy()
    m["qid"], c["qid"] = m["qid"].astype(str), c["qid"].astype(str)
    x = m.merge(c, on=["qid", "action", "seed"], how="left")
    x.insert(0, "dataset", d)
    parts.append(x)
    print(d, len(x))

Path("results").mkdir(exist_ok=True)
out = pd.concat(parts, ignore_index=True)
out.to_csv("results/bm25_perquery.csv", index=False)
print("wrote results/bm25_perquery.csv", out.shape)