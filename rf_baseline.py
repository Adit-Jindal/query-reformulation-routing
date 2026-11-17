#!/usr/bin/env python3
"""Script 2/3: RRF fusion baselines over the cached BM25 runs (no LLM calls, CPU only).
Run from the project root:
    python rrf_baseline.py
Output: results/bm25_rrf.csv  (dataset, qid, variant, seed, ndcg10)
Variants (RRF k=60, same as the decomp action):
    rrf_all5      a0 + prf + query2doc + hyde + decomp
    rrf_no_decomp a0 + prf + query2doc + hyde
    rrf_a0_prf    a0 + prf                  (no LLM at all)
    rrf_a0_q2d    a0 + query2doc
    rrf_a0_q2d_hyde  a0 + query2doc + hyde  (one shared LLM passage)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
import pytrec_eval
from run_dataset import Store, rrf

DATASETS = ["scifact", "trec-covid", "nfcorpus", "fiqa", "webis-touche2020"]
VARIANTS = {
    "rrf_all5": ["a0", "prf", "query2doc", "hyde", "decomp"],
    "rrf_no_decomp": ["a0", "prf", "query2doc", "hyde"],
    "rrf_a0_prf": ["a0", "prf"],
    "rrf_a0_q2d": ["a0", "query2doc"],
    "rrf_a0_q2d_hyde": ["a0", "query2doc", "hyde"],
}
SEEDLESS = {"a0", "prf"}

rows = []
for d in DATASETS:
    s = Store(f"store/{d}")
    ids = s.corpus_ids
    qrels = {}
    for q, doc, r in zip(s.qrels.qid.astype(str), s.qrels.doc_id.astype(str), s.qrels.rel):
        qrels.setdefault(q, {})[doc] = int(r)
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
    seeds = sorted(int(x) for x in s.manifest["seeds"])
    r = s.runs
    r = r[(r.retriever == "bm25") & (r.subq_idx == -1) & r.action.isin(["a0", "prf", "query2doc", "hyde", "decomp"])]
    r = r.sort_values(["qid", "action", "seed", "rank"])
    lists = {k: g["doc_idx"].to_numpy() for k, g in r.groupby(["qid", "action", "seed"], sort=False)}
    qids = s.queries.qid.astype(str).tolist()
    for name, acts in VARIANTS.items():
        for seed in ([-1] if all(a in SEEDLESS for a in acts) else seeds):
            run = {}
            for q in qids:
                ls = [lists.get((q, a, -1 if a in SEEDLESS else seed)) for a in acts]
                if any(l is None for l in ls):
                    continue
                idx, sc = rrf(ls)
                run[q] = {ids[int(i)]: float(v) for i, v in zip(idx, sc)}
            for q, m in ev.evaluate(run).items():
                rows.append((d, q, name, seed, m["ndcg_cut_10"]))
    print(d, "done", flush=True)

Path("results").mkdir(exist_ok=True)
out = pd.DataFrame(rows, columns=["dataset", "qid", "variant", "seed", "ndcg10"])
out.to_csv("results/bm25_rrf.csv", index=False)
print(out.groupby("variant").ndcg10.mean().round(4))
print("wrote results/bm25_rrf.csv", out.shape)