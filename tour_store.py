#!/usr/bin/env python3
"""
tour_store.py -- follow ONE query through everything run_dataset.py stored.

    python tour_store.py --store store/scifact              # first query
    python tour_store.py --store store/scifact --qid 1 --seed 1

Reads the store (and data/<dataset>/corpus.jsonl for document titles) and prints, in order:
  A the question and its ground truth   B what the LLM wrote   C what each action searched for
  D what came back (top docs)           E how each action scored   F what it cost   G the router features
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import Store  # noqa: E402

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 70)


def t(s, n=100):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "..."


def head(letter, title):
    print(f"\n{'=' * 4} {letter}. {title} " + "=" * max(4, 70 - len(title)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--qid", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    store_dir = Path(a.store)
    s = Store(store_dir)
    ds, seed = s.manifest["dataset"], a.seed
    qid = a.qid or s.queries.qid.iloc[0]

    print(f"STORE FOLDER: {store_dir}")
    for f in sorted(store_dir.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:9.2f} MB")

    titles = {}
    corpus = store_dir.parent.parent / "data" / ds / "corpus.jsonl"
    need = set(s.doc_ids(s.runs[s.runs.qid == qid].doc_idx.unique())) if corpus.exists() else set()
    if need:
        with open(corpus, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if str(d["_id"]) in need:
                    titles[str(d["_id"])] = d.get("title") or d.get("text", "")[:60]

    # A ---------------------------------------------------------------
    head("A", "THE QUESTION (queries.parquet, qrels.parquet)")
    q = s.queries[s.queries.qid == qid].text.iloc[0]
    rel = s.qrels[(s.qrels.qid == qid)]
    relset = set(rel[rel.rel > 0].doc_id)
    print(f"qid={qid}\nquery: {q}\nrelevant docs: {len(relset)} (ground truth; every score below is measured against these)")

    # B ---------------------------------------------------------------
    head("B", f"WHAT THE LLM WROTE (generations.jsonl, seed={seed})")
    g = s.generations
    for kind in ("passage", "decomp"):
        r = g[(g.qid == qid) & (g.seed == seed) & (g.kind == kind)]
        if r.empty:
            print(f"{kind}: (no generation for this seed)")
            continue
        r = r.iloc[0]
        body = r.text if kind == "passage" else r.subqs
        print(f"{kind}: ok={r.ok} attempts={r.attempt + 1} tokens_out={r.eval_count} total_ms={r.total_duration_ms:.0f}")
        print(f"   -> {t(body, 300)}")

    # C ---------------------------------------------------------------
    head("C", "WHAT EACH ACTION SENT TO THE RETRIEVERS (search_texts.parquet)")
    st = s.search_texts
    st = st[(st.qid == qid) & ((st.seed == seed) | (st.seed == -1))]
    st = st[st.text != ""]
    print(st[["action", "retriever", "subq_idx", "text"]].assign(text=st.text.map(lambda x: t(x, 80))).to_string(index=False))
    print("(seed=-1 actions do not use the LLM; subq_idx>=0 rows are the individual decomposition sub-questions)")

    # D ---------------------------------------------------------------
    head("D", "WHAT CAME BACK: top-3 documents, bm25 (runs.parquet; doc_idx -> doc id via corpus_ids.json)")
    for action, sd in (("a0", -1), ("hyde", seed), ("decomp", seed)):
        r = s.run(action, "bm25", seed=sd, subq_idx=-1, qid=qid).head(3)
        print(f"[{action}]")
        for _, row in r.iterrows():
            did = s.doc_ids([row.doc_idx])[0]
            print(f"   rank {int(row['rank'])} score {row.score:7.3f} {'RELEVANT ' if did in relset else '         '}{t(titles.get(did, did), 70)}")

    # E ---------------------------------------------------------------
    head("E", "HOW EACH ACTION SCORED ON THIS QUERY (metrics.parquet)")
    m = s.metrics
    m = m[(m.qid == qid) & ((m.seed == seed) | (m.seed == -1))]
    print(m.pivot_table(index="action", columns="retriever", values="ndcg10").reindex(s.actions(True)).round(3).dropna(how="all").to_string())
    print("(nDCG@10: 1.0 = perfect top 10, 0 = no relevant doc in the top 10; gain tau = action minus a0)")

    # F ---------------------------------------------------------------
    head("F", "WHAT IT COST, in ms (costs.parquet + generations.jsonl via Store.cost_table())")
    c = s.cost_table()
    c = c[(c.qid == qid) & ((c.seed == seed) | (c.seed == -1))]
    print(c.pivot_table(index="action", columns="retriever", values="total_ms").reindex(s.actions(True)).round(1).dropna(how="all").to_string())

    # G ---------------------------------------------------------------
    head("G", "ROUTER FEATURES (features.parquet; s0_/s1_/s2_ = known before / after first retrieval / after LLM)")
    f = s.features[s.features.qid == qid].iloc[0].drop("qid")
    for stage in (0, 1, 2):
        cols = [c for c in f.index if s.feature_stages.get(c) == stage][:5]
        print(f"stage {stage}: " + ", ".join(f"{c[3:]}={f[c]:.3g}" for c in cols))
    print("\nNext: Store.gain_matrix('ndcg10','bm25') gives this same E-table for every query at once.")


if __name__ == "__main__":
    main()