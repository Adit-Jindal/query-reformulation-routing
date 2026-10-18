# run_dataset.py -- one script per dataset

## Setup
    pip install -r requirements.txt        # if pyarrow has no wheel for your Python, use a 3.12/3.13 venv
    OLLAMA_NUM_PARALLEL=3 ollama serve     # in a separate terminal; lets --workers 3 run concurrently
    ollama list                            # copy the exact model tag for --model

## Always dry-run first (10 queries, writes to store/<ds>__limit10, touches nothing else)
    python run_dataset.py --dataset scifact --model <tag> --limit 10 --workers 3
Check: no errors, "Generation finished ... failed=0 (or few)", summary line prints mean nDCG@10 per action.

## Full runs (priority order; each is resumable -- just re-run the same command)
    python run_dataset_.py --dataset scifact          --model <tag> --workers 3 --reuse-legacy-passages
    python run_dataset_.py --dataset trec-covid       --model <tag> --workers 3 --reuse-legacy-passages
    python run_dataset_.py --dataset nfcorpus         --model <tag> --workers 3
    python run_dataset_.py --dataset fiqa             --model <tag> --workers 3
    python run_dataset_.py --dataset webis-touche2020 --model <tag> --workers 3
`--reuse-legacy-passages` imports the passages from your earlier cache/<ds>/generations.jsonl. Use it ONLY if the
passage prompt and model were identical to this script's (see PASSAGE_PROMPT); otherwise omit it and regenerate.
Existing cache/<ds>/ indexes (BM25, dense) are reused; the script checks the corpus order matches.

## Stages / flags
    --stages index,generate,retrieve,metrics,features     run a subset
    --force retrieve,metrics,features                      recompute derived tables (never re-calls the LLM)
    --seeds 0 1 2   --temperature 0.7   --max-attempts 3   --split test

## What lands in store/<dataset>/
    manifest.json        config, prompt hashes, model digest, library versions, git commit, stage state
    queries.parquet, qrels.parquet, corpus_ids.json
    generations.jsonl    every LLM call: prompt, raw output, parsed passage / sub-questions, tokens, timings, ok/empty/truncated, attempt
    search_texts.parquet exact string sent to each retriever per (qid, action, seed, retriever, subq_idx) + aux (PRF terms, ...)
    runs.parquet         top-100 for every ranked list: qid, action, seed(-1=seedless), retriever, subq_idx(-1=action-level), rank, doc_idx, score
    metrics.parquet      per (qid, action, seed, retriever): ndcg10, ndcg100, recall10, recall100, map100, p10, mrr10, judged10, n_ret
    costs.parquet        ms per stage (encode/search/prf_expand/fuse) for every list; LLM cost lives in generations.jsonl
    features.parquet     per-query routing features, columns prefixed s0_/s1_/s2_ = available before retrieval / after a0 retrieval / after LLM
    features_seed.parquet, feature_stages.json
    text_embeddings.npy + text_index.parquet   embeddings of every dense search text
    run.log

## Using it
    from run_dataset import Store
    s = Store("store/scifact")
    s.metric_matrix("ndcg10", "bm25")      # qid x action (seed-averaged); include_extra=True adds prf_k5/rocchio/decomp_orig
    s.gain_matrix("ndcg10", "dense")       # tau_a(q) = metric(a) - metric(a0)
    s.cost_table()                         # per (qid, action, seed, retriever); LLM cost charged to every action that uses it
    s.feature_columns(max_stage=1)         # features usable by a router that only sees a0 retrieval
    s.run("hyde", "bm25", seed=0, qid="...")  # ranked list; s.doc_ids(df.doc_idx) maps to BEIR doc ids
Primary actions: a0, prf, query2doc, hyde, decomp. Extras (keep out of headline numbers): prf_k5, rocchio (dense only), decomp_orig.