#!/usr/bin/env python3
"""
run_dataset.py -- run ALL actions on ONE dataset and store everything needed
for later routing / statistics / failure-analysis work, in analysis-ready tables.

Usage (from the project folder that holds data/ and cache/):

    # 10-query dry run first (writes to store/<dataset>__limit10, nothing else touched)
    python run_dataset.py --dataset scifact --model <ollama-tag> --limit 10

    # full run (resumable: re-run the same command after any interruption)
    python run_dataset.py --dataset scifact --model <ollama-tag> --workers 3

Datasets: any BEIR name, e.g. scifact, trec-covid, nfcorpus, fiqa, webis-touche2020.

Pipeline (each stage is resumable and skipped when its outputs are up to date):
  index     build/verify BM25 + dense indexes and IDF statistics     -> cache/<ds>/
  generate  LLM calls (passage per query+seed, decomposition per query+seed)  [slow]
  retrieve  run every action x seed x retriever, record top-100 + costs
  metrics   per-query metrics for every run
  features  routing features (tagged by the stage at which they become available)

Everything after `generate` is cheap and deterministic and can be recomputed
with --force retrieve,metrics,features without touching the LLM outputs.

Use the Store class at the bottom for loading, e.g.

    from run_dataset import Store
    s = Store("store/scifact")
    gains = s.gain_matrix("ndcg10", "bm25")        # qid x action gains over a0
    costs = s.cost_table()                         # per (qid, action, seed, retriever)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants (changing any prompt/parameter changes results: they are hashed into
# the manifest and every generation row)
# ----------------------------------------------------------------------------
SCHEMA_VERSION = 1
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 100
RRF_K = 60
ROCCHIO_BETA = 0.5
ROCCHIO_K = 5
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

PRIMARY_ACTIONS = ["a0", "prf", "query2doc", "hyde", "decomp"]
EXTRA_ACTIONS = ["prf_k5", "rocchio", "decomp_orig"]  # cheap variants; keep out of headline numbers
ALL_ACTIONS = PRIMARY_ACTIONS + EXTRA_ACTIONS
SEEDLESS_ACTIONS = {"a0", "prf", "prf_k5", "rocchio"}  # stored with seed = -1
RETRIEVERS = ["bm25", "dense"]
ACTION_RETRIEVERS = {"rocchio": ["dense"]}
PRF_VARIANTS = {"prf": (10, 10), "prf_k5": (5, 5)}  # name -> (feedback docs, expansion terms)

PASSAGE_PROMPT = (
    "Write a short, factual passage (3-5 sentences) that directly answers "
    "the following question or claim. Do not repeat the question itself, "
    "just write the passage.\n\nQuestion: {query}\n\nPassage:"
)
DECOMP_PROMPT = (
    "Break the following question or claim into 2-4 simpler, self-contained "
    "sub-questions that together cover what is needed to answer it. If it is "
    "already simple, return a single sub-question. Return ONLY a JSON list of "
    "strings, nothing else.\n\nQuestion: {query}\n\nJSON:"
)
PROMPTS = {"passage": PASSAGE_PROMPT, "decomp": DECOMP_PROMPT}
NUM_PREDICT = {"passage": 200, "decomp": 150}
GEN_KINDS = ["passage", "decomp"]
STAGES = ["index", "generate", "retrieve", "metrics", "features"]

STOPWORDS = set(
    """a an the of in on at to for and or is are was were be been being this that
    these those it its as by with from into over under between among we you they he
    she i not no but if then than so such can could will would should may might do
    does did has have had which who whom what when where why how all any some more
    most other own same very just""".split()
)
WORD_RE = re.compile(r"[a-z]+")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
QUESTION_WORDS = ("who", "what", "when", "where", "why", "how", "which", "is", "are", "do", "does", "can")

log = logging.getLogger("rd")


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False, compression="zstd")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def check_dependencies() -> None:
    missing = []
    for mod, pip in [("pyarrow", "pyarrow"), ("bm25s", "bm25s==0.2.6"), ("pytrec_eval", "pytrec_eval-terrier"),
                     ("sentence_transformers", "sentence-transformers"), ("tqdm", "tqdm"), ("requests", "requests")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip)
    if missing:
        sys.exit("Missing packages: pip install " + " ".join(missing))


def tqdm_(it, **kw):
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:  # pragma: no cover
        return it


def setup_logging(path: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="a")):
        h.setFormatter(fmt)
        log.addHandler(h)


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else {}

    def update(self, **kw):
        self.d.update(kw)
        self.save()

    def set_stage(self, name: str, **info):
        self.d.setdefault("stages", {})[name] = {**info, "finished_at": now_iso()}
        self.save()

    def stage(self, name: str) -> dict:
        return self.d.get("stages", {}).get(name, {})

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=2, default=str))
        os.replace(tmp, self.path)


class Paths:
    def __init__(self, root: str, dataset: str, limit: int | None):
        self.root = Path(root)
        self.dataset = dataset
        self.tag = dataset if not limit else f"{dataset}__limit{limit}"
        self.data = self.root / "data" / dataset
        self.cache = self.root / "cache" / dataset
        self.store = self.root / "store" / self.tag
        for p in (self.data.parent, self.cache, self.store):
            p.mkdir(parents=True, exist_ok=True)

    def s(self, name: str) -> Path:
        return self.store / name


# ----------------------------------------------------------------------------
# Dataset loading (BEIR layout; no beir package needed)
# ----------------------------------------------------------------------------
def ensure_dataset(paths: Paths) -> None:
    if (paths.data / "corpus.jsonl").exists():
        return
    import requests

    url = BEIR_URL.format(paths.dataset)
    zpath = paths.data.parent / f"{paths.dataset}.zip"
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(paths.data.parent)
    zpath.unlink()
    if not (paths.data / "corpus.jsonl").exists():
        sys.exit(f"Downloaded archive did not produce {paths.data}/corpus.jsonl")


def load_dataset(paths: Paths, split: str, limit: int | None):
    corpus_ids, corpus_texts = [], []
    with open(paths.data / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus_ids.append(str(d["_id"]))
            corpus_texts.append((d.get("title", "") + " " + d.get("text", "")).strip())
    qrels_df = pd.read_csv(paths.data / "qrels" / f"{split}.tsv", sep="\t", dtype=str)
    qrels_df.columns = ["qid", "doc_id", "rel"]
    qrels_df["rel"] = qrels_df["rel"].astype(int)
    qrels: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for q, d, r in zip(qrels_df.qid, qrels_df.doc_id, qrels_df.rel):
        qrels[q][d] = int(r)
    queries = {}
    n_no_rel = 0
    with open(paths.data / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            qid = str(d["_id"])
            if qid not in qrels:
                continue
            if not any(v > 0 for v in qrels[qid].values()):
                n_no_rel += 1
                continue
            queries[qid] = d["text"]
    if n_no_rel:
        log.info("Dropped %d queries with no relevant document in qrels", n_no_rel)
    if limit:
        queries = dict(list(queries.items())[:limit])
    qrels = {q: qrels[q] for q in queries}
    return corpus_ids, corpus_texts, queries, qrels


# ----------------------------------------------------------------------------
# Model wrappers (module-level classes so they can be swapped in tests)
# ----------------------------------------------------------------------------
class Encoder:
    def __init__(self, name: str = DENSE_MODEL):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name)
        self.encode_one("warm up")

    def encode_many(self, texts, batch_size=64, progress=False) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=progress),
            dtype=np.float32,
        )

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]


class OllamaClient:
    """Plain HTTP client (works with any ollama python-package version)."""

    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: int = 300):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.digest = None

    def check(self) -> None:
        import requests

        try:
            tags = requests.get(self.host + "/api/tags", timeout=10).json().get("models", [])
        except Exception as e:
            sys.exit(f"Cannot reach Ollama at {self.host} ({e}). Is `ollama serve` running?")
        names = {m["name"]: m for m in tags}
        for cand in (self.model, self.model + ":latest"):
            if cand in names:
                self.model = cand
                self.digest = names[cand].get("digest")
                return
        sys.exit(f"Model '{self.model}' not found in Ollama. Available: {sorted(names)}")

    def generate(self, prompt: str, seed: int, num_predict: int, temperature: float) -> dict:
        import requests

        payload = {"model": self.model, "prompt": prompt, "stream": False, "think": False,
                   "options": {"seed": seed, "temperature": temperature, "num_predict": num_predict}}
        r = requests.post(self.host + "/api/generate", json=payload, timeout=self.timeout)
        if r.status_code == 400 and "think" in r.text.lower():
            payload.pop("think")
            r = requests.post(self.host + "/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        ms = lambda k: (d.get(k) or 0) / 1e6
        return {
            "response": d.get("response", "") or "",
            "thinking_chars": len(d.get("thinking", "") or ""),
            "done_reason": d.get("done_reason"),
            "eval_count": d.get("eval_count"),
            "prompt_eval_count": d.get("prompt_eval_count"),
            "total_duration_ms": ms("total_duration"),
            "load_duration_ms": ms("load_duration"),
            "prompt_eval_duration_ms": ms("prompt_eval_duration"),
            "eval_duration_ms": ms("eval_duration"),
        }


# ----------------------------------------------------------------------------
# Stage: index
# ----------------------------------------------------------------------------
def stage_index(ctx) -> None:
    import bm25s

    p = ctx.paths
    ids_path = p.cache / "corpus_ids.json"
    if ids_path.exists():
        cached = json.loads(ids_path.read_text())
        if cached != ctx.corpus_ids:
            sys.exit(f"{ids_path} does not match the corpus order on disk. Delete {p.cache} and re-run.")
    else:
        ids_path.write_text(json.dumps(ctx.corpus_ids))

    tok = None
    if not (p.cache / "bm25_index").exists():
        log.info("Building BM25 index (%d docs)", len(ctx.corpus_texts))
        tok = bm25s.tokenize(ctx.corpus_texts, stopwords="en", show_progress=True)
        r = bm25s.BM25()
        r.index(tok)
        r.save(str(p.cache / "bm25_index"))
    if not (p.cache / "df.npy").exists():
        log.info("Computing document frequencies")
        if tok is None:
            tok = bm25s.tokenize(ctx.corpus_texts, stopwords="en", show_progress=True)
        df = np.zeros(len(tok.vocab), dtype=np.int64)
        uniq = [np.unique(np.asarray(ids, dtype=np.int64)) for ids in tok.ids if len(ids)]
        if uniq:
            df += np.bincount(np.concatenate(uniq), minlength=len(df))
        np.save(p.cache / "df.npy", df)
        (p.cache / "vocab.json").write_text(json.dumps(tok.vocab))
    if not (p.cache / "dense_emb.npy").exists():
        log.info("Encoding corpus with %s", DENSE_MODEL)
        enc = ctx.get_encoder()
        np.save(p.cache / "dense_emb.npy", enc.encode_many(ctx.corpus_texts, progress=True))
    emb = np.load(p.cache / "dense_emb.npy", mmap_mode="r")
    if emb.shape[0] != len(ctx.corpus_ids):
        sys.exit("dense_emb.npy row count does not match the corpus. Delete the cache and re-run.")
    ctx.manifest.set_stage("index", n_docs=len(ctx.corpus_ids))


# ----------------------------------------------------------------------------
# Stage: generate
# ----------------------------------------------------------------------------
def prompt_hash(kind: str, model: str, temperature: float) -> str:
    return sha(json.dumps([PROMPTS[kind], model, temperature, NUM_PREDICT[kind]]))[:10]


def clean_passage(resp: str) -> str:
    t = THINK_RE.sub("", resp).strip()
    return re.sub(r"^\s*passage\s*:\s*", "", t, flags=re.I).strip()


def parse_subqs(resp: str, max_n: int = 4) -> list[str]:
    resp = THINK_RE.sub("", resp).strip()
    cands = None
    m = re.search(r"\[.*\]", resp, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                cands = [str(x) for x in arr]
        except Exception:
            cands = None
    if cands is None:
        # fallback: accept only bulleted/numbered lines or lines phrased as questions
        # (a refusal such as "I cannot help with that." must NOT count as a sub-question)
        cands = []
        for line in resp.splitlines():
            m2 = re.match(r"^\s*(?:[-*•]|\d+[.)])\s*(.*)$", line)
            body = (m2.group(1) if m2 else line).strip().strip('",').strip()
            if len(body) > 8 and (m2 or body.endswith("?")):
                cands.append(body)
    out, seen = [], set()
    for c in cands:
        c = c.strip()
        if len(c) > 3 and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out[:max_n]


def load_generations(path: Path) -> dict:
    rows = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[(r["qid"], int(r["seed"]), r["kind"])] = r  # latest line wins
    return rows


def count_ok_generations(path: Path) -> int:
    return sum(1 for r in load_generations(path).values() if r.get("ok"))


def gen_fingerprint(path: Path) -> str:
    """Content hash of all successful generations: changes whenever any generated text changes."""
    h = hashlib.sha1()
    rows = load_generations(path)
    for key in sorted(rows):
        r = rows[key]
        if r.get("ok"):
            h.update(repr((key, r.get("text"), r.get("subqs"))).encode("utf-8"))
    return h.hexdigest()[:16]


def stage_generate(ctx) -> None:
    args, gpath = ctx.args, ctx.paths.s("generations.jsonl")
    existing = load_generations(gpath)

    if args.reuse_legacy_passages:
        legacy = ctx.paths.cache / "generations.jsonl"
        n = 0
        if legacy.exists():
            with open(legacy) as f, open(gpath, "a", encoding="utf-8") as out:
                for line in f:
                    r = json.loads(line)
                    qid, seed, txt = str(r["qid"]), int(r["seed"]), (r.get("text") or "").strip()
                    key = (qid, seed, "passage")
                    if qid not in ctx.queries or seed not in args.seeds or not txt:
                        continue
                    if key in existing and existing[key].get("ok"):
                        continue
                    row = {"gen_id": f"{qid}|{seed}|passage", "qid": qid, "seed": seed, "kind": "passage",
                           "ok": True, "text": txt, "subqs": None, "parse_ok": True, "prompt_hash": "legacy",
                           "model": "legacy", "attempt": 0, "timestamp": now_iso()}
                    out.write(json.dumps(row) + "\n")
                    existing[key] = row
                    n += 1
        log.info("Imported %d legacy passages (prompt/model unverified -- see README)", n)

    todo = [(q, s, k) for q in ctx.qids for s in args.seeds for k in GEN_KINDS
            if not existing.get((q, s, k), {}).get("ok")]
    log.info("Generation: %d to do (%d already ok)", len(todo), len(ctx.qids) * len(args.seeds) * len(GEN_KINDS) - len(todo))
    if not todo:
        return
    client = ctx.get_llm()
    client.check()
    ctx.manifest.update(llm={"model": client.model, "digest": client.digest, "temperature": args.temperature,
                             "workers": args.workers, "prompt_hashes": {k: prompt_hash(k, client.model, args.temperature) for k in GEN_KINDS}})
    lock = threading.Lock()

    def work(item):
        qid, seed, kind = item
        prompt = PROMPTS[kind].format(query=ctx.queries[qid])
        last = None
        for attempt in range(args.max_attempts):
            eff_seed = seed + 1000 * attempt
            t0 = time.perf_counter()
            d, err = None, None
            try:
                d = client.generate(prompt, eff_seed, NUM_PREDICT[kind], args.temperature)
            except Exception as e:  # network / server error
                err = str(e)[:300]
            wall = (time.perf_counter() - t0) * 1000
            row = {"gen_id": f"{qid}|{seed}|{kind}", "qid": qid, "seed": seed, "kind": kind, "attempt": attempt,
                   "effective_seed": eff_seed, "prompt": prompt, "prompt_hash": prompt_hash(kind, client.model, args.temperature),
                   "model": client.model, "temperature": args.temperature, "num_predict": NUM_PREDICT[kind],
                   "wall_ms": wall, "workers": args.workers, "error": err, "timestamp": now_iso(),
                   "raw_response": None, "text": None, "subqs": None, "ok": False, "empty": None,
                   "truncated": None, "parse_ok": None}
            if d is not None:
                row.update({k: v for k, v in d.items() if k != "response"})
                row["raw_response"] = d["response"]
                row["truncated"] = d.get("done_reason") == "length"
                if kind == "passage":
                    txt = clean_passage(d["response"])
                    row.update(empty=not txt, text=txt or None, parse_ok=bool(txt), ok=bool(txt))
                else:
                    sq = parse_subqs(d["response"])
                    row.update(empty=not d["response"].strip(), subqs=sq or None, parse_ok=bool(sq), ok=bool(sq))
            last = row
            if row["ok"]:
                break
        return last

    done = 0
    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futs = [ex.submit(work, it) for it in todo]
        with open(gpath, "a", encoding="utf-8") as out:
            for fut in tqdm_(as_completed(futs), total=len(futs), desc="generate"):
                row = fut.result()
                with lock:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                done += 1
    except KeyboardInterrupt:
        log.warning("Interrupted; completed generations are saved. Re-run to resume.")
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    rows = load_generations(gpath)
    n_ok = sum(1 for r in rows.values() if r.get("ok"))
    log.info("Generation finished: %d ok, %d failed", n_ok, len(rows) - n_ok)
    for kind in GEN_KINDS:
        kr = [r for r in rows.values() if r["kind"] == kind]
        retried = sum(1 for r in kr if r.get("ok") and r.get("attempt", 0) > 0)
        trunc = sum(1 for r in kr if r.get("truncated"))
        log.info("  %-8s ok=%d failed=%d needed-retry=%d truncated=%d", kind, sum(1 for r in kr if r.get("ok")),
                 sum(1 for r in kr if not r.get("ok")), retried, trunc)
    if len(rows) - n_ok:
        log.warning("%d generations failed after %d attempts; re-run the same command to retry them "
                    "(queries missing a generation simply have no row for that action/seed)", len(rows) - n_ok, args.max_attempts)


# ----------------------------------------------------------------------------
# Retrieval helpers
# ----------------------------------------------------------------------------
class Retrievers:
    def __init__(self, ctx):
        import bm25s

        self.bm25s = bm25s
        self.bm25 = bm25s.BM25.load(str(ctx.paths.cache / "bm25_index"), load_corpus=False)
        self.dense = np.ascontiguousarray(np.load(ctx.paths.cache / "dense_emb.npy"), dtype=np.float32)
        self.n_docs = self.dense.shape[0]

    def bm25_search(self, text: str, k: int = TOP_K):
        t0 = time.perf_counter()
        k = min(k, self.n_docs)
        tok = self.bm25s.tokenize([text], stopwords="en", show_progress=False)
        if len(tok.ids[0]) == 0:
            tok = self.bm25s.tokenize([text], stopwords=None, show_progress=False)
        if len(tok.ids[0]) == 0:
            log.warning("Empty BM25 query tokens for text %r", text[:60])
            idx, sc = np.arange(k, dtype=np.int32), np.zeros(k, dtype=np.float32)
        else:
            res = self.bm25.retrieve(tok, k=k, show_progress=False)
            idx, sc = np.asarray(res[0][0], dtype=np.int32), np.asarray(res[1][0], dtype=np.float32)
        return idx, sc, (time.perf_counter() - t0) * 1000

    def dense_search(self, qvec: np.ndarray, k: int = TOP_K):
        t0 = time.perf_counter()
        scores = self.dense @ qvec
        n = scores.shape[0]
        k = min(k, n)
        part = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
        order = part[np.argsort(-scores[part], kind="stable")]
        return order.astype(np.int32), scores[order].astype(np.float32), (time.perf_counter() - t0) * 1000


def prf_expand(query: str, top_idx, top_scores, corpus_texts, n_terms: int):
    qterms = set(WORD_RE.findall(query.lower()))
    weights = collections.Counter()
    for i, s in zip(top_idx, top_scores):
        for t in WORD_RE.findall(corpus_texts[int(i)].lower()):
            if len(t) > 2 and t not in qterms and t not in STOPWORDS:
                weights[t] += float(s)
    terms = weights.most_common(n_terms)
    return query + " " + " ".join(t for t, _ in terms), [[t, round(w, 4)] for t, w in terms]


def rrf(lists, k: int = RRF_K, top: int = TOP_K):
    scores: dict[int, float] = {}
    for idx in lists:
        for r, d in enumerate(idx.tolist(), 1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + r)
    items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top]
    return np.array([d for d, _ in items], dtype=np.int32), np.array([s for _, s in items], dtype=np.float32)


# ----------------------------------------------------------------------------
# Stage: retrieve
# ----------------------------------------------------------------------------
def stage_retrieve(ctx) -> None:
    p = ctx.paths
    gens = load_generations(p.s("generations.jsonl"))
    passages = {(r["qid"], int(r["seed"])): r["text"] for r in gens.values() if r["kind"] == "passage" and r.get("ok")}
    subqs = {(r["qid"], int(r["seed"])): r["subqs"] for r in gens.values() if r["kind"] == "decomp" and r.get("ok")}
    seeds = ctx.args.seeds
    R = Retrievers(ctx)
    enc = ctx.get_encoder()

    emb: dict[str, np.ndarray] = {}
    emb_text: dict[str, str] = {}
    emb_ms: dict[str, float] = {}
    meta, idx_list, sc_list = [], [], []
    search_rows, cost_rows = [], []

    def add_run(qid, action, seed, retr, subq, idx, sc):
        meta.append((qid, action, seed, retr, subq))
        idx_list.append(idx)
        sc_list.append(sc)

    def add_cost(qid, action, seed, retr, subq, stage, ms):
        cost_rows.append((qid, action, seed, retr, subq, stage, float(ms)))

    def add_text(qid, action, seed, retr, subq, text, aux=None):
        search_rows.append({"qid": qid, "action": action, "seed": seed, "retriever": retr, "subq_idx": subq,
                            "text_id": sha(text) if text else "", "text": text, "aux": json.dumps(aux) if aux else ""})

    def run_bm25(qid, action, seed, subq, text, aux=None):
        idx, sc, ms = R.bm25_search(text)
        add_run(qid, action, seed, "bm25", subq, idx, sc)
        add_cost(qid, action, seed, "bm25", subq, "search", ms)
        add_text(qid, action, seed, "bm25", subq, text, aux)
        return idx, sc

    def encode(text):
        h = sha(text)
        if h not in emb:
            t0 = time.perf_counter()
            emb[h] = enc.encode_one(text)
            emb_text[h] = text
            emb_ms[h] = (time.perf_counter() - t0) * 1000
        return h, emb[h], emb_ms[h]

    def run_dense(qid, action, seed, subq, text, aux=None, vec=None):
        if vec is None:
            h, vec, ems = encode(text)
            add_cost(qid, action, seed, "dense", subq, "encode", ems)
        idx, sc, ms = R.dense_search(vec)
        add_run(qid, action, seed, "dense", subq, idx, sc)
        add_cost(qid, action, seed, "dense", subq, "search", ms)
        add_text(qid, action, seed, "dense", subq, text, aux)
        return idx, sc, vec

    for qid in tqdm_(ctx.qids, desc="retrieve"):
        q = ctx.queries[qid]
        a0 = {"bm25": run_bm25(qid, "a0", -1, -1, q)}
        a0_dn_idx, a0_dn_sc, qvec = run_dense(qid, "a0", -1, -1, q)
        a0["dense"] = (a0_dn_idx, a0_dn_sc)

        for name, (kd, nt) in PRF_VARIANTS.items():
            t0 = time.perf_counter()
            text, terms = prf_expand(q, a0["bm25"][0][:kd], a0["bm25"][1][:kd], ctx.corpus_texts, nt)
            ms = (time.perf_counter() - t0) * 1000
            aux = {"terms": terms, "k_docs": kd, "n_terms": nt}
            for retr in RETRIEVERS:
                add_cost(qid, name, -1, retr, -1, "prf_expand", ms)
            run_bm25(qid, name, -1, -1, text, aux)
            run_dense(qid, name, -1, -1, text, aux)

        t0 = time.perf_counter()
        v = qvec + ROCCHIO_BETA * R.dense[a0_dn_idx[:ROCCHIO_K]].mean(axis=0)
        v = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
        add_cost(qid, "rocchio", -1, "dense", -1, "prf_expand", (time.perf_counter() - t0) * 1000)
        run_dense(qid, "rocchio", -1, -1, q, {"beta": ROCCHIO_BETA, "k_docs": ROCCHIO_K}, vec=v)

        for seed in seeds:
            passage = passages.get((qid, seed))
            if passage:
                run_bm25(qid, "query2doc", seed, -1, (q + " ") * 5 + passage)
                run_dense(qid, "query2doc", seed, -1, q + " " + passage)
                run_bm25(qid, "hyde", seed, -1, passage)
                run_dense(qid, "hyde", seed, -1, passage)
            sq = subqs.get((qid, seed))
            if sq:
                per = {"bm25": [], "dense": []}
                for i, s in enumerate(sq):
                    per["bm25"].append(run_bm25(qid, "decomp", seed, i, s)[0])
                    per["dense"].append(run_dense(qid, "decomp", seed, i, s)[0])
                for retr in RETRIEVERS:
                    for action, lists in (("decomp", per[retr]), ("decomp_orig", per[retr] + [a0[retr][0]])):
                        t0 = time.perf_counter()
                        idx, sc = rrf(lists)
                        add_run(qid, action, seed, retr, -1, idx, sc)
                        add_cost(qid, action, seed, retr, -1, "fuse", (time.perf_counter() - t0) * 1000)
                        add_text(qid, action, seed, retr, -1, "", {"n_lists": len(lists), "rrf_k": RRF_K, "n_subq": len(sq)})

    # ---- write ----
    lens = np.array([len(a) for a in idx_list], dtype=np.int64)
    mq, ma, ms_, mr, mu = zip(*meta)
    runs = pd.DataFrame({
        "qid": np.repeat(np.array(mq, dtype=object), lens),
        "action": np.repeat(np.array(ma, dtype=object), lens),
        "seed": np.repeat(np.array(ms_, dtype=np.int16), lens),
        "retriever": np.repeat(np.array(mr, dtype=object), lens),
        "subq_idx": np.repeat(np.array(mu, dtype=np.int8), lens),
        "rank": np.concatenate([np.arange(1, n + 1, dtype=np.int16) for n in lens]),
        "doc_idx": np.concatenate(idx_list).astype(np.int32),
        "score": np.concatenate(sc_list).astype(np.float32),
    })
    write_table(runs, p.s("runs.parquet"))
    write_table(pd.DataFrame(search_rows), p.s("search_texts.parquet"))
    write_table(pd.DataFrame(cost_rows, columns=["qid", "action", "seed", "retriever", "subq_idx", "stage", "ms"]), p.s("costs.parquet"))
    hs = list(emb.keys())
    np.save(p.s("text_embeddings.npy"), np.stack([emb[h] for h in hs]).astype(np.float32))
    write_table(pd.DataFrame({"row": range(len(hs)), "text_id": hs, "text": [emb_text[h] for h in hs],
                              "encode_ms": [emb_ms[h] for h in hs]}), p.s("text_index.parquet"))
    log.info("Retrieval done: %d ranked lists, %d rows", len(meta), len(runs))
    ctx.manifest.set_stage("retrieve", n_gen_ok=count_ok_generations(p.s("generations.jsonl")),
                           gen_fp=gen_fingerprint(p.s("generations.jsonl")), n_lists=len(meta))


# ----------------------------------------------------------------------------
# Stage: metrics
# ----------------------------------------------------------------------------
def stage_metrics(ctx) -> None:
    import pytrec_eval

    p = ctx.paths
    runs = read_table(p.s("runs.parquet"))
    runs = runs[runs.subq_idx == -1].sort_values(["action", "seed", "retriever", "qid", "rank"], kind="stable")
    qrels = ctx.qrels
    ev = pytrec_eval.RelevanceEvaluator({q: {d: int(r) for d, r in v.items()} for q, v in qrels.items()},
                                        {"ndcg_cut_10", "ndcg_cut_100", "recall_10", "recall_100", "map_cut_100", "P_10"})
    ids = ctx.corpus_ids
    rows = []
    for (action, seed, retr), g in tqdm_(runs.groupby(["action", "seed", "retriever"], sort=False), desc="metrics"):
        run = {}
        extra = {}
        for qid, gq in g.groupby("qid", sort=False):
            docs = [ids[i] for i in gq["doc_idx"].to_numpy()]
            run[qid] = {d: float(s) for d, s in zip(docs, gq["score"].to_numpy())}
            top10 = docs[:10]
            q = qrels[qid]
            rr = next((1.0 / (i + 1) for i, d in enumerate(top10) if q.get(d, 0) > 0), 0.0)
            extra[qid] = (rr, sum(d in q for d in top10) / 10.0, len(docs))
        res = ev.evaluate(run)
        for qid, m in res.items():
            rr, j10, n = extra[qid]
            rows.append({"qid": qid, "action": action, "seed": int(seed), "retriever": retr,
                         "ndcg10": m["ndcg_cut_10"], "ndcg100": m["ndcg_cut_100"], "recall10": m["recall_10"],
                         "recall100": m["recall_100"], "map100": m["map_cut_100"], "p10": m["P_10"],
                         "mrr10": rr, "judged10": j10, "n_ret": n})
    write_table(pd.DataFrame(rows), p.s("metrics.parquet"))
    log.info("Metrics done: %d rows", len(rows))
    ctx.manifest.set_stage("metrics", n_rows=len(rows))


# ----------------------------------------------------------------------------
# Stage: features
# ----------------------------------------------------------------------------
def _entropy(s: np.ndarray) -> float:
    s = np.clip(s.astype(np.float64), 0, None)
    tot = s.sum()
    if tot <= 0:
        return float("nan")
    pr = s / tot
    pr = pr[pr > 0]
    return float(-(pr * np.log(pr)).sum())


def _cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def stage_features(ctx) -> None:
    import bm25s

    p = ctx.paths
    runs = read_table(p.s("runs.parquet"))
    tindex = read_table(p.s("text_index.parquet"))
    E = np.load(p.s("text_embeddings.npy"))
    row_of = dict(zip(tindex.text_id, tindex.row))
    dense = np.load(p.cache / "dense_emb.npy", mmap_mode="r")
    df = np.load(p.cache / "df.npy")
    vocab = json.loads((p.cache / "vocab.json").read_text())
    n_docs = len(ctx.corpus_ids)
    gens = load_generations(p.s("generations.jsonl"))

    def emb_of(text):
        r = row_of.get(sha(text))
        return None if r is None else E[r]

    a0 = runs[(runs.action == "a0") & (runs.subq_idx == -1)]
    a0_lists = {}
    for (retr, qid), g in a0.groupby(["retriever", "qid"], sort=False):
        a0_lists[(retr, qid)] = (g["doc_idx"].to_numpy(), g["score"].to_numpy())

    feats = {}
    for qid in ctx.qids:
        q = ctx.queries[qid]
        f = {}
        tok = bm25s.tokenize([q], stopwords="en", show_progress=False)
        inv = {v: k for k, v in tok.vocab.items()}
        toks = [inv[i] for i in tok.ids[0]]
        dfs = np.array([df[vocab[t]] if t in vocab else 0 for t in toks], dtype=np.float64)
        idf = np.log((n_docs - dfs + 0.5) / (dfs + 0.5) + 1.0) if len(toks) else np.array([np.nan])
        words = q.split()
        f.update({
            "s0_len_chars": len(q), "s0_len_words": len(words), "s0_n_tokens": len(toks),
            "s0_mean_word_len": float(np.mean([len(w) for w in words])) if words else float("nan"),
            "s0_idf_mean": float(np.nanmean(idf)), "s0_idf_max": float(np.nanmax(idf)),
            "s0_idf_min": float(np.nanmin(idf)), "s0_idf_std": float(np.nanstd(idf)),
            "s0_oov_frac": float(np.mean(dfs == 0)) if len(toks) else float("nan"),
            "s0_n_digit_tokens": sum(any(c.isdigit() for c in t) for t in toks),
            "s0_has_qmark": int("?" in q),
            "s0_starts_qword": int(bool(words) and words[0].lower() in QUESTION_WORDS),
        })
        tops = {}
        for retr in RETRIEVERS:
            idx, sc = a0_lists[(retr, qid)]
            tops[retr] = idx
            s = sc.astype(np.float64)
            f.update({
                f"s1_{retr}_top1": s[0], f"s1_{retr}_mean10": s[:10].mean(), f"s1_{retr}_std10": s[:10].std(),
                f"s1_{retr}_std100": s.std(), f"s1_{retr}_gap1_10": s[0] - s[min(9, len(s) - 1)],
                f"s1_{retr}_gap1_2": s[0] - s[1] if len(s) > 1 else float("nan"),
                f"s1_{retr}_wig": s[:10].mean() - s.mean(), f"s1_{retr}_nqc": s.std() / (abs(s.mean()) + 1e-9),
                f"s1_{retr}_norm_mean10": s[:10].mean() / (abs(s[0]) + 1e-9), f"s1_{retr}_entropy100": _entropy(s),
            })
        b, d = set(tops["bm25"][:10].tolist()), set(tops["dense"][:10].tolist())
        f["s1_overlap10"] = len(b & d) / 10.0
        f["s1_overlap100"] = len(set(tops["bm25"].tolist()) & set(tops["dense"].tolist())) / float(TOP_K)
        feats[qid] = f

    seed_rows = []
    qv_cache = {}
    for qid in ctx.qids:
        q = ctx.queries[qid]
        qe = emb_of(q)
        qv_cache[qid] = qe
        cent = {r: _centroid(dense, tops_idx[:10]) for r, tops_idx in
                {r: a0_lists[(r, qid)][0] for r in RETRIEVERS}.items()}
        qtoks = {t for t in WORD_RE.findall(q.lower()) if len(t) > 2 and t not in STOPWORDS}
        pass_embs, dec_embs = [], []
        for seed in ctx.args.seeds:
            r = {"qid": qid, "seed": seed}
            g = gens.get((qid, seed, "passage"))
            if g and g.get("ok"):
                txt = g["text"]
                pe = emb_of(txt)
                ptoks = set(WORD_RE.findall(txt.lower()))
                r.update({"s2_pass_len_words": len(txt.split()), "s2_pass_len_chars": len(txt),
                          "s2_pass_truncated": int(bool(g.get("truncated"))),
                          "s2_pass_q_overlap": len(qtoks & ptoks) / len(qtoks) if qtoks else float("nan")})
                if pe is not None:
                    pass_embs.append(pe)
                    r["s2_pass_q_cos"] = _cos(pe, qe) if qe is not None else float("nan")
                    for retr in RETRIEVERS:
                        r[f"s2_pass_{retr}_centroid_cos"] = _cos(pe, cent[retr])
            g = gens.get((qid, seed, "decomp"))
            if g and g.get("ok"):
                sq = g["subqs"]
                se = [emb_of(s) for s in sq]
                se = [e for e in se if e is not None]
                r.update({"s2_n_subq": len(sq), "s2_subq_total_chars": sum(len(s) for s in sq)})
                if se:
                    mean_e = np.mean(se, axis=0)
                    dec_embs.append(mean_e)
                    if qe is not None:
                        r["s2_subq_q_cos_mean"] = float(np.mean([_cos(e, qe) for e in se]))
                    if len(se) > 1:
                        r["s2_subq_pairwise_cos"] = float(np.mean([_cos(se[i], se[j]) for i in range(len(se)) for j in range(i + 1, len(se))]))
            seed_rows.append(r)
        agree = lambda es: float(np.mean([_cos(es[i], es[j]) for i in range(len(es)) for j in range(i + 1, len(es))])) if len(es) > 1 else float("nan")
        feats[qid]["s2_seed_agree_passage"] = agree(pass_embs)
        feats[qid]["s2_seed_agree_decomp"] = agree(dec_embs)

    ftab = pd.DataFrame.from_dict(feats, orient="index").rename_axis("qid").reset_index()
    write_table(ftab, p.s("features.parquet"))
    write_table(pd.DataFrame(seed_rows), p.s("features_seed.parquet"))
    stage_of = {c: int(c[1]) for c in list(ftab.columns) + list(pd.DataFrame(seed_rows).columns) if re.match(r"s[012]_", c)}
    p.s("feature_stages.json").write_text(json.dumps(stage_of, indent=2))
    log.info("Features done: %d queries, %d per-query columns", len(ftab), ftab.shape[1] - 1)
    ctx.manifest.set_stage("features", n_queries=len(ftab))


def _centroid(dense, idx):
    c = np.asarray(dense[np.sort(idx)], dtype=np.float32).mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-12)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
class Ctx:
    pass


def git_commit(root: Path):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def lib_versions() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    for m in ("pyarrow", "bm25s", "sentence_transformers", "torch", "pytrec_eval", "requests"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return out


def outputs_exist(ctx, stage: str) -> bool:
    p = ctx.paths
    need = {"retrieve": ["runs.parquet", "search_texts.parquet", "costs.parquet", "text_embeddings.npy", "text_index.parquet"],
            "metrics": ["metrics.parquet"], "features": ["features.parquet", "features_seed.parquet"]}[stage]
    return all(p.s(n).exists() for n in need)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N queries (writes to a separate store folder)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", default=None, help="Ollama model tag (see `ollama list`); required for generation")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=1, help="parallel LLM requests (start ollama with OLLAMA_NUM_PARALLEL=N)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--force", default="", help="comma list of stages to recompute (retrieve,metrics,features)")
    ap.add_argument("--reuse-legacy-passages", action="store_true",
                    help="import passages from cache/<ds>/generations.jsonl (only valid if prompt+model were identical)")
    args = ap.parse_args(argv)
    args.stages = [s for s in args.stages.split(",") if s]
    args.force = [s for s in args.force.split(",") if s]

    ctx = Ctx()
    ctx.args = args
    ctx.paths = Paths(args.root, args.dataset, args.limit)
    setup_logging(ctx.paths.s("run.log"))
    check_dependencies()
    # ensure_dataset(ctx.paths)
    ctx.corpus_ids, ctx.corpus_texts, ctx.queries, ctx.qrels = load_dataset(ctx.paths, args.split, args.limit)
    ctx.qids = list(ctx.queries)
    log.info("Dataset %s: %d docs, %d queries -> %s", args.dataset, len(ctx.corpus_ids), len(ctx.qids), ctx.paths.store)

    _enc = {}
    ctx.get_encoder = lambda: _enc.setdefault("e", Encoder())
    ctx.get_llm = lambda: OllamaClient(args.model, args.host, args.timeout)
    if "generate" in args.stages and not args.model:
        gen_missing = any(not load_generations(ctx.paths.s("generations.jsonl")).get((q, s, k), {}).get("ok")
                          for q in ctx.qids for s in args.seeds for k in GEN_KINDS)
        if gen_missing:
            sys.exit("--model is required (generations are missing). Run `ollama list` to see tags.")

    ctx.manifest = Manifest(ctx.paths.s("manifest.json"))
    started = ctx.manifest.d.get("started_at") or now_iso()
    ctx.manifest.update(
        schema_version=SCHEMA_VERSION, dataset=args.dataset, split=args.split, started_at=started, last_run_at=now_iso(),
        n_queries=len(ctx.qids), n_docs=len(ctx.corpus_ids), seeds=args.seeds, limit=args.limit,
        actions={"primary": PRIMARY_ACTIONS, "extra": EXTRA_ACTIONS, "seedless": sorted(SEEDLESS_ACTIONS)},
        params={"top_k": TOP_K, "rrf_k": RRF_K, "prf_variants": PRF_VARIANTS, "rocchio_beta": ROCCHIO_BETA,
                "rocchio_k": ROCCHIO_K, "dense_model": DENSE_MODEL, "prompts": PROMPTS, "num_predict": NUM_PREDICT},
        env={"platform": platform.platform(), "machine": platform.machine(), "libs": lib_versions(), "git_commit": git_commit(ctx.paths.root)},
    )
    # dataset tables (cheap, always rewritten)
    qdf = pd.DataFrame({"qid": ctx.qids, "text": [ctx.queries[q] for q in ctx.qids]})
    write_table(qdf, ctx.paths.s("queries.parquet"))
    write_table(pd.DataFrame([(q, d, r) for q, v in ctx.qrels.items() for d, r in v.items()], columns=["qid", "doc_id", "rel"]),
                ctx.paths.s("qrels.parquet"))
    ctx.paths.s("corpus_ids.json").write_text(json.dumps(ctx.corpus_ids))

    changed = False
    gpath = ctx.paths.s("generations.jsonl")
    for stage in STAGES:
        if stage not in args.stages:
            continue
        if stage == "index":
            stage_index(ctx)
        elif stage == "generate":
            stage_generate(ctx)
            ctx.manifest.set_stage("generate", n_gen_ok=count_ok_generations(gpath))
        else:
            stale = (ctx.manifest.stage("retrieve").get("gen_fp") != gen_fingerprint(gpath)) if stage == "retrieve" else False
            if stage in args.force or changed or stale or not outputs_exist(ctx, stage):
                log.info("Running stage: %s", stage)
                {"retrieve": stage_retrieve, "metrics": stage_metrics, "features": stage_features}[stage](ctx)
                changed = True
            else:
                log.info("Stage %s up to date, skipping", stage)
    ctx.manifest.update(finished_at=now_iso())
    print_summary(ctx.paths.store)


def print_summary(store_path: Path) -> None:
    try:
        s = Store(store_path)
        m = s.metrics
    except Exception as e:  # pragma: no cover
        log.info("(no summary: %s)", e)
        return
    g = s.generations
    log.info("---- Summary ----")
    log.info("generations ok: %d / %d", int(g.ok.sum()), len(g))
    for retr in RETRIEVERS:
        mat = s.metric_matrix("ndcg10", retr)
        if mat.empty:
            continue
        means = mat.mean().round(4).to_dict()
        oracle = mat.max(axis=1).mean()
        log.info("[%s] mean nDCG@10 by action: %s | naive oracle gap vs a0: %.4f", retr, means, oracle - mat["a0"].mean())
    log.info("Store: %s", store_path)


# ----------------------------------------------------------------------------
# Store: loader for downstream analysis code
# ----------------------------------------------------------------------------
class Store:
    """Read-only access to everything a run wrote. All tables are plain DataFrames."""

    def __init__(self, path):
        self.path = Path(path)

    def _t(self, name):
        return read_table(self.path / name)

    @cached_property
    def manifest(self) -> dict:
        return json.loads((self.path / "manifest.json").read_text())

    @cached_property
    def queries(self) -> pd.DataFrame:
        return self._t("queries.parquet")

    @cached_property
    def qrels(self) -> pd.DataFrame:
        return self._t("qrels.parquet")

    @cached_property
    def corpus_ids(self) -> list:
        return json.loads((self.path / "corpus_ids.json").read_text())

    @cached_property
    def generations(self) -> pd.DataFrame:
        rows = list(load_generations(self.path / "generations.jsonl").values())
        df = pd.DataFrame(rows)
        for c in ("ok", "empty", "truncated", "parse_ok"):
            if c in df:
                df[c] = df[c].fillna(False).astype(bool)
        return df

    @cached_property
    def runs(self) -> pd.DataFrame:
        """qid, action, seed(-1 = seedless), retriever, subq_idx(-1 = action-level list), rank, doc_idx, score"""
        return self._t("runs.parquet")

    @cached_property
    def search_texts(self) -> pd.DataFrame:
        return self._t("search_texts.parquet")

    @cached_property
    def metrics(self) -> pd.DataFrame:
        return self._t("metrics.parquet")

    @cached_property
    def costs(self) -> pd.DataFrame:
        return self._t("costs.parquet")

    @cached_property
    def features(self) -> pd.DataFrame:
        return self._t("features.parquet")

    @cached_property
    def features_seed(self) -> pd.DataFrame:
        return self._t("features_seed.parquet")

    @cached_property
    def feature_stages(self) -> dict:
        return json.loads((self.path / "feature_stages.json").read_text())

    def feature_columns(self, max_stage: int) -> list:
        """Per-query feature columns available by `max_stage` (0 pre-retrieval, 1 after a0 retrieval, 2 after LLM)."""
        return [c for c in self.features.columns if c != "qid" and self.feature_stages.get(c, 9) <= max_stage]

    def embeddings(self):
        """(matrix, text_index DataFrame) -- row i of the matrix is the embedding of text_index.row == i."""
        return np.load(self.path / "text_embeddings.npy"), self._t("text_index.parquet")

    def actions(self, include_extra: bool = False) -> list:
        return PRIMARY_ACTIONS + (EXTRA_ACTIONS if include_extra else [])

    def doc_ids(self, doc_idx):
        ids = self.corpus_ids
        return [ids[int(i)] for i in doc_idx]

    def run(self, action, retriever, seed=-1, subq_idx=-1, qid=None) -> pd.DataFrame:
        r = self.runs
        m = (r.action == action) & (r.retriever == retriever) & (r.seed == seed) & (r.subq_idx == subq_idx)
        if qid is not None:
            m &= r.qid == qid
        return r[m]

    def metric_matrix(self, metric="ndcg10", retriever="bm25", seed="mean", include_extra=False) -> pd.DataFrame:
        """qid x action matrix. seed='mean' averages the LLM seeds; or pass an int seed."""
        m = self.metrics
        m = m[m.retriever == retriever]
        acts = self.actions(include_extra)
        m = m[m.action.isin(acts)]
        if seed != "mean":
            m = m[(m.seed == seed) | (m.seed == -1)]
        mat = m.groupby(["qid", "action"])[metric].mean().unstack("action")
        return mat[[a for a in acts if a in mat.columns]]

    def gain_matrix(self, metric="ndcg10", retriever="bm25", seed="mean", include_extra=False) -> pd.DataFrame:
        """tau_a(q) = metric(a) - metric(a0); a0 column dropped."""
        mat = self.metric_matrix(metric, retriever, seed, include_extra)
        return mat.drop(columns="a0").sub(mat["a0"], axis=0)

    def cost_table(self, llm_col: str = "total_duration_ms") -> pd.DataFrame:
        """Per (qid, action, seed, retriever): own retrieval ms + dependency ms (first-stage runs the action reuses)
        + LLM ms (full generation cost charged to every action that uses it). Latency under --workers>1 is inflated."""
        c = self.costs
        own = c.groupby(["qid", "action", "seed", "retriever"], as_index=False)["ms"].sum().rename(columns={"ms": "retrieval_ms"})
        a0 = own[own.action == "a0"][["qid", "retriever", "retrieval_ms"]].rename(columns={"retrieval_ms": "a0_ms"})
        a0_bm = a0[a0.retriever == "bm25"][["qid", "a0_ms"]].rename(columns={"a0_ms": "a0_bm25_ms"})
        t = own.merge(a0, on=["qid", "retriever"], how="left").merge(a0_bm, on="qid", how="left")
        t["dependency_ms"] = (np.where(t.action.isin(["prf", "prf_k5"]), t.a0_bm25_ms.fillna(0.0), 0.0)
                              + np.where(t.action.isin(["rocchio", "decomp_orig"]), t.a0_ms.fillna(0.0), 0.0))
        kind = {"query2doc": "passage", "hyde": "passage", "decomp": "decomp", "decomp_orig": "decomp"}
        t["kind"] = t.action.map(kind)
        g = self.generations
        g = g[g.ok][["qid", "seed", "kind", llm_col, "eval_count"]].rename(columns={llm_col: "llm_ms", "eval_count": "llm_tokens_out"})
        t = t.merge(g, on=["qid", "seed", "kind"], how="left")
        t["llm_ms"] = t.llm_ms.fillna(0.0)
        t["llm_tokens_out"] = t.llm_tokens_out.fillna(0.0)
        t["total_ms"] = t.retrieval_ms + t.dependency_ms + t.llm_ms
        return t.drop(columns=["a0_ms", "a0_bm25_ms", "kind"])


if __name__ == "__main__":
    main()