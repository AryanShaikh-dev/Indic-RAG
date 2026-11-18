"""T1 + T2 — Retrieval reproduction and baseline comparison.

All systems see IDENTICAL queries and IDENTICAL chunk corpora (production RecursiveChunker(1000, 200)).
Dense search is exact cosine (torch matmul) == Qdrant local-mode exact search (verified for e5-small on DS1).

Datasets / splits
  DS1-all   : 627 labelled queries (reproduces stored benchmark_results.json)
  DS1-test  : held-out half of DS1 (comparisons reported here; DS1-dev reserved for tuning in T3/T5/T6)
  DSV-test  : 80% of labelled queries from a fresh 5,000-query sample of the MS MARCO-XI *validation* split
  DSV-test-EN: same DSV-test queries in ENGLISH (Eng_Query) against the Hindi corpus (cross-lingual)

Usage: .venv-gpu/Scripts/python audit/scripts/t1_t2_retrieval.py
"""
import time
import numpy as np
import torch
from common import (load_ds1, ds1_split, load_dsv, dsv_split, chunk_corpus, BM25, encode, dense_topk,
                    relevance_matrix, per_query_metrics, summarize_metrics, mcnemar, paired_bootstrap_diff,
                    load_bi_encoder, environment, write_results, CACHE_DIR)
from src.core.chunking import RecursiveChunker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH = 20
MODELS = [  # name, hf_id, query_prefix, passage_prefix, max_seq_length (None = model default)
    ("minilm_l6_en", "sentence-transformers/all-MiniLM-L6-v2", "", "", None),
    ("e5_small", "intfloat/multilingual-e5-small", "query: ", "passage: ", None),
    ("para_mminilm_l12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", "", None),
    ("e5_base", "intfloat/multilingual-e5-base", "query: ", "passage: ", None),
]

ds1, dsv = load_ds1(), load_dsv()
s1, sv = ds1_split(), dsv_split(dsv)
chunker = RecursiveChunker()
corp = {}
for ds in (ds1, dsv):
    chunks, owner = chunk_corpus(ds.passages, chunker)
    corp[ds.name] = (chunks, owner)
    print(f"[DATA] {ds.name}: {len(ds.queries)} queries, {len(ds.labelled_ids)} labelled, {len(ds.passages)} passages, {len(chunks)} chunks")

# evaluation slices: (slice_name, dataset, qids, query_text_getter)
slices = [
    ("DS1-all", ds1, ds1.labelled_ids, lambda q: ds1.queries[q]),
    ("DS1-dev", ds1, s1["dev"], lambda q: ds1.queries[q]),
    ("DS1-test", ds1, s1["test"], lambda q: ds1.queries[q]),
    ("DSV-test", dsv, sv["test"], lambda q: dsv.queries[q]),
    ("DSV-test-EN", dsv, [q for q in sv["test"] if dsv.eng_queries.get(q)], lambda q: dsv.eng_queries[q]),
]

per_query = {}   # (system, slice) -> dict(metric -> array)
rows, meta = [], {}


import pickle
DONE_DIR = CACHE_DIR / "t1_t2_systems"
DONE_DIR.mkdir(exist_ok=True)


def load_done(name):
    f = DONE_DIR / f"{name}.pkl"
    if not f.exists():
        return False
    d = pickle.load(open(f, "rb"))
    per_query.update(d["per_query"]); rows.extend(d["rows"]); meta.update(d["meta"])
    print(f"  [cached] {name}")
    return True


def save_done(name):
    pickle.dump({"per_query": {k: v for k, v in per_query.items() if k[0] == name},
                 "rows": [r for r in rows if r["system"] == name], "meta": {name: meta.get(name, {})}},
                open(DONE_DIR / f"{name}.pkl", "wb"))


def score_system(name, rank_fn):
    for sl, ds, qids, getq in slices:
        chunks, owner = corp[ds.name]
        idx = rank_fn(ds.name, [getq(q) for q in qids])
        rel = relevance_matrix(idx, owner, qids, ds.gold)
        pq = per_query_metrics(rel, qids, ds.gold, ranked_owner=owner[idx])
        per_query[(name, sl)] = pq
        rows.extend(summarize_metrics(name, ds.name, sl, pq))
        if name in ("e5_small", "bm25") and sl in ("DS1-all", "DS1-dev", "DS1-test", "DSV-test"):
            np.save(CACHE_DIR / f"rank_{name}_{sl}.npy", idx)
        print(f"  {name:18s} {sl:12s} n={len(qids):5d} Hit@1={100*pq['Hit@1'].mean():5.2f} Hit@5={100*pq['Hit@5'].mean():5.2f} "
              f"Hit@10={100*pq['Hit@10'].mean():5.2f} MRR@10={100*pq['MRR@10'].mean():5.2f}")


# ---------------- BM25
bm = {}
t = time.perf_counter()
for name, (chunks, _) in corp.items():
    bm[name] = BM25(chunks)
meta["bm25"] = {"build_s_both_corpora": round(time.perf_counter() - t, 1), "k1": 1.2, "b": 0.75,
                "tokenizer": "NFC + lowercase + regex [\\u0900-\\u097F\\w]+ (no stopwords, no stemming)"}
if not load_done("bm25"):
    score_system("bm25", lambda c, qs: np.array([bm[c].search(q, DEPTH)[0] for q in qs]))
    save_done("bm25")

# ---------------- dense models
dense_rank_cache = {}
for name, hf, qp, pp, msl in MODELS:
    if load_done(name):
        continue
    t = time.perf_counter()
    model = load_bi_encoder(hf, DEVICE, msl)
    load_s = time.perf_counter() - t
    emb, thr = {}, {}
    for cname, (chunks, _) in corp.items():
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        emb[cname] = encode(model, chunks, pp, batch_size=64)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        thr[cname] = round(len(chunks) / (time.perf_counter() - t), 1)
        if name == "e5_small":
            np.save(CACHE_DIR / f"emb_e5_small_{cname}.npy", emb[cname])
    meta[name] = {"hf_id": hf, "query_prefix": qp, "passage_prefix": pp, "dim": int(emb["DS1"].shape[1]),
                  "max_seq_length": model.max_seq_length, "load_s": round(load_s, 1),
                  "index_throughput_chunks_per_s_gpu": thr, "device": DEVICE}

    def rank(cname, qs, model=model, qp=qp, emb=emb):
        qe = encode(model, qs, qp, batch_size=128)
        idx, _ = dense_topk(qe, emb[cname], DEPTH, DEVICE)
        return idx

    score_system(name, rank)
    save_done(name)
    del model, emb
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ---------------- hybrid: Reciprocal Rank Fusion of BM25 + e5_small (k=60, untuned standard constant)
_E5 = {}


def rrf_rank(cname, qs):
    e5 = _E5.setdefault("m", load_bi_encoder("intfloat/multilingual-e5-small", DEVICE))
    ce = np.load(CACHE_DIR / f"emb_e5_small_{cname}.npy")
    di, _ = dense_topk(encode(e5, qs, "query: "), ce, 100, DEVICE)
    out = []
    for q, d in zip(qs, di):
        b, _ = bm[cname].search(q, 100)
        s = {}
        for r, c in enumerate(d):
            s[c] = s.get(c, 0) + 1 / (60 + r + 1)
        for r, c in enumerate(b):
            s[c] = s.get(c, 0) + 1 / (60 + r + 1)
        out.append(sorted(s, key=s.get, reverse=True)[:DEPTH])
    return np.array(out)


if not load_done("rrf_bm25_e5small"):
    meta["rrf_bm25_e5small"] = {"fusion": "Reciprocal Rank Fusion, k=60, top-100 from each retriever", "tuned": False}
    score_system("rrf_bm25_e5small", rrf_rank)
    save_done("rrf_bm25_e5small")

# ---------------- paired comparisons vs production (e5_small)
comparisons = []
systems = ["bm25"] + [m[0] for m in MODELS if m[0] != "e5_small"] + ["rrf_bm25_e5small"]
for sl in ["DS1-test", "DSV-test", "DSV-test-EN"]:
    base = per_query[("e5_small", sl)]
    for s in systems:
        other = per_query[(s, sl)]
        for k in ("Hit@1", "Hit@5"):
            r = mcnemar(base[k], other[k])
            comparisons.append({"slice": sl, "baseline": "e5_small", "system": s, "metric": k, "test": "McNemar exact",
                                "baseline_pct": round(100 * base[k].mean(), 2), "system_pct": round(100 * other[k].mean(), 2),
                                "delta_pp": round(100 * (other[k].mean() - base[k].mean()), 2), **r})
        d = paired_bootstrap_diff(base["MRR@10"], other["MRR@10"])
        comparisons.append({"slice": sl, "baseline": "e5_small", "system": s, "metric": "MRR@10", "test": "paired bootstrap (2000)",
                            "baseline_pct": round(100 * base["MRR@10"].mean(), 2), "system_pct": round(100 * other["MRR@10"].mean(), 2),
                            "delta_pp": round(100 * d["mean_diff"], 2), "ci_low_pp": round(100 * d["ci_low"], 2),
                            "ci_high_pp": round(100 * d["ci_high"], 2), "p_value": d["p_two_sided"]})

# ---------------- complementarity BM25 vs e5_small (Hit@10)
comp = {}
for sl in ["DS1-test", "DSV-test"]:
    a, b = per_query[("e5_small", sl)]["Hit@10"].astype(bool), per_query[("bm25", sl)]["Hit@10"].astype(bool)
    comp[sl] = {"bm25_hits_e5_misses_pct": round(100 * (b & ~a).mean(), 2), "e5_hits_bm25_misses_pct": round(100 * (a & ~b).mean(), 2),
                "both_miss_pct": round(100 * (~a & ~b).mean(), 2)}

payload = {
    "test_id": "T1_T2_retrieval", "status": "NEWLY EXECUTED", "environment": environment(DEVICE),
    "datasets": {
        "DS1": {"source": "benchmark_artifacts (MS MARCO-XI train sample, seed 42)", "passages": len(ds1.passages), "chunks": len(corp["DS1"][0]),
                "labelled_queries": len(ds1.labelled_ids), "dev": len(s1["dev"]), "test": len(s1["test"])},
        "DSV": {"source": "MS MARCO-XI validation/hinval.parquet (97,941 rows), 5,000-query sample seed 7", "passages": len(dsv.passages),
                "chunks": len(corp["DSV"][0]), "labelled_queries": len(dsv.labelled_ids), "dev": len(sv["dev"]), "test": len(sv["test"])},
    },
    "excluded_systems": {"bge_m3 (BAAI/bge-m3, 568M params, 1024-d)": "NOT EVALUATED - excluded due to hardware memory limits: embedding the 60.6k-chunk corpora exceeded the laptop's Windows commit limit (15.3 GB RAM, ~20 GB of 28 GB commit already used by other applications); the run was stopped by the Claude Code memory guard and, per user decision, not restarted."},
    "notes": ["Hit@k == repo's 'Recall@k' (>=1 relevant chunk in top-k). Recall@k = distinct gold passages found / gold passages.",
              "Closed-pool corpora (each query's ~10 candidate passages pooled): absolute numbers are higher than open-corpus retrieval.",
              "Relevance = chunk belongs to an is_selected=1 passage (single-label for ~95% of queries; label incompleteness not corrected)."],
    "systems": meta, "comparisons_vs_e5_small": comparisons, "complementarity_hit10": comp,
}
write_results("T1_T2_retrieval", payload, rows)
import csv
with open(CACHE_DIR.parent / "results" / "T1_T2_retrieval_comparisons.csv", "w", newline="", encoding="utf-8") as f:
    keys = sorted({k for c in comparisons for k in c})
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(comparisons)
print("complementarity:", comp)
