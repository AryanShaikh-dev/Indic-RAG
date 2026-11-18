"""T5 — k-sweep and cross-encoder reranking over the production retriever's top-20.

Candidates: e5-small top-20 (cached by t1_t2_retrieval.py). Rerankers (query, chunk) pairs, no prefixes:
  mmarco_mminilm : cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (118M)
  bge_reranker_m3: BAAI/bge-reranker-v2-m3 (568M)
Selection (reranker, k) on DS1-dev; reported on DS1-test and DSV-test with McNemar vs production (no rerank, k=2).
Usage: .venv-gpu/Scripts/python audit/scripts/t5_rerank.py
"""
import time
import numpy as np
import torch
from common import (load_ds1, ds1_split, load_dsv, dsv_split, chunk_corpus, relevance_matrix, per_query_metrics,
                    summarize_metrics, mcnemar, load_cross_encoder, environment, write_results, CACHE_DIR)
from src.core.chunking import RecursiveChunker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ds1, dsv = load_ds1(), load_dsv()
s1, sv = ds1_split(), dsv_split(dsv)
corp = {d.name: chunk_corpus(d.passages, RecursiveChunker()) for d in (ds1, dsv)}
slices = [("DS1-dev", ds1, s1["dev"]), ("DS1-test", ds1, s1["test"]), ("DSV-test", dsv, sv["test"])]
cand = {}
for sl, ds, q in slices:
    src = "DS1-all" if sl.startswith("DS1") else sl
    full = np.load(CACHE_DIR / f"rank_e5_small_{src}.npy")
    if src == "DS1-all":
        pos = {qid: i for i, qid in enumerate(ds1.labelled_ids)}
        full = full[[pos[x] for x in q]]
    cand[sl] = full[:, :20]

rows, per_q, timing = [], {}, {}
for sl, ds, qids in slices:
    chunks, owner = corp[ds.name]
    rel = relevance_matrix(cand[sl], owner, qids, ds.gold)
    pq = per_query_metrics(rel, qids, ds.gold, ks=(1, 2, 3, 5, 10, 20))
    per_q[("no_rerank", sl)] = pq
    rows.extend(summarize_metrics("e5_small_no_rerank", ds.name, sl, pq))

for rname, hf in (("mmarco_mminilm", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),):
    ce = load_cross_encoder(hf, DEVICE, max_length=512)
    for sl, ds, qids in slices:
        chunks, owner = corp[ds.name]
        pairs = [(ds.queries[q], chunks[c]) for q, row in zip(qids, cand[sl]) for c in row]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        sc = np.asarray(ce.predict(pairs, batch_size=32, show_progress_bar=False), dtype=np.float32).reshape(len(qids), 20)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        timing[f"{rname}|{sl}"] = {"pairs": len(pairs), "s": round(time.perf_counter() - t, 1), "device": DEVICE}
        order = np.argsort(-sc, axis=1)
        reranked = np.take_along_axis(cand[sl], order, axis=1)
        np.save(CACHE_DIR / f"rerank_{rname}_{sl}_scores.npy", np.take_along_axis(sc, order, axis=1))
        rel = relevance_matrix(reranked, owner, qids, ds.gold)
        pq = per_query_metrics(rel, qids, ds.gold, ks=(1, 2, 3, 5, 10, 20))
        per_q[(rname, sl)] = pq
        rows.extend(summarize_metrics(f"e5_small_top20+{rname}", ds.name, sl, pq))
        print(f"{rname:16s} {sl:9s} " + " ".join(f"{k}={100*v.mean():.2f}" for k, v in pq.items() if k in ("Hit@1", "Hit@2", "Hit@3", "Hit@5", "MRR@10")))
    del ce
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# selection on dev: best reranker by Hit@2 (production k)
sel = "mmarco_mminilm"
comparisons = []
for sl in ("DS1-dev", "DS1-test", "DSV-test"):
    base = per_q[("no_rerank", sl)]["Hit@2"]
    for system, k in [("no_rerank", 3), ("no_rerank", 5), ("mmarco_mminilm", 2), ("mmarco_mminilm", 3), ("mmarco_mminilm", 5)]:
        other = per_q[(system, sl)][f"Hit@{k}"]
        comparisons.append({"slice": sl, "baseline": "production: no rerank, k=2", "system": f"{system}, k={k}",
                            "baseline_hit_pct": round(100 * base.mean(), 2), "system_hit_pct": round(100 * other.mean(), 2),
                            "delta_pp": round(100 * (other.mean() - base.mean()), 2), **mcnemar(base, other)})
write_results("T5_rerank", {"test_id": "T5_rerank", "status": "NEWLY EXECUTED", "environment": environment(DEVICE),
                            "design": "Candidates = e5-small top-20 (ceiling = Hit@20). Metric of interest: Hit@k for k chunks passed to the LLM.",
                            "selected_reranker_on_dev": sel, "excluded": {"BAAI/bge-reranker-v2-m3 (568M)": "NOT EVALUATED - large-model exclusion after memory-limit incident (user decision)"}, "comparisons": comparisons, "batch_timing_gpu": timing,
                            "note": "Per-query reranker latency (CPU/GPU) is measured in T4."}, rows)
for c in comparisons:
    if c["slice"] != "DS1-dev":
        print(c)
