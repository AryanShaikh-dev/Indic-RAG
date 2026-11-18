"""T3 — Chunking strategy comparison that actually exercises chunking.

Stored comparison (max_chars=1000) touched only 19/9,943 passages. Here max_chars in {200, 400, 1000}
x chunker in {recursive (production), recursive_danda (audit-only variant: also splits on । ॥), sentence}.
Model fixed: e5-small (production). Corpus: DS1. Relevance is passage-level: a chunk is relevant if it
comes from a gold passage. Selection on DS1-dev, reported on DS1-test.
Usage: .venv-gpu/Scripts/python audit/scripts/t3_chunking.py
"""
import re
import numpy as np
import torch
from common import (load_ds1, ds1_split, chunk_corpus, encode, dense_topk, relevance_matrix, per_query_metrics,
                    summarize_metrics, mcnemar, load_bi_encoder, environment, write_results)
from src.core.chunking import RecursiveChunker, SentenceChunker


class RecursiveDandaChunker(RecursiveChunker):
    """Identical to production RecursiveChunker except the sentence regex includes the Devanagari danda."""
    SENT = re.compile(r'(?<=[.!?।॥]) +')

    def chunk(self, text):
        import src.core.chunking as m
        orig = m.re.split
        m.re.split = lambda pat, s, *a, **k: self.SENT.split(s) if pat == r'(?<=[.!?]) +' else orig(pat, s, *a, **k)
        try:
            return super().chunk(text)
        finally:
            m.re.split = orig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
END_RE = re.compile(r'[.!?।॥"\')\]]\s*$')
ds = load_ds1(); split = ds1_split()
model = load_bi_encoder("intfloat/multilingual-e5-small", DEVICE)
rows, configs = [], []
per_q = {}
for max_chars in (200, 400, 1000):
    overlap = max_chars // 5
    for cname, cls in (("recursive", RecursiveChunker), ("recursive_danda", RecursiveDandaChunker), ("sentence", SentenceChunker)):
        ch = cls(max_chars=max_chars, overlap_chars=overlap)
        chunks, owner = chunk_corpus(ds.passages, ch)
        last = np.r_[owner[1:] != owner[:-1], True]
        cut = np.array([not END_RE.search(c) for c in chunks]) & ~last
        emb = encode(model, chunks, "passage: ", batch_size=128)
        name = f"{cname}_{max_chars}"
        info = {"config": name, "chunker": cname, "max_chars": max_chars, "overlap_chars": overlap, "n_chunks": len(chunks),
                "avg_chunk_chars": round(float(np.mean([len(c) for c in chunks])), 1),
                "passages_split": int(np.sum(np.bincount(owner) > 1)),
                "mid_sentence_cut_pct": round(100 * float(cut.mean()), 2)}
        for sl in ("dev", "test"):
            qids = split[sl]
            idx, _ = dense_topk(encode(model, [ds.queries[q] for q in qids], "query: "), emb, 10, DEVICE)
            rel = relevance_matrix(idx, owner, qids, ds.gold)
            pq = per_query_metrics(rel, qids, ds.gold, ks=(1, 2, 3, 5, 10))
            per_q[(name, sl)] = pq
            info[f"avg_context_chars_top2_{sl}"] = round(float(np.mean([sum(len(chunks[c]) for c in r[:2]) for r in idx])), 1)
            rows.extend(summarize_metrics(name, "DS1", f"DS1-{sl}", pq, {"max_chars": max_chars, "chunker": cname}))
        configs.append(info)
        print(info, {k: round(100 * v.mean(), 2) for k, v in per_q[(name, 'test')].items() if k in ('Hit@2', 'Hit@5', 'MRR@10')})

comparisons = []
for sl in ("dev", "test"):
    base = per_q[("recursive_1000", sl)]
    for c in configs:
        if c["config"] == "recursive_1000":
            continue
        for k in ("Hit@2", "Hit@5"):
            comparisons.append({"split": f"DS1-{sl}", "baseline": "recursive_1000 (production)", "system": c["config"], "metric": k,
                                "baseline_pct": round(100 * base[k].mean(), 2), "system_pct": round(100 * per_q[(c['config'], sl)][k].mean(), 2),
                                **mcnemar(base[k], per_q[(c["config"], sl)][k])})
best_dev = max(configs, key=lambda c: per_q[(c["config"], "dev")]["Hit@2"].mean())["config"]
write_results("T3_chunking", {
    "test_id": "T3_chunking", "status": "NEWLY EXECUTED", "environment": environment(DEVICE),
    "design": "Model fixed (e5-small). Passage-level relevance. Overlap = max_chars/5 (production chunkers only apply overlap in the fixed-size fallback). "
              "Note: at equal k, smaller chunks give the LLM less context (see avg_context_chars_top2).",
    "configs": configs, "comparisons_vs_production": comparisons,
    "selected_on_dev_by_Hit@2": best_dev}, rows)
print("best on dev (Hit@2):", best_dev)
