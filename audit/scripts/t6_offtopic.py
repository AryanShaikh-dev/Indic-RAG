"""T6 — Off-topic guardrail evaluation against the production index (DS1 corpus).

Positives (should PASS) : DS1 labelled queries (answerable from the corpus by construction). dev / test halves.
Negatives (should BLOCK):
  natural_ooc : MS MARCO-XI *validation* queries whose gold passages are NOT in the DS1 corpus (out-of-corpus by
                construction; topical overlap means a few may still be answerable -> label noise, biases FAR upward).
                dev = DSV-dev, test = 600 random DSV-test queries.
  handwritten : audit/data/offtopic_queries.jsonl (60 queries, hi/en/hinglish), test only.
Signals:
  cos_top1      : production signal (e5-small top-1 cosine; production threshold 0.826)
  cos_gap       : top-1 minus mean(top-2..10)
  rerank_top1   : cross-encoder max score over e5 top-20 (mmarco-mMiniLMv2 and bge-reranker-v2-m3)
Threshold for each signal chosen on dev at FRR <= 5%, then FAR/FRR reported on test.
Usage: .venv-gpu/Scripts/python audit/scripts/t6_offtopic.py
"""
import json
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from common import (PROJECT_ROOT, load_ds1, ds1_split, load_dsv, dsv_split, chunk_corpus, encode, dense_topk,
                    load_bi_encoder, load_cross_encoder, bootstrap_ci, environment, write_results, CACHE_DIR)
from src.core.chunking import RecursiveChunker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROD_T = 0.826
ds1, dsv = load_ds1(), load_dsv()
s1, sv = ds1_split(), dsv_split(dsv)
chunks, owner = chunk_corpus(ds1.passages, RecursiveChunker())
cemb = np.load(CACHE_DIR / "emb_e5_small_DS1.npy")
ds1_set = set(ds1.passages)


def ooc(qids):
    return [q for q in qids if not any(dsv.passages[p] in ds1_set for p in dsv.gold[q])]


rng = np.random.default_rng(11)
neg_dev = ooc(sv["dev"])
neg_test = sorted(rng.choice(ooc(sv["test"]), size=600, replace=False).tolist())
hand = [json.loads(l) for l in open(PROJECT_ROOT / "audit/data/offtopic_queries.jsonl", encoding="utf-8")]
sets = {
    "pos_dev": [ds1.queries[q] for q in s1["dev"]], "pos_test": [ds1.queries[q] for q in s1["test"]],
    "neg_natural_dev": [dsv.queries[q] for q in neg_dev], "neg_natural_test": [dsv.queries[q] for q in neg_test],
    "neg_handwritten": [h["text"] for h in hand],
}
e5 = load_bi_encoder("intfloat/multilingual-e5-small", DEVICE)
sig, top20 = {}, {}
for name, qs in sets.items():
    idx, sc = dense_topk(encode(e5, qs, "query: "), cemb, 20, DEVICE)
    top20[name] = idx
    sig[name] = {"cos_top1": sc[:, 0], "cos_gap": sc[:, 0] - sc[:, 1:10].mean(1)}
for rname, hf in (("rerank_mmarco_top1", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),):
    ce = load_cross_encoder(hf, DEVICE)
    for name, qs in sets.items():
        pairs = [(q, chunks[c]) for q, row in zip(qs, top20[name]) for c in row]
        s = np.asarray(ce.predict(pairs, batch_size=32, show_progress_bar=False)).reshape(len(qs), 20)
        sig[name][rname] = s.max(1)
    del ce
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def auroc(pos, neg):
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


results, rows = {}, []
for s in ["cos_top1", "cos_gap", "rerank_mmarco_top1"]:
    pd_, pt = sig["pos_dev"][s], sig["pos_test"][s]
    t_dev = float(np.percentile(pd_, 5))      # FRR <= 5% on dev
    r = {"threshold_selected_on_dev_FRR5": round(t_dev, 4)}
    for neg in ["neg_natural_test", "neg_handwritten"]:
        nt = sig[neg][s]
        r[f"AUROC_pos_test_vs_{neg}"] = round(auroc(pt, nt), 4)
        far = (nt >= t_dev).astype(float)
        r[f"FAR_pct_{neg}@dev_threshold"] = round(100 * far.mean(), 1)
        r[f"FAR_ci95_{neg}@dev_threshold"] = [round(100 * x, 1) for x in bootstrap_ci(far)]
    frr = (pt < t_dev).astype(float)
    r["FRR_pct_pos_test@dev_threshold"] = round(100 * frr.mean(), 1)
    r["FRR_ci95_pos_test@dev_threshold"] = [round(100 * x, 1) for x in bootstrap_ci(frr)]
    if s == "cos_top1":
        r["production_threshold"] = PROD_T
        r["FRR_pct_pos_test@0.826"] = round(100 * float((pt < PROD_T).mean()), 1)
        for neg in ["neg_natural_test", "neg_handwritten"]:
            far = (sig[neg][s] >= PROD_T).astype(float)
            r[f"FAR_pct_{neg}@0.826"] = round(100 * far.mean(), 1)
            r[f"FAR_ci95_{neg}@0.826"] = [round(100 * x, 1) for x in bootstrap_ci(far)]
        r["handwritten_FAR@0.826_by_lang"] = {lang: round(100 * float(np.mean([v >= PROD_T for v, h in zip(sig["neg_handwritten"][s], hand) if h["lang"] == lang])), 1)
                                             for lang in ("hi", "en", "hinglish")}
    r["score_distribution"] = {k: {"p5": round(float(np.percentile(v[s], 5)), 4), "p50": round(float(np.median(v[s])), 4),
                                   "p95": round(float(np.percentile(v[s], 95)), 4)} for k, v in sig.items()}
    results[s] = r
    rows.append({"signal": s, **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}})
    print(s, {k: v for k, v in r.items() if not isinstance(v, (dict, list))})

write_results("T6_offtopic", {
    "test_id": "T6_offtopic", "status": "NEWLY EXECUTED", "environment": environment(DEVICE),
    "index": "DS1 corpus (identical content to production collection hindi_rag_production), exact cosine",
    "sets": {k: len(v) for k, v in sets.items()},
    "caveats": ["natural_ooc negatives are out-of-corpus by construction but may be topically answerable (label noise inflates FAR).",
                "handwritten negatives authored by the auditor (an AI); n=60.",
                "Positives are in-distribution MS MARCO-XI queries; real user queries may score lower."],
    "results": results}, rows)
