"""
audit/scripts/common.py
=======================
Shared utilities for the Indic-RAG audit. Read-only with respect to the
project: imports production code from src/ but never modifies it.

Datasets
  DS1      : existing benchmark artifacts (1,000 MS MARCO-XI *train* queries, seed 42)
  DS1-dev  : 50% of DS1's 627 labelled queries  (tuning / selection only)
  DS1-test : other 50%                            (reporting)
  DSV      : fresh sample from MS MARCO-XI *validation* split (disjoint query set)
             with its own corpus built by the production chunker  (held-out reporting)
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

AUDIT_DIR = PROJECT_ROOT / "audit"
RESULTS_DIR = AUDIT_DIR / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "benchmark_artifacts"
CACHE_DIR = AUDIT_DIR / "cache"            # embeddings / intermediate files (gitignored by audit/.gitignore)
HF_CACHE_C = Path.home() / ".cache" / "huggingface" / "hub"   # pre-existing cache (reused, not duplicated)
HF_CACHE_D = Path("D:/hf-cache-audit")                         # new downloads
MSMARCO_REV = "bf5cdc1f26e581e519018e434db14edd1b77602b"

SPLIT_SEED = 2026
DSV_SEED = 7
DSV_N_QUERIES = 5000

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #
def environment(device: str | None = None) -> dict:
    from importlib.metadata import version, PackageNotFoundError

    def v(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None

    try:
        commit = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    env = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": "AMD Ryzen 7 7445HS (6C/12T)",
        "ram_gb": 15.3,
        "packages": {p: v(p) for p in ["torch", "sentence-transformers", "transformers", "qdrant-client", "groq", "numpy", "scipy", "jiwer", "flask"]},
        "device": device,
    }
    try:
        import torch
        env["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


# --------------------------------------------------------------------------- #
# Model loading without duplicating downloads
# --------------------------------------------------------------------------- #
def _cache_for(model_id: str) -> str:
    folder = "models--" + model_id.replace("/", "--")
    if (HF_CACHE_C / folder).exists():
        return str(HF_CACHE_C)
    return str(HF_CACHE_D)


def load_bi_encoder(model_id: str, device: str, max_seq_length: int | None = None):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device=device, cache_folder=_cache_for(model_id))
    if max_seq_length:
        m.max_seq_length = max_seq_length
    return m


def load_cross_encoder(model_id: str, device: str, max_length: int = 512):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_id, device=device, max_length=max_length, cache_folder=_cache_for(model_id))


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class RetrievalDataset:
    """passages: list[str] (unique, stripped); queries: {qid: text}; gold: {qid: set(passage_idx)}"""

    def __init__(self, name: str, passages: List[str], queries: Dict[int, str], gold: Dict[int, Set[int]],
                 eng_queries: Dict[int, str] | None = None, answers: Dict[int, str] | None = None):
        self.name = name
        self.passages = passages
        self.queries = queries
        self.gold = gold
        self.eng_queries = eng_queries or {}
        self.answers = answers or {}

    @property
    def labelled_ids(self) -> List[int]:
        return sorted(q for q in self.queries if self.gold.get(q))


def _build(name, rows) -> RetrievalDataset:
    """rows: iterable of (qid, query, [(passage_text, is_selected), ...])"""
    pid: Dict[str, int] = {}
    passages: List[str] = []
    queries, gold = {}, {}
    for qid, q, plist in rows:
        queries[qid] = q
        g = set()
        for text, sel in plist:
            text = (text or "").strip()
            if not text:
                continue
            if text not in pid:
                pid[text] = len(passages)
                passages.append(text)
            if int(sel) == 1:
                g.add(pid[text])
        gold[qid] = g
    return RetrievalDataset(name, passages, queries, gold)


def load_ds1() -> RetrievalDataset:
    q = {int(k): v for k, v in json.load(open(ARTIFACTS_DIR / "queries.json", encoding="utf-8")).items()}
    e = {int(k): v for k, v in json.load(open(ARTIFACTS_DIR / "eval_map.json", encoding="utf-8")).items()}
    rows = ((qid, q[qid], [(x["passage_text"], x["is_selected"]) for x in e[qid]]) for qid in e)
    return _build("DS1", rows)


def ds1_split() -> Dict[str, List[int]]:
    ds = load_ds1()
    ids = np.array(ds.labelled_ids)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(ids))
    half = len(ids) // 2
    return {"dev": sorted(ids[perm[:half]].tolist()), "test": sorted(ids[perm[half:]].tolist())}


def hinval_path() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("ai4bharat/MSMARCO-XI", "validation/hinval.parquet", repo_type="dataset",
                                revision=MSMARCO_REV, cache_dir=str(HF_CACHE_D), local_files_only=True))


def load_dsv(n: int = DSV_N_QUERIES, seed: int = DSV_SEED) -> RetrievalDataset:
    cache = CACHE_DIR / f"dsv_n{n}_s{seed}.json"
    if cache.exists():
        d = json.load(open(cache, encoding="utf-8"))
        return RetrievalDataset("DSV", d["passages"], {int(k): v for k, v in d["queries"].items()},
                                {int(k): set(v) for k, v in d["gold"].items()},
                                {int(k): v for k, v in d["eng_queries"].items()},
                                {int(k): v for k, v in d["answers"].items()})
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(hinval_path())
    recs = []
    for batch in pf.iter_batches(columns=["query_id", "query", "passages", "Eng_Query", "Answer"]):
        for r in batch.to_pylist():
            recs.append(r)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(recs), size=min(n, len(recs)), replace=False)
    rows, eng, ans = [], {}, {}
    for i in sorted(idx):
        r = recs[i]
        qid = int(r["query_id"])
        rows.append((qid, r["query"], list(zip(r["passages"]["Translated_passages"], r["passages"]["is_selected"]))))
        eng[qid] = r["Eng_Query"]
        ans[qid] = r["Answer"]
    ds = _build("DSV", rows)
    ds.eng_queries, ds.answers = eng, ans
    json.dump({"passages": ds.passages, "queries": ds.queries, "gold": {k: sorted(v) for k, v in ds.gold.items()},
               "eng_queries": eng, "answers": ans,
               "source": "ai4bharat/MSMARCO-XI validation/hinval.parquet", "revision": MSMARCO_REV,
               "n_total_rows": len(recs), "seed": seed}, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return ds


def dsv_split(ds: "RetrievalDataset") -> Dict[str, List[int]]:
    """DSV-dev (20%) for LLM/generation diagnostics; DSV-test (80%) untouched held-out retrieval reporting."""
    ids = np.array(ds.labelled_ids)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(ids))
    cut = len(ids) // 5
    return {"dev": sorted(ids[perm[:cut]].tolist()), "test": sorted(ids[perm[cut:]].tolist())}


# --------------------------------------------------------------------------- #
# Chunking (production chunker) -> chunk-level corpus with passage mapping
# --------------------------------------------------------------------------- #
def chunk_corpus(passages: Sequence[str], chunker) -> tuple[list[str], np.ndarray]:
    chunks, owner = [], []
    for i, p in enumerate(passages):
        for c in chunker.chunk(p):
            chunks.append(c)
            owner.append(i)
    return chunks, np.array(owner)


# --------------------------------------------------------------------------- #
# Text normalisation / BM25
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[\u0900-\u097F\w]+")


def hi_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


class BM25:
    """Okapi BM25 (k1=1.2, b=0.75) on a scipy sparse matrix — exact, vectorised."""

    def __init__(self, docs: Sequence[str], k1: float = 1.2, b: float = 0.75):
        from scipy import sparse
        vocab: Dict[str, int] = {}
        rows, cols, vals = [], [], []
        dl = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            toks = hi_tokenize(d)
            dl[i] = len(toks)
            tf: Dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                tf[j] = tf.get(j, 0) + 1
            for j, c in tf.items():
                rows.append(i); cols.append(j); vals.append(c)
        tf_m = sparse.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(docs), len(vocab)))
        df = np.bincount(np.array(cols), minlength=len(vocab)).astype(np.float32)
        n = len(docs)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = dl.mean()
        tf_m = tf_m.tocoo()
        denom = tf_m.data + k1 * (1 - b + b * dl[tf_m.row] / avgdl)
        w = tf_m.data * (k1 + 1) / denom
        self.W = sparse.csc_matrix((w.astype(np.float32), (tf_m.row, tf_m.col)), shape=(n, len(vocab)))
        self.vocab = vocab

    def search(self, query: str, k: int):
        ids = [self.vocab[t] for t in set(hi_tokenize(query)) if t in self.vocab]
        if not ids:
            return np.arange(k), np.zeros(k, dtype=np.float32)
        scores = np.asarray(self.W[:, ids] @ self.idf[ids]).ravel()
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]


# --------------------------------------------------------------------------- #
# Dense exact search (equivalent to Qdrant local-mode exact cosine search)
# --------------------------------------------------------------------------- #
def encode(model, texts: Sequence[str], prefix: str = "", batch_size: int = 128) -> np.ndarray:
    return model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def dense_topk(q_emb: np.ndarray, c_emb: np.ndarray, k: int, device: str = "cpu"):
    import torch
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    C = torch.from_numpy(c_emb).to(dev)
    I_all, S_all = [], []
    for s in range(0, len(q_emb), 512):
        Q = torch.from_numpy(q_emb[s:s + 512]).to(dev)
        S = Q @ C.T
        v, i = torch.topk(S, k, dim=1)
        I_all.append(i.cpu().numpy()); S_all.append(v.cpu().numpy())
    return np.vstack(I_all), np.vstack(S_all)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def relevance_matrix(ranked_chunk_ids: np.ndarray, chunk_owner: np.ndarray, qids: Sequence[int],
                     gold: Dict[int, Set[int]]) -> np.ndarray:
    """bool [n_queries, depth]: chunk at rank r belongs to a gold passage."""
    rel = np.zeros(ranked_chunk_ids.shape, dtype=bool)
    for i, q in enumerate(qids):
        g = gold[q]
        rel[i] = [chunk_owner[c] in g for c in ranked_chunk_ids[i]]
    return rel


def per_query_metrics(rel: np.ndarray, qids: Sequence[int], gold: Dict[int, Set[int]],
                      ranked_owner: np.ndarray | None = None, ks=(1, 2, 3, 5, 10, 20)) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    depth = rel.shape[1]
    for k in ks:
        if k <= depth:
            out[f"Hit@{k}"] = rel[:, :k].any(1).astype(float)
    # true Recall@k: distinct gold passages found / gold passages
    if ranked_owner is not None:
        for k in (5, 10):
            if k <= depth:
                out[f"Recall@{k}"] = np.array([len(set(ranked_owner[i, :k][rel[i, :k]])) / len(gold[q]) for i, q in enumerate(qids)])
    first = np.where(rel.any(1), rel.argmax(1) + 1, 0)
    for k in (5, 10):
        if k <= depth:
            out[f"MRR@{k}"] = np.where((first > 0) & (first <= k), 1.0 / np.maximum(first, 1), 0.0)
    if depth >= 10:
        disc = 1.0 / np.log2(np.arange(2, 12))
        dcg = (rel[:, :10] * disc).sum(1)
        ideal = np.array([disc[:min(len(gold[q]), 10)].sum() for q in qids])
        out["nDCG@10"] = dcg / ideal
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import binomtest
    a = a.astype(bool); b = b.astype(bool)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"b_wins": n01, "a_wins": n10, "p_value": float(p)}


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return {"mean_diff": float(d.mean()), "ci_low": float(np.percentile(m, 2.5)), "ci_high": float(np.percentile(m, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))}


def latency_summary(x: Sequence[float]) -> dict:
    a = np.asarray(x, dtype=float)
    n = len(a)
    s = {"n": n, "mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
         "std": round(float(a.std(ddof=1)) if n > 1 else 0.0, 3), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}
    s["p95"] = round(float(np.percentile(a, 95)), 3) if n >= 100 else None   # rule: P95 only with n>=100
    s["p99"] = round(float(np.percentile(a, 99)), 3) if n >= 1000 else None  # rule: P99 only with n>=1000
    return s


# --------------------------------------------------------------------------- #
# Results I/O
# --------------------------------------------------------------------------- #
def write_results(test_id: str, payload: dict, rows: List[dict] | None = None) -> None:
    jp = RESULTS_DIR / f"{test_id}.json"
    json.dump(payload, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    if rows:
        keys: List[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(RESULTS_DIR / f"{test_id}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"[SAVE] {jp}")


def summarize_metrics(system: str, dataset: str, split: str, per_q: Dict[str, np.ndarray], extra: dict | None = None) -> List[dict]:
    rows = []
    for m, arr in per_q.items():
        lo, hi = bootstrap_ci(arr)
        r = {"system": system, "dataset": dataset, "split": split, "n": len(arr), "metric": m,
             "value_pct": round(100 * float(arr.mean()), 2), "ci95_low_pct": round(100 * lo, 2), "ci95_high_pct": round(100 * hi, 2)}
        if extra:
            r.update(extra)
        rows.append(r)
    return rows
