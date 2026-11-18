"""T4 — Local latency & resource breakdown (no network).

IMPORTANT: Qdrant here is LOCAL EMBEDDED MODE (QdrantClient(path=...), exact search in Python, no HNSW, no network).
It is NOT representative of Qdrant Cloud (network RTT + server-side HNSW). Cloud was unavailable and is not measured.

Usage:
  .venv/Scripts/python     audit/scripts/t4_latency.py <qdrant_copy_dir>      # CPU (deployment-like)
  .venv-gpu/Scripts/python audit/scripts/t4_latency.py <qdrant_copy_dir>      # GPU (RTX 3050 laptop)
"""
import json, os, subprocess, sys, time
import numpy as np
import psutil, torch
from common import (PROJECT_ROOT, load_dsv, load_ds1, chunk_corpus, encode, load_bi_encoder, load_cross_encoder,
                    latency_summary, environment, write_results, CACHE_DIR)
from src.core.chunking import RecursiveChunker

QPATH = sys.argv[1]
os.environ.pop("QDRANT_URL", None); os.environ.pop("QDRANT_API", None)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
proc = psutil.Process()
rss = lambda: round(proc.memory_info().rss / 2**20, 1)
res = {"device": DEV, "torch_threads": torch.get_num_threads(), "rss_mb_start": rss()}
dsv, ds1 = load_dsv(), load_ds1()
qs = [dsv.queries[q] for q in np.random.default_rng(4).choice(sorted(dsv.queries), 1000, replace=False)]


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def timeit(fn, items, warm=10, reps=1):
    for x in items[:warm]:
        fn(x)
    out = []
    for _ in range(reps):
        for x in items:
            t = time.perf_counter(); fn(x); sync(); out.append((time.perf_counter() - t) * 1000)
    return out


# 1. cold start in fresh subprocesses (production EmbeddingModel + QdrantVectorStore init)
cold = []
code = ("import time,os,sys; t0=time.perf_counter(); sys.path.insert(0,r'%s'); os.environ.pop('QDRANT_URL',None);"
        "from src.core.embeddings import EmbeddingModel; t1=time.perf_counter(); m=EmbeddingModel('intfloat/multilingual-e5-small'); t2=time.perf_counter();"
        "m.embed(['query: test']); t3=time.perf_counter();"
        "from src.core.vectorstore import QdrantVectorStore; t4=time.perf_counter(); db=QdrantVectorStore(collection_name='hindi_rag_production', persist_directory=r'%s', vector_dim=384); t5=time.perf_counter();"
        "print(t1-t0,t2-t1,t3-t2,t5-t4); db.client.close()") % (PROJECT_ROOT, QPATH)
for _ in range(3):
    o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(PROJECT_ROOT)).stdout.strip().splitlines()[-1]
    a = [float(x) * 1000 for x in o.split()]
    cold.append({"import_ms": a[0], "model_load_ms": a[1], "first_embed_ms": a[2], "qdrant_vectorstore_init_ms_incl_2nd_model_load": a[3]})
res["cold_start_3_runs"] = cold

# 2. warm single-query embedding
e5 = load_bi_encoder("intfloat/multilingual-e5-small", DEV)
res["rss_mb_after_e5_small"] = rss()
res["embed_query_ms"] = {"e5_small": latency_summary(timeit(lambda q: e5.encode("query: " + q), qs, reps=3))}
for name, hf, pre, n in ():
    m = load_bi_encoder(hf, DEV, 512)
    res["embed_query_ms"][name] = latency_summary(timeit(lambda q: m.encode(pre + q), qs[:n]))
    res[f"rss_mb_after_{name}"] = rss()
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()

# 3. vector search only (precomputed vectors)
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
qv = encode(e5, qs, "query: ")
client = QdrantClient(path=QPATH)
res["search_ms"] = {"qdrant_local_10k_production_collection": latency_summary(
    timeit(lambda v: client.query_points("hindi_rag_production", query=v.tolist(), limit=2), list(qv), reps=3))}
c10 = np.load(CACHE_DIR / "emb_e5_small_DS1.npy")
res["search_ms"]["numpy_exact_10k"] = latency_summary(timeit(lambda v: np.argpartition(-(c10 @ v), 2)[:2], list(qv), reps=3))
c50 = None
if False:
  c50 = np.load(CACHE_DIR / "emb_e5_small_DSV.npy")
  p50 = os.path.join(os.path.dirname(QPATH), f"qdrant_t4_50k_{DEV}")
  c2 = QdrantClient(path=p50)
  if not c2.collection_exists("dsv"):
    c2.create_collection("dsv", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    for s in range(0, len(c50), 1000):
        c2.upsert("dsv", [PointStruct(id=i, vector=c50[i].tolist()) for i in range(s, min(s + 1000, len(c50)))])
  res["search_ms"]["qdrant_local_50k"] = latency_summary(timeit(lambda v: c2.query_points("dsv", query=v.tolist(), limit=2), list(qv)))
  res["search_ms"]["numpy_exact_50k"] = latency_summary(timeit(lambda v: np.argpartition(-(c50 @ v), 2)[:2], list(qv)))
  c2.close()
client.close()

# 4. production retrieval call: QdrantVectorStore.query (embed + search, as logged under 'retrieval')
from src.core.vectorstore import QdrantVectorStore
from src.core.embeddings import EmbeddingModel
emb = EmbeddingModel("intfloat/multilingual-e5-small")
db = QdrantVectorStore(collection_name="hindi_rag_production", persist_directory=QPATH, vector_dim=384, query_prefix="query: ", passage_prefix="passage: ")
db.embedding_model = emb
res["production_vectorstore_query_ms"] = latency_summary(timeit(lambda q: db.query(q, k=2, with_scores=True), qs, reps=1))

# 5. groundedness check (production function), answers ~ passage-length text
from src.core.guardrails import check_groundedness
ch1, _ = chunk_corpus(ds1.passages, RecursiveChunker())
trip = [(ch1[i], [ch1[i + 1], ch1[i + 2]]) for i in range(0, 900, 3)]
res["production_groundedness_check_ms"] = latency_summary(timeit(lambda t: check_groundedness(t[0], t[1], emb), trip))
db.client.close()

# 6. reranking 20 candidates per query
cands = [[ch1[j] for j in range(i, i + 20)] for i in range(0, 6000, 20)]
for name, hf, n in ():
    ce = load_cross_encoder(hf, DEV)
    items = list(zip(qs[:n], cands[:n]))
    res.setdefault("rerank_20_candidates_ms", {})[name] = latency_summary(timeit(lambda t: ce.predict([(t[0], c) for c in t[1]], batch_size=32), items, warm=5))
    res[f"rss_mb_after_{name}"] = rss()
    del ce
res["rss_mb_end"] = rss()
if DEV == "cuda":
    res["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
write_results(f"T4_latency_{DEV}", {"test_id": f"T4_latency_{DEV}", "status": "NEWLY EXECUTED", "environment": environment(DEV),
                                    "qdrant_mode": "LOCAL EMBEDDED (path=...), exact search, no network — NOT Qdrant Cloud",
                                    "query_set": "1,000 random MS MARCO-XI validation Hindi queries", "results": res})
print(json.dumps(res, indent=1))
