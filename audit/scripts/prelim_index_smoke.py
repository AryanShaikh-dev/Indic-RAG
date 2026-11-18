import sys, json, time, numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
qpath = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
t=time.perf_counter(); m = SentenceTransformer("intfloat/multilingual-e5-small", device=dev); print(f"device={dev} load={1000*(time.perf_counter()-t):.0f}ms")
c = QdrantClient(path=qpath)
for col in c.get_collections().collections:
    print(col.name, c.count(col.name).count)
pts,_ = c.scroll("hindi_rag_production", limit=20, with_vectors=True, with_payload=True)
re = m.encode(["passage: "+p.payload["chunk"] for p in pts], normalize_embeddings=True)
st = np.array([p.vector for p in pts]); st = st/np.linalg.norm(st,axis=1,keepdims=True)
print("stored-vs-reembedded cosine: min=%.5f mean=%.5f" % ((re*st).sum(1).min(), (re*st).sum(1).mean()))
print("payload keys:", sorted(pts[0].payload.keys()))
q = json.load(open("benchmark_artifacts/queries.json", encoding="utf-8"))
qt = list(q.values())[0]
h = c.query_points("hindi_rag_production", query=m.encode("query: "+qt).tolist(), limit=3)
print("top3 scores:", [round(x.score,4) for x in h.points])
