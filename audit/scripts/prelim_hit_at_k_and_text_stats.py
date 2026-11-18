import sys, json, unicodedata, numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
q={int(k):v for k,v in json.load(open("benchmark_artifacts/queries.json",encoding="utf-8")).items()}
e={int(k):v for k,v in json.load(open("benchmark_artifacts/eval_map.json",encoding="utf-8")).items()}
gt={k:{x["passage_text"].strip() for x in v if x["is_selected"]==1} for k,v in e.items()}
P={x["passage_text"].strip() for v in e.values() for x in v}
print("passages not NFC:", sum(unicodedata.normalize("NFC",p)!=p for p in P), "/", len(P), "| queries not NFC:", sum(unicodedata.normalize("NFC",t)!=t for t in q.values()))
print("passages with ZWJ/ZWNJ:", sum(("\u200d" in p) or ("\u200c" in p) for p in P))
m=SentenceTransformer("intfloat/multilingual-e5-small",device="cpu"); c=QdrantClient(path=sys.argv[1])
ids=[k for k in sorted(q) if gt[k]]; V=m.encode(["query: "+q[k] for k in ids],batch_size=64)
H={2:[],3:[],20:[]}
for k,v in zip(ids,V):
    pts=c.query_points("hindi_rag_production",query=v.tolist(),limit=20).points
    rel=[any(p.payload["chunk"].strip() in g for g in gt[k]) for p in pts]
    for K in H: H[K].append(any(rel[:K]))
for K in H: print(f"Hit@{K}={100*np.mean(H[K]):.2f}%")
tok=m.tokenizer; L=[len(tok(t)["input_ids"]) for t in list(P)[:3000]]
print("e5 token length of passages (3000 sample): p50=%d p95=%d max=%d, >512: %d" % (np.percentile(L,50),np.percentile(L,95),max(L),sum(l>512 for l in L)))
