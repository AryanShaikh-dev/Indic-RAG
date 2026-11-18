import sys, json, numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
qpath=sys.argv[1]
q={int(k):v for k,v in json.load(open("benchmark_artifacts/queries.json",encoding="utf-8")).items()}
e={int(k):v for k,v in json.load(open("benchmark_artifacts/eval_map.json",encoding="utf-8")).items()}
gt={k:{x["passage_text"].strip() for x in v if x["is_selected"]==1} for k,v in e.items()}
m=SentenceTransformer("intfloat/multilingual-e5-small",device="cpu"); c=QdrantClient(path=qpath)
ids=[k for k in sorted(q) if gt[k]]
V=m.encode(["query: "+q[k] for k in ids],batch_size=64)
hit={1:[],5:[],10:[]}; rr=[]; top1=[]; top1_rel=[]
for k,v in zip(ids,V):
    pts=c.query_points("hindi_rag_production",query=v.tolist(),limit=10).points
    rel=[any(p.payload["chunk"].strip() in g for g in gt[k]) for p in pts]
    for K in hit: hit[K].append(any(rel[:K]))
    rr.append(next((1/(i+1) for i,r in enumerate(rel[:5]) if r),0)); top1.append(pts[0].score); top1_rel.append(rel[0])
n=len(ids); print("n=",n)
for K in hit:
    p=np.mean(hit[K]); b=[np.mean(np.random.default_rng(i).choice(hit[K],n)) for i in range(1000)]
    print(f"Recall@{K}={100*p:.2f}%  95%CI=[{100*np.percentile(b,2.5):.1f},{100*np.percentile(b,97.5):.1f}]")
print(f"MRR@5={100*np.mean(rr):.2f}%")
t=np.array(top1); print("top1 score pctl 1/5/50:",np.round(np.percentile(t,[1,5,50]),4))
for th in [0.80,0.826,0.85]: print(f"answerable queries refused at {th}: {100*np.mean(t<th):.1f}%")
# in-corpus queries (incl. no-label ones) count of top1 below threshold
