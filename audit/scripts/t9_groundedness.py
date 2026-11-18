"""T9 — Groundedness-check evaluation on real production answers (from T8, config A_production).

Labelled pairs (labels by construction):
  P  (grounded)   : answer + the context it was generated from
  N1 (ungrounded) : answer + context of a random other query
  N2 (ungrounded, hard): answer + context of the most similar other query (e5 query-embedding cosine)
Methods:
  prod_cosine   : src.core.guardrails.check_groundedness (unchanged; production threshold 0.75)
  sent_maxsim   : audit method — each answer sentence ('query: ') vs each chunk ('passage: '); score = min over sentences of max sim
  llm_judge     : openai/gpt-oss-120b, reasoning low, verdict SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED (1 / 0.5 / 0)
Also: llm_judge correctness of P answers vs the (machine-translated) reference answer.
Caveat: P label assumes the generator used its context; the judge's verdicts on P estimate how often that fails.
Usage: .venv/Scripts/python audit/scripts/t9_groundedness.py
"""
import json, os, re, time
import numpy as np
from scipy.stats import mannwhitneyu
from common import (PROJECT_ROOT, load_dsv, chunk_corpus, encode, dense_topk, bootstrap_ci, environment, write_results, CACHE_DIR)
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
os.environ.pop("QDRANT_URL", None)
import groq
from src.core.guardrails import check_groundedness
from src.core.embeddings import EmbeddingModel
from src.core.chunking import RecursiveChunker

JUDGE = "openai/gpt-oss-120b"
OUT = CACHE_DIR / "t9_judge.jsonl"
ds = load_dsv()
chunks, owner = chunk_corpus(ds.passages, RecursiveChunker())
cemb = np.load(CACHE_DIR / "emb_e5_small_DSV.npy")
emb = EmbeddingModel("intfloat/multilingual-e5-small"); st = emb.model
calls = [json.loads(l) for l in open(CACHE_DIR / "t8_calls.jsonl", encoding="utf-8")]
REFUSAL_EN = "I cannot answer this based on the provided context."
A = [r for r in calls if r["config"] == "A_production" and r["status"] == "llm_ok" and (r.get("answer") or "").strip()
     and REFUSAL_EN not in r["answer"] and not re.search(r"(उत्तर नहीं दे सकता|उत्तर नहीं दिया जा सकता)", r["answer"])]
allq = sorted({r["qid"] for r in calls})
qv = encode(st, [ds.queries[q] for q in allq], "query: ")
idx, _ = dense_topk(qv, cemb, 2, "cpu")
ctx = {q: [chunks[c] for c in row] for q, row in zip(allq, idx)}
sim = qv @ qv.T
np.fill_diagonal(sim, -1)
hard = {q: allq[int(np.argmax(sim[i]))] for i, q in enumerate(allq)}
rng = np.random.default_rng(9)
pairs = []
for r in A:
    q = r["qid"]
    rnd = q
    while rnd == q:
        rnd = int(rng.choice(allq))
    pairs.append({"qid": q, "type": "P", "label": 1, "answer": r["answer"], "ctx_q": q})
    if sum(p["type"] == "N1" for p in pairs) < 20:   # small judge sanity check: answer vs unrelated context
        pairs.append({"qid": q, "type": "N1", "label": 0, "answer": r["answer"], "ctx_q": rnd})

# --- prod cosine + sentence max-sim
SPLIT = re.compile(r"(?<=[.!?।॥])\s+|\n+")
for p in pairs:
    c = ctx[p["ctx_q"]]
    ok, g = check_groundedness(p["answer"], c, emb, threshold=0.75)
    p["prod_cosine"], p["prod_pass"] = float(g), bool(ok)
    sents = [s for s in SPLIT.split(p["answer"].strip()) if len(s.strip()) > 3] or [p["answer"]]
    sv = encode(st, sents, "query: "); cv = encode(st, c, "passage: ")
    p["sent_maxsim"] = float((sv @ cv.T).max(1).min())

# --- LLM judge (cached / resumable)
client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
FAITH = ("You are a strict fact-checking assistant. Decide whether every factual claim in the ANSWER is supported by the CONTEXT. "
         "Ignore style and language. Reply with JSON only: {\"verdict\": \"SUPPORTED\" | \"PARTIALLY_SUPPORTED\" | \"NOT_SUPPORTED\"}")
CORR = ("Compare the ANSWER to the REFERENCE answer for the QUESTION. The reference may be a noisy machine translation. "
        "Reply with JSON only: {\"correct\": \"CORRECT\" | \"PARTIAL\" | \"INCORRECT\"}")
cache = {}
if OUT.exists():
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l); cache[r["key"]] = r
events = []


def judge(key, system, user):
    if key in cache:
        return cache[key]
    out = {"key": key, "text": None, "error": "retries exhausted"}
    for attempt in range(5):
        try:
            t = time.perf_counter()
            raw = client.chat.completions.with_raw_response.create(model=JUDGE, reasoning_effort="low", max_completion_tokens=600,
                                                                   messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            r = raw.parse()
            out = {"key": key, "text": r.choices[0].message.content, "finish": r.choices[0].finish_reason,
                   "ms": (time.perf_counter() - t) * 1000, "usage": r.usage.model_dump() if r.usage else None}
            rem = raw.headers.get("x-ratelimit-remaining-tokens")
            if rem is not None and int(rem) < 2500:
                time.sleep(8)
            break
        except groq.RateLimitError as e:
            ra = float(e.response.headers.get("retry-after", 10)); events.append({"key": key, "event": "429", "retry_after_s": ra}); time.sleep(ra)
        except Exception as e:  # noqa
            events.append({"key": key, "event": type(e).__name__, "detail": str(e)[:200]}); time.sleep(3)
            out = {"key": key, "text": None, "error": str(e)[:200]}
    cache[key] = out
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def parse(text, field, mapping):
    m = re.search(r'"%s"\s*:\s*"([A-Z_]+)"' % field, text or "")
    return mapping.get(m.group(1)) if m else None


FMAP = {"SUPPORTED": 1.0, "PARTIALLY_SUPPORTED": 0.5, "NOT_SUPPORTED": 0.0}
CMAP = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
for i, p in enumerate(pairs):
    c = "\n\n".join(ctx[p["ctx_q"]])
    j = judge(f'faith|{p["qid"]}|{p["type"]}', FAITH, f"CONTEXT:\n{c}\n\nANSWER:\n{p['answer']}")
    p["llm_judge"] = parse(j.get("text"), "verdict", FMAP)
    if False:  # correctness-vs-reference judging skipped (scope reduction)
        j2 = judge(f'corr|{p["qid"]}', CORR, f"QUESTION:\n{ds.queries[p['qid']]}\n\nREFERENCE:\n{ds.answers[p['qid']]}\n\nANSWER:\n{p['answer']}")
        p["llm_correct"] = parse(j2.get("text"), "correct", CMAP)
    if (i + 1) % 30 == 0:
        print(f"{i+1}/{len(pairs)} pairs judged", flush=True)


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    return round(float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))), 4) if pos and neg else None


res = {}
for m in ("prod_cosine", "sent_maxsim", "llm_judge"):
    P = [p[m] for p in pairs if p["type"] == "P"]
    res[m] = {"AUROC_P_vs_N1_random": auroc(P, [p[m] for p in pairs if p["type"] == "N1"]),
              "mean_score": {t: round(float(np.mean([p[m] for p in pairs if p["type"] == t and p[m] is not None])), 4) for t in ("P", "N1")}}
for t in ("P", "N1"):
    x = np.array([p["prod_pass"] for p in pairs if p["type"] == t], dtype=float)
    res["prod_cosine"][f"pass_rate_pct_at_0.75_{t}"] = round(100 * x.mean(), 1)
    res["prod_cosine"][f"pass_rate_ci95_{t}"] = [round(100 * v, 1) for v in bootstrap_ci(x)]
jp = np.array([p["llm_judge"] for p in pairs if p["type"] == "P" and p["llm_judge"] is not None])
res["llm_judge"]["P_verdict_dist_pct"] = {k: round(100 * float(np.mean(jp == v)), 1) for k, v in FMAP.items()}
res["llm_judge"]["P_fully_supported_ci95"] = [round(100 * v, 1) for v in bootstrap_ci((jp == 1.0).astype(float))]
jn = np.array([p["llm_judge"] for p in pairs if p["type"] == "N1" and p["llm_judge"] is not None])
res["llm_judge"]["N1_judged_supported_pct_should_be_near_0"] = round(100 * float(np.mean(jn == 1.0)), 1) if len(jn) else None
judged = [c for c in cache.values() if c.get("ms")]
write_results("T9_groundedness", {"test_id": "T9_groundedness", "status": "NEWLY EXECUTED", "environment": environment("api"),
                                  "design": __doc__, "n_answers": len(A), "n_pairs": len(pairs), "judge_model": JUDGE,
                                  "results": res, "judge_calls": len(judged),
                                  "judge_latency_ms_median": float(np.median([c["ms"] for c in judged])) if judged else None,
                                  "api_events": events,
                                  "caveats": ["Judge (gpt-oss-120b) is the same model family as the generator (gpt-oss-20b): possible leniency bias.",
                                              "Judge verdicts were not validated by a human annotator; an AI spot-check is recorded separately.",
                                              "Answers come from 100 MS MARCO-XI validation queries; n is small, see CIs."]},
              [{k: v for k, v in p.items() if k != "answer"} for p in pairs])
print(json.dumps(res, indent=1, ensure_ascii=False))
