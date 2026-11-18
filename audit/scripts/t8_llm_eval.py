"""T8 — Real LLM generation evaluation (Groq), production prompt and parameters.

Mirrors src/core/orchestrator.py stages 3-6 exactly (system prompt imported from production, same prompt template,
k=2 retrieval with e5-small, off-topic gate 0.826, streaming), and additionally records finish_reason, token usage
(incl. reasoning tokens), Groq queue time, rate-limit headers and 429s.

Configs (interleaved per query, order alternated to cancel drift):
  A_production : openai/gpt-oss-20b, reasoning_effort=low, max_completion_tokens=150   (api.py)
  B_cap1024    : openai/gpt-oss-20b, reasoning_effort=low, max_completion_tokens=1024  (diagnostic)
Queries: 100 random DSV-dev labelled queries (MS MARCO-XI validation), corpus = DSV corpus.
Usage: .venv/Scripts/python audit/scripts/t8_llm_eval.py
"""
import json, os, re, time
import numpy as np
from common import (PROJECT_ROOT, load_dsv, dsv_split, chunk_corpus, encode, dense_topk, load_bi_encoder,
                    hi_tokenize, latency_summary, mcnemar, paired_bootstrap_diff, environment, write_results, CACHE_DIR)
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
os.environ.pop("QDRANT_URL", None); os.environ.pop("QDRANT_API", None)
import groq, httpx
from src.core.orchestrator import _SYSTEM_PROMPT
from src.core.guardrails import check_groundedness
from src.core.embeddings import EmbeddingModel
from src.core.chunking import RecursiveChunker

N_QUERIES = 100
CONFIGS = {"A_production": dict(model="openai/gpt-oss-20b", reasoning_effort="low", max_completion_tokens=150),
           "B_cap1024": dict(model="openai/gpt-oss-20b", reasoning_effort="low", max_completion_tokens=1024)}
REFUSAL_EN = "I cannot answer this based on the provided context."
OUT = CACHE_DIR / "t8_calls.jsonl"

ds = load_dsv(); split = dsv_split(ds)
pool = [q for q in split["dev"] if ds.answers.get(q) and ds.answers[q].strip() not in ("सहमत हैं तो",)]
qids = sorted(np.random.default_rng(8).choice(pool, size=N_QUERIES, replace=False).tolist())
chunks, owner = chunk_corpus(ds.passages, RecursiveChunker())
cemb = np.load(CACHE_DIR / "emb_e5_small_DSV.npy")
emb_model = EmbeddingModel("intfloat/multilingual-e5-small")          # production wrapper (CPU), reused for groundedness
st = emb_model.model
idx, sc = dense_topk(encode(st, [ds.queries[q] for q in qids], "query: "), cemb, 2, "cpu")

client = groq.Groq(api_key=os.environ["GROQ_API_KEY"],
                   http_client=httpx.Client(limits=httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=60.0)))


def parse_reset(s):
    if not s:
        return 0.0
    tot = 0.0
    for v, u in re.findall(r"([\d.]+)(ms|s|m|h)", s):
        tot += float(v) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[u]
    return tot


done = set()
if OUT.exists():
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l); done.add((r["qid"], r["config"]))
rate_events = []
with open(OUT, "a", encoding="utf-8") as fout:
    for i, (q, row, s) in enumerate(zip(qids, idx, sc)):
        ctx = [chunks[c] for c in row]
        gold_in_ctx = any(owner[c] in ds.gold[q] for c in row)
        prompt = f"<context>\n{chr(10).join([ctx[0], '', ctx[1]])}\n</context>\n\nQuestion: {ds.queries[q]}\nAnswer:"
        order = list(CONFIGS) if i % 2 == 0 else list(CONFIGS)[::-1]
        for cname in order:
            if (q, cname) in done:
                continue
            rec = {"qid": q, "config": cname, "top1_score": float(s[0]), "gold_in_top2": bool(gold_in_ctx)}
            if s[0] < 0.826:
                rec["status"] = "refused_off_topic"
            else:
                for attempt in range(4):
                    try:
                        t0 = time.perf_counter()
                        raw = client.chat.completions.with_raw_response.create(
                            messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                            stream=True, **CONFIGS[cname])
                        hdr = raw.headers
                        stream = raw.parse()
                        first, text, reasoning, finish, usage, xg = None, "", "", None, None, None
                        for ch in stream:
                            if ch.choices:
                                d = ch.choices[0].delta
                                if d.content is not None:
                                    if first is None:
                                        first = time.perf_counter()
                                    text += d.content
                                if getattr(d, "reasoning", None):
                                    reasoning += d.reasoning
                                if ch.choices[0].finish_reason:
                                    finish = ch.choices[0].finish_reason
                            x = getattr(ch, "x_groq", None)
                            if x is not None and getattr(x, "usage", None) is not None:
                                usage = x.usage.model_dump()
                            if getattr(ch, "usage", None) is not None:
                                usage = ch.usage.model_dump()
                        t1 = time.perf_counter()
                        rec.update({"status": "llm_ok", "answer": text, "finish_reason": finish, "usage": usage,
                                    "ttft_ms": (first - t0) * 1000 if first else None, "total_ms": (t1 - t0) * 1000,
                                    "reasoning_chars": len(reasoning), "attempts": attempt + 1,
                                    "rl_remaining_tokens": hdr.get("x-ratelimit-remaining-tokens"),
                                    "rl_reset_tokens": hdr.get("x-ratelimit-reset-tokens")})
                        break
                    except groq.RateLimitError as e:
                        ra = float(e.response.headers.get("retry-after", 10))
                        rate_events.append({"qid": q, "config": cname, "event": "429", "retry_after_s": ra}); time.sleep(ra)
                    except Exception as e:  # noqa
                        rate_events.append({"qid": q, "config": cname, "event": type(e).__name__, "detail": str(e)[:200]})
                        rec.update({"status": "error", "error": str(e)[:200]}); time.sleep(3)
                # token-aware pacing to stay under the 8K TPM limit
                rem = rec.get("rl_remaining_tokens")
                if rem is not None and int(rem) < 3000:
                    time.sleep(parse_reset(rec.get("rl_reset_tokens")) + 0.5)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(qids)}", flush=True)

# ------------------------------------------------------------------ analysis
recs = [json.loads(l) for l in open(OUT, encoding="utf-8")]
recs = [r for r in recs if r["qid"] in set(qids)]
ctx_by_q = {q: [chunks[c] for c in row] for q, row in zip(qids, idx)}


def classify(r):
    if r["status"] != "llm_ok":
        return r["status"]
    a = (r.get("answer") or "").strip()
    if not a:
        return "empty"
    if REFUSAL_EN in a:
        return "refusal_en"
    if re.search(r"(उत्तर नहीं दे सकता|उत्तर नहीं दिया जा सकता|संदर्भ में .*नहीं)", a):
        return "refusal_hi"
    if r.get("finish_reason") == "length":
        return "answered_truncated"
    return "answered"


def f1(pred, ref):
    p, g = hi_tokenize(pred), hi_tokenize(ref)
    if not p or not g:
        return 0.0
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


for r in recs:
    r["class"] = classify(r)
    if r["status"] == "llm_ok":
        ok, g = check_groundedness(r.get("answer") or "", ctx_by_q[r["qid"]], emb_model, threshold=0.75)
        r["prod_groundedness_pass"], r["prod_groundedness_score"] = bool(ok), float(g)
        r["token_f1_vs_reference"] = f1(r.get("answer") or "", ds.answers[r["qid"]])

summary = {}
for cname in CONFIGS:
    rs = [r for r in recs if r["config"] == cname]
    ok = [r for r in rs if r["status"] == "llm_ok"]
    cls = {}
    for r in rs:
        cls[r["class"]] = cls.get(r["class"], 0) + 1
    u = [r["usage"] for r in ok if r.get("usage")]
    reas = [x.get("completion_tokens_details", {}) or {} for x in u]
    gin = [r for r in ok if r["gold_in_top2"]]; gout = [r for r in ok if not r["gold_in_top2"]]
    is_ref = lambda r: r["class"].startswith("refusal")
    summary[cname] = {
        "n_queries": len(rs), "n_llm_calls_ok": len(ok), "class_counts": cls,
        "finish_reason_length_pct": round(100 * np.mean([r.get("finish_reason") == "length" for r in ok]), 1) if ok else None,
        "empty_answer_pct": round(100 * np.mean([r["class"] == "empty" for r in ok]), 1) if ok else None,
        "refusal_pct": round(100 * np.mean([is_ref(r) for r in ok]), 1) if ok else None,
        "refusal_pct_when_gold_in_top2": round(100 * np.mean([is_ref(r) for r in gin]), 1) if gin else None,
        "refusal_pct_when_gold_NOT_in_top2": round(100 * np.mean([is_ref(r) for r in gout]), 1) if gout else None,
        "n_gold_in_top2": len(gin), "n_gold_not_in_top2": len(gout),
        "prod_groundedness_pass_pct": round(100 * np.mean([r["prod_groundedness_pass"] for r in ok]), 1) if ok else None,
        "token_f1_vs_reference_mean_answered": round(float(np.mean([r["token_f1_vs_reference"] for r in ok if r["class"] in ("answered", "answered_truncated")] or [0])), 4),
        "ttft_ms": latency_summary([r["ttft_ms"] for r in ok if r.get("ttft_ms")]) if ok else None,
        "total_ms": latency_summary([r["total_ms"] for r in ok]) if ok else None,
        "groq_queue_time_ms": latency_summary([1000 * x["queue_time"] for x in u if x.get("queue_time") is not None]) if u else None,
        "prompt_tokens_mean": round(float(np.mean([x["prompt_tokens"] for x in u])), 1) if u else None,
        "completion_tokens_mean": round(float(np.mean([x["completion_tokens"] for x in u])), 1) if u else None,
        "reasoning_tokens_mean": round(float(np.mean([d.get("reasoning_tokens") or 0 for d in reas])), 1) if reas else None,
        "retries": sum(r.get("attempts", 1) - 1 for r in ok),
    }
pa = {r["qid"]: r for r in recs if r["config"] == "A_production" and r["status"] == "llm_ok"}
pb = {r["qid"]: r for r in recs if r["config"] == "B_cap1024" and r["status"] == "llm_ok"}
both = sorted(set(pa) & set(pb))
paired = {"n_pairs": len(both)}
if both:
    paired["truncation_A_vs_B"] = mcnemar(np.array([pa[q]["finish_reason"] == "length" for q in both]), np.array([pb[q]["finish_reason"] == "length" for q in both]))
    paired["total_ms_B_minus_A"] = paired_bootstrap_diff(np.array([pa[q]["total_ms"] for q in both]), np.array([pb[q]["total_ms"] for q in both]))
    paired["answered_A_vs_B"] = mcnemar(np.array([pa[q]["class"] == "answered" for q in both]), np.array([pb[q]["class"] == "answered" for q in both]))

write_results("T8_llm_eval", {"test_id": "T8_llm_eval", "status": "NEWLY EXECUTED", "environment": environment("api"),
                              "design": __doc__, "n_queries": len(qids), "summary": summary, "paired_A_vs_B": paired,
                              "rate_limit_and_error_events": rate_events,
                              "caveats": ["Reference answers are machine-translated MS MARCO answers; token-F1 is indicative only.",
                                          "Hindi refusal detection is regex-based (refusal_hi) and may miss paraphrases.",
                                          "Latency measured from this laptop over the public internet to Groq."]},
              [{"config": c, **{k: v for k, v in s.items() if not isinstance(v, dict)}} for c, s in summary.items()])
json.dump([{k: r.get(k) for k in ("qid", "config", "class", "answer", "finish_reason", "gold_in_top2", "prod_groundedness_score", "token_f1_vs_reference")}
           for r in recs], open(CACHE_DIR / "t8_analysed.json", "w", encoding="utf-8"), ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
