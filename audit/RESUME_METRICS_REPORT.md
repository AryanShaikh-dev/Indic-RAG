# Indic-RAG — Verified Resume Metrics

_All numbers below were **newly measured on 2026-11-18** by the scripts in `audit/scripts/`. Raw results: `audit/results/*.json|csv`. No application code was modified._

**Common conditions**
- **Hardware:** laptop, AMD Ryzen 7 7445HS (6C/12T), 15.3 GB RAM, NVIDIA RTX 3050 Laptop 4 GB, Windows 11.
- **Software:** Python 3.12, torch 2.13, sentence-transformers 5.7, qdrant-client 1.19, groq 1.6 (locked by `uv.lock`).
- **Vector DB:** **local embedded Qdrant** (`QdrantClient(path=...)`), not Qdrant Cloud, which was unavailable. Cloud latency would add network round-trips.
- **LLM / STT:** Groq API (`openai/gpt-oss-20b`, `whisper-large-v3`), free tier, called over the public internet from the laptop.
- **Production config evaluated:** e5-small embeddings, top-2 retrieval, gpt-oss-20b with `reasoning_effort=low` and `max_completion_tokens=150`, guardrails as in `api.py`.

---

## 1. Strongest verified metrics

| # | Metric | Value | Sample / conditions |
|---|---|---|---|
| 1 | **Hindi retrieval Recall@5 / MRR@5** (e5-small) | **70.3 % / 46.1 %** (95 % CI 68.4–72.2 / 44.4–47.8) | 2,159 held-out Hindi queries, corpus of 50,493 chunks (MS MARCO-XI *validation* split) |
| 2 | **Improvement over the BM25 lexical baseline** | **+27.8 pp Recall@5** (42.5 % → 70.3 %), MRR@5 26.0 % → 46.1 % | Same 2,159 queries and corpus; McNemar p < 1e-100 |
| 3 | **Cross-lingual retrieval** (English question → Hindi documents) | **Recall@5 67.9 %**, MRR@5 45.0 % (BM25: 0.5 %) | Same 2,159 queries in their original English form |
| 4 | **Speech recognition, Hindi** | **WER 29.6 %, CER 14.5 %** (95 % CI 27.6–31.6 / 12.6–16.8); median STT latency 636 ms, real-time factor 0.06 | Full FLEURS `hi_in` test set: 418 utterances, 80.5 min of audio |
| 5 | **End-to-end API latency** (text query → answer, real `api.py`) | **median 734 ms, P95 1,042 ms**; 100/100 HTTP 200, 0 errors | 100 sequential requests (90 answerable + 10 off-topic), local Flask server on CPU |
| 6 | **Answer groundedness** | **86.6 %** of answers fully supported by the retrieved context (95 % CI 78.0–93.9) | 82 non-refusal production answers, judged by `gpt-oss-120b` |

Supporting component latencies, measured on CPU with the app's own `QdrantVectorStore.query` (embed + search):
- **Retrieval:** median **47 ms**, P95 58 ms, P99 74 ms (n = 1,000).
- **Query embedding alone:** median 25 ms, P95 33 ms (n = 3,000).

---

## 2. Methodology

**Retrieval (metrics 1–3).**
- **Data:** queries come from the MS MARCO-XI Hindi *validation* split. It shares no queries with the training-split sample behind the stored benchmark, and nothing was tuned on it.
- **Corpus:** a 5,000-query random sample (seed 7). Each query's ~10 candidate passages were pooled into one corpus and chunked with the production `RecursiveChunker`.
- **Queries scored:** the 2,159 test-split queries that have a relevance label.
- **Fairness:** every system saw identical queries and chunks. Dense search is exact cosine, the same as Qdrant local mode.
- **Metrics:** "Recall@5" = share of queries with at least one relevant chunk in the top 5 (the repo's definition, also called hit-rate@5). MRR@5 = mean reciprocal rank of the first relevant chunk within the top 5.
- **Statistics:** bootstrap 95 % CIs; McNemar test for paired comparisons.
- **Reproduction check:** the stored benchmark (DS1, 627 queries) reproduces at Recall@5 72.9–73.1 % and MRR@5 45.4–45.5 %.

**Speech (metric 4).**
- The unmodified production `GroqSTT` class (`whisper-large-v3`, `language="hi"`) ran on all 418 FLEURS Hindi test utterances.
- Reference and hypothesis were normalised the same way: NFC, lowercase, punctuation and danda removed. No number normalisation.
- WER/CER are corpus-level, computed with jiwer.

**API latency (metric 5).**
- The real `api.py` ran from a copy of the working tree, served by the Flask dev server (`python api.py`) on CPU.
- Client and server were on the same laptop. Requests were paced at 1 every 8 s to stay inside Groq's 8K tokens/min limit.
- Latency is client-measured round-trip time over all 100 requests. Off-topic refusals return in about 70 ms and are included.

**Groundedness (metric 6).**
- **Answers:** 100 random validation-dev queries went through the exact production prompt and parameters. The 82 answers that were neither refusals nor empty were judged.
- **Judging:** `gpt-oss-120b` labelled each answer, against the context it was generated from, as SUPPORTED, PARTIALLY_SUPPORTED or NOT_SUPPORTED.
- **Sanity check:** 0 of 20 answers paired with an unrelated context were labelled SUPPORTED.

---

## 3. Baselines and comparisons (held-out set, 2,159 queries)

| System | Recall@5 | MRR@5 | Δ Recall@5 vs e5-small |
|---|---|---|---|
| **e5-small (production, 118M)** | **70.3 %** | **46.1 %** | — |
| multilingual-e5-base (278M) | 70.5 % | 47.1 % | +0.2 pp, **not significant** (p = 0.79) |
| BM25 (Hindi tokenised) | 42.5 % | 26.0 % | −27.8 pp (p < 1e-100) |
| paraphrase-multilingual-MiniLM-L12 | 38.2 % | 23.5 % | −32.1 pp |
| Hybrid BM25 + e5-small (RRF, untuned) | 59.6 % | 37.2 % | −10.7 pp |
| all-MiniLM-L6-v2 (English-only) | 1.2 % | 0.8 % | −69.2 pp |

**What this shows:** e5-small matches a model 2.4× its size, beats lexical and other multilingual baselines by a wide margin, and works cross-lingually. Adding BM25 did not help: BM25 finds a passage that e5 misses for only 2.6 % of queries.

**LLM configuration (100 queries, interleaved A/B):**

| | Production cap (150 tokens) | Cap 1,024 tokens |
|---|---|---|
| Complete answers | 53 / 99 (53.5 %) | 85 / 99 (85.9 %) |
| Truncated by token limit | 31 % (+2 empty) | 0 % |
| LLM latency, median | 538 ms | 542 ms (Δ not significant, p = 0.28) |

The reasoning model spends ~47 tokens per answer on hidden reasoning, and those count against the 150-token cap. **This was measured, not deployed.** Phrase it as a finding, not as a shipped improvement.

---

## 4. Limitations you must disclose

1. **The corpus is translated MS MARCO, pooled per query.** About 50k chunks, not a real Hindi knowledge base. Absolute recall would be lower on a full open corpus.
2. **The vector DB was local embedded Qdrant.** Cloud latency is unmeasured.
3. **Latency was measured on a single laptop, paced at under 8 requests/min.** No concurrency test was run. Groq's free tier (8K tokens/min) caps throughput at roughly 10 RAG requests per minute.
4. **End-to-end latency is the text path.** STT (median 636 ms) is measured separately, not as one voice request.
5. **Groundedness labels come from an LLM judge,** not humans, and the judge is the same model family as the generator. n = 82, so the CI is wide.
6. **The production LLM config truncates 31 % of answers.** The API's "answered" status (91 % in T10) includes those truncations and refusals, so don't quote it as a success rate.
7. **FLEURS is read Wikipedia speech.** Conversational or code-mixed Hindi is untested. 9 of 418 transcripts came back in English instead of Hindi.
8. **Not evaluated:** bge-m3 and bge-reranker-v2-m3 (excluded because of laptop memory limits), chunking variants, reranking, off-topic guardrail rates.
9. **The pinned `requirements.txt` / Docker build cannot serve queries.** With those versions, qdrant-client 1.9.1 can't open the DB and groq 0.8.0 rejects `reasoning_effort`. The Docker build itself was not run. Don't claim a working deployment without fixing this.

### Metrics that must NOT appear on a resume
- README "~600 ms end-to-end / ~450 ms LLM". It comes from an incomplete, cherry-picked run.
- `gradio_app.py` metrics (350 / 45 / 450 / 845 ms). These are hard-coded placeholders.
- "+70 pp Recall@5" over all-MiniLM-L6-v2 as an *improvement*. That baseline is English-only; use the BM25 comparison instead.
- The stored chunking comparison. It affected only 0.19 % of passages.
- "100 % groundedness pass rate" or the cosine guardrail as hallucination prevention. It passes 55 % of unrelated answer/context pairs.
- Anything about Qdrant Cloud performance or bge-m3.

---

## 5. Suggested resume bullets (verified numbers only)

- Built a Hindi voice-RAG pipeline (Whisper STT → multilingual-e5 embeddings → Qdrant → Groq LLM, Flask API) reaching **70.3 % Recall@5 / 46.1 % MRR@5** on 2,159 held-out Hindi queries over a 50K-chunk corpus.
- Benchmarked 6 retrieval systems on identical held-out data and selected multilingual-e5-small, which beat a BM25 baseline by **+27.8 pp Recall@5** (p < 0.001) and matched a 2.4× larger model, and which supports **cross-lingual English→Hindi retrieval (67.9 % Recall@5)**.
- Evaluated Hindi speech recognition on the full FLEURS test set (418 utterances, 80 min): **14.5 % CER / 29.6 % WER** at a median **636 ms** transcription latency.
- Measured end-to-end API latency of **734 ms median / 1.04 s P95** over 100 real requests, with **0 errors**. Component profiling showed 47 ms median retrieval on CPU.
- Built an LLM-judge evaluation showing **86.6 %** of generated answers fully grounded in retrieved context. Also found that the token budget truncated 31 % of answers, and that raising it lifts complete answers from **54 % to 86 %** with no significant latency cost.

_Only use the last bullet's second sentence as written ("found… lifts"), unless you actually ship the change._
