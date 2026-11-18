# Indic-RAG — Improvement Analysis & Prioritized Roadmap

_Date: 2026-11-18 · Scope: analysis only — no production code has been modified._

Every claim below is tagged with its evidence level:

| Tag | Meaning |
|---|---|
| **[MEASURED]** | Newly executed during this audit; reproducible via `audit/scripts/` → `audit/results/preliminary_checks.json` |
| **[LOG]** | Derived from a stored log/artifact in the repo (not re-run) |
| **[CODE]** | Follows directly from reading the source (file:line given) |
| **[UNMEASURED]** | Plausible but no evidence yet — the measurement is listed |

Complexity: **Low** ≈ < 1 day · **Medium** ≈ 1–3 days · **High** ≈ > 3 days or needs new labelled data.

---

## Key evidence driving the priorities

| Fact | Value | Source |
|---|---|---|
| Stored Recall@5 (e5-small) reproduces | 73.05 % (95 % CI 69.7–76.4), n=627 | [MEASURED] |
| **Gold passage in the top-2 chunks the LLM actually receives** | **Hit@2 = 46.09 %** | [MEASURED] |
| Gold passage anywhere in top-20 | Hit@20 = 89.63 % | [MEASURED] |
| Off-topic threshold margin on answerable queries | P1 top-1 score 0.8332 vs threshold 0.826 | [MEASURED] |
| Off-topic queries' top-1 scores (manual, n=4) | 0.7924–0.8356; 1/4 leaked | [LOG] `threshold_test_output.txt` |
| "Answered + grounded" results that were actually refusals | 9 / 62 (≈15 %) | [LOG] `benchmark_*.txt` |
| Groundedness gate rejected a real answer | 0 / 62 | [LOG] |
| LLM latency, last complete-ish run (n=27) | P50 567 ms, P95 4,643 ms | [LOG] `benchmark_optimized.txt` |
| Pinned deploy deps lack APIs used by code | `query_points`, `reasoning_effort` absent | [MEASURED] (wheel inspection) |
| Passages longer than chunker `max_chars` | 19 / 9,943 (0.19 %) | [MEASURED] |
| Non-NFC passages | 79 / 9,943 (0.8 %) | [MEASURED] |

---

## 1. Retrieval accuracy

### R1 · Only 2 chunks reach the LLM — gold context is missing for most answerable queries
- **Problem.** `max_retrieval_results=2` (`src/core/orchestrator.py:210`). Hit@2 = 46.1 % [MEASURED]: for ~54 % of answerable queries the LLM never sees the relevant passage, so it must refuse or hallucinate. This caps answer quality more than any other factor.
- **Solution.** Two-stage retrieval: dense top-N (N≈20–30; Hit@20 = 89.6 %) → multilingual cross-encoder reranker (candidates to evaluate: `BAAI/bge-reranker-v2-m3`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) → pass top-k (k∈{2,3,5}) to the LLM. Evaluate "just raise k" as the zero-cost baseline first (Hit@3 = 58.9 %, Hit@5 = 73.1 %), since it trades prompt tokens for recall without a new model.
- **Complexity.** Medium (reranker is ~50 lines + a model; k change is a parameter).
- **Measure.** Hit@k-to-LLM and MRR@10 on a held-out split; reranker latency P50/P95 on **CPU** (deployment target); LLM prompt tokens; downstream answer-correctness / refusal-on-answerable rate (see Evaluation Plan §4). Accept only if gain is significant (McNemar p<0.05) and P95 latency budget holds.

### R2 · No credible baseline — model choice is unvalidated beyond an English-only straw man
- **Problem.** The only comparison is vs `all-MiniLM-L6-v2`, an English-only model ([LOG] Recall@5 2.71 %). `bge-m3` is configured in `scripts/benchmark_embeddings.py:83` but was never run. No lexical baseline exists.
- **Solution.** Run the existing harness logic on identical artifacts with: BM25 (Hindi-tokenized), `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-base`, `bge-m3` (dense). Choose the production model on a quality/CPU-latency Pareto front, not on Recall alone.
- **Complexity.** Low–Medium (new script in `audit/`; ~3.5 GB model downloads).
- **Measure.** Hit@1/5/10, MRR@10, nDCG@10 with bootstrap CIs; paired McNemar vs e5-small; CPU query-embed P50/P95; index build time; RAM.

### R3 · Dense-only retrieval (hybrid is conditional)
- **Problem.** Single dense retriever; Hindi queries with named entities, numbers or transliterated English terms are where dense models commonly miss. [UNMEASURED] for this corpus.
- **Solution.** Only if R2 shows BM25 retrieves gold passages that dense misses (measure complementarity: % queries where BM25 hits@10 and dense misses), add BM25 + dense fusion (Reciprocal Rank Fusion). Qdrant supports sparse vectors natively.
- **Complexity.** Medium.
- **Measure.** Complementarity first; then fused Hit@k vs dense with McNemar. Skip if complementarity < ~5 % of queries.

### R4 · Evaluation methodology gaps that make reported numbers hard to defend
- **Problem.** (a) Metrics called "Recall@k" are hit-rate@k [CODE] `scripts/benchmark_embeddings.py:334`; (b) no CIs, single run; (c) all tuning (thresholds, k) and testing use the same 627 queries → selection bias once you start tuning; (d) closed 10k-passage pool inflates absolute scores vs open-corpus retrieval; (e) chunking comparison exercises 0.19 % of passages [MEASURED].
- **Solution.** Rename metrics; add bootstrap CIs; create dev/test split (or sample a fresh test set with a different seed); add a scaled corpus run (e.g. 5–10k sampled queries → ~50–100k passages) to show how quality degrades with corpus size; redesign the chunking experiment (see H1).
- **Complexity.** Low (naming/CI/split) to Medium (larger corpus needs the parquet download).
- **Measure.** N/A — this *is* measurement infrastructure. Success = every reported number has n, CI, split and config recorded.

---

## 2. Hindi & multilingual support

### H1 · Recursive chunker ignores the Devanagari danda (।)
- **Problem.** `RecursiveChunker` splits sentences on `[.!?]` only [CODE] `src/core/chunking.py:30`; long Hindi paragraphs fall through to blind fixed-char splits that cut mid-word/mid-sentence. Negligible on the current corpus (19 long passages) but matters for any real document corpus.
- **Solution.** Add `।` and `॥` to the sentence regex; apply overlap at sentence granularity, not only in the fixed-size fallback.
- **Complexity.** Low.
- **Measure.** A chunking experiment that actually exercises chunking: `max_chars ∈ {200, 400, 1000}` × {recursive, recursive+danda, sentence}, Hit@k mapped back to passage-level relevance; plus % chunks ending mid-sentence.

### H2 · STT is hard-coded to Hindi
- **Problem.** `language="hi"` always [CODE] `src/core/stt.py:120`. Code-mixed (Hinglish) speech and other Indic languages are forced through Hindi decoding; the project name suggests "Indic" but only Hindi is supported or evaluated. STT quality has **never** been measured [LOG — no reference transcripts exist].
- **Solution.** Make language a request parameter (default `hi`), optionally auto-detect; keep Devanagari output normalization. Do not add more STT vendors — two unused STT classes already exist.
- **Complexity.** Low (code) / Medium (evaluation data).
- **Measure.** WER and **CER** (CER is more stable for Hindi) on a public Hindi test set (e.g. FLEURS `hi_in`) and a small Hinglish set; script-correctness rate (% Devanagari output); STT latency P50/P95.

### H3 · Input-safety guardrail is English-only
- **Problem.** All patterns are English regexes [CODE] `src/core/guardrails.py:53-73`; transcripts are Devanagari, so Hindi/Hinglish injection attempts are not covered, while benign English phrases ("act as a …") are blocked.
- **Solution.** Build a labelled prompt set first; then compare (a) extended Hindi/Hinglish regex list vs (b) an LLM safety classifier call. Pick based on measured block rate / false-block rate / latency.
- **Complexity.** Medium.
- **Measure.** Attack block rate (TPR) and benign false-block rate (FPR) on ~50 attack + ~50 benign prompts in Hindi/Hinglish/English.

### H4 · Refusal is an English sentinel string
- **Problem.** The prompt forces the English sentence "I cannot answer this based on the provided context." even for Hindi questions; groundedness bypass is an exact English substring match [CODE] `src/core/guardrails.py:239`. A Hindi-phrased refusal would be blocked as "ungrounded"; an English refusal is returned to a Hindi user.
- **Solution.** Ask for structured output (e.g. JSON `{"answerable": bool, "answer": str}`) or a language-neutral sentinel token; localize the user-facing refusal message.
- **Complexity.** Low–Medium.
- **Measure.** Refusal detection accuracy on a labelled set of model outputs; % responses in the query's language.

### H5 · Unicode normalization (low priority)
- **Problem.** No NFC normalization before embedding. Affects 79 passages / 6 queries in this corpus [MEASURED].
- **Solution.** `unicodedata.normalize("NFC", …)` at ingest and query time.
- **Complexity.** Low.
- **Measure.** Hit@k before/after (expected ≈ no change on this corpus; justified mainly for robustness on user/STT input).

---

## 3. Guardrails & answer reliability

### G1 · Refusals are reported as successful, grounded answers
- **Problem.** Refusal bypass returns `(True, 1.0)` [CODE] `src/core/guardrails.py:239-241`, so status = `answered`, groundedness = 1.0. 9/62 logged "answered" results were refusals [LOG]. Any "pass rate" from these runs is inflated.
- **Solution.** New status `refused_no_answer` (or `answerable=false`), groundedness `null`.
- **Complexity.** Low.
- **Measure.** Status distribution on the evaluation set; refusal-on-answerable rate becomes a reportable metric.

### G2 · Groundedness check does not demonstrably detect hallucinations
- **Problem.** Single cosine between the answer and the concatenated context, embedded **without** E5 prefixes [CODE] `src/core/guardrails.py:253`; E5 similarities are compressed into a narrow high band; threshold 0.75 never rejected a real answer (0/62, scores 0.76–1.0) [LOG]. There is no labelled data showing it separates grounded from ungrounded answers.
- **Solution.** Build a labelled groundedness set (Eval Plan §4.3), then compare on AUROC and latency: (a) current method, calibrated; (b) sentence-level max-similarity (each answer sentence vs each chunk, with prefixes); (c) LLM-as-judge via Groq. Adopt the cheapest method that achieves acceptable AUROC.
- **Complexity.** Medium–High (dominated by labelling).
- **Measure.** AUROC, precision/recall at chosen threshold (chosen on dev split), added latency P50/P95.

### G3 · Off-topic threshold has near-zero margin and an unmeasured false-accept rate
- **Problem.** Calibrated on random query–passage pairs [CODE] `calculate_thresholds.py:34-37`, but applied to the *maximum* score over 10k passages — a different distribution. Answerable P1 = 0.8332 vs threshold 0.826 [MEASURED]; off-topic examples reached 0.8356 [LOG].
- **Solution.** Build an off-topic set; recalibrate on the correct statistic (top-1 over the corpus). Evaluate alternative signals: reranker score (R1, typically better calibrated), top-1 minus mean(top-k) gap.
- **Complexity.** Medium.
- **Measure.** False-accept rate (off-topic passed) and false-refusal rate (answerable blocked); ROC/AUROC; threshold selected on dev, reported on test.

### G4 · Token cap likely truncates reasoning-model answers
- **Problem.** `max_completion_tokens=150` with `gpt-oss-20b` reasoning [CODE] `api.py:93`; reasoning tokens count against the cap. Logged answers of 4, 24, 28 chars [LOG] are consistent with truncation but **unconfirmed** — `finish_reason` and reasoning token counts are not recorded [CODE] `src/core/llm.py:47-52`.
- **Solution.** Record `finish_reason`, usage (reasoning/completion tokens) and Groq timing fields; then set the cap from data.
- **Complexity.** Low.
- **Measure.** `finish_reason == "length"` rate; empty/very short answer rate; latency impact of the new cap.

### G5 · No citations (optional, enables a reliability metric)
- **Problem.** Chunks are concatenated without IDs [CODE] `src/core/orchestrator.py:340`; answers cannot be traced to sources.
- **Solution.** Number chunks in the prompt and request citation markers.
- **Complexity.** Low.
- **Measure.** Citation precision (cited chunk supports claim) on a manually checked sample.

---

## 4. Performance & latency

### P1 · Latency instrumentation is not trustworthy
- **Problem.** (a) `retrieval` timing includes query embedding [CODE] `src/core/vectorstore.py:87-98`; (b) API's `embedding` value is the P50 of two different calls (query + groundedness); (c) one global `LatencyLogger` is reset per request while Flask is multithreaded → concurrent requests corrupt each other [CODE] `api.py:148-155`; (d) `end_to_end` is logged only on the `answered` path; (e) existing "E2E" excludes STT and HTTP [CODE] `scripts/evaluate_llm_latency.py:79`.
- **Solution.** Per-request timing context (pass a `Timings` object or use `contextvars`), separate spans: `stt`, `embed_query`, `vector_search`, `llm_ttft`, `llm_total`, `guard_*`, `e2e`, logged on every exit path.
- **Complexity.** Low–Medium.
- **Measure.** Sum of spans ≈ e2e (± a few ms); identical per-request numbers under 1 vs 4 concurrent clients.

### P2 · LLM tail latency dominates and is unmanaged
- **Problem.** LLM P50 567 ms but P95 4,643 ms; the run shows a step change after query 15 (E2E P50 608 → 3,813 ms) consistent with rate-limit queuing [LOG]. Retry wrapper retries all exceptions with fixed sleeps [CODE] `src/core/orchestrator.py:98-143` and does not honour `retry-after`.
- **Solution.** Record Groq queue time and 429s; retry only on 429/5xx/timeouts honouring `retry-after`; set an explicit request timeout. Stream tokens to the client (SSE) only if the UX target is perceived latency — measure TTFT first.
- **Complexity.** Low (instrumentation/retry) / Medium (streaming to browser).
- **Measure.** LLM P50/P95 with n≥100 at a fixed request rate; 429 count; client-side TTFT.

### P3 · Embedding model loaded twice at startup
- **Problem.** `QdrantVectorStore.__init__` always constructs `EmbeddingModel()` [CODE] `src/core/vectorstore.py:35`; `api.py:88` then replaces it. Logs show two loads [LOG].
- **Solution.** Inject the model via the constructor.
- **Complexity.** Low.
- **Measure.** Startup time and peak RSS before/after.

### P4 · CPU embedding & Qdrant Cloud latency are unmeasured (conditional work)
- **Problem.** Stored latency is GPU (RTX 3050) + local embedded Qdrant (exact search in Python) [LOG]; deployment is CPU + Qdrant Cloud. [UNMEASURED].
- **Solution.** Measure first. Only if CPU query embedding is a significant share of E2E, consider ONNX/int8 export of e5-small; only if Cloud RTT dominates search, consider region/co-location.
- **Complexity.** Low (measure) / Medium (optimize).
- **Measure.** CPU embed P50/P95/P99 (n≥1000, cheap locally); Cloud search P50/P95 (needs credentials).

---

## 5. Code quality & architecture

| ID | Problem (evidence) | Solution | Complexity | Measure |
|---|---|---|---|---|
| C1 | Text pipeline re-implemented in `api.py:233-309`, duplicating `orchestrator.py:238-422` — behaviour can drift (it already omits per-stage prints and some error text) | `orchestrator.process_text_query()`; voice path = STT + text path | Low–Med | Shared tests pass for both paths |
| C2 | Point IDs `int(time.time()*1000)+i` [CODE] `vectorstore.py:61` — two `add()` calls < len(chunks) ms apart overwrite each other | Deterministic UUIDv5 from content/source | Low | Unit test: two consecutive `add()` → count = sum |
| C3 | Hard-coded thresholds duplicated in `api.py`, `main.py`, scripts; absolute Windows parquet path in 2 scripts | Single config module / env settings | Low | grep shows one source of truth |
| C4 | Benchmark logic copied in 3 scripts (`build_ground_truth`, `check_relevant`) | Shared `eval` module | Low | Scripts produce identical numbers |
| C5 | Repo hygiene: dummy `gradio_app.py` with fake metrics; unused deps (chromadb, ollama, elevenlabs); ~250 MB of local DBs and `.txt` logs tracked despite `.gitignore` | Remove/flag dummy app; prune deps; untrack generated files | Low | Clone size; dependency count |
| C6 | `print`-based logging | `logging` with request IDs | Low | — |

---

## 6. Testing & deployment readiness

| ID | Problem (evidence) | Solution | Complexity | Measure |
|---|---|---|---|---|
| D1 | **Deployment is broken as pinned**: `requirements.txt` pins `qdrant-client==1.9.1` (no `query_points`) and `groq==0.8.0` (no `reasoning_effort`) [MEASURED]; Dockerfile uses Python 3.10 vs `pyproject` ≥3.12 | Generate requirements from `uv.lock` (`uv export`), Python 3.12 base image, bake model weights into image | Low | `docker build` + one real `/api/query/text` call succeeds |
| D2 | No automated tests — `test_*.py` are manual scripts without assertions | pytest: chunkers, guardrails, orchestrator status routing (fakes acceptable **for unit tests only**), retrieval regression test on the artifact set (Hit@5 ≥ agreed floor) | Medium | CI pass; coverage of guardrail branches |
| D3 | Flask dev server, no upload size limit, no timeouts, CORS `*`, startup deletes Qdrant lock file [CODE] `api.py:76-79,114,319` | Production WSGI server (waitress/gunicorn), `MAX_CONTENT_LENGTH`, request timeouts, restricted CORS, remove lock deletion | Low–Med | Load test P95 at 1/4/8 concurrency; oversized upload rejected |
| D4 | API returns HTTP 200 for `status="error"`; no health endpoint | Proper status codes; `/healthz` | Low | Contract tests |
| D5 | No CI | GitHub Actions: lint + unit tests + CPU retrieval regression | Medium | Green pipeline on PR |

---

## Prioritized implementation roadmap

Ordering rule: fix what invalidates measurement first, then the largest measured quality gap, then reliability, then production hardening. Each phase lists its exit criterion.

### Phase 0 — Measurement foundation (prerequisite for every claim)
1. **Evaluation harness in `audit/`** (Eval Plan) — dev/test split, CIs, result schema. _(R4)_
2. **Per-request latency spans** _(P1)_ and **LLM telemetry** (`finish_reason`, usage, queue time) _(G4 instrumentation)_.
3. **Labelled sets**: off-topic, groundedness, safety prompts, STT references _(enables G2, G3, H2, H3)_.
- _Exit:_ baseline report with CIs for retrieval, per-span latency (n≥100 LLM, n≥1000 local), guardrail FAR/FRR.

### Phase 1 — Correctness quick wins (all Low)
4. **D1** fix dependency pins / Dockerfile (deployment currently cannot serve queries).
5. **G1** separate refusal status.
6. **C2** deterministic point IDs · **P3** single model load · **H1** danda splitting.
7. **C1** unify text/voice pipeline (makes later changes land in one place).
- _Exit:_ Docker image answers a real query; unit tests for these fixes.

### Phase 2 — Retrieval accuracy (largest measured gap: Hit@2 = 46 %)
8. **R2** baselines (BM25 + multilingual dense models) on the dev split.
9. **R1** k-sweep, then reranker over top-20; select on dev, report on test.
10. **R3** hybrid only if complementarity justifies it.
- _Exit:_ statistically significant Hit@k-to-LLM gain within an agreed CPU P95 budget.

### Phase 3 — Guardrails & reliability
11. **G3** recalibrate off-topic on correct statistic (consider reranker score from Phase 2).
12. **G2** groundedness method comparison on labelled set.
13. **H4** structured/localized refusal · **G4** token cap from telemetry · **H3** multilingual safety.
- _Exit:_ reported AUROC / FAR / FRR on test split for each guardrail.

### Phase 4 — Speech & multilingual
14. **H2** language parameter + WER/CER evaluation (Hindi, Hinglish).
15. **H5** NFC normalization.

### Phase 5 — Production hardening
16. **D2/D5** pytest suite + CI with retrieval regression gate.
17. **D3/D4** WSGI server, limits, status codes, health check.
18. **P2** retry policy; client streaming only if TTFT target requires it. **P4** optimize CPU/Cloud only if measured to matter.
19. **C3–C6** config, shared eval module, repo hygiene, logging.

### Explicitly **not** recommended (no evidence of need)
- Adding more STT/LLM vendors (unused ones already exist).
- Switching vector DB, adding a cache layer, or quantizing embeddings before P4 measurements show a bottleneck.
- Fine-tuning embeddings before R1/R2 baselines are exhausted.
