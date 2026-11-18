# Indic-RAG — Testing & Evaluation Metrics Plan

_Date: 2026-11-18 · Companion to `IMPROVEMENT_ROADMAP.md`._

Goal: every number that may appear in the final `TECHNICAL_AUDIT.md` or a resume is produced by a script in `audit/`, on a recorded dataset/config/environment, with a sample size and an uncertainty estimate. **No mocked components in any end-to-end or latency result** (fakes are allowed only in unit tests, which report pass/fail, not performance).

---

## 1. Environment (set up and verified 2026-11-18)

| Item | Value |
|---|---|
| Hardware | AMD Ryzen 7 7445HS (6C/12T), 15.3 GB RAM, NVIDIA RTX 3050 Laptop 4 GB |
| OS | Windows 11 Home 10.0.26300 |
| `.venv` (CPU, **locked**) | `uv sync --frozen` from `uv.lock` → Python 3.12, torch 2.13.0 (CPU), sentence-transformers 5.7.0, transformers 5.15.0, qdrant-client 1.19.0, groq 1.6.0, flask 3.1.3. Audit-only extras: rank-bm25 0.2.2, psutil 7.2.2, jiwer 4.0.0, scipy 1.18.0 |
| `.venv-gpu` | Same locked versions except torch 2.13.0+cu130; CUDA verified on RTX 3050 |
| Qdrant | Local embedded mode, using a **copy** of `qdrant_hindi_benchmark/` (repo DB and lock files untouched) |
| Verified | Production index vectors == fresh e5-small embeddings (cos = 1.0); stored retrieval metrics reproduce exactly |

Notes: the audit extras are installed into the venvs only — `pyproject.toml`/`uv.lock` are unchanged. `.venv-gpu/` is not covered by `.gitignore` (`.venv/` is); do not commit it.

**CPU is the primary latency target** (the Docker / HF Spaces deployment has no GPU). GPU numbers are reported separately and labelled as such.

---

## 2. Datasets

| ID | Dataset | Purpose | Status |
|---|---|---|---|
| DS1 | Existing artifacts: 1,000 MS MARCO-XI Hindi queries (seed 42), 9,943 passages / 10,096 chunks, 627 queries with a relevance label | Retrieval quality; reproduction of stored results | ✅ Available |
| DS1-split | DS1's 627 labelled queries split **dev 50 % / test 50 %** (stratified-random, fixed seed) | Tune k, thresholds, model choices on dev; report on test only | ✅ Can create now |
| DS1-large | Fresh sample (different seed, e.g. 5–10k queries → ~50–100k passages) from `hintrain.parquet` | Scale sensitivity; untouched test set | ⛔ Needs dataset download (`ai4bharat/MSMARCO-XI`, not cached locally) |
| DS2 | Off-topic queries: (a) ~50 hand-written out-of-domain queries (Hindi / English / Hinglish); (b) MS MARCO-XI queries whose passages are **not** in the corpus (natural out-of-corpus) | Off-topic guardrail FAR/FRR | (a) ✅ can write now · (b) ⛔ needs download |
| DS3 | Groundedness set: for ~100 answerable queries — grounded answers (LLM with gold context) and ungrounded answers (LLM with mismatched context + entity/number-perturbed answers); manual spot-labelling of a subset | Groundedness AUROC; LLM-judge agreement | ⛔ Needs Groq API key |
| DS4 | Safety prompts: ~50 injection/jailbreak (Hindi, Hinglish, English) + ~50 benign look-alikes | Input-safety TPR/FPR | ✅ can write now |
| DS5 | Speech: public Hindi set with references (e.g. ~100 utterances from FLEURS `hi_in` test) + optional ~20 Hinglish recordings with transcripts | STT WER/CER, latency | ⛔ Needs Groq key + audio download / your recordings |

**Leakage rules.** (1) Nothing is tuned on the test split. (2) The off-topic threshold, reranker choice, k and groundedness threshold are all selected on dev. (3) LLM-generated DS3 answers must not be judged by the same prompt that produced them.

---

## 3. Metrics — definitions

### 3.1 Retrieval
| Metric | Definition | Notes |
|---|---|---|
| Hit@k (repo's "Recall@k") | Fraction of queries with ≥1 relevant chunk in top-k | Report k = 1, 2, 3, 5, 10, 20. **Hit@2 is the production-relevant value** (k=2 is passed to the LLM) |
| Recall@k (true) | Relevant items retrieved / relevant items, averaged | Differs from Hit@k only for the ~29 multi-label queries |
| MRR@10 | Mean reciprocal rank of first relevant chunk | Keep MRR@5 for continuity with stored results |
| nDCG@10 | Binary-relevance nDCG | Standard IR comparability |
| Complementarity | % queries where system A hits@10 and B misses | Decides whether hybrid retrieval (R3) is worth it |

Relevance: chunk text is a substring of an `is_selected=1` passage (existing rule; correct for this chunker).

### 3.2 Latency (per request, ms)
Spans: `stt`, `embed_query`, `vector_search`, `rerank` (if added), `llm_ttft`, `llm_total`, `guard_input`, `guard_off_topic`, `guard_groundedness`, `e2e_text` (no STT), `e2e_voice` (with STT), `http_roundtrip` (client-measured).

Statistics: mean, median, P95, P99, min, max, std. **Minimum sample sizes:** P95 reported only with n ≥ 100; **P99 only with n ≥ 1,000** (otherwise P99 is ≈ the single slowest request and is marked "insufficient n"). Cold start (model load + first call) is reported separately and excluded from warm statistics.

### 3.3 Guardrails
| Guardrail | Metrics |
|---|---|
| Off-topic | False-accept rate (off-topic passed), false-refusal rate (answerable blocked), ROC/AUROC; threshold picked on dev at a stated operating point |
| Groundedness | AUROC (grounded vs ungrounded), precision/recall at chosen threshold, added latency |
| Input safety | Attack block rate (TPR), benign false-block rate (FPR), per language |
| Refusal | Refusal-on-answerable rate; refusal detection accuracy |

### 3.4 Generation (needs Groq)
Status distribution (`answered` / each refusal / `error`); `finish_reason="length"` rate; empty/very-short answer rate; answer language match rate; faithfulness via LLM-judge with **manual agreement check on ≥30 samples** (report Cohen's κ). Reference-answer correctness only if the MS MARCO-XI parquet contains an answers field (to verify after download).

### 3.5 Speech
WER and CER (jiwer; Devanagari-normalized: NFC, punctuation stripped), Devanagari-script rate, STT latency P50/P95.

### 3.6 Resources
Model load time, peak RSS (psutil), peak VRAM (`torch.cuda.max_memory_allocated`), index build time, embedding throughput (chunks/s, CPU and GPU), on-disk index size.

---

## 4. Statistical protocol

- **Uncertainty:** 95 % bootstrap CIs (1,000+ resamples) for every quality metric.
- **Paired comparisons** (same queries, two systems): McNemar's test for Hit@k; paired bootstrap / Wilcoxon signed-rank for MRR and latency.
- **Repetitions:** latency experiments run ≥3 times; report per-run and pooled statistics.
- **Warm-up:** ≥5 discarded requests per configuration.
- **A/B fairness:** configurations interleaved (A, B, A, B…) rather than run back-to-back, to cancel network/rate-limit drift — this is exactly what invalidated the stored baseline-vs-optimized LLM comparison.
- **Rate limits:** Groq experiments run at a fixed, recorded request rate; 429s are logged and reported, not silently retried away.
- **Environment capture** written into every result file: git commit, package versions, device, hardware, timestamp, config.

---

## 5. Test matrix

| # | Test | Metrics | Device | Needs | Status |
|---|---|---|---|---|---|
| T1 | Reproduce stored retrieval (e5-small, MiniLM) | Hit@k, MRR, CIs | CPU + GPU | — | ✅ e5 reproduced (prelim); MiniLM pending |
| T2 | Baselines: BM25, paraphrase-multilingual-MiniLM-L12-v2, multilingual-e5-base, bge-m3 | §3.1 + throughput, CPU latency | CPU + GPU | ~3.5 GB downloads | Ready |
| T3 | Chunking: max_chars {200, 400, 1000} × {recursive, recursive+danda*, sentence} | Hit@k (passage-level), % mid-sentence cuts | GPU (quality) | — | Ready (*danda variant implemented in `audit/`, not in `src/`) |
| T4 | Latency decomposition: embed vs search, cold vs warm, n ≥ 1,000 | §3.2, §3.6 | CPU + GPU | — | Ready |
| T5 | k-sweep + reranker over top-20 | Hit@k-to-LLM, MRR, rerank CPU latency | CPU + GPU | reranker download | Ready |
| T6 | Off-topic guardrail | FAR, FRR, AUROC | CPU | DS2(a) now; DS2(b) download | Partly ready |
| T7 | Input-safety guardrail | TPR / FPR by language | CPU | DS4 | Ready |
| T8 | LLM latency + telemetry: random ~100 dev queries × 3 runs, current vs alternative configs interleaved | §3.2 LLM spans, §3.4 | — | **GROQ_API_KEY** | ⛔ Blocked |
| T9 | Groundedness method comparison | §3.3 | CPU | **GROQ_API_KEY** (DS3) | ⛔ Blocked |
| T10 | Real HTTP benchmark of `api.py`: 1 / 4 / 8 concurrent clients | `http_roundtrip`, per-span consistency (logger race) | CPU | **GROQ_API_KEY** | ⛔ Blocked |
| T11 | STT | WER, CER, latency | — | **GROQ_API_KEY** + DS5 | ⛔ Blocked |
| T12 | Qdrant Cloud search latency | `vector_search` P50/P95 | — | **QDRANT_URL / QDRANT_API** | ⛔ Blocked |
| T13 | Deployment smoke: `docker build` with current `requirements.txt`, one real query | pass/fail + error | — | Docker (installed) | Ready (expected to fail per wheel inspection) |

T1–T7 and T13 run entirely locally and don't touch production code.

---

## 6. Outputs

```
audit/
  scripts/                 # one script per test, CLI-configurable, seed-fixed
  results/
    preliminary_checks.json
    retrieval_<test>.json / .csv     # one row per (system, split, metric)
    latency_<test>.json / .csv       # raw per-request samples + summary
    guardrails_<test>.json / .csv
    environment.json
  IMPROVEMENT_ROADMAP.md
  EVALUATION_PLAN.md
TECHNICAL_AUDIT.md         # final report (Phase 5), every number linked to a results file
```

Every summary row carries: `test_id, system, config, split, n, metric, value, ci_low, ci_high, device, run_id, git_commit, timestamp`. Raw latency samples are kept so percentiles can be recomputed.

---

## 7. Execution order

1. T1, T4 (CPU+GPU), T13 → validated baseline + deployment status.
2. T2, T3, T5 → retrieval evidence for roadmap Phase 2.
3. T6(a), T7 → guardrail evidence that needs no API.
4. T8–T12 once credentials and audio data are available.
5. Write `TECHNICAL_AUDIT.md` with the "resume-safe vs not-safe" metric lists.
