"""T11 — Speech-to-text evaluation on FLEURS Hindi (hi_in) test split.

System: production src.core.stt.GroqSTT (unchanged): whisper-large-v3, language="hi",
        prompt "The audio may contain Hindi or English."
Data  : google/fleurs parquet-data/hi_in/test (418 utterances, read speech, Wikipedia-domain sentences).
Metrics: corpus WER and CER (jiwer) after identical normalisation of reference and hypothesis,
         bootstrap 95% CI over utterances, Devanagari-script rate, latency, real-time factor.
Usage: .venv/Scripts/python audit/scripts/t11_stt_fleurs.py
"""
import glob, io, json, os, re, tempfile, time, unicodedata
import numpy as np
import pyarrow.parquet as pq
import jiwer
from common import PROJECT_ROOT, environment, write_results, latency_summary, CACHE_DIR
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
os.environ.pop("QDRANT_URL", None)
import groq
from src.core.stt import GroqSTT

PARQUET = glob.glob("D:/hf-cache-audit/datasets--google--fleurs/snapshots/*/parquet-data/hi_in/test-*.parquet")[0]
tbl = pq.read_table(PARQUET).to_pylist()
stt = GroqSTT()
PACE_S = 3.2          # Groq whisper free tier: 20 requests/min
partial = CACHE_DIR / "t11_partial.jsonl"
done = {}
if partial.exists():
    for l in open(partial, encoding="utf-8"):
        r = json.loads(l); done[r["id"]] = r

PUNCT = re.compile(r"[\u0964\u0965]")  # danda, double danda


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = PUNCT.sub(" ", s)
    s = "".join(ch if not unicodedata.category(ch).startswith("P") else " " for ch in s)
    return " ".join(s.split())


def deva_rate(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum("\u0900" <= c <= "\u097F" for c in letters) / len(letters) if letters else 0.0


errors = []
with open(partial, "a", encoding="utf-8") as fout:
    for i, row in enumerate(tbl):
        uid = f'{row["id"]}_{i}'
        if uid in done:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(row["audio"]["bytes"]); path = f.name
        rec = {"id": uid, "duration_s": row["num_samples"] / 16000, "ref": row["transcription"], "ref_raw": row["raw_transcription"]}
        for attempt in range(4):
            try:
                t = time.perf_counter()
                hyp = stt.transcribe(path)
                rec["latency_ms"] = (time.perf_counter() - t) * 1000
                rec["hyp"] = hyp
                break
            except groq.RateLimitError as e:
                ra = float(e.response.headers.get("retry-after", 20))
                errors.append({"id": uid, "type": "429", "retry_after_s": ra}); time.sleep(ra)
            except Exception as e:  # noqa
                errors.append({"id": uid, "type": type(e).__name__, "detail": str(e)[:200]}); time.sleep(5)
        os.unlink(path)
        if "hyp" in rec:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            done[uid] = rec
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(tbl)} done, errors={len(errors)}", flush=True)
        time.sleep(PACE_S)

recs = list(done.values())
refs = [norm(r["ref"]) for r in recs]; hyps = [norm(r["hyp"]) for r in recs]
refs_raw = [norm(r["ref_raw"]) for r in recs]


def corpus(ref, hyp, idx):
    w = jiwer.process_words([ref[i] for i in idx], [hyp[i] for i in idx])
    c = jiwer.process_characters([ref[i] for i in idx], [hyp[i] for i in idx])
    return w.wer, c.cer


n = len(recs)
wer, cer = corpus(refs, hyps, range(n))
wer_raw, cer_raw = corpus(refs_raw, hyps, range(n))
rng = np.random.default_rng(0)
boots = [corpus(refs, hyps, rng.integers(0, n, n)) for _ in range(500)]
bw, bc = np.array(boots).T
lat = [r["latency_ms"] for r in recs]
rtf = [r["latency_ms"] / 1000 / r["duration_s"] for r in recs]
dr = np.array([deva_rate(r["hyp"]) for r in recs])
per_utt = []
for r, rf, hy in zip(recs, refs, hyps):
    per_utt.append({"id": r["id"], "duration_s": round(r["duration_s"], 2), "latency_ms": round(r["latency_ms"], 1),
                    "wer": round(jiwer.wer(rf, hy), 4) if rf else None, "cer": round(jiwer.cer(rf, hy), 4) if rf else None,
                    "ref": r["ref"], "hyp": r["hyp"]})

payload = {
    "test_id": "T11_stt_fleurs_hi", "status": "NEWLY EXECUTED", "environment": environment("api"),
    "system": {"class": "src.core.stt.GroqSTT (unchanged)", "model": "whisper-large-v3", "language": "hi",
               "prompt": "The audio may contain Hindi or English."},
    "dataset": {"name": "google/fleurs hi_in test", "file": PARQUET, "n_total": len(tbl), "n_evaluated": n,
                "total_audio_s": round(sum(r["duration_s"] for r in recs), 1), "domain": "read Wikipedia sentences (FLoRes), not conversational/code-mixed"},
    "normalisation": "NFC, lowercase, danda/double-danda and all Unicode punctuation removed, whitespace collapsed; same for ref & hyp. "
                     "Reference = FLEURS 'transcription' field. No numeral/word normalisation (digits vs words count as errors).",
    "results": {
        "WER_pct": round(100 * wer, 2), "WER_ci95_pct": [round(100 * np.percentile(bw, 2.5), 2), round(100 * np.percentile(bw, 97.5), 2)],
        "CER_pct": round(100 * cer, 2), "CER_ci95_pct": [round(100 * np.percentile(bc, 2.5), 2), round(100 * np.percentile(bc, 97.5), 2)],
        "WER_vs_raw_transcription_pct": round(100 * wer_raw, 2), "CER_vs_raw_transcription_pct": round(100 * cer_raw, 2),
        "devanagari_script_rate_mean": round(float(dr.mean()), 4), "utterances_below_90pct_devanagari": int((dr < 0.9).sum()),
        "latency_ms": latency_summary(lat), "real_time_factor": latency_summary(rtf),
    },
    "api_errors": errors, "per_utterance": per_utt,
}
rows = [{k: v for k, v in u.items() if k not in ("ref", "hyp")} for u in per_utt]
write_results("T11_stt_fleurs_hi", payload, rows)
print(json.dumps({k: v for k, v in payload["results"].items()}, ensure_ascii=False, indent=1))
