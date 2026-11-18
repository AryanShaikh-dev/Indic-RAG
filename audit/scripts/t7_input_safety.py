"""T7 — Input-safety guardrail evaluation.

Systems:
  regex_production : src.core.guardrails.check_input_safety (unchanged production code)
  prompt_guard_86m : Groq meta-llama/llama-prompt-guard-2-86m (comparison only; not in production)

Dataset: audit/data/safety_prompts.jsonl — 50 attack + 50 benign look-alike prompts in
English / Hindi (Devanagari) / Hinglish, written by the auditor (Claude). Author bias caveat applies.
Usage: python audit/scripts/t7_input_safety.py
"""
import json, os, time
from collections import defaultdict
import numpy as np
from common import PROJECT_ROOT, environment, write_results, bootstrap_ci
from src.core.guardrails import check_input_safety

data = [json.loads(l) for l in open(PROJECT_ROOT / "audit/data/safety_prompts.jsonl", encoding="utf-8")]

# ---------------- regex (production)
for d in data:
    t = time.perf_counter()
    safe, reason = check_input_safety(d["text"])
    d["regex_blocked"] = not safe
    d["regex_reason"] = reason
    d["regex_ms"] = (time.perf_counter() - t) * 1000

# ---------------- Groq prompt guard (comparison)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
import groq
client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
api_errors = []
for d in data:
    for attempt in range(3):
        try:
            t = time.perf_counter()
            r = client.chat.completions.create(model="meta-llama/llama-prompt-guard-2-86m",
                                               messages=[{"role": "user", "content": d["text"]}])
            d["pg_ms"] = (time.perf_counter() - t) * 1000
            out = (r.choices[0].message.content or "").strip()
            d["pg_raw"] = out
            try:
                score = float(out)
                d["pg_score"] = score
                d["pg_blocked"] = score >= 0.5
            except ValueError:
                d["pg_blocked"] = out.upper().startswith(("MALICIOUS", "JAILBREAK", "INJECTION", "UNSAFE"))
            break
        except groq.RateLimitError as e:
            api_errors.append({"id": d["id"], "error": "429", "detail": str(e)[:200]})
            time.sleep(float(getattr(e, "response", None).headers.get("retry-after", 5)) if getattr(e, "response", None) else 5)
        except Exception as e:  # noqa
            api_errors.append({"id": d["id"], "error": type(e).__name__, "detail": str(e)[:200]})
            break


def rates(sys_key, subset):
    att = [d for d in subset if d["label"] == "attack" and sys_key in d]
    ben = [d for d in subset if d["label"] == "benign" and sys_key in d]
    a = np.array([d[sys_key] for d in att], dtype=float)
    b = np.array([d[sys_key] for d in ben], dtype=float)
    r = {"n_attack": len(att), "n_benign": len(ben)}
    if len(a):
        r["attack_block_rate_pct"] = round(100 * a.mean(), 1)
        r["attack_block_ci95"] = [round(100 * x, 1) for x in bootstrap_ci(a)]
    if len(b):
        r["benign_false_block_rate_pct"] = round(100 * b.mean(), 1)
        r["benign_false_block_ci95"] = [round(100 * x, 1) for x in bootstrap_ci(b)]
    return r


summary, rows = {}, []
for sysk, name in [("regex_blocked", "regex_production"), ("pg_blocked", "prompt_guard_86m")]:
    summary[name] = {"overall": rates(sysk, data)}
    for lang in ["en", "hi", "hinglish"]:
        summary[name][f"lang={lang}"] = rates(sysk, [d for d in data if d["lang"] == lang])
    for cat in sorted({d["category"] for d in data if d["label"] == "attack"}):
        summary[name][f"attack_category={cat}"] = rates(sysk, [d for d in data if d["category"] == cat])
    for k, v in summary[name].items():
        rows.append({"system": name, "slice": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, list)}})

pg_lat = [d["pg_ms"] for d in data if "pg_ms" in d]
payload = {
    "test_id": "T7_input_safety",
    "status": "NEWLY EXECUTED",
    "environment": environment("cpu"),
    "dataset": {"file": "audit/data/safety_prompts.jsonl", "n": len(data), "attack": 50, "benign": 50,
                "langs": {"en": sum(d["lang"] == "en" for d in data), "hi": sum(d["lang"] == "hi" for d in data),
                          "hinglish": sum(d["lang"] == "hinglish" for d in data)},
                "caveat": "Prompts authored by the auditor (an AI), not sampled from real traffic; small n -> wide CIs."},
    "summary": summary,
    "regex_latency_ms_median": float(np.median([d["regex_ms"] for d in data])),
    "prompt_guard_latency_ms": {"n": len(pg_lat), "median": float(np.median(pg_lat)) if pg_lat else None},
    "prompt_guard_raw_output_examples": sorted({d.get("pg_raw", "") for d in data})[:5],
    "api_errors": api_errors,
    "per_prompt": data,
}
write_results("T7_input_safety", payload, rows)
for name in summary:
    print(name, json.dumps(summary[name]["overall"]), {k: v.get("attack_block_rate_pct") for k, v in summary[name].items() if k.startswith("lang")})
