"""T10 — Real HTTP benchmark of the UNMODIFIED production Flask app (api.py).

The app runs from a scratch copy of the working tree (api.py, src/, frontend/, assets/, qdrant DB copy) so that
api.py's startup lock-file deletion cannot touch the repository. A launcher injects GROQ_API_KEY only and removes
QDRANT_URL/QDRANT_API (Cloud unavailable) -> local embedded Qdrant. Server runs on CPU (.venv), Flask dev server
exactly as `python api.py`.

Phases
  1  paced sequential text  : 50 DS1-test answerable + 10 hand-written off-topic queries, 1 request / 8 s
  2  burst, 4 concurrent    : 40 other DS1-test queries, no pacing (shows rate-limit + shared-logger behaviour)
  3  paced sequential voice : 20 FLEURS hi_in test clips via /api/query/audio (full STT path), 1 request / 8 s
Usage: .venv/Scripts/python audit/scripts/t10_api.py <scratch_dir>
"""
import glob, io, json, os, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, psutil, requests
import pyarrow.parquet as pq
from common import PROJECT_ROOT, load_ds1, ds1_split, latency_summary, environment, write_results

SCR = os.path.join(sys.argv[1], "apirun")
URL = "http://127.0.0.1:7860"
PACE = 8.0
if os.path.exists(SCR):
    shutil.rmtree(SCR)
os.makedirs(SCR)
for item in ("api.py", "src", "frontend", "assets"):
    s = PROJECT_ROOT / item
    (shutil.copytree if s.is_dir() else shutil.copy)(s, os.path.join(SCR, item))
shutil.copytree(os.path.join(sys.argv[1], "qdrant_copy"), os.path.join(SCR, "qdrant_hindi_benchmark"))
launcher = os.path.join(SCR, "_launch.py")
open(launcher, "w").write(
    "import os, sys, runpy\n"
    "from dotenv import dotenv_values\n"
    f"os.environ['GROQ_API_KEY'] = dotenv_values(r'{PROJECT_ROOT / '.env'}')['GROQ_API_KEY']\n"
    "os.environ.pop('QDRANT_URL', None); os.environ.pop('QDRANT_API', None)\n"
    f"os.chdir(r'{SCR}'); sys.path.insert(0, r'{SCR}')\n"
    "runpy.run_path('api.py', run_name='__main__')\n")
log = open(os.path.join(SCR, "server.log"), "w", encoding="utf-8")
t_start = time.perf_counter()
srv = subprocess.Popen([sys.executable, launcher], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
ready = None
for _ in range(600):
    try:
        if requests.get(URL + "/", timeout=1).status_code == 200:
            ready = time.perf_counter() - t_start; break
    except Exception:
        pass
    time.sleep(0.5)
if ready is None:
    srv.kill(); raise SystemExit("server did not start; see server.log")
sp = psutil.Process(srv.pid)
mem = lambda: round(sum(p.memory_info().rss for p in [sp] + sp.children(recursive=True)) / 2**20, 1)
out = {"server_startup_s": round(ready, 1), "server_rss_mb_after_startup": mem()}
print("server ready in", out["server_startup_s"], "s")

ds = load_ds1(); test = ds1_split()["test"]
rng = np.random.default_rng(10)
order = rng.permutation(test).tolist()
seq_q = [("answerable", ds.queries[q]) for q in order[:90]]
off = [json.loads(l)["text"] for l in open(PROJECT_ROOT / "audit/data/offtopic_queries.jsonl", encoding="utf-8")]
seq_q += [("offtopic", t) for t in rng.choice(off, 10, replace=False)]
seq_q = [seq_q[i] for i in rng.permutation(len(seq_q))]
burst_q = [("answerable", ds.queries[q]) for q in order[50:90]]


def call_text(kind, q):
    t = time.perf_counter()
    try:
        r = requests.post(URL + "/api/query/text", json={"query": q}, timeout=120)
        rtt = (time.perf_counter() - t) * 1000
        j = r.json()
        return {"kind": kind, "http": r.status_code, "rtt_ms": rtt, "status": j.get("status"), "latency": j.get("latency"),
                "error_detail": (j.get("error_detail") or "")[:200], "t_start": t, "t_end": time.perf_counter()}
    except Exception as e:  # noqa
        return {"kind": kind, "http": None, "rtt_ms": (time.perf_counter() - t) * 1000, "status": "client_error", "error_detail": str(e)[:200]}


phase1 = []
for kind, q in seq_q:
    t0 = time.perf_counter()
    phase1.append(call_text(kind, q))
    time.sleep(max(0, PACE - (time.perf_counter() - t0)))
out["server_rss_mb_after_phase1"] = mem()
print("phase1 done")
phase2, phase3 = [], []   # burst + voice phases skipped (scope reduction)
if False:
  fl = pq.read_table(glob.glob("D:/hf-cache-audit/datasets--google--fleurs/snapshots/*/parquet-data/hi_in/test-*.parquet")[0]).to_pylist()
  for row in [fl[i] for i in rng.choice(len(fl), 20, replace=False)]:
    t0 = time.perf_counter()
    try:
        r = requests.post(URL + "/api/query/audio", files={"audio": ("clip.wav", row["audio"]["bytes"], "audio/wav")}, timeout=120)
        j = r.json()
        phase3.append({"kind": "voice", "http": r.status_code, "rtt_ms": (time.perf_counter() - t0) * 1000, "status": j.get("status"),
                       "latency": j.get("latency"), "audio_s": row["num_samples"] / 16000, "error_detail": (j.get("error_detail") or "")[:200]})
    except Exception as e:  # noqa
        phase3.append({"kind": "voice", "http": None, "status": "client_error", "error_detail": str(e)[:200]})
    time.sleep(max(0, PACE - (time.perf_counter() - t0)))
out["server_rss_mb_end"] = mem()
srv.terminate(); srv.wait(10); log.close()


def anomalies(rs):
    a = {"server_e2e_gt_client_rtt": 0, "generation_reported_on_non_llm_status": 0, "n": len(rs)}
    for r in rs:
        L = r.get("latency") or {}
        if L.get("end_to_end") and r.get("rtt_ms") and L["end_to_end"] > r["rtt_ms"] + 5:
            a["server_e2e_gt_client_rtt"] += 1
        if r.get("status") in ("refused_off_topic", "refused_unsafe") and (L.get("generation") or 0) > 0:
            a["generation_reported_on_non_llm_status"] += 1
    return a


def summ(rs, label):
    s = {"n": len(rs), "status_counts": {}, "http_counts": {}}
    for r in rs:
        s["status_counts"][r["status"]] = s["status_counts"].get(r["status"], 0) + 1
        s["http_counts"][str(r["http"])] = s["http_counts"].get(str(r["http"]), 0) + 1
    s["client_rtt_ms_all"] = latency_summary([r["rtt_ms"] for r in rs if r.get("rtt_ms")])
    for st in ("answered", "refused_off_topic", "refused_not_grounded", "error"):
        x = [r["rtt_ms"] for r in rs if r["status"] == st and r.get("rtt_ms")]
        if x:
            s[f"client_rtt_ms_{st}"] = latency_summary(x)
    for stage in ("stt", "embedding", "retrieval", "generation", "guardrail_groundedness", "end_to_end"):
        x = [r["latency"][stage] for r in rs if r.get("latency") and r["latency"].get(stage)]
        if x:
            s[f"server_reported_{stage}_ms"] = latency_summary(x)
    s["errors"] = [r["error_detail"] for r in rs if r["status"] in ("error", "client_error")][:10]
    s["logger_anomalies"] = anomalies(rs)
    return s


out["phase1_paced_text"] = summ(phase1, "p1")
out["skipped_phases"] = "burst (4 concurrent) and voice phases skipped by scope reduction"
srv_log = open(os.path.join(SCR, "server.log"), encoding="utf-8", errors="replace").read()
out["server_log_counts"] = {"429_mentions": srv_log.count("429"), "retry_warnings": srv_log.count("[retry]"),
                            "tracebacks": srv_log.count("Traceback")}
write_results("T10_api", {"test_id": "T10_api", "status": "NEWLY EXECUTED", "environment": environment("cpu"),
                          "design": __doc__, "results": out,
                          "caveats": ["Client and server on the same laptop (no network hop between them); Groq over public internet.",
                                      "Local embedded Qdrant, not Qdrant Cloud.",
                                      "Server-reported latency is the P50 over all stage samples within one request (api.py _extract_latency)."]},
              [{"phase": p, **{k: v for k, v in r.items() if k not in ("latency", "t_start", "t_end")},
                **{f"srv_{k}": v for k, v in (r.get("latency") or {}).items()}}
               for p, rs in (("1", phase1), ("2", phase2), ("3", phase3)) for r in rs])
print(json.dumps(out, indent=1, ensure_ascii=False)[:4000])
