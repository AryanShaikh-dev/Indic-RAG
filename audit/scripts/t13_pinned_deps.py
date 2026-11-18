"""T13 — Dependency compatibility of requirements.txt (the file the Dockerfile installs).

Run with the Python 3.10 venv built from requirements.txt EXACTLY (matches Dockerfile's python:3.10-slim
minus the OS): D:/indic-rag-pinned-venv/Scripts/python audit/scripts/t13_pinned_deps.py <qdrant_copy_dir>
Exercises the unmodified production classes used by api.py, one step at a time.
"""
import json, os, sys, time, traceback, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
qdrant_copy = sys.argv[1]
from dotenv import dotenv_values
os.environ["GROQ_API_KEY"] = dotenv_values(ROOT / ".env")["GROQ_API_KEY"]   # never printed
os.environ.pop("QDRANT_URL", None); os.environ.pop("QDRANT_API", None)       # Cloud unavailable -> local mode

steps, ctx = [], {}


def step(name, fn):
    t = time.perf_counter()
    try:
        out = fn()
        steps.append({"step": name, "ok": True, "ms": round((time.perf_counter() - t) * 1000), "detail": str(out)[:200]})
    except BaseException as e:  # noqa
        steps.append({"step": name, "ok": False, "ms": round((time.perf_counter() - t) * 1000),
                      "error_type": type(e).__name__, "error": str(e)[:300],
                      "where": traceback.extract_tb(e.__traceback__)[-1].filename.split("site-packages")[-1] + ":" +
                               str(traceback.extract_tb(e.__traceback__)[-1].lineno)})
    print(steps[-1])


def imports():
    from src.core.stt import GroqSTT
    from src.core.embeddings import EmbeddingModel
    from src.core.vectorstore import QdrantVectorStore
    from src.core.llm import GroqLLMBackend
    from src.core.orchestrator import VoiceRAGOrchestrator
    ctx.update(locals()); return "imported"


step("import production modules", imports)
step("EmbeddingModel load + encode", lambda: ctx.setdefault("emb", ctx["EmbeddingModel"]("intfloat/multilingual-e5-small")).embed(["query: परीक्षण"], "")[0][:3])


def vs():
    db = ctx["QdrantVectorStore"](collection_name="hindi_rag_production", persist_directory=qdrant_copy, vector_dim=384,
                                  query_prefix="query: ", passage_prefix="passage: ")
    db.embedding_model = ctx["emb"]; ctx["db"] = db
    return db.client.count("hindi_rag_production")


step("QdrantVectorStore open local collection", vs)
step("QdrantVectorStore.query (retrieval)", lambda: ctx["db"].query("काउंटी मार्शल क्या करते हैं", k=2, with_scores=True)[0]["score"])
step("GroqLLMBackend init (gpt-oss-20b, reasoning=low, max_tokens=150)",
     lambda: ctx.setdefault("llm", ctx["GroqLLMBackend"](model_name="openai/gpt-oss-20b", reasoning_effort="low", max_completion_tokens=150)) and "ok")
step("GroqLLMBackend.generate (real API call)", lambda: ctx["llm"].generate("Say OK.", "You are helpful.")["text"][:50])
step("GroqSTT.transcribe sample_record.wav (real API call)", lambda: ctx["GroqSTT"]().transcribe(str(ROOT / "sample_record.wav"))[:60])

from importlib.metadata import version
res = {"test_id": "T13_dependency_compat", "status": "NEWLY EXECUTED",
       "method": "Python 3.10 venv with requirements.txt installed exactly (uv pip install -r requirements.txt); unmodified src/ classes as wired in api.py. "
                 "Docker build itself NOT executed (Docker daemon not running; C: has 8.4 GB free).",
       "python": platform.python_version(), "platform": platform.platform(),
       "resolved": {p: version(p) for p in ["torch", "sentence-transformers", "transformers", "huggingface-hub", "qdrant-client", "groq", "httpx", "numpy", "flask"]},
       "steps": steps, "all_passed": all(s["ok"] for s in steps)}
json.dump(res, open(ROOT / "audit/results/T13_dependency_compat.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("ALL PASSED" if res["all_passed"] else "FAILURES PRESENT")
