# llm/inference.py
# LLM inference via Ollama — tuned for the College Advisor use case.
#
# ── One-time setup ───────────────────────────────────────────────────────────
#   1. Install Ollama:
#      Linux / macOS:  curl -fsSL https://ollama.com/install.sh | sh
#      Windows:        https://ollama.com/download
#
#   2. Pull a model (pick one):
#      ollama pull mistral       (~4 GB, good balance of speed + quality)
#      ollama pull llama3        (~5 GB, stronger reasoning)
#      ollama pull tinyllama     (~600 MB, fastest, for quick testing)
#
#   3. Ollama runs as a background service on port 11434 automatically.
#      Test it:  curl http://localhost:11434/api/tags
#
#   4. pip install requests
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME      = "mistral"    # change to "llama3" or "tinyllama" if needed
REQUEST_TIMEOUT = 120          # seconds — LLM can be slow on CPU
MAX_TOKENS      = 512          # longer answers for advisor context

# ---------------------------------------------------------------------------
# System prompt — defines the LLM's persona for every conversation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an official academic advisor for a university engineering faculty.
You have been given excerpts from the official college bylaws and regulations as context.

Your job:
- Answer the student's question accurately based ONLY on the provided context.
- If the context contains the answer, quote or paraphrase the relevant rule clearly.
- If the context does not contain enough information, say:
  "I could not find a clear answer in the bylaws provided. Please visit the academic affairs office for clarification."
- Be concise, professional, and helpful.
- Do not make up rules or policies that are not in the context.
- When referencing a rule, mention its source document so the student can verify it.
"""


# ---------------------------------------------------------------------------
# Public function — called by gpu_worker.py
# ---------------------------------------------------------------------------
def run_llm(query: str, context: str) -> str:
    """
    Send (student query + retrieved bylaw context) to the Ollama LLM
    and return the advisor's answer.

    Falls back to a stub response if Ollama is not running,
    so the rest of the distributed system keeps working.
    """
    if not _REQUESTS_OK:
        return _stub_response(query, context, reason="'requests' library missing")

    prompt = _build_prompt(query, context)

    try:
        return _call_ollama(prompt)
    except requests.exceptions.ConnectionError:
        logger.warning("[LLM] Ollama not reachable — using stub fallback.")
        return _stub_response(query, context, reason="Ollama not running")
    except requests.exceptions.Timeout:
        logger.warning("[LLM] Ollama timed out.")
        return _stub_response(query, context, reason="Ollama timeout")
    except Exception as exc:
        logger.error(f"[LLM] Unexpected error: {exc}")
        return _stub_response(query, context, reason=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_prompt(query: str, context: str) -> str:
    """
    Combine the retrieved bylaw excerpts with the student's question.
    This is the core of RAG: the LLM sees the relevant rules before answering.
    """
    return (
        "### Relevant Excerpts from the College Bylaws:\n"
        f"{context}\n\n"
        "### Student Question:\n"
        f"{query}\n\n"
        "### Advisor Answer:"
    )


def _call_ollama(prompt: str) -> str:
    """POST to Ollama /api/generate and return the model's text."""
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.2,   # low = more factual, less creative
            "top_p": 0.9,
        },
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _stub_response(query: str, context: str, reason: str = "") -> str:
    """
    Graceful degradation stub — returned when Ollama is unavailable.
    Clearly labelled so you know it's not a real LLM answer.
    """
    time.sleep(0.05)
    note = f" [{reason}]" if reason else ""
    preview = context[:120].replace("\n", " ")
    return (
        f"[STUB ADVISOR RESPONSE{note}]\n"
        f"Question : {query}\n"
        f"Context  : {preview}…\n"
        f"(Start Ollama and pull a model to get real answers)"
    )


# ---------------------------------------------------------------------------
# Health check — called by main.py at startup
# ---------------------------------------------------------------------------
def check_ollama_health() -> dict:
    result = {"reachable": False, "model_available": False, "models": []}
    if not _REQUESTS_OK:
        return result
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        result["reachable"]        = True
        result["models"]           = models
        result["model_available"]  = any(MODEL_NAME in m for m in models)
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Self-test:  python -m llm.inference
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=== LLM Inference Self-Test (College Advisor) ===\n")
    health = check_ollama_health()
    print(f"Ollama reachable      : {health['reachable']}")
    print(f"Model '{MODEL_NAME}' available : {health['model_available']}")
    print(f"Available models      : {health['models']}\n")

    sample_context = (
        "[Source: bylaws.pdf | Chunk 4]\n"
        "A student must complete a minimum of 136 credit hours to graduate. "
        "At least 60 of these must be upper-division courses. "
        "A minimum cumulative GPA of 2.0 is required for graduation.\n\n"
        "[Source: bylaws.pdf | Chunk 12]\n"
        "A student whose GPA falls below 2.0 for two consecutive semesters "
        "will be placed on academic probation. Continued failure may result "
        "in academic dismissal."
    )
    sample_query = "How many credit hours do I need to graduate?"

    print(f"Query:\n{sample_query}\n")
    answer = run_llm(sample_query, sample_context)
    print(f"Answer:\n{answer}")