# llm/inference.py
# LLM inference via Ollama — tuned for the Diffusion Models Expert use case.
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
MODEL_NAME      = "llama3.2-fast"   # change to "mistral" or "llama3" for better quality
REQUEST_TIMEOUT = 200           # seconds — LLM can be slow on CPU
MAX_TOKENS      = 256

# ---------------------------------------------------------------------------
# System prompt — diffusion model expert persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
    "You are an expert engineering assistant. If context is provided, use it to answer. "
    "If the user asks a general question or the context is empty, use your own knowledge "
    "to provide a helpful and creative response."
"""


# ---------------------------------------------------------------------------
# Public function — called by gpu_worker.py
# ---------------------------------------------------------------------------
def run_llm(query: str, context: str) -> str:
    """
    Send (user query + retrieved paper context) to the Ollama LLM
    and return the expert's answer.

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
    Combine the retrieved paper excerpts with the user's question.
    This is the core of RAG: the LLM sees the relevant context before answering.
    """
    return (
        "### Relevant Excerpts from Diffusion Model Research Papers:\n"
        f"{context}\n\n"
        "### Question:\n"
        f"{query}\n\n"
        "### Expert Answer:"
    )


def _call_ollama(prompt: str) -> str:
    """POST to Ollama /api/generate and return the model's text."""
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "num_ctx": 2048,
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
        f"[STUB RESPONSE{note}]\n"
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

    print("=== LLM Inference Self-Test (Diffusion Models Expert) ===\n")
    health = check_ollama_health()
    print(f"Ollama reachable          : {health['reachable']}")
    print(f"Model '{MODEL_NAME}' available : {health['model_available']}")
    print(f"Available models          : {health['models']}\n")

    sample_context = (
        "[Source: 2307.01952v1.pdf | Chunk 79]\n"
        "Classifier-free guidance is a technique to guide the iterative sampling "
        "process of a diffusion model towards a conditioning signal c by mixing the "
        "predictions of a conditional and an unconditional model: "
        "Dw(x; σ, c) = (1 + w)D(x; σ, c) − wD(x; σ), "
        "where w >= 0 is the guidance strength.\n\n"
        "[Source: 2112.10752v2.pdf | Chunk 3]\n"
        "Latent Diffusion Models (LDMs) apply the diffusion process in the latent space "
        "of a pretrained autoencoder rather than directly in pixel space. "
        "This reduces computational cost significantly while preserving image quality."
    )
    sample_query = "What is classifier-free guidance and how does it work?"

    print(f"Query:\n{sample_query}\n")
    answer = run_llm(sample_query, sample_context)
    print(f"Answer:\n{answer}")