# llm/inference.py
#
# LLM inference via vLLM (OpenAI-compatible API).
#
# ── Setup ───────────────────────────────────────────────────────────────────
#   1. Install vLLM:
#      pip install vllm
#
#   2. Start the server:
#      python -m vllm.entrypoints.openai.api_server \
#          --model Qwen/Qwen2.5-14B-Instruct \
#          --port 5000 --dtype auto --max-model-len 4096
#
#   3. Test it:
#      curl http://localhost:5000/v1/models
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_BASE_URL   = "http://localhost:8000/v1"
MODEL_NAME      = "Qwen/Qwen2.5-0.5B-Instruct"
REQUEST_TIMEOUT = 120
MAX_TOKENS      = 512

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert assistant specialized in diffusion models and generative AI research. "
    "You have been given excerpts from research papers on diffusion models as context.\n\n"
    "Your job:\n"
    "- Answer the user's question accurately based ONLY on the provided context.\n"
    "- If the context contains the answer, explain it clearly and technically.\n"
    "- If the context does not contain enough information, say: "
    "\"I could not find that information in the provided papers.\"\n"
    "- Be precise and technical. You are talking to researchers or ML engineers.\n"
    "- When referencing a concept, mention its source paper so the user can verify it.\n"
    "- Do not make up facts or numbers not present in the context."
)

# ---------------------------------------------------------------------------
# vLLM client (lazy-init)
# ---------------------------------------------------------------------------
_client: "OpenAI | None" = None

def _get_client() -> "OpenAI":
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed",
        )
    return _client


# ---------------------------------------------------------------------------
# Public function — called by gpu_worker.py
# ---------------------------------------------------------------------------
def run_llm(query: str, context: str) -> str:
    if not _OPENAI_OK:
        return _stub_response(query, context, reason="'openai' library missing")

    prompt = _build_prompt(query, context)

    try:
        return _call_vllm(prompt)
    except Exception as exc:
        reason = str(exc)
        if "Connection" in reason or "refused" in reason:
            logger.warning("[LLM] vLLM not reachable — using stub fallback.")
            return _stub_response(query, context, reason="vLLM not running")
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            logger.warning("[LLM] vLLM timed out.")
            return _stub_response(query, context, reason="vLLM timeout")
        logger.error(f"[LLM] Unexpected error: {exc}")
        return _stub_response(query, context, reason=reason)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_prompt(query: str, context: str) -> str:
    return (
        "### Relevant Excerpts from Diffusion Model Research Papers:\n"
        f"{context}\n\n"
        "### Question:\n"
        f"{query}\n\n"
        "### Expert Answer:"
    )


def _call_vllm(prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        top_p=0.9,
        timeout=REQUEST_TIMEOUT,
    )
    return response.choices[0].message.content.strip()


def _stub_response(query: str, context: str, reason: str = "") -> str:
    time.sleep(0.05)
    note = f" [{reason}]" if reason else ""
    preview = context[:120].replace("\n", " ")
    return (
        f"[STUB RESPONSE{note}]\n"
        f"Question : {query}\n"
        f"Context  : {preview}…\n"
        f"(Start vLLM to get real answers)"
    )


# ---------------------------------------------------------------------------
# Health check — called by main.py at startup
# ---------------------------------------------------------------------------
def check_ollama_health() -> dict:
    result = {"reachable": False, "model_available": False, "models": []}
    if not _REQUESTS_OK:
        return result
    try:
        resp = requests.get(f"http://localhost:5000/v1/models", timeout=5)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        result["reachable"] = True
        result["models"] = models
        result["model_available"] = any(MODEL_NAME in m for m in models)
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Self-test:  python -m llm.inference
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=== LLM Inference Self-Test (vLLM) ===\n")
    health = check_ollama_health()
    print(f"vLLM reachable             : {health['reachable']}")
    print(f"Model '{MODEL_NAME}' available : {health['model_available']}")
    print(f"Available models           : {health['models']}\n")

    sample_context = (
        "[Source: 2307.01952v1.pdf | Chunk 79]\n"
        "Classifier-free guidance is a technique to guide the iterative sampling "
        "process of a diffusion model towards a conditioning signal c by mixing the "
        "predictions of a conditional and an unconditional model.\n\n"
        "[Source: 2112.10752v2.pdf | Chunk 3]\n"
        "Latent Diffusion Models (LDMs) apply the diffusion process in the latent space "
        "of a pretrained autoencoder rather than directly in pixel space."
    )
    sample_query = "What is classifier-free guidance and how does it work?"

    print(f"Query:\n{sample_query}\n")
    answer = run_llm(sample_query, sample_context)
    print(f"Answer:\n{answer}")
