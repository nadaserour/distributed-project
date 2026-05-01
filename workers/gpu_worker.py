# workers/gpu_worker.py
# Updated GPU Worker – uses real RAG retrieval + real LLM inference.
# Replaces the skeleton version which only had stub imports.

import time
import threading
import logging

from llm.inference import run_llm, check_ollama_health
from rag.retriever import retrieve_context

logger = logging.getLogger(__name__)


class GPUWorker:
    """
    Simulates a GPU worker node in the distributed cluster.

    Each worker:
      1. Receives a Request from the load balancer / scheduler.
      2. Calls the RAG retriever to fetch relevant context.
      3. Calls the LLM with (query + context) to generate an answer.
      4. Returns a response dict with id, result, and latency.

    The 'active_requests' counter is used by Least-Connections load balancing.
    The 'alive' flag supports fault-tolerance simulation (set to False to
    simulate a node failure).
    """

    def __init__(self, node_id: int):
        self.id              = node_id
        self.active_requests = 0        # used by least-connections balancer
        self.alive           = True     # set False to simulate node failure
        self._lock           = threading.Lock()
        self.total_processed = 0
        self.total_latency   = 0.0

    # ------------------------------------------------------------------
    def process(self, request) -> dict:
        """
        Main processing pipeline:  RAG  →  LLM  →  Response
        Thread-safe: multiple threads can call this simultaneously.
        """
        if not self.alive:
            raise RuntimeError(f"Worker {self.id} is DOWN – cannot process request.")

        with self._lock:
            self.active_requests += 1

        start = time.time()
        logger.info(f"[Worker {self.id}] Processing request {request.id}: '{request.query}'")

        try:
            # ── Step 1: RAG – retrieve relevant context ──────────────────
            context = retrieve_context(request.query)
            logger.debug(f"[Worker {self.id}] RAG context for req {request.id}:\n{context}")

            # ── Step 2: LLM – generate answer with context ────────────────
            result = run_llm(request.query, context)
            logger.debug(f"[Worker {self.id}] LLM answer for req {request.id}: {result[:80]}…")

        except Exception as exc:
            logger.error(f"[Worker {self.id}] Error processing request {request.id}: {exc}")
            result  = f"[ERROR] Worker {self.id} failed: {exc}"
            context = ""

        finally:
            with self._lock:
                self.active_requests -= 1

        latency = time.time() - start

        with self._lock:
            self.total_processed += 1
            self.total_latency   += latency

        logger.info(
            f"[Worker {self.id}] Completed request {request.id} | "
            f"latency={latency:.3f}s | active={self.active_requests}"
        )

        return {
            "id":      request.id,
            "result":  result,
            "latency": latency,
            "worker":  self.id,
        }

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return runtime statistics for this worker (useful for monitoring)."""
        avg_lat = (
            self.total_latency / self.total_processed
            if self.total_processed > 0 else 0.0
        )
        return {
            "worker_id":       self.id,
            "alive":           self.alive,
            "active_requests": self.active_requests,
            "total_processed": self.total_processed,
            "avg_latency":     round(avg_lat, 4),
        }

    # ------------------------------------------------------------------
    def simulate_failure(self):
        """Mark this worker as dead (for fault-tolerance testing)."""
        self.alive = False
        logger.warning(f"[Worker {self.id}] *** NODE FAILURE SIMULATED ***")

    def recover(self):
        """Bring a failed worker back online."""
        self.alive = True
        logger.info(f"[Worker {self.id}] Node recovered and back online.")


# ---------------------------------------------------------------------------
# Startup health check – called once from main.py
# ---------------------------------------------------------------------------
def print_startup_banner():
    health = check_ollama_health()
    print("\n" + "="*60)
    print("  GPU Worker Node – LLM + RAG Startup Check")
    print("="*60)
    if health["reachable"]:
        status = "✓  REAL inference" if health["model_available"] \
                 else "⚠  Ollama running but model not found – stub fallback"
    else:
        status = "✗  Ollama not running – stub fallback active"
    print(f"  LLM backend : {status}")
    print(f"  RAG backend : FAISS semantic search (sentence-transformers)")
    print("="*60 + "\n")