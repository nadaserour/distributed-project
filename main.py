# main.py — College Advisor distributed system entry point

import sys
import logging
import time
from pathlib import Path

# Configure logging before any project imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

from workers.gpu_worker    import GPUWorker
from lb.load_balancer      import LoadBalancer
from master.scheduler      import Scheduler
from client.load_generator import run_load_test
from llm.inference         import check_ollama_health
from rag.retriever         import list_indexed_sources


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------
def startup_checks() -> bool:
    """
    Verify that the RAG index exists and Ollama is reachable before
    starting the load test.  Returns False if a critical check fails.
    """
    print("\n" + "="*60)
    print("  College Advisor — Distributed System Startup")
    print("="*60)

    all_ok = True

    # ── Check 1: RAG index ────────────────────────────────────────────────
    index_path  = Path("rag/index/faiss.index")
    chunks_path = Path("rag/index/chunks.pkl")

    if index_path.exists() and chunks_path.exists():
        try:
            sources = list_indexed_sources()
            print(f"\n  [✓] RAG index found")
            print(f"      Indexed documents ({len(sources)}):")
            for s in sources:
                print(f"        • {s}")
        except Exception as e:
            print(f"\n  [✗] RAG index exists but failed to load: {e}")
            all_ok = False
    else:
        print("\n  [✗] RAG index NOT found!")
        print("      Build it first by running:")
        print("        python -m rag.build_index")
        print("      Then drop your bylaws PDFs into:  rag/docs/")
        all_ok = False

    # ── Check 2: Ollama / LLM ─────────────────────────────────────────────
    health = check_ollama_health()
    if health["reachable"] and health["model_available"]:
        print(f"\n  [✓] Ollama running — model ready")
        print(f"      Available models: {health['models']}")
    elif health["reachable"]:
        print(f"\n  [!] Ollama running but model not found — stub fallback active")
        print(f"      Pull a model:  ollama pull mistral")
        print(f"      Available    : {health['models']}")
    else:
        print("\n  [!] Ollama not running — stub fallback active")
        print("      For real LLM answers: install Ollama and run  ollama serve")

    print("\n" + "="*60)
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ok = startup_checks()

    if not ok:
        print("\n[!] Critical startup check failed. Exiting.")
        print("    Fix the issues above, then re-run:  python main.py\n")
        sys.exit(1)

    # ── Create GPU worker nodes ───────────────────────────────────────────
    NUM_WORKERS = 4     # simulate 4 GPU nodes
    workers = [GPUWorker(i) for i in range(NUM_WORKERS)]
    print(f"\n[Main] {NUM_WORKERS} GPU worker nodes created.")

    # ── Load balancer ─────────────────────────────────────────────────────
    # Strategy options: "round_robin" | "least_connections" | "load_aware"
    lb = LoadBalancer(workers, strategy="round_robin")
    print(f"[Main] Load balancer ready  (strategy: round_robin)")

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = Scheduler(lb)

    # ── Run load test ─────────────────────────────────────────────────────
    # Start small (10–20) to verify the full pipeline works end-to-end,
    # then scale up to 1000 for the actual performance evaluation.
    NUM_USERS = 20
    print(f"[Main] Starting load test: {NUM_USERS} concurrent users\n")

    t0 = time.time()
    run_load_test(scheduler, num_users=NUM_USERS)
    elapsed = time.time() - t0

    # ── Per-worker statistics ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Worker Statistics")
    print("="*60)
    total_requests = 0
    for w in workers:
        s = w.stats()
        total_requests += s["total_processed"]
        print(
            f"  Worker {s['worker_id']}  |  "
            f"processed={s['total_processed']:>4}  |  "
            f"avg_latency={s['avg_latency']:.3f}s  |  "
            f"alive={s['alive']}"
        )
    print(f"\n  Total requests handled : {total_requests}")
    print(f"  Total wall-clock time  : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"  Throughput             : {total_requests/elapsed:.1f} req/s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()