# workers/gpu_worker.py
# Each Worker is a standalone FastAPI HTTP server.
# Run one instance per machine / VM:
#   uvicorn workers.gpu_worker:app --host 0.0.0.0 --port 8001
#
# Multiple workers run on DIFFERENT ports (or different machines):
#   Worker 0 → port 8001
#   Worker 1 → port 8002
#   Worker 2 → port 8003  ... etc.
#
# The Master's Load Balancer knows the addresses of all workers and
# routes tasks to them over HTTP.

import asyncio
import time
import logging
import os
from uuid import uuid4

import httpx
import psutil
from fastapi import FastAPI

from common.models import LB_To_Worker, Worker_To_Master, Worker_Heartbeat
from llm.inference import run_llm
from rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

# ── Optional NVIDIA GPU monitoring ───────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    pynvml   = None
    _NVML_OK = False
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Configuration  (override with environment variables when deploying)
# ---------------------------------------------------------------------------
MY_NODE_ID  = str(uuid4())                              # unique ID for this worker
MASTER_URL  = os.getenv("MASTER_URL", "http://localhost:8000")   # where to send heartbeats
WORKER_PORT = int(os.getenv("WORKER_PORT", 8001))       # this worker's own port
HEARTBEAT_INTERVAL = 10                                  # seconds between heartbeats

# ---------------------------------------------------------------------------
# In-flight task counter  (used for Least-Connections load balancing)
# ---------------------------------------------------------------------------
active_tasks_counter: int = 0


# ---------------------------------------------------------------------------
# Hardware monitoring helpers
# ---------------------------------------------------------------------------
def get_gpu_vram_free() -> float:
    """Return free VRAM in GB for GPU 0, or 0.0 if no NVIDIA GPU."""
    if _NVML_OK:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free / (1024 ** 3)
        except Exception:
            pass
    return 0.0


def get_worker_status() -> str:
    """Simple status string based on current load."""
    if active_tasks_counter == 0:
        return "ready"
    if active_tasks_counter < 5:
        return "busy"
    return "overloaded"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title=f"GPU Worker {MY_NODE_ID[:8]}")


# ── Health check (Load Balancer pings this to verify the worker is alive) ───
@app.get("/health")
async def health_check():
    return {
        "node_id":      MY_NODE_ID,
        "status":       get_worker_status(),
        "active_tasks": active_tasks_counter,
        "cpu_percent":  psutil.cpu_percent(),
        "vram_free_gb": get_gpu_vram_free(),
    }


# ── Main inference endpoint ──────────────────────────────────────────────────
@app.post("/generate", response_model=Worker_To_Master)
async def generate_task(data: LB_To_Worker):
    """
    Receives a task from the Load Balancer, runs RAG + LLM, returns result.

    Flow:
      1. Load Balancer POSTs a LB_To_Worker payload here
      2. Worker runs retrieve_context(query) → RAG retrieval
      3. Worker runs run_llm(query, context)  → LLM inference
      4. Worker returns Worker_To_Master with the answer + timing metadata
      5. Master saves the result and returns it to the original user
    """
    global active_tasks_counter
    active_tasks_counter += 1

    worker_received_at = time.time()
    inference_start    = time.time()

    logger.info(
        f"[Worker {MY_NODE_ID[:8]}] Received task {data.task_id} | "
        f"active={active_tasks_counter}"
    )

    status        = "success"
    error_message = None
    response_text = ""

    try:
        # ── Step 1: RAG — retrieve relevant bylaw excerpts ────────────────
        context = retrieve_context(data.instruction)

        # ── Step 2: LLM — generate the advisor answer ─────────────────────
        # run_llm is synchronous (blocking). We run it in a thread pool so
        # it doesn't block the entire async event loop while other requests
        # are waiting — this is critical for handling concurrent load.
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            None, run_llm, data.instruction, context
        )

    except Exception as exc:
        logger.error(f"[Worker {MY_NODE_ID[:8]}] Task {data.task_id} failed: {exc}")
        status        = "error"
        error_message = str(exc)
        response_text = f"[Worker error] {exc}"

    finally:
        active_tasks_counter -= 1

    inference_end = time.time()
    logger.info(
        f"[Worker {MY_NODE_ID[:8]}] Completed task {data.task_id} | "
        f"latency={inference_end - inference_start:.3f}s | active={active_tasks_counter}"
    )

    return Worker_To_Master(
        task_id            = data.task_id,
        worker_id          = MY_NODE_ID,
        response_text      = response_text,
        worker_received_at = worker_received_at,
        inference_start    = inference_start,
        inference_end      = inference_end,
        status             = status,
        error_message      = error_message,
    )


# ── Background heartbeat loop ────────────────────────────────────────────────
async def heartbeat_loop():
    """
    Runs forever in the background after the worker starts.
    Every HEARTBEAT_INTERVAL seconds it POSTs current stats to the Master.
    The Master's Load Balancer uses these stats for routing decisions:
      - current_load_count  → Least Connections algorithm
      - gpu_vram_free       → VRAM-aware routing
      - cpu_usage_percent   → Load-aware routing
    If this POST fails (Master is down) we just log and keep running —
    the worker continues serving tasks even without heartbeat delivery.
    """
    async with httpx.AsyncClient() as client:
        while True:
            heartbeat = Worker_Heartbeat(
                node_id            = MY_NODE_ID,
                status             = get_worker_status(),
                current_load_count = active_tasks_counter,
                cpu_usage_percent  = psutil.cpu_percent(interval=None),
                gpu_vram_free      = get_gpu_vram_free(),
                last_seen          = time.time(),
            )

            try:
                await client.post(
                    f"{MASTER_URL}/heartbeat",
                    json=heartbeat.model_dump(),
                    timeout=5.0,
                )
                logger.debug(
                    f"[Worker {MY_NODE_ID[:8]}] Heartbeat sent → "
                    f"load={active_tasks_counter} cpu={heartbeat.cpu_usage_percent:.1f}%"
                )
            except Exception as e:
                logger.warning(f"[Worker {MY_NODE_ID[:8]}] Heartbeat failed: {e}")

            await asyncio.sleep(HEARTBEAT_INTERVAL)


@app.on_event("startup")
async def startup_event():
    """Called automatically by uvicorn when the server starts."""
    logger.info(f"[Worker {MY_NODE_ID[:8]}] Starting up on port {WORKER_PORT}")
    logger.info(f"[Worker {MY_NODE_ID[:8]}] Will send heartbeats to: {MASTER_URL}")
    asyncio.create_task(heartbeat_loop())


# ---------------------------------------------------------------------------
# Entry point for running directly:  python workers/gpu_worker.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run("workers.gpu_worker:app", host="0.0.0.0", port=WORKER_PORT, reload=False)