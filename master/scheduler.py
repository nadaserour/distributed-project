# master/scheduler.py
#
# Gateway / Master Node — the single entry point for all client traffic.
#
# Responsibilities (per architecture spec):
#   1. User Management     — API-key validation on every request.
#   2. Admission Control   — Rejects new requests when the LB queue is full
#                            (backpressure), preventing system overload.
#   3. Data Logging        — Writes per-request timing rows to a CSV file
#                            and logs them for live monitoring.
#   4. Result Persistence  — Caches LLM answers in memory (+ CSV) so a
#                            brief connection flicker doesn't lose a result.
#   5. Worker Registry     — Accepts worker registrations and heartbeats,
#                            forwards them to the Load Balancer.
#   6. Admin Visibility    — Exposes /admin/stats for dashboards.
#
# The Master is a FastAPI application.  Launch with:
#   uvicorn master.scheduler:app --host 0.0.0.0 --port 8000

import asyncio
import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Header, Request # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from pydantic import BaseModel # type: ignore

from common.models import (
    Final_Response,
    LB_To_Worker,
    Master_Message_To_LB,
    User_Request,
    Worker_Heartbeat,
)
from lb.load_balancer import LoadBalancer
from fault_tolerance.fault_handler import FaultHandler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("master")

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
VALID_API_KEYS: set[str] = set(
    os.getenv("API_KEYS", "dev-key-1,dev-key-2,test-key").split(",")
)
LOG_DIR   = Path(os.getenv("LOG_DIR",  "logs"))
LOG_CSV   = LOG_DIR / "request_log.csv"
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "500"))   # must match LB's QUEUE_MAXSIZE

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CSV log setup — write header only if the file is new
# ---------------------------------------------------------------------------
_CSV_HEADER = [
    "request_id", "user_id", "query_snippet",
    "user_sent_at", "master_received_at", "dispatched_at",
    "worker_id", "provider", "model_used",
    "worker_received_at", "inference_start", "inference_end",
    "master_responded_at", "total_latency_s", "inference_latency_s",
    "status",
]

def _ensure_csv_header() -> None:
    if not LOG_CSV.exists() or LOG_CSV.stat().st_size == 0:
        with LOG_CSV.open("w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)

_ensure_csv_header()


def _append_csv_row(row: dict) -> None:
    with LOG_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_HEADER, extrasaction="ignore")
        w.writerow(row)


# ---------------------------------------------------------------------------
# In-memory result cache (request_id → Final_Response)
# Simple LRU-ish: evict oldest when size exceeds limit.
# ---------------------------------------------------------------------------
_CACHE_MAX = 5_000
_result_cache: dict[str, Final_Response] = {}

def _cache_result(response: Final_Response) -> None:
    if len(_result_cache) >= _CACHE_MAX:
        oldest_key = next(iter(_result_cache))
        del _result_cache[oldest_key]
    _result_cache[response.request_id] = response


# ---------------------------------------------------------------------------
# FastAPI app + shared LB instance
# ---------------------------------------------------------------------------
app = FastAPI(title="CSE354 — Master Scheduler", version="1.0.0")
lb  = LoadBalancer()
fh  = FaultHandler(lb)


@app.on_event("startup")
async def _startup() -> None:
    lb.start()
    fh.start()
    logger.info("[Master] Started — Load Balancer and Fault Handler are live.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    lb.stop()
    fh.stop()
    logger.info("[Master] Shutting down.")


# ---------------------------------------------------------------------------
# Pydantic request/response bodies for FastAPI
# ---------------------------------------------------------------------------
class WorkerRegistrationRequest(BaseModel):
    node_id: str   # UUID as string
    url: str       # e.g. "http://192.168.1.5:8001"


# ---------------------------------------------------------------------------
# Helper: API-key guard
# ---------------------------------------------------------------------------
def _require_api_key(x_api_key: Optional[str]) -> None:
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# 1.  /query  — main user-facing endpoint
# ---------------------------------------------------------------------------
@app.post("/query", response_model=dict)
async def handle_query(
    user_request: User_Request,
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    """
    Accepts a user query, routes it through the LB to a GPU worker,
    persists the result, and returns the answer.

    Returns 429 if the system is overloaded (backpressure).
    Returns 401 if the API key is missing or invalid.
    """
    # ── 1. Auth ──────────────────────────────────────────────────────────
    _require_api_key(x_api_key)

    master_received_at = time.time()
    request_id = str(uuid4())

    # ── 2. Admission Control ─────────────────────────────────────────────
    if lb.queue_depth >= MAX_QUEUE:
        logger.warning(
            f"[Master] Backpressure — queue depth {lb.queue_depth}/{MAX_QUEUE}. "
            f"Rejecting request {request_id}."
        )
        raise HTTPException(
            status_code=429,
            detail="System is under heavy load. Please try again shortly.",
        )

    logger.info(
        f"[Master] → Request {request_id} from user '{user_request.user_id}' | "
        f"query: '{user_request.query[:60]}…'"
    )

    # ── 3. Build the internal task for the LB ────────────────────────────
    master_msg = Master_Message_To_LB(
        request_id    = UUID(request_id),
        query         = user_request.query,
        parameters    = user_request.parameters or {},
        priority_level= user_request.parameters.get("priority", 1)
            if user_request.parameters else 1,
    )

    lb_task = LB_To_Worker(
        task_id         = master_msg.request_id,
        lb_dispatched_at= time.time(),
        instruction     = master_msg.query,
        parameters      = master_msg.parameters,
    )

    dispatched_at = lb_task.lb_dispatched_at

    # ── 4. Dispatch via Load Balancer ─────────────────────────────────────
    try:
        worker_result = await lb.dispatch(lb_task)
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=429,
            detail="Request queue is full. Please try again shortly.",
        )
    except RuntimeError as exc:
        logger.error(f"[Master] Task {request_id} failed: {exc}")
        await fh.complete_task(lb_task.task_id)   # clean up FH tracking on failure
        _log_failed_row(
            request_id, user_request, master_received_at, dispatched_at, str(exc)
        )
        raise HTTPException(status_code=503, detail="All GPU workers are unavailable.")

    await fh.complete_task(lb_task.task_id)   # task finished — remove from watchlist

    master_responded_at = time.time()

    # ── 5. Build Final Response ───────────────────────────────────────────
    total_latency     = master_responded_at - user_request.user_sent_at
    inference_latency = worker_result.inference_end - worker_result.inference_start

    final = Final_Response(
        request_id    = request_id,
        status        = worker_result.status,
        answer        = worker_result.response_text,
        total_latency = round(total_latency, 4),
    )

    # ── 6. Persist result ─────────────────────────────────────────────────
    _cache_result(final)

    # ── 7. Log to CSV ──────────────────────────────────────────────────────
    _append_csv_row({
        "request_id":          request_id,
        "user_id":             user_request.user_id,
        "query_snippet":       user_request.query[:80],
        "user_sent_at":        round(user_request.user_sent_at, 4),
        "master_received_at":  round(master_received_at, 4),
        "dispatched_at":       round(dispatched_at, 4),
        "worker_id":           worker_result.worker_id,
        "provider":            worker_result.provider,
        "model_used":          worker_result.model_used,
        "worker_received_at":  round(worker_result.worker_received_at, 4),
        "inference_start":     round(worker_result.inference_start, 4),
        "inference_end":       round(worker_result.inference_end, 4),
        "master_responded_at": round(master_responded_at, 4),
        "total_latency_s":     round(total_latency, 4),
        "inference_latency_s": round(inference_latency, 4),
        "status":              worker_result.status,
    })

    logger.info(
        f"[Master] ✓ Request {request_id} completed | "
        f"total={total_latency:.3f}s | inference={inference_latency:.3f}s | "
        f"worker={worker_result.worker_id}"
    )

    return {
        "request_id":    final.request_id,
        "status":        final.status,
        "answer":        final.answer,
        "total_latency": final.total_latency,
    }


# ---------------------------------------------------------------------------
# 2.  /result/{request_id}  — retrieve a cached answer
# ---------------------------------------------------------------------------
@app.get("/result/{request_id}")
async def get_result(
    request_id: str,
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    """
    Returns a previously completed result from the in-memory cache.
    Useful if the client's connection dropped after the worker finished.
    """
    _require_api_key(x_api_key)
    result = _result_cache.get(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or expired.")
    return {
        "request_id":    result.request_id,
        "status":        result.status,
        "answer":        result.answer,
        "total_latency": result.total_latency,
    }


# ---------------------------------------------------------------------------
# 3.  /workers/register  — worker self-registration on startup
# ---------------------------------------------------------------------------
@app.post("/workers/register")
async def register_worker(body: WorkerRegistrationRequest) -> dict:
    """
    Called by each GPU worker when it boots up.
    No API key required — workers run inside the trusted network.
    """
    node_id = UUID(body.node_id)
    await lb.register_worker(node_id, body.url)
    logger.info(f"[Master] Worker registered: {node_id} @ {body.url}")
    return {"status": "registered", "node_id": body.node_id}


# ---------------------------------------------------------------------------
# 4.  /heartbeat  — periodic worker health update
# ---------------------------------------------------------------------------
@app.post("/heartbeat")
async def receive_heartbeat(heartbeat: Worker_Heartbeat) -> dict:
    """
    Workers POST here every ~10 seconds with their current load metrics.
    The Master forwards the data to the Load Balancer for routing decisions.
    """
    await lb.update_worker_state(heartbeat)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 5.  /admin/stats  — system-wide snapshot for monitoring dashboards
# ---------------------------------------------------------------------------
@app.get("/admin/stats")
async def admin_stats(
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    """
    Returns a real-time snapshot of:
      - Queue depth
      - Alive worker count
      - Per-worker metrics (tasks, CPU, VRAM, WLC score)
    """
    _require_api_key(x_api_key)
    return {
        "queue_depth":         lb.queue_depth,
        "alive_workers":       lb.alive_worker_count,
        "total_workers":       len(lb.get_worker_stats()),
        "backpressure_active": lb.queue_depth >= MAX_QUEUE,
        "workers":             lb.get_worker_stats(),
        "fault_stats":         fh.get_fault_stats(),
    }


# ---------------------------------------------------------------------------
# 6.  /admin/mark-dead/{node_id}  — manual eviction (testing / fault injection)
# ---------------------------------------------------------------------------
@app.post("/admin/mark-dead/{node_id}")
async def mark_dead(
    node_id: str,
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    """
    Manually marks a worker as dead.
    Useful for fault-tolerance testing: simulate a node failure mid-run.
    """
    _require_api_key(x_api_key)
    await lb.mark_worker_dead(UUID(node_id))
    return {"status": "marked_dead", "node_id": node_id}


# ---------------------------------------------------------------------------
# Internal helper: log a failed request row to CSV
# ---------------------------------------------------------------------------
def _log_failed_row(
    request_id: str,
    user_request: User_Request,
    master_received_at: float,
    dispatched_at: float,
    error: str,
) -> None:
    _append_csv_row({
        "request_id":          request_id,
        "user_id":             user_request.user_id,
        "query_snippet":       user_request.query[:80],
        "user_sent_at":        round(user_request.user_sent_at, 4),
        "master_received_at":  round(master_received_at, 4),
        "dispatched_at":       round(dispatched_at, 4),
        "worker_id":           "N/A",
        "provider":            "N/A",
        "model_used":          "N/A",
        "worker_received_at":  0,
        "inference_start":     0,
        "inference_end":       0,
        "master_responded_at": round(time.time(), 4),
        "total_latency_s":     round(time.time() - user_request.user_sent_at, 4),
        "inference_latency_s": 0,
        "status":              f"error: {error[:120]}",
    })