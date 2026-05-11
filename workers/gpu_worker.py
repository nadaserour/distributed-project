import asyncio

import hashlib

import logging

import sys

import time

import uuid
 
import httpx

import psutil

from fastapi import FastAPI

from fastapi.encoders import jsonable_encoder
 
from common.models import LB_To_Worker, Worker_Heartbeat, Worker_To_Master

from llm.inference import run_llm

from rag.retriever import retrieve_context
 
# Hardware monitoring

try:

    import pynvml

    pynvml.nvmlInit()

except Exception:

    pynvml = None
 
# ---------------------------------------------------------------------------

# Fix 1: proper logger

# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
 
app = FastAPI()
 
# ---------------------------------------------------------------------------

# Worker identity

# ---------------------------------------------------------------------------

def get_port() -> int:

    for i, arg in enumerate(sys.argv):

        if arg == "--port" and i + 1 < len(sys.argv):

            return int(sys.argv[i + 1])

    return 8001
 
def get_consistent_id(port: int) -> uuid.UUID:

    seed = f"worker-at-{port}"

    return uuid.UUID(hashlib.md5(seed.encode()).hexdigest())
 
CURRENT_PORT    = get_port()

MY_NODE_ID      = get_consistent_id(CURRENT_PORT)

MASTER_URL      = "http://localhost:8000"   # change if master is on a different machine

active_tasks_counter = 0
 
 
def get_gpu_vram() -> float:

    if pynvml:

        try:

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)

            return round(info.free / (1024 ** 3), 2)

        except Exception:

            return 0.0

    return 0.0
 
 
# ---------------------------------------------------------------------------

# Health check

# ---------------------------------------------------------------------------

@app.get("/health")

async def health_check():

    return {

        "status":       "healthy",

        "node_id":      str(MY_NODE_ID),

        "active_tasks": active_tasks_counter,

    }
 
 
# ---------------------------------------------------------------------------

# Fix 2: async endpoint + awaited run_llm + executor for blocking RAG call

# ---------------------------------------------------------------------------

@app.post("/generate")

async def generate_task(data: LB_To_Worker):

    global active_tasks_counter

    active_tasks_counter += 1
 
    received_at     = time.time()

    inference_start = time.time()
 
    try:

        # retrieve_context is synchronous (FAISS) — run in thread pool so we

        # don't block the event loop

        logger.info(f"[Worker] Fetching RAG context for: {data.instruction[:50]}...")

        loop    = asyncio.get_event_loop()

        context = await loop.run_in_executor(None, retrieve_context, data.instruction)

        logger.debug(f"[Worker] Context snippet: {context[:120]}...")
 
        # Fix 3: run_llm is async — must be awaited

        ai_answer = await run_llm(data.instruction, context)
 
        inference_end = time.time()
 
        response_obj = Worker_To_Master(

            task_id            = str(data.task_id),

            worker_id          = str(MY_NODE_ID),

            response_text      = ai_answer,

            model_used         = "qwen2.5:14b",

            provider           = "Ollama-Distributed-RAG",

            worker_received_at = received_at,

            inference_start    = inference_start,

            inference_end      = inference_end,

            metrics            = {"rag_active": True},

            status             = "success",

        )

        return jsonable_encoder(response_obj)
 
    except Exception as exc:

        logger.error(f"[Worker] RAG/LLM failure: {exc}")

        return jsonable_encoder({"status": "error", "detail": str(exc)})
 
    finally:

        active_tasks_counter -= 1
 
 
# ---------------------------------------------------------------------------

# Heartbeat loop

# ---------------------------------------------------------------------------

async def heartbeat_loop(master_url: str) -> None:

    await asyncio.sleep(2)

    async with httpx.AsyncClient() as client:

        while True:

            heartbeat = Worker_Heartbeat(

                node_id             = MY_NODE_ID,

                status              = "ready",

                current_load_count  = active_tasks_counter,

                cpu_usage_percent   = psutil.cpu_percent(interval=None),

                gpu_vram_free       = get_gpu_vram(),

                last_seen           = time.time(),

            )

            try:

                payload           = heartbeat.__dict__.copy()

                payload["node_id"] = str(payload["node_id"])

                resp = await client.post(

                    f"{master_url}/heartbeat", json=payload, timeout=2.0

                )

                if resp.status_code == 200:

                    logger.debug(f"[Worker] Heartbeat sent. load={active_tasks_counter}")

            except Exception:

                logger.debug("[Worker] Master unreachable for heartbeat — will retry.")

            await asyncio.sleep(5)
 
 
# ---------------------------------------------------------------------------

# Startup: register + start heartbeat

# ---------------------------------------------------------------------------

@app.on_event("startup")

async def startup_event() -> None:

    port = CURRENT_PORT
 
    async def register() -> None:

        while True:

            try:

                async with httpx.AsyncClient() as client:

                    payload = {

                        "node_id": str(MY_NODE_ID),

                        "url":     f"http://localhost:{port}",

                    }

                    resp = await client.post(

                        f"{MASTER_URL}/workers/register", json=payload

                    )

                    if resp.status_code == 200:

                        logger.info(f"[Worker] Registered with Master on port {port}.")

                        return

            except Exception:

                logger.debug("[Worker] Master not ready, retrying registration...")

            await asyncio.sleep(2)
 
    asyncio.create_task(register())

    asyncio.create_task(heartbeat_loop(MASTER_URL))
 