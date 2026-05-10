import asyncio
import sys
import time
import uuid
from venv import logger
import psutil
import httpx
import random
from uuid import uuid4
from fastapi import FastAPI
from common.models import LB_To_Worker, Worker_To_Master, Worker_Heartbeat
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException
import hashlib
from llm.inference import run_llm
from rag.retriever import retrieve_context
# Hardware monitoring imports
try:
    import pynvml  # For AWS NVIDIA GPUs
    pynvml.nvmlInit()
except Exception:
    pynvml = None

app = FastAPI()

# 1. Get the port first (defaulting to 8001)
def get_port():
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            return int(sys.argv[i+1])
    return 8001

CURRENT_PORT = get_port()

# 2. Generate a "Deterministic" ID (Same port = Same ID)
def get_consistent_id(port):
    seed = f"worker-at-{port}"
    hash_val = hashlib.md5(seed.encode()).hexdigest()
    return uuid.UUID(hash_val)

MY_NODE_ID = get_consistent_id(CURRENT_PORT)
MASTER_URL = "http://192.168.8.186:8000"  # Change to your Master's actual IP
active_tasks_counter = 0

def get_gpu_vram():
    """Retrieves free VRAM in GB or 0.0 if not available."""
    if pynvml:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return round(info.free / (1024**3), 2)
        except Exception:
            return 0.0
    return 0.0

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "node_id": str(MY_NODE_ID),
        "active_tasks": active_tasks_counter
    }



@app.post("/generate")
def generate_task(data: LB_To_Worker):
    global active_tasks_counter
    active_tasks_counter += 1
    
    received_at = time.time()
    inference_start = time.time()
    
    try:
        # STEP 1: Get the RAG context using your pre-built retriever
        # This will automatically load the FAISS index on the first request.
        logger.info(f"[Worker] Fetching context for: {data.instruction[:50]}...")
        context = retrieve_context(data.instruction)
        print(f"DEBUG: Context sent to LLM: {context[:200]}...")
        
        # STEP 2: Pass the context to your LLM expert
        ai_answer = run_llm(data.instruction, context)
        
        inference_end = time.time()

        # Build your response (Expert-RAG Mode)
        response_obj = Worker_To_Master(
            task_id=str(data.task_id),
            worker_id=str(MY_NODE_ID),
            response_text=ai_answer,
            model_used="qwen2.5:14b ",
            provider="Ollama-Distributed-RAG",
            worker_received_at=received_at,
            inference_start=inference_start,
            inference_end=inference_end,
            metrics={"rag_active": True},
            status="success"
        )
        return jsonable_encoder(response_obj)

    except Exception as e:
        logger.error(f"[Worker] RAG/LLM Failure: {e}")
        # Return an error object if something breaks
        return jsonable_encoder({"status": "error", "detail": str(e)})
    finally:
        active_tasks_counter -= 1

        
async def heartbeat_loop(master_url: str):
    """Background task to notify Master of worker status."""
    await asyncio.sleep(2)  # Give the server a second to start up
    async with httpx.AsyncClient() as client:
        while True:
            heartbeat = Worker_Heartbeat(
                node_id=MY_NODE_ID,
                status="ready",
                current_load_count=active_tasks_counter,
                cpu_usage_percent=psutil.cpu_percent(interval=None),  
                gpu_vram_free=get_gpu_vram(),
                last_seen=time.time()
            )
            
            try:
                payload = heartbeat.__dict__.copy()
                payload['node_id'] = str(payload['node_id'])
                
                response = await client.post(f"{master_url}/heartbeat", json=payload, timeout=2.0)
                if response.status_code == 200:
                    print(f"DEBUG: Heartbeat sent. Load: {active_tasks_counter}")
            except Exception:
                # Silently fail to console so it doesn't spam, but keep trying
                print("DEBUG: Master node unreachable for heartbeat...")
            
            await asyncio.sleep(5)  

@app.on_event("startup")
async def startup_event():
    # Get the port from the command line arguments or default to 8001
    import sys
    port = 8001
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i+1])

    async def register():
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    # Dynamic URL based on the port this worker is running on
                    reg_payload = {
                        "node_id": str(MY_NODE_ID), 
                        "url": f"http://localhost:{port}" 
                    }
                    resp = await client.post(f"{MASTER_URL}/workers/register", json=reg_payload)
                    if resp.status_code == 200:
                        print(f"DEBUG: Registered with Master on port {port}.")
                        return
            except Exception:
                print("DEBUG: Master not ready, retrying...")
            await asyncio.sleep(2)

    asyncio.create_task(register())
    asyncio.create_task(heartbeat_loop(MASTER_URL))