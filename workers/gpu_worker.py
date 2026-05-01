import asyncio
import time
import psutil
import httpx
from uuid import uuid4
from fastapi import FastAPI
from common.models import LB_To_Worker, Worker_To_Master, Worker_Heartbeat

# Hardware monitoring imports
try:
    import pynvml  # For AWS NVIDIA GPUs
    pynvml.nvmlInit()
except ImportError:
    pynvml = None

app = FastAPI()

# --- Configuration & State ---
MY_NODE_ID = uuid4()
MASTER_URL = "http://MASTER_IP:8000"  
active_tasks_counter = 0

def get_gpu_vram():
    """Retrieves free VRAM in GB."""
    if pynvml:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free / (1024**3)  # Convert to GB
        except:
            return 0.0
    # Fallback for Radeon/Non-NVIDIA: can use a placeholder or system memory
    return 0.0

@app.post("/generate")
async def generate_task(data: LB_To_Worker):
    global active_tasks_counter
    active_tasks_counter += 1
    
    received_at = time.time()
    inference_start = time.time()
    
    # Simulate inference or call your pre-loaded model
    # response_text = model.generate(data.instruction)
    await asyncio.sleep(0.5) # Simulate workload
    response_text = f"Processed: {data.instruction[:20]}..."
    
    inference_end = time.time()
    active_tasks_counter -= 1
    
    return Worker_To_Master(
        task_id=data.task_id,
        worker_id=str(MY_NODE_ID),
        response_text=response_text,
        worker_received_at=received_at,
        inference_start=inference_start,
        inference_end=inference_end,
        status="success"
    )

async def heartbeat_loop(master_url: str):
    async with httpx.AsyncClient() as client:
        while True:
            # 1. Gather real-time stats
            heartbeat = Worker_Heartbeat(
                node_id=MY_NODE_ID,
                status="ready",
                current_load_count=active_tasks_counter,
                cpu_usage_percent=psutil.cpu_percent(),
                gpu_vram_free=get_gpu_vram(),
                last_seen=time.time()
            )
            
            try:
                # 2. POST to the Master/LB (ensure UUID is stringified for JSON)
                payload = heartbeat.__dict__.copy()
                payload['node_id'] = str(payload['node_id'])
                await client.post(f"{master_url}/heartbeat", json=payload)
            except Exception as e:
                print(f"Heartbeat failed to reach Master: {e}")
            
            await asyncio.sleep(10)

@app.on_event("startup")
async def startup_event():
    # Automatically starts when you run uvicorn
    asyncio.create_task(heartbeat_loop(MASTER_URL))