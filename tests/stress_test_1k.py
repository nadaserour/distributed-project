import asyncio
import time
import random
import csv
import os
import aiohttp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MASTER_URL = "http://localhost:8000/query"  # Adjust endpoint path to match your API
TOTAL_REQUESTS = 10                        # Scaled down from 1000 for safe step-up testing
CONCURRENCY_LIMIT = 5                      # Max active simultaneous TCP connections
API_KEY = "dev-key-1"                      # Your master node's required auth key
CSV_FILENAME = "latency_test_results.csv"

test_prompts = [
    "What are the design constraints of embedded systems?",
    "Explain the difference between PID and PD controllers.",
    "How does a master-worker architecture reduce inference bottlenecks?",
    "What is the role of a vector database in a RAG pipeline?",
    "How does edge connectivity differ from vertex connectivity?",
    "What is the physical significance of steady-state error?"
]

# Thread-safe lists to collect telemetry in memory
results_data = []
success_counts = 0
failure_counts = 0

async def send_single_request(session, sem, request_id):
    global success_counts, failure_counts
    
    prompt = random.choice(test_prompts)
    
    # Payload matching the exact User_Request dataclass schema
    payload = {
        "user_id": f"stress_test_user_{request_id:04d}",
        "query": prompt,
        "user_sent_at": float(time.time()),
        "parameters": {}
    }
    
    headers = {
        "x-api-key": API_KEY
    }
    
    async with sem:
        start_time = time.time()
        status_code = -1
        error_msg = "None"
        worker_id = "N/A"
        response_size = 0
        
        try:
            async with session.post(MASTER_URL, json=payload, headers=headers, timeout=120) as response:
                status_code = response.status
                duration = time.time() - start_time
                
                if status_code == 200:
                    try:
                        res_json = await response.json()
                        worker_id = res_json.get("worker_id", "Unknown")
                        response_size = len(res_json.get("answer", ""))
                    except Exception:
                        # Fallback if response is text or dictionary keys differ
                        res_text = await response.text()
                        response_size = len(res_text)
                    
                    success_counts += 1
                    if request_id % 10 == 0 or request_id == 1:
                        print(f"✅ [Req {request_id:04d}] Status: 200 | Latency: {duration:.2f}s | Worker: {worker_id}")
                else:
                    failure_counts += 1
                    error_msg = await response.text()
                    print(f"⚠️ [Req {request_id:04d}] Failed with Status {status_code} | Res: {error_msg[:100]}")
                    
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            failure_counts += 1
            status_code = 408
            error_msg = "Request Timeout"
            print(f"❌ [Req {request_id:04d}] Timeout Error after {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            failure_counts += 1
            error_msg = str(e)
            print(f"❌ [Req {request_id:04d}] Network Error: {e}")
            
        # Collect metric row
        results_data.append({
            "request_id": request_id,
            "prompt": prompt,
            "status_code": status_code,
            "latency_s": round(duration, 4),
            "worker_id": worker_id,
            "response_char_size": response_size,
            "error_msg": error_msg[:150]
        })

def write_to_csv(filepath, data):
    file_exists = os.path.isfile(filepath)
    keys = ["request_id", "prompt", "status_code", "latency_s", "worker_id", "response_char_size", "error_msg"]
    
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

async def main():
    print(f"🚀 Starting Step-up Performance Test ({TOTAL_REQUESTS} requests total)...")
    print(f"⚡ Concurrency level set to: {CONCURRENCY_LIMIT}")
    print(f"📁 Local telemetry logging target: {CSV_FILENAME}")
    print("-" * 65)
    
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # Configure high-throughput socket pool limits
    conn = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT, ttl_dns_cache=300)
    
    async with aiohttp.ClientSession(connector=conn) as session:
        start_test_time = time.time()
        
        tasks = [
            asyncio.create_task(send_single_request(session, sem, i + 1))
            for i in range(TOTAL_REQUESTS)
        ]
        
        await asyncio.gather(*tasks)
        total_test_time = time.time() - start_test_time
        
    # Sort data by request_id before saving
    results_data.sort(key=lambda x: x["request_id"])
    write_to_csv(CSV_FILENAME, results_data)
    
    # Calculate performance metrics
    latencies = [r["latency_s"] for r in results_data if r["status_code"] == 200]
    
    print("" + "="*50)
    print("📊 HIGH-THROUGHPUT SYSTEM METRICS")
    print("="*50)
    print(f"Total Requests Dispatched:  {TOTAL_REQUESTS}")
    print(f"Successful Responses:      {success_counts} ({success_counts/TOTAL_REQUESTS*100:.1f}%)")
    print(f"Failed / Dropped Tasks:     {failure_counts}")
    print(f"Total Execution Time:       {total_test_time:.2f} seconds")
    print(f"System Throughput:          {TOTAL_REQUESTS/total_test_time:.2f} RPS (Requests/Sec)")
    
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99_lat = sorted_lat[int(len(sorted_lat) * 0.99)]
        print(f"Average Response Latency:   {avg_lat:.2f} seconds")
        print(f"95th Percentile Latency:    {p95_lat:.2f} seconds")
        print(f"99th Percentile Latency:    {p99_lat:.2f} seconds")
    print(f"💾 Results saved successfully to: {CSV_FILENAME}")
    print("="*50)

if __name__ == '__main__':
    asyncio.run(main())