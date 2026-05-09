# tests/test_load.py
#
# Load Testing Script
# Simulates increasing concurrent users (100 → 1000) and measures:
#   - Per-request latency (avg, p95, p99, max)
#   - Throughput (requests/second)
#   - Success / failure / backpressure (429) rates
#
# Run:
#   python -m tests.test_load
#   python -m tests.test_load --users 500        (single run)
#   python -m tests.test_load --ramp             (100,250,500,1000 step ramp)

import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration — adjust to match your deployment
# ---------------------------------------------------------------------------
MASTER_URL  = "http://localhost:8000"
API_KEY     = "dev-key-1"
OUTPUT_DIR  = Path("tests/results")
TIMEOUT_S   = 120.0   # per-request timeout

# Sample queries that mimic real college-advisor traffic
SAMPLE_QUERIES = [
    "What are the admission requirements for the computer science program?",
    "How do I apply for financial aid at Ain Shams University?",
    "What is the GPA requirement to stay enrolled in the engineering faculty?",
    "Can I transfer credits from another university?",
    "What elective courses are available in the 3rd year?",
    "How do I register for graduation?",
    "What is the deadline for course withdrawal this semester?",
    "Who is my academic advisor and how do I contact them?",
    "Is there a co-op or internship program available?",
    "What are the lab hours for the computer engineering department?",
]


# ---------------------------------------------------------------------------
# Single user coroutine
# ---------------------------------------------------------------------------
async def simulate_user(
    client: httpx.AsyncClient,
    user_id: int,
    results: list,
) -> None:
    query = SAMPLE_QUERIES[user_id % len(SAMPLE_QUERIES)]
    payload = {
        "user_id":      f"load_test_user_{user_id}",
        "query":        query,
        "user_sent_at": time.time(),
        "parameters":   {},
    }
    t0 = time.time()
    try:
        resp = await client.post(
            f"{MASTER_URL}/query",
            json=payload,
            headers={"x-api-key": API_KEY},
            timeout=TIMEOUT_S,
        )
        latency = time.time() - t0
        results.append({
            "user_id":     user_id,
            "status_code": resp.status_code,
            "latency_s":   round(latency, 4),
            "error":       None,
        })
    except httpx.TimeoutException:
        results.append({
            "user_id":     user_id,
            "status_code": 0,
            "latency_s":   round(time.time() - t0, 4),
            "error":       "timeout",
        })
    except Exception as exc:
        results.append({
            "user_id":     user_id,
            "status_code": 0,
            "latency_s":   round(time.time() - t0, 4),
            "error":       str(exc),
        })


# ---------------------------------------------------------------------------
# Run one batch of N concurrent users
# ---------------------------------------------------------------------------
async def run_batch(num_users: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Load Test — {num_users} concurrent users")
    print(f"{'='*60}")

    results = []
    # Use a semaphore to avoid hammering the OS with 1000 simultaneous
    # TCP connections; 200 is a safe concurrency ceiling for a laptop.
    sem = asyncio.Semaphore(200)

    async def guarded_user(client, uid):
        async with sem:
            await simulate_user(client, uid, results)

    async with httpx.AsyncClient() as client:
        t_start = time.time()
        await asyncio.gather(
            *[guarded_user(client, i) for i in range(num_users)]
        )
        wall_time = time.time() - t_start

    # ── Aggregate metrics ─────────────────────────────────────────────
    total      = len(results)
    successes  = [r for r in results if r["status_code"] == 200]
    failures   = [r for r in results if r["status_code"] not in (200, 429) and r["status_code"] != 0]
    bp_hits    = [r for r in results if r["status_code"] == 429]
    timeouts   = [r for r in results if r["error"] == "timeout"]
    latencies  = [r["latency_s"] for r in successes]

    throughput = len(successes) / wall_time if wall_time > 0 else 0

    metrics = {
        "num_users":        num_users,
        "total_requests":   total,
        "successes":        len(successes),
        "failures":         len(failures),
        "backpressure_429": len(bp_hits),
        "timeouts":         len(timeouts),
        "wall_time_s":      round(wall_time, 3),
        "throughput_rps":   round(throughput, 2),
        "latency_avg_s":    round(statistics.mean(latencies), 4)    if latencies else 0,
        "latency_median_s": round(statistics.median(latencies), 4)  if latencies else 0,
        "latency_p95_s":    round(_percentile(latencies, 95), 4)    if latencies else 0,
        "latency_p99_s":    round(_percentile(latencies, 99), 4)    if latencies else 0,
        "latency_max_s":    round(max(latencies), 4)                if latencies else 0,
    }

    # ── Print summary ─────────────────────────────────────────────────
    print(f"  Users sent         : {total}")
    print(f"  Successes (200)    : {metrics['successes']}")
    print(f"  Failures           : {metrics['failures']}")
    print(f"  Backpressure (429) : {metrics['backpressure_429']}")
    print(f"  Timeouts           : {metrics['timeouts']}")
    print(f"  Wall time          : {metrics['wall_time_s']}s")
    print(f"  Throughput         : {metrics['throughput_rps']} req/s")
    if latencies:
        print(f"  Latency avg        : {metrics['latency_avg_s']}s")
        print(f"  Latency p95        : {metrics['latency_p95_s']}s")
        print(f"  Latency p99        : {metrics['latency_p99_s']}s")
        print(f"  Latency max        : {metrics['latency_max_s']}s")

    return metrics


# ---------------------------------------------------------------------------
# Ramp test: 100 → 250 → 500 → 1000
# ---------------------------------------------------------------------------
async def run_ramp_test() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    levels    = [100, 250, 500, 1000]
    all_metrics = []

    for n in levels:
        m = await run_batch(n)
        all_metrics.append(m)
        # Brief pause between levels so workers drain the queue
        print(f"\n  [Ramp] Waiting 15s before next level...")
        await asyncio.sleep(15)

    # ── Save ramp summary CSV ─────────────────────────────────────────
    csv_path = OUTPUT_DIR / f"load_ramp_{int(time.time())}.csv"
    _write_csv(csv_path, all_metrics)
    print(f"\n[Load Test] Ramp results saved → {csv_path}")
    _print_ramp_table(all_metrics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _print_ramp_table(metrics: list[dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  Ramp Test Summary")
    print(f"{'='*80}")
    header = f"{'Users':>6} | {'OK':>5} | {'Fail':>5} | {'429':>5} | {'RPS':>7} | {'Avg(s)':>7} | {'p95(s)':>7} | {'p99(s)':>7}"
    print(f"  {header}")
    print(f"  {'-'*70}")
    for m in metrics:
        print(
            f"  {m['num_users']:>6} | "
            f"{m['successes']:>5} | "
            f"{m['failures']:>5} | "
            f"{m['backpressure_429']:>5} | "
            f"{m['throughput_rps']:>7.2f} | "
            f"{m['latency_avg_s']:>7.4f} | "
            f"{m['latency_p95_s']:>7.4f} | "
            f"{m['latency_p99_s']:>7.4f}"
        )
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the Master scheduler.")
    parser.add_argument("--users", type=int, default=None, help="Run a single batch with N users.")
    parser.add_argument("--ramp",  action="store_true",    help="Ramp: 100 → 250 → 500 → 1000.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.ramp or args.users is None:
        asyncio.run(run_ramp_test())
    else:
        m = asyncio.run(run_batch(args.users))
        csv_path = OUTPUT_DIR / f"load_{args.users}users_{int(time.time())}.csv"
        _write_csv(csv_path, [m])
        print(f"\n[Load Test] Results saved → {csv_path}")