# tests/test_metrics.py
#
# Performance Metrics Collector
# Polls /admin/stats every second during a live run and records:
#   - Per-worker CPU usage
#   - Per-worker GPU VRAM free (proxy for GPU load)
#   - Queue depth
#   - Active task counts
# Also parses logs/request_log.csv to produce a final latency/throughput report.
#
# Run DURING a load test (in a separate terminal):
#   python -m tests.test_metrics --watch          (live polling)
#   python -m tests.test_metrics --report         (post-run report from CSV)

import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MASTER_URL    = "http://localhost:8000"
API_KEY       = "dev-key-1"
POLL_INTERVAL = 1.0          # seconds between admin/stats polls
OUTPUT_DIR    = Path("tests/results")
REQUEST_LOG   = Path("logs/request_log.csv")


# ---------------------------------------------------------------------------
# 1.  Live watcher — polls /admin/stats while the load test runs
# ---------------------------------------------------------------------------
async def watch_metrics(duration_s: int = 120) -> None:
    """
    Poll /admin/stats every second for `duration_s` seconds and write
    every snapshot to a timestamped CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"metrics_live_{int(time.time())}.csv"

    print(f"\n[Metrics] Live monitoring started ({duration_s}s). Output → {out_path}")
    print(f"[Metrics] Start your load test now in another terminal.\n")

    rows = []
    headers_written = False

    async with httpx.AsyncClient() as client:
        deadline = time.time() + duration_s
        while time.time() < deadline:
            ts = time.time()
            try:
                resp = await client.get(
                    f"{MASTER_URL}/admin/stats",
                    headers={"x-api-key": API_KEY},
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    snapshot = _flatten_stats(ts, data)
                    rows.append(snapshot)
                    _print_snapshot(snapshot)

                    # Write header once, then append rows
                    with out_path.open("a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=snapshot.keys())
                        if not headers_written:
                            w.writeheader()
                            headers_written = True
                        w.writerow(snapshot)
                else:
                    print(f"[Metrics] /admin/stats returned {resp.status_code}")
            except Exception as exc:
                print(f"[Metrics] Poll error: {exc}")

            await asyncio.sleep(POLL_INTERVAL)

    print(f"\n[Metrics] Monitoring done. {len(rows)} snapshots saved → {out_path}")
    _summarise_live(rows)


def _flatten_stats(ts: float, data: dict) -> dict:
    """Turn the nested /admin/stats JSON into a flat CSV row."""
    row: dict = {
        "timestamp":          round(ts, 3),
        "queue_depth":        data.get("queue_depth", 0),
        "alive_workers":      data.get("alive_workers", 0),
        "total_workers":      data.get("total_workers", 0),
        "backpressure_active":int(data.get("backpressure_active", False)),
    }
    # Per-worker metrics (up to 8 workers)
    for i, w in enumerate(data.get("workers", [])[:8]):
        prefix = f"w{i}"
        row[f"{prefix}_active_tasks"]  = w.get("active_task_count", 0)
        row[f"{prefix}_cpu_pct"]       = w.get("cpu_usage_percent", 0)
        row[f"{prefix}_vram_free_gb"]  = w.get("gpu_vram_free_gb", 0)
        row[f"{prefix}_wlc_score"]     = w.get("wlc_score", "inf")
        row[f"{prefix}_alive"]         = int(w.get("is_alive", True))
    return row


def _print_snapshot(snap: dict) -> None:
    ts_str = time.strftime("%H:%M:%S", time.localtime(snap["timestamp"]))
    workers_alive = snap["alive_workers"]
    queue         = snap["queue_depth"]
    bp            = "⚠ BP" if snap["backpressure_active"] else "   "

    # Collect per-worker lines
    worker_parts = []
    for i in range(snap.get("total_workers", 0)):
        prefix = f"w{i}"
        if f"{prefix}_active_tasks" not in snap:
            break
        alive = "✓" if snap[f"{prefix}_alive"] else "✗"
        tasks = snap[f"{prefix}_active_tasks"]
        cpu   = snap[f"{prefix}_cpu_pct"]
        vram  = snap[f"{prefix}_vram_free_gb"]
        worker_parts.append(f"W{i}[{alive} t={tasks} cpu={cpu:.0f}% vram={vram:.1f}GB]")

    workers_str = "  ".join(worker_parts)
    print(f"[{ts_str}] queue={queue:>3} alive={workers_alive} {bp}  {workers_str}")


def _summarise_live(rows: list[dict]) -> None:
    if not rows:
        return
    queue_depths = [r["queue_depth"] for r in rows]
    print(f"\n{'='*60}")
    print(f"  Live Metrics Summary")
    print(f"{'='*60}")
    print(f"  Snapshots collected : {len(rows)}")
    print(f"  Max queue depth     : {max(queue_depths)}")
    print(f"  Avg queue depth     : {statistics.mean(queue_depths):.1f}")
    backpressure_secs = sum(1 for r in rows if r["backpressure_active"])
    print(f"  Backpressure active : {backpressure_secs}s / {len(rows)}s")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 2.  Post-run report — reads logs/request_log.csv and prints a full summary
# ---------------------------------------------------------------------------
def generate_report() -> None:
    if not REQUEST_LOG.exists():
        print(f"[Metrics] {REQUEST_LOG} not found. Run a load test first.")
        return

    rows = []
    with REQUEST_LOG.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("[Metrics] Request log is empty.")
        return

    total      = len(rows)
    successful = [r for r in rows if r["status"] == "success"]
    failed     = [r for r in rows if r["status"] != "success"]

    total_latencies     = [float(r["total_latency_s"])     for r in successful]
    inference_latencies = [float(r["inference_latency_s"]) for r in successful]

    # Time range
    start_ts = min(float(r["user_sent_at"])        for r in rows)
    end_ts   = max(float(r["master_responded_at"])  for r in successful) if successful else start_ts
    duration = end_ts - start_ts
    throughput = len(successful) / duration if duration > 0 else 0

    print(f"\n{'='*65}")
    print(f"  Post-Run Performance Report")
    print(f"{'='*65}")
    print(f"  Total requests logged  : {total}")
    print(f"  Successful             : {len(successful)}")
    print(f"  Failed                 : {len(failed)}")
    print(f"  Test duration          : {duration:.2f}s")
    print(f"  Throughput             : {throughput:.2f} req/s")

    if total_latencies:
        print(f"\n  -- End-to-end Latency --")
        print(f"  Average   : {statistics.mean(total_latencies):.4f}s")
        print(f"  Median    : {statistics.median(total_latencies):.4f}s")
        print(f"  Stdev     : {statistics.stdev(total_latencies):.4f}s" if len(total_latencies)>1 else "")
        print(f"  p95       : {_percentile(total_latencies, 95):.4f}s")
        print(f"  p99       : {_percentile(total_latencies, 99):.4f}s")
        print(f"  Max       : {max(total_latencies):.4f}s")

    if inference_latencies:
        print(f"\n  -- Inference (LLM only) Latency --")
        print(f"  Average   : {statistics.mean(inference_latencies):.4f}s")
        print(f"  p95       : {_percentile(inference_latencies, 95):.4f}s")
        print(f"  Max       : {max(inference_latencies):.4f}s")

    # Per-worker breakdown
    from collections import defaultdict
    per_worker: dict[str, list] = defaultdict(list)
    for r in successful:
        per_worker[r["worker_id"]].append(float(r["total_latency_s"]))

    print(f"\n  -- Per-Worker Task Distribution --")
    for wid, lats in sorted(per_worker.items()):
        print(
            f"  Worker {wid[:8]}…  tasks={len(lats):>4}  "
            f"avg={statistics.mean(lats):.3f}s  "
            f"max={max(lats):.3f}s"
        )

    print(f"{'='*65}\n")

    # Save report CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"perf_report_{int(time.time())}.csv"
    summary = {
        "total": total, "successes": len(successful), "failures": len(failed),
        "duration_s": round(duration, 3), "throughput_rps": round(throughput, 2),
        "latency_avg": round(statistics.mean(total_latencies), 4) if total_latencies else 0,
        "latency_p95": round(_percentile(total_latencies, 95), 4) if total_latencies else 0,
        "latency_p99": round(_percentile(total_latencies, 99), 4) if total_latencies else 0,
        "latency_max": round(max(total_latencies), 4) if total_latencies else 0,
        "inference_avg": round(statistics.mean(inference_latencies), 4) if inference_latencies else 0,
    }
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary.keys())
        w.writeheader()
        w.writerow(summary)
    print(f"[Metrics] Report saved → {out_path}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance metrics collector.")
    parser.add_argument("--watch",    action="store_true", help="Live-poll /admin/stats.")
    parser.add_argument("--report",   action="store_true", help="Post-run report from CSV.")
    parser.add_argument("--duration", type=int, default=120, help="Watch duration in seconds (default 120).")
    args = parser.parse_args()

    if args.report:
        generate_report()
    elif args.watch or True:   # default action
        asyncio.run(watch_metrics(duration_s=args.duration))