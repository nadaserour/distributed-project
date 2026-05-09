# tests/test_fault_tolerance.py
#
# Fault Tolerance Test Suite
# Tests three scenarios:
#   1. Single worker killed mid-run  → system continues, tasks reassigned
#   2. Multiple workers killed       → system degrades gracefully, no crash
#   3. Worker recovers               → system auto-reregisters it
#
# Run:
#   python -m tests.test_fault_tolerance --scenario single
#   python -m tests.test_fault_tolerance --scenario multi
#   python -m tests.test_fault_tolerance --scenario recovery
#   python -m tests.test_fault_tolerance --scenario all

import argparse
import asyncio
import csv
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MASTER_URL   = "http://localhost:8000"
API_KEY      = "dev-key-1"
OUTPUT_DIR   = Path("tests/results")
TIMEOUT_S    = 120.0

# Worker ports — must match how you launched the workers
WORKER_PORTS = [8001, 8002, 8003, 8004]

SAMPLE_QUERIES = [
    "What are the admission requirements for computer science?",
    "How do I apply for a scholarship?",
    "What courses are required in year 2?",
    "How do I appeal a grade?",
    "What is the late withdrawal policy?",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
async def send_query(client: httpx.AsyncClient, user_id: int) -> dict:
    payload = {
        "user_id":      f"fault_test_user_{user_id}",
        "query":        SAMPLE_QUERIES[user_id % len(SAMPLE_QUERIES)],
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
        return {
            "user_id":     user_id,
            "status_code": resp.status_code,
            "latency_s":   round(time.time() - t0, 4),
            "error":       None,
        }
    except Exception as exc:
        return {
            "user_id":     user_id,
            "status_code": 0,
            "latency_s":   round(time.time() - t0, 4),
            "error":       str(exc),
        }


async def get_admin_stats(client: httpx.AsyncClient) -> dict:
    try:
        resp = await client.get(
            f"{MASTER_URL}/admin/stats",
            headers={"x-api-key": API_KEY},
            timeout=5.0,
        )
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


async def mark_worker_dead(client: httpx.AsyncClient, node_id: str) -> bool:
    """
    Uses the Master's /admin/mark-dead endpoint to simulate a node failure
    without actually killing the OS process.
    This is the safest way to test fault tolerance in a dev environment.
    """
    try:
        resp = await client.post(
            f"{MASTER_URL}/admin/mark-dead/{node_id}",
            headers={"x-api-key": API_KEY},
            timeout=5.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


async def get_worker_node_ids(client: httpx.AsyncClient) -> list[dict]:
    stats = await get_admin_stats(client)
    return stats.get("workers", [])


def print_stats_snapshot(label: str, stats: dict) -> None:
    alive   = stats.get("alive_workers", "?")
    total   = stats.get("total_workers", "?")
    queue   = stats.get("queue_depth", "?")
    print(f"\n  [{label}]  alive={alive}/{total}  queue={queue}")
    for w in stats.get("workers", []):
        status = "✓ alive" if w["is_alive"] else "✗ DEAD"
        print(
            f"    Worker {w['node_id'][:8]}…  {status}  "
            f"tasks={w['active_task_count']}  "
            f"cpu={w['cpu_usage_percent']}%  "
            f"vram={w['gpu_vram_free_gb']}GB"
        )


def save_results(scenario: str, data: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"fault_{scenario}_{int(time.time())}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data.keys())
        w.writeheader()
        w.writerow(data)
    print(f"\n  [Results] Saved → {path}")


# ---------------------------------------------------------------------------
# Scenario 1: Single worker killed mid-run
# ---------------------------------------------------------------------------
async def scenario_single_failure() -> None:
    print(f"\n{'='*65}")
    print("  SCENARIO 1: Single Worker Failure")
    print("  Kill one worker after 50 requests. Verify remaining requests succeed.")
    print(f"{'='*65}")

    TOTAL_USERS  = 150
    KILL_AFTER   = 50     # kill first worker after this many requests launch

    results      = []
    kill_done    = False
    killed_id    = None

    async with httpx.AsyncClient() as client:
        # ── Baseline snapshot ────────────────────────────────────────
        before_stats = await get_admin_stats(client)
        print_stats_snapshot("BEFORE", before_stats)
        workers = before_stats.get("workers", [])

        if len(workers) < 2:
            print("\n  [!] Need at least 2 alive workers to run this test. Aborting.")
            return

        kill_target_id = workers[0]["node_id"]

        sem = asyncio.Semaphore(50)

        async def user_task(uid: int) -> None:
            nonlocal kill_done, killed_id
            async with sem:
                # Trigger the kill exactly once after KILL_AFTER requests
                if uid == KILL_AFTER and not kill_done:
                    kill_done = True
                    ok = await mark_worker_dead(client, kill_target_id)
                    killed_id = kill_target_id
                    print(f"\n  *** FAULT INJECTED: Worker {kill_target_id[:8]}… marked dead "
                          f"(after request #{uid}) — ok={ok} ***\n")

                result = await send_query(client, uid)
                results.append(result)

        t0 = time.time()
        await asyncio.gather(*[user_task(i) for i in range(TOTAL_USERS)])
        wall = time.time() - t0

        # ── Post-kill snapshot ───────────────────────────────────────
        await asyncio.sleep(2)
        after_stats = await get_admin_stats(client)
        print_stats_snapshot("AFTER KILL", after_stats)

    # ── Analyse ──────────────────────────────────────────────────────
    pre_kill  = [r for r in results if r["user_id"] < KILL_AFTER]
    post_kill = [r for r in results if r["user_id"] >= KILL_AFTER]

    pre_ok  = sum(1 for r in pre_kill  if r["status_code"] == 200)
    post_ok = sum(1 for r in post_kill if r["status_code"] == 200)

    print(f"\n  ── Results ──────────────────────────────────────────")
    print(f"  Pre-kill  requests : {len(pre_kill):>4}  successes: {pre_ok}")
    print(f"  Post-kill requests : {len(post_kill):>4}  successes: {post_ok}")
    print(f"  Total wall time    : {wall:.2f}s")

    passed = post_ok > 0
    print(f"\n  TEST {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(f"  System {'continued processing' if passed else 'dropped all'} "
          f"requests after the node failure.")

    save_results("single_failure", {
        "scenario": "single_failure",
        "total_users": TOTAL_USERS,
        "killed_worker": killed_id,
        "pre_kill_ok": pre_ok,
        "post_kill_ok": post_ok,
        "post_kill_fail": len(post_kill) - post_ok,
        "wall_time_s": round(wall, 3),
        "passed": passed,
    })


# ---------------------------------------------------------------------------
# Scenario 2: Multiple workers killed
# ---------------------------------------------------------------------------
async def scenario_multi_failure() -> None:
    print(f"\n{'='*65}")
    print("  SCENARIO 2: Multiple Worker Failures")
    print("  Kill workers one-by-one. Verify graceful degradation.")
    print(f"{'='*65}")

    TOTAL_USERS = 200

    async with httpx.AsyncClient() as client:
        before_stats = await get_admin_stats(client)
        print_stats_snapshot("BEFORE", before_stats)
        workers = before_stats.get("workers", [])

        if len(workers) < 3:
            print("\n  [!] Need at least 3 workers for this test. Aborting.")
            return

        # We'll keep the last worker alive intentionally
        to_kill = [w["node_id"] for w in workers[:-1]]

        results   = []
        kill_log  = []
        sem       = asyncio.Semaphore(50)
        kill_idx  = 0
        kill_lock = asyncio.Lock()

        async def user_task(uid: int) -> None:
            nonlocal kill_idx
            async with sem:
                # Kill a new worker every 50 requests
                async with kill_lock:
                    if kill_idx < len(to_kill) and uid % 50 == 0 and uid > 0:
                        target = to_kill[kill_idx]
                        ok = await mark_worker_dead(client, target)
                        kill_log.append({"at_request": uid, "worker": target, "ok": ok})
                        print(f"\n  *** FAULT: Worker {target[:8]}… killed at request #{uid} ***\n")
                        kill_idx += 1

                result = await send_query(client, uid)
                results.append(result)

        t0 = time.time()
        await asyncio.gather(*[user_task(i) for i in range(TOTAL_USERS)])
        wall = time.time() - t0

        await asyncio.sleep(2)
        after_stats = await get_admin_stats(client)
        print_stats_snapshot("AFTER ALL KILLS", after_stats)

    successes = sum(1 for r in results if r["status_code"] == 200)
    failures  = len(results) - successes
    passed    = successes > 0 and after_stats.get("alive_workers", 0) >= 1

    print(f"\n  ── Results ──────────────────────────────────────────")
    print(f"  Workers killed     : {len(kill_log)}")
    for kl in kill_log:
        print(f"    At request #{kl['at_request']}: worker {kl['worker'][:8]}…")
    print(f"  Total successes    : {successes}/{TOTAL_USERS}")
    print(f"  Alive workers left : {after_stats.get('alive_workers', 0)}")
    print(f"  Wall time          : {wall:.2f}s")
    print(f"\n  TEST {'PASSED ✓' if passed else 'FAILED ✗'}")

    save_results("multi_failure", {
        "scenario": "multi_failure",
        "total_users": TOTAL_USERS,
        "workers_killed": len(kill_log),
        "successes": successes,
        "failures": failures,
        "alive_at_end": after_stats.get("alive_workers", 0),
        "wall_time_s": round(wall, 3),
        "passed": passed,
    })


# ---------------------------------------------------------------------------
# Scenario 3: Worker recovery
# ---------------------------------------------------------------------------
async def scenario_recovery() -> None:
    print(f"\n{'='*65}")
    print("  SCENARIO 3: Worker Recovery")
    print("  Kill a worker, wait for FaultHandler to detect it,")
    print("  then re-register it and verify it receives new tasks.")
    print(f"{'='*65}")

    async with httpx.AsyncClient() as client:
        before_stats = await get_admin_stats(client)
        print_stats_snapshot("BEFORE", before_stats)
        workers = before_stats.get("workers", [])

        if not workers:
            print("\n  [!] No workers registered. Aborting.")
            return

        target    = workers[0]
        target_id = target["node_id"]
        target_url= target["url"]

        # ── Step 1: Kill worker ───────────────────────────────────────
        print(f"\n  Step 1: Killing worker {target_id[:8]}…")
        await mark_worker_dead(client, target_id)
        await asyncio.sleep(2)

        mid_stats = await get_admin_stats(client)
        print_stats_snapshot("AFTER KILL", mid_stats)

        dead_confirmed = not any(
            w["is_alive"] for w in mid_stats.get("workers", [])
            if w["node_id"] == target_id
        )
        print(f"\n  Worker confirmed dead: {dead_confirmed}")

        # ── Step 2: Send 20 requests — they should go to other workers ─
        print(f"\n  Step 2: Sending 20 requests while worker is dead...")
        results_while_dead = []
        sem = asyncio.Semaphore(20)
        async def q(uid):
            async with sem:
                results_while_dead.append(await send_query(client, uid))
        await asyncio.gather(*[q(i) for i in range(20)])
        ok_while_dead = sum(1 for r in results_while_dead if r["status_code"] == 200)
        print(f"  Requests while dead: {ok_while_dead}/20 succeeded (served by remaining workers)")

        # ── Step 3: Re-register the dead worker (simulate recovery) ───
        print(f"\n  Step 3: Re-registering worker {target_id[:8]}… (simulating recovery)...")
        resp = await client.post(
            f"{MASTER_URL}/workers/register",
            json={"node_id": target_id, "url": target_url},
            timeout=5.0,
        )
        recovered = resp.status_code == 200
        print(f"  Re-registration response: {resp.status_code} — recovered={recovered}")
        await asyncio.sleep(3)

        # ── Step 4: Verify recovered worker gets tasks ─────────────────
        print(f"\n  Step 4: Sending 30 requests after recovery...")
        results_after = []
        async def q2(uid):
            async with sem:
                results_after.append(await send_query(client, uid + 100))
        await asyncio.gather(*[q2(i) for i in range(30)])

        after_stats = await get_admin_stats(client)
        print_stats_snapshot("AFTER RECOVERY", after_stats)

        revived_worker = next(
            (w for w in after_stats.get("workers", []) if w["node_id"] == target_id), None
        )
        revived_alive = revived_worker["is_alive"] if revived_worker else False
        ok_after = sum(1 for r in results_after if r["status_code"] == 200)

        print(f"\n  ── Results ──────────────────────────────────────────")
        print(f"  Worker marked dead          : {dead_confirmed}")
        print(f"  Requests OK while dead      : {ok_while_dead}/20")
        print(f"  Worker re-registered        : {recovered}")
        print(f"  Worker alive after recovery : {revived_alive}")
        print(f"  Requests OK after recovery  : {ok_after}/30")

        passed = dead_confirmed and ok_while_dead > 0 and recovered and revived_alive
        print(f"\n  TEST {'PASSED ✓' if passed else 'FAILED ✗'}")

        save_results("recovery", {
            "scenario": "recovery",
            "target_worker": target_id,
            "dead_confirmed": dead_confirmed,
            "ok_while_dead": ok_while_dead,
            "re_registered": recovered,
            "alive_after_recovery": revived_alive,
            "ok_after_recovery": ok_after,
            "passed": passed,
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fault tolerance test suite.")
    parser.add_argument(
        "--scenario",
        choices=["single", "multi", "recovery", "all"],
        default="all",
        help="Which scenario to run."
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async def main():
        if args.scenario in ("single", "all"):
            await scenario_single_failure()
            await asyncio.sleep(5)
        if args.scenario in ("multi", "all"):
            await scenario_multi_failure()
            await asyncio.sleep(5)
        if args.scenario in ("recovery", "all"):
            await scenario_recovery()

    asyncio.run(main())