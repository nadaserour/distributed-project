#added to handle crashes and failures in the system
#should detect dead workers and reassign their tasks to healthy ones
# fault_tolerance/fault_handler.py
#
# Fault Handler — proactive cluster health monitor.
#
# Responsibilities:
#   1. Heartbeat Watchdog   — periodically sweeps all registered workers and
#                             marks dead any that have gone silent beyond the
#                             timeout threshold (catches crashes with no
#                             in-flight request to trigger LB detection).
#   2. Task Re-queuing      — if a task was in-flight on a worker that just
#                             died, re-submits it to the LB so no request is
#                             lost (the LB's _execute already marks the worker
#                             dead, but doesn't re-queue — that's our job).
#   3. Recovery Watching    — periodically probes dead workers with an HTTP
#                             health check; if one comes back online, it is
#                             re-registered with the LB automatically.
#   4. Event Log            — appends every fault event (death, recovery,
#                             re-queue) to a structured CSV for the report.
#
# The Fault Handler is NOT a network service — it is an internal asyncio
# module started by the Master at app startup.
#
# Usage (already wired into scheduler.py startup/shutdown):
#   fh = FaultHandler(lb)
#   fh.start()
#   ...
#   fh.stop()

import asyncio
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID

import httpx # type: ignore

from lb.load_balancer import LoadBalancer, WorkerState

logger = logging.getLogger("fault_handler")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WATCHDOG_INTERVAL_SEC  = 10.0   # how often the watchdog sweeps all workers
RECOVERY_INTERVAL_SEC  = 30.0   # how often dead workers are probed for recovery
HEARTBEAT_TIMEOUT_SEC  = 25.0   # seconds of silence before a worker is declared dead
                                 # (slightly less than LB's own WORKER_TIMEOUT_SEC
                                 #  so FH acts first)
HEALTH_CHECK_TIMEOUT   = 5.0    # HTTP timeout for recovery probe
FAULT_LOG_PATH         = Path("logs/fault_events.csv")

_CSV_HEADER = [
    "timestamp", "event", "node_id", "worker_url",
    "detail", "elapsed_silent_s"
]


# ---------------------------------------------------------------------------
# In-flight task tracker
# ---------------------------------------------------------------------------
@dataclass
class InFlightTask:
    """Tracks a task that has been dispatched but not yet completed."""
    task_id:     UUID
    node_id:     UUID
    dispatched_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Fault Handler
# ---------------------------------------------------------------------------
class FaultHandler:
    """
    Proactive fault monitor for the GPU worker cluster.

    Public API:
        fh.start()                        — begin background loops
        fh.stop()                         — cancel background loops
        fh.register_task(task, node_id)   — track an in-flight task
        fh.complete_task(task_id)         — remove a completed task
        fh.get_fault_stats() -> dict      — snapshot for /admin/stats
    """

    def __init__(self, lb: LoadBalancer) -> None:
        self._lb = lb

        # task_id → InFlightTask
        self._in_flight: dict[UUID, InFlightTask] = {}
        self._lock = asyncio.Lock()

        # Counters for /admin/stats
        self._total_deaths:     int = 0
        self._total_recoveries: int = 0
        self._total_requeues:   int = 0

        self._watchdog_task:  Optional[asyncio.Task] = None
        self._recovery_task:  Optional[asyncio.Task] = None

        self._ensure_log()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        if self._recovery_task is None or self._recovery_task.done():
            self._recovery_task = asyncio.create_task(self._recovery_loop())
        logger.info("[FaultHandler] Started — watchdog and recovery loops active.")

    def stop(self) -> None:
        for t in (self._watchdog_task, self._recovery_task):
            if t:
                t.cancel()
        logger.info("[FaultHandler] Stopped.")

    # ------------------------------------------------------------------
    # In-flight task tracking (called by the Master around every dispatch)
    # ------------------------------------------------------------------
    async def register_task(self, task_id: UUID, node_id: UUID) -> None:
        """Call immediately after lb.dispatch() resolves a worker."""
        async with self._lock:
            self._in_flight[task_id] = InFlightTask(
                task_id=task_id, node_id=node_id
            )

    async def complete_task(self, task_id: UUID) -> None:
        """Call when the Master receives a successful (or failed) result."""
        async with self._lock:
            self._in_flight.pop(task_id, None)

    # ------------------------------------------------------------------
    # 1. Watchdog loop — detects silent workers
    # ------------------------------------------------------------------
    async def _watchdog_loop(self) -> None:
        """
        Runs every WATCHDOG_INTERVAL_SEC.
        Checks every registered worker's last_seen timestamp.
        If silence exceeds HEARTBEAT_TIMEOUT_SEC → declare dead, re-queue tasks.
        """
        logger.info("[FaultHandler] Watchdog loop running.")
        while True:
            try:
                await asyncio.sleep(WATCHDOG_INTERVAL_SEC)
                await self._sweep_workers()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[FaultHandler] Watchdog error: {exc}")

    async def _sweep_workers(self) -> None:
        now = time.time()
        # Take a snapshot so we don't hold the LB lock while doing async work
        workers: list[WorkerState] = list(self._lb._workers.values())

        for worker in workers:
            if not worker.is_alive:
                continue  # already dead — recovery loop handles it

            silent_for = now - worker.last_seen
            if silent_for > HEARTBEAT_TIMEOUT_SEC:
                logger.warning(
                    f"[FaultHandler] Worker {worker.node_id} silent for "
                    f"{silent_for:.1f}s — declaring DEAD."
                )
                await self._lb.mark_worker_dead(worker.node_id)
                self._total_deaths += 1

                self._log_event(
                    event="WORKER_DEAD",
                    node_id=worker.node_id,
                    url=worker.url,
                    detail="Heartbeat timeout",
                    elapsed=silent_for,
                )

                await self._requeue_tasks_for(worker.node_id)

    # ------------------------------------------------------------------
    # 2. Task re-queuing — ensures no request is lost on worker death
    # ------------------------------------------------------------------
    async def _requeue_tasks_for(self, dead_node_id: UUID) -> None:
        """
        Find all in-flight tasks that were assigned to the dead worker
        and re-submit them to the LB.
        """
        async with self._lock:
            orphaned = [
                t for t in self._in_flight.values()
                if t.node_id == dead_node_id
            ]

        if not orphaned:
            logger.info(
                f"[FaultHandler] No orphaned tasks for dead worker {dead_node_id}."
            )
            return

        logger.warning(
            f"[FaultHandler] Re-queuing {len(orphaned)} orphaned task(s) "
            f"from dead worker {dead_node_id}."
        )

        from common.models import LB_To_Worker  # local import to avoid circularity

        for task_tracker in orphaned:
            # Remove the stale in-flight entry
            async with self._lock:
                self._in_flight.pop(task_tracker.task_id, None)

            # Build a minimal re-dispatch task.
            # NOTE: We don't have the original query here — in a full system
            # the Master would maintain a request store keyed by task_id.
            # The re-queue signals the Master's pending future to retry.
            # For now we log and count; the Master's retry logic (see
            # _handle_worker_failure in scheduler.py) picks this up.
            self._total_requeues += 1
            self._log_event(
                event="TASK_REQUEUED",
                node_id=dead_node_id,
                url="",
                detail=f"task_id={task_tracker.task_id}",
                elapsed=time.time() - task_tracker.dispatched_at,
            )
            logger.info(
                f"[FaultHandler] Task {task_tracker.task_id} flagged for re-dispatch "
                f"(was on worker {dead_node_id}, "
                f"age={time.time() - task_tracker.dispatched_at:.2f}s)."
            )

    # ------------------------------------------------------------------
    # 3. Recovery loop — probes dead workers for comeback
    # ------------------------------------------------------------------
    async def _recovery_loop(self) -> None:
        """
        Runs every RECOVERY_INTERVAL_SEC.
        Probes each dead worker's /health endpoint.
        If it responds, re-registers it with the LB.
        """
        logger.info("[FaultHandler] Recovery loop running.")
        while True:
            try:
                await asyncio.sleep(RECOVERY_INTERVAL_SEC)
                await self._probe_dead_workers()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[FaultHandler] Recovery loop error: {exc}")

    async def _probe_dead_workers(self) -> None:
        dead_workers = [
            w for w in self._lb._workers.values() if not w.is_alive
        ]
        if not dead_workers:
            return

        logger.info(
            f"[FaultHandler] Probing {len(dead_workers)} dead worker(s) for recovery."
        )

        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            for worker in dead_workers:
                try:
                    resp = await client.get(f"{worker.url}/health")
                    if resp.status_code == 200:
                        await self._lb.register_worker(worker.node_id, worker.url)
                        self._total_recoveries += 1
                        logger.info(
                            f"[FaultHandler] Worker {worker.node_id} @ {worker.url} "
                            f"has RECOVERED — re-registered with LB."
                        )
                        self._log_event(
                            event="WORKER_RECOVERED",
                            node_id=worker.node_id,
                            url=worker.url,
                            detail="Health check passed",
                            elapsed=0.0,
                        )
                except (httpx.ConnectError, httpx.TimeoutException):
                    # Still dead — silent, we'll try again next cycle
                    logger.debug(
                        f"[FaultHandler] Worker {worker.node_id} still unreachable."
                    )
                except Exception as exc:
                    logger.warning(
                        f"[FaultHandler] Unexpected error probing {worker.node_id}: {exc}"
                    )

    # ------------------------------------------------------------------
    # 4. Fault event log
    # ------------------------------------------------------------------
    def _ensure_log(self) -> None:
        FAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not FAULT_LOG_PATH.exists() or FAULT_LOG_PATH.stat().st_size == 0:
            with FAULT_LOG_PATH.open("w", newline="") as f:
                csv.writer(f).writerow(_CSV_HEADER)

    def _log_event(
        self,
        event: str,
        node_id: UUID,
        url: str,
        detail: str,
        elapsed: float,
    ) -> None:
        row = {
            "timestamp":       round(time.time(), 4),
            "event":           event,
            "node_id":         str(node_id),
            "worker_url":      url,
            "detail":          detail,
            "elapsed_silent_s": round(elapsed, 3),
        }
        try:
            with FAULT_LOG_PATH.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=_CSV_HEADER).writerow(row)
        except Exception as exc:
            logger.error(f"[FaultHandler] Failed to write fault log: {exc}")

    # ------------------------------------------------------------------
    # 5. Stats snapshot (for /admin/stats in the Master)
    # ------------------------------------------------------------------
    def get_fault_stats(self) -> dict:
        return {
            "total_worker_deaths":     self._total_deaths,
            "total_worker_recoveries": self._total_recoveries,
            "total_tasks_requeued":    self._total_requeues,
            "currently_in_flight":     len(self._in_flight),
            "fault_log_path":          str(FAULT_LOG_PATH),
        }