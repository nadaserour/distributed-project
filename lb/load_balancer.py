# lb/load_balancer.py
#
# Dynamic Load Balancer — Weighted Least Connections (WLC)
#
# Strategy:
#   For each available worker, compute a score:
#       score = active_tasks / weight
#   where weight is derived from the worker's free VRAM and inverse CPU usage.
#   The worker with the LOWEST score receives the next task.
#
#   If ALL workers are saturated (at capacity), requests are queued in a
#   FIFO buffer and re-dispatched as soon as any worker frees up.
#
# The Master calls   lb.dispatch(task)   and awaits the result.
# Workers push heartbeats to lb.update_worker_state(heartbeat).
# The Fault Handler calls lb.mark_worker_dead(node_id) to evict a node.

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import httpx # type: ignore

from common.models import LB_To_Worker, Worker_Heartbeat, Worker_To_Master

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables, we can configure these later based on real-world performance and load patterns.
# ---------------------------------------------------------------------------
MAX_TASKS_PER_WORKER = 10        # hard cap before a worker is "saturated"
QUEUE_MAXSIZE         = 500      # max pending requests before backpressure kicks in
WORKER_TIMEOUT_SEC    = 300     # seconds before a silent worker is considered dead
HTTP_REQUEST_TIMEOUT  = 300    # timeout for the actual LLM call to a worker
RETRY_ATTEMPTS        = 2        # how many times to retry on transient failures


# ---------------------------------------------------------------------------
# Internal worker state record
# ---------------------------------------------------------------------------
@dataclass
class WorkerState:
    node_id: UUID
    url: str                          # e.g. "http://192.168.1.x:8001"
    active_task_count: int  = 0
    cpu_usage_percent: float = 0.0
    gpu_vram_free: float     = 100.0  # GB — high default so new workers get traffic first
    last_seen: float         = field(default_factory=time.time)
    is_alive: bool           = True

    # ------------------------------------------------------------------
    # Weight: higher free VRAM + lower CPU = higher weight = lower score
    # Weight is clamped to [0.1, 10.0] to avoid division issues.
    # ------------------------------------------------------------------
    @property
    def weight(self) -> float:
        vram_score = max(0.1, self.gpu_vram_free)           # more free VRAM → heavier
        cpu_score  = max(0.1, 100.0 - self.cpu_usage_percent)  # lower CPU → heavier
        raw = (vram_score * cpu_score) / 1000.0
        return max(0.1, min(raw, 10.0))

    # ------------------------------------------------------------------
    # WLC score — lower is better (we pick the minimum).
    # A saturated worker gets +inf so it is never chosen.
    # ------------------------------------------------------------------
    @property
    def wlc_score(self) -> float:
        if not self.is_alive:
            return float("inf")
        if self.active_task_count >= MAX_TASKS_PER_WORKER:
            return float("inf")
        return self.active_task_count / self.weight

    def is_stale(self) -> bool:
        return (time.time() - self.last_seen) > WORKER_TIMEOUT_SEC


# ---------------------------------------------------------------------------
# Load Balancer
# ---------------------------------------------------------------------------
class LoadBalancer:
    """
    Thread-safe (asyncio) load balancer.

    Public API (called by Master / Fault Handler):
        await lb.dispatch(task: LB_To_Worker) -> Worker_To_Master
        lb.register_worker(node_id, url)
        lb.update_worker_state(heartbeat: Worker_Heartbeat)
        lb.mark_worker_dead(node_id: UUID)
        lb.get_worker_stats() -> list[dict]
    """

    def __init__(self) -> None:
        self._workers: dict[UUID, WorkerState] = {}
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[tuple[LB_To_Worker, asyncio.Future]] = asyncio.Queue(
            maxsize=QUEUE_MAXSIZE
        )
        self._dispatcher_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background dispatcher loop. Call once from app startup."""
        if self._dispatcher_task is None or self._dispatcher_task.done():
            self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())
            logger.info("[LB] Dispatcher loop started.")

    def stop(self) -> None:
        if self._dispatcher_task:
            self._dispatcher_task.cancel()

    # ------------------------------------------------------------------
    # Worker registration & state management
    # ------------------------------------------------------------------
    async def register_worker(self, node_id: UUID, url: str) -> None:
        async with self._lock:
            if node_id not in self._workers:
                self._workers[node_id] = WorkerState(node_id=node_id, url=url)
                logger.info(f"[LB] Registered worker {node_id} @ {url}")
            else:
                # Re-registration after a restart — mark alive again
                self._workers[node_id].url       = url
                self._workers[node_id].is_alive  = True
                self._workers[node_id].last_seen = time.time()
                logger.info(f"[LB] Re-registered worker {node_id} @ {url}")

    async def update_worker_state(self, heartbeat: Worker_Heartbeat) -> None:
        """Called by the Master every time a heartbeat arrives from a worker."""
        async with self._lock:
            node_id = heartbeat.node_id
            if node_id not in self._workers:
                logger.warning(
                    f"[LB] Heartbeat from unknown worker {node_id} — ignoring. "
                    "Register the worker first."
                )
                return
            w = self._workers[node_id]
            w.cpu_usage_percent = heartbeat.cpu_usage_percent
            w.gpu_vram_free     = heartbeat.gpu_vram_free
            w.active_task_count = heartbeat.current_load_count
            w.last_seen         = heartbeat.last_seen
            w.is_alive          = (heartbeat.status != "dead")
        logger.debug(
            f"[LB] Heartbeat from {node_id}: "
            f"tasks={heartbeat.current_load_count}, "
            f"cpu={heartbeat.cpu_usage_percent:.1f}%, "
            f"vram_free={heartbeat.gpu_vram_free:.2f}GB"
        )

    async def mark_worker_dead(self, node_id: UUID) -> None:
        """Called by the Fault Handler when a worker stops responding."""
        async with self._lock:
            if node_id in self._workers:
                self._workers[node_id].is_alive = False
                logger.warning(f"[LB] Worker {node_id} marked DEAD.")

    # ------------------------------------------------------------------
    # Core: pick the best worker (Weighted Least Connections)
    # ------------------------------------------------------------------
    async def _pick_worker(self) -> Optional[WorkerState]:
        """
        Returns the WorkerState with the lowest WLC score, or None if all
        workers are saturated / dead / stale.
        """
        async with self._lock:
            # Evict workers that have gone silent
            stale = [nid for nid, w in self._workers.items() if w.is_stale()]
            for nid in stale:
                self._workers[nid].is_alive = False
                logger.warning(f"[LB] Worker {nid} evicted — no heartbeat for {WORKER_TIMEOUT_SEC}s.")

            candidates = [w for w in self._workers.values() if w.is_alive]
            if not candidates:
                return None

            best = min(candidates, key=lambda w: w.wlc_score)
            if best.wlc_score == float("inf"):
                return None   # all saturated

            # Optimistically increment so the next pick sees this worker as busier
            best.active_task_count += 1
            return best

    async def _release_worker(self, node_id: UUID) -> None:
        """Decrement task counter after a task completes (or fails)."""
        async with self._lock:
            if node_id in self._workers:
                self._workers[node_id].active_task_count = max(
                    0, self._workers[node_id].active_task_count - 1
                )

    # ------------------------------------------------------------------
    # HTTP call to a worker
    # ------------------------------------------------------------------
    async def _call_worker(
        self, worker: WorkerState, task: LB_To_Worker
    ) -> Worker_To_Master:
        """
        POST the task to the chosen worker's /generate endpoint.
        Retries RETRY_ATTEMPTS times on transient errors, then raises.
        """
        payload = {
            "task_id":         str(task.task_id),
            "lb_dispatched_at": task.lb_dispatched_at,
            "instruction":     task.instruction,
            "parameters":      task.parameters,
        }

        last_exc: Exception = RuntimeError("No attempts made")
        async with httpx.AsyncClient(timeout=HTTP_REQUEST_TIMEOUT) as client:
            for attempt in range(1, RETRY_ATTEMPTS + 2):   # +1 for initial try
                try:
                    resp = await client.post(
                        f"{worker.url}/generate",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # Reconstruct the dataclass from JSON
                    return Worker_To_Master(
                        task_id              = UUID(data["task_id"]),
                        worker_id            = data["worker_id"],
                        response_text        = data["response_text"],
                        model_used           = data.get("model_used", "unknown"),
                        provider             = data.get("provider", "unknown"),
                        worker_received_at   = data["worker_received_at"],
                        inference_start      = data["inference_start"],
                        inference_end        = data["inference_end"],
                        metrics              = data.get("metrics", {}),
                        status               = data.get("status", "success"),
                    )
                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    last_exc = exc
                    logger.warning(
                        f"[LB] Worker {worker.node_id} attempt {attempt} failed: {exc}. "
                        f"{'Retrying…' if attempt <= RETRY_ATTEMPTS else 'Giving up.'}"
                    )
                    if attempt <= RETRY_ATTEMPTS:
                        await asyncio.sleep(0.5 * attempt)   # brief back-off
                except httpx.HTTPStatusError as exc:
                    # 4xx errors are not retryable
                    raise RuntimeError(
                        f"Worker {worker.node_id} returned HTTP {exc.response.status_code}"
                    ) from exc

        raise RuntimeError(
            f"Worker {worker.node_id} failed after {RETRY_ATTEMPTS + 1} attempts"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public dispatch — called by the Master
    # ------------------------------------------------------------------
    async def dispatch(self, task: LB_To_Worker) -> Worker_To_Master:
        """
        Route the task to the best available worker.
        If all workers are at capacity, park the task in the FIFO queue
        and await until a slot opens.

        Raises:
            asyncio.QueueFull  — if even the queue is full (signal backpressure
                                 to the Master so it can reject the request).
            RuntimeError       — if the selected worker fails after retries.
        """
        worker = await self._pick_worker()
        if worker is not None:
            # Fast path: a worker is ready right now
            return await self._execute(worker, task)

        # Slow path: queue the request
        logger.info(f"[LB] All workers busy — queuing task {task.task_id}")
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        try:
            self._queue.put_nowait((task, future))
        except asyncio.QueueFull:
            raise asyncio.QueueFull(
                "[LB] Request buffer full — backpressure active. Reject this request."
            )

        # Await the future; the dispatcher loop will resolve it
        return await future

    async def _execute(
        self, worker: WorkerState, task: LB_To_Worker
    ) -> Worker_To_Master:
        """Run the HTTP call and always release the worker slot afterwards."""
        node_id = worker.node_id
        try:
            logger.info(
                f"[LB] → Dispatching task {task.task_id} to worker {node_id} "
                f"(score={worker.wlc_score:.3f})"
            )
            result = await self._call_worker(worker, task)
            logger.info(
                f"[LB] ✓ Task {task.task_id} completed by worker {node_id} "
                f"in {result.inference_end - result.inference_start:.3f}s"
            )
            return result
        except Exception as exc:
            logger.error(f"[LB] ✗ Task {task.task_id} failed on worker {node_id}: {exc}")
            # Mark worker dead so the Fault Handler can take over
            await self.mark_worker_dead(node_id)
            raise
        finally:
            await self._release_worker(node_id)

    # ------------------------------------------------------------------
    # Background dispatcher loop — drains the queue as workers free up
    # ------------------------------------------------------------------
    async def _dispatcher_loop(self) -> None:
        """
        Continuously pulls queued tasks and sends them to the first available
        worker.  Sleeps briefly when all workers are still busy.
        """
        logger.info("[LB] Background dispatcher loop running.")
        while True:
            try:
                task, future = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue   # nothing in queue — loop back
            except asyncio.CancelledError:
                break

            # Keep trying until a worker is available
            while True:
                worker = await self._pick_worker()
                if worker is not None:
                    break
                await asyncio.sleep(0.1)   # brief wait, then retry

            # Run the task without blocking the loop
            asyncio.create_task(self._dispatch_queued(worker, task, future))
            self._queue.task_done()

    async def _dispatch_queued(
        self,
        worker: WorkerState,
        task: LB_To_Worker,
        future: asyncio.Future,
    ) -> None:
        """Execute a queued task and resolve (or reject) its future."""
        try:
            result = await self._execute(worker, task)
            if not future.done():
                future.set_result(result)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    def get_worker_stats(self) -> list[dict]:
        """
        Returns a snapshot of all worker states.
        Useful for the Master's /admin/stats endpoint.
        """
        return [
            {
                "node_id":           str(w.node_id),
                "url":               w.url,
                "is_alive":          w.is_alive,
                "active_task_count": w.active_task_count,
                "cpu_usage_percent": round(w.cpu_usage_percent, 1),
                "gpu_vram_free_gb":  round(w.gpu_vram_free, 2),
                "weight":            round(w.weight, 4),
                "wlc_score":         round(w.wlc_score, 4) if w.wlc_score != float("inf") else "inf",
                "last_seen_ago_s":   round(time.time() - w.last_seen, 1),
            }
            for w in self._workers.values()
        ]

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def alive_worker_count(self) -> int:
        return sum(1 for w in self._workers.values() if w.is_alive)