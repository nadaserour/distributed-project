import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import httpx  # type: ignore

from common.models import LB_To_Worker, Worker_Heartbeat, Worker_To_Master

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optimizing Tunables for Parallel GPU Batching & Connection Pooling
# ---------------------------------------------------------------------------
MAX_TASKS_PER_WORKER = 32       # Raised to leverage GPU continuous batching (vLLM/Ollama)
QUEUE_MAXSIZE         = 1000     # Expanded buffer size for high-throughput testing
WORKER_TIMEOUT_SEC    = 30       # Faster eviction of silent/dead workers (was 300s)
HTTP_REQUEST_TIMEOUT  = 120      # Defensive timeout limit for generation (was 300s)
RETRY_ATTEMPTS        = 1        # Reduced retries to avoid compounding queue delays


@dataclass
class WorkerState:
    node_id: UUID
    url: str                          
    active_task_count: int  = 0
    cpu_usage_percent: float = 0.0
    gpu_vram_free: float     = 100.0  
    last_seen: float         = field(default_factory=time.time)
    is_alive: bool           = True

    @property
    def weight(self) -> float:
        vram_score = max(0.1, self.gpu_vram_free)           
        cpu_score  = max(0.1, 100.0 - self.cpu_usage_percent)  
        raw = (vram_score * cpu_score) / 1000.0
        return max(0.1, min(raw, 10.0))

    @property
    def wlc_score(self) -> float:
        if not self.is_alive:
            return float("inf")
        if self.active_task_count >= MAX_TASKS_PER_WORKER:
            return float("inf")
        return self.active_task_count / self.weight

    def is_stale(self) -> bool:
        return (time.time() - self.last_seen) > WORKER_TIMEOUT_SEC


class LoadBalancer:
    def __init__(self) -> None:
        self._workers: dict[UUID, WorkerState] = {}
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[tuple[LB_To_Worker, asyncio.Future]] = asyncio.Queue(
            maxsize=QUEUE_MAXSIZE
        )
        self._dispatcher_task: Optional[asyncio.Task] = None
        
        # ── SUCCESS FIX: Long-lived persistent connection pool ────────────────
        # This keeps TCP connections alive (keep-alive) to completely bypass connection handshakes!
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
        self._client = httpx.AsyncClient(limits=limits, timeout=HTTP_REQUEST_TIMEOUT)

    def start(self) -> None:
        if self._dispatcher_task is None or self._dispatcher_task.done():
            self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())
            logger.info("[LB] Dynamic Dispatcher loop running with Keep-Alive pool.")

    def stop(self) -> None:
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
        # Clean up client session close on shutdown
        asyncio.create_task(self._client.aclose())

    async def register_worker(self, node_id: UUID, url: str) -> None:
        async with self._lock:
            if node_id not in self._workers:
                self._workers[node_id] = WorkerState(node_id=node_id, url=url)
                logger.info(f"[LB] Registered worker {node_id} @ {url}")
            else:
                self._workers[node_id].url       = url
                self._workers[node_id].is_alive  = True
                self._workers[node_id].last_seen = time.time()
                logger.info(f"[LB] Re-registered worker {node_id} @ {url}")

    async def update_worker_state(self, heartbeat: Worker_Heartbeat) -> None:
        async with self._lock:
            node_id = heartbeat.node_id
            if node_id not in self._workers:
                return
            w = self._workers[node_id]
            w.cpu_usage_percent = heartbeat.cpu_usage_percent
            w.gpu_vram_free     = heartbeat.gpu_vram_free
            w.active_task_count = heartbeat.current_load_count
            w.last_seen         = heartbeat.last_seen
            w.is_alive          = (heartbeat.status != "dead")

    async def mark_worker_dead(self, node_id: UUID) -> None:
        async with self._lock:
            if node_id in self._workers:
                self._workers[node_id].is_alive = False
                logger.warning(f"[LB] Worker {node_id} marked DEAD.")

    async def _pick_worker(self) -> Optional[WorkerState]:
        async with self._lock:
            stale = [nid for nid, w in self._workers.items() if w.is_stale()]
            for nid in stale:
                self._workers[nid].is_alive = False
                logger.warning(f"[LB] Worker {nid} evicted due to silence.")

            candidates = [w for w in self._workers.values() if w.is_alive]
            if not candidates:
                return None

            best = min(candidates, key=lambda w: w.wlc_score)
            if best.wlc_score == float("inf"):
                return None

            best.active_task_count += 1
            return best

    async def _release_worker(self, node_id: UUID) -> None:
        async with self._lock:
            if node_id in self._workers:
                self._workers[node_id].active_task_count = max(
                    0, self._workers[node_id].active_task_count - 1
                )

    # ── SUCCESS FIX: Uses global keep-alive client session instead of spinning one up ──
    async def _call_worker(self, worker: WorkerState, task: LB_To_Worker) -> Worker_To_Master:
        payload = {
            "task_id":          str(task.task_id),
            "lb_dispatched_at": task.lb_dispatched_at,
            "instruction":      task.instruction,
            "parameters":       task.parameters,
        }

        last_exc: Exception = RuntimeError("No attempts made")
        
        for attempt in range(1, RETRY_ATTEMPTS + 2):
            try:
                # Reusing client session across all tasks bypassing DNS and TCP handshakes!
                resp = await self._client.post(
                    f"{worker.url}/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                if "task_id" not in data or "worker_id" not in data:
                    raise RuntimeError(
                        f"Worker {worker.node_id} returned malformed response: "
                        f"status={data.get('status')}, detail={data.get('detail', 'n/a')}"
                    )

                return Worker_To_Master(
                    task_id              = UUID(data["task_id"]),
                    worker_id            = data["worker_id"],
                    response_text        = data.get("response_text", ""),
                    model_used           = data.get("model_used", "unknown"),
                    provider             = data.get("provider", "unknown"),
                    worker_received_at   = data.get("worker_received_at", 0),
                    inference_start      = data.get("inference_start", 0),
                    inference_end        = data.get("inference_end", 0),
                    metrics              = data.get("metrics", {}),
                    status               = data.get("status", "success"),
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning(f"[LB] Connection issue on worker {worker.node_id} (Attempt {attempt})")
                if attempt <= RETRY_ATTEMPTS:
                    await asyncio.sleep(0.2 * attempt)
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(f"Worker {worker.node_id} returned HTTP {exc.response.status_code}") from exc

        raise RuntimeError(f"Worker {worker.node_id} offline/timed out.") from last_exc

    async def dispatch(self, task: LB_To_Worker) -> Worker_To_Master:
        worker = await self._pick_worker()
        if worker is not None:
            return await self._execute(worker, task)

        logger.info(f"[LB] Saturated! Queuing task {task.task_id}")
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        try:
            self._queue.put_nowait((task, future))
        except asyncio.QueueFull:
            raise asyncio.QueueFull("[LB] Capacity buffer full.")

        return await future

    async def _execute(self, worker: WorkerState, task: LB_To_Worker) -> Worker_To_Master:
        node_id = worker.node_id
        try:
            result = await self._call_worker(worker, task)
            return result
        except Exception as exc:
            logger.error(f"[LB] ✗ Task {task.task_id} failed: {exc}")
            await self.mark_worker_dead(node_id)
            raise
        finally:
            await self._release_worker(node_id)

    async def _dispatcher_loop(self) -> None:
        """Continuously processes queued tasks as worker capacities open up."""
        while True:
            try:
                task, future = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            while True:
                worker = await self._pick_worker()
                if worker is not None:
                    break
                # Tuned queue sleep down to 10ms to maximize pipeline execution speed
                await asyncio.sleep(0.01)

            asyncio.create_task(self._dispatch_queued(worker, task, future))
            self._queue.task_done()

    async def _dispatch_queued(
        self, worker: WorkerState, task: LB_To_Worker, future: asyncio.Future
    ) -> None:
        try:
            result = await self._execute(worker, task)
            if not future.done():
                future.set_result(result)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)

    def get_worker_stats(self) -> list[dict]:
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