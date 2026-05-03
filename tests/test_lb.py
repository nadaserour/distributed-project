# you know the drill, testing wip
# tests/test_lb.py
#
# Unit tests for lb/load_balancer.py
#
# Covers:
#   1.  Worker registration
#   2.  WLC scoring and routing correctness
#   3.  Heartbeat updates shift routing
#   4.  Saturated workers are skipped
#   5.  Dead workers are skipped
#   6.  Stale workers are auto-evicted
#   7.  Queue fills when all workers saturated
#   8.  Queue drains when a worker frees up
#   9.  Retry on transient HTTP failure
#   10. Worker release always happens (even on exception)

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, patch, MagicMock

import pytest # type: ignore
import pytest_asyncio # type: ignore
from httpx import ConnectError # type: ignore

from lb.load_balancer import LoadBalancer, WorkerState, MAX_TASKS_PER_WORKER
from common.models import LB_To_Worker, Worker_To_Master, Worker_Heartbeat
from tests.conftest import make_task, make_heartbeat


# ─────────────────────────────────────────────────────────────
# 1. Worker Registration
# ─────────────────────────────────────────────────────────────
class TestWorkerRegistration:

    @pytest.mark.asyncio
    async def test_register_new_worker(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w1:8001")
        assert nid in lb._workers
        assert lb._workers[nid].url == "http://w1:8001"
        assert lb._workers[nid].is_alive is True

    @pytest.mark.asyncio
    async def test_register_same_worker_twice_is_idempotent(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w1:8001")
        await lb.register_worker(nid, "http://w1:8001")
        assert len(lb._workers) == 1

    @pytest.mark.asyncio
    async def test_reregister_dead_worker_revives_it(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w1:8001")
        await lb.mark_worker_dead(nid)
        assert lb._workers[nid].is_alive is False
        await lb.register_worker(nid, "http://w1:8001")
        assert lb._workers[nid].is_alive is True

    @pytest.mark.asyncio
    async def test_alive_worker_count(self, lb):
        ids = [uuid.uuid4() for _ in range(3)]
        for i, nid in enumerate(ids):
            await lb.register_worker(nid, f"http://w{i}:8001")
        assert lb.alive_worker_count == 3
        await lb.mark_worker_dead(ids[0])
        assert lb.alive_worker_count == 2


# ─────────────────────────────────────────────────────────────
# 2. WLC Scoring
# ─────────────────────────────────────────────────────────────
class TestWLCScoring:

    def test_idle_worker_has_zero_score(self):
        w = WorkerState(node_id=uuid.uuid4(), url="http://x:1")
        w.active_task_count = 0
        assert w.wlc_score == 0.0

    def test_dead_worker_score_is_inf(self):
        w = WorkerState(node_id=uuid.uuid4(), url="http://x:1")
        w.is_alive = False
        assert w.wlc_score == float("inf")

    def test_saturated_worker_score_is_inf(self):
        w = WorkerState(node_id=uuid.uuid4(), url="http://x:1")
        w.active_task_count = MAX_TASKS_PER_WORKER
        assert w.wlc_score == float("inf")

    def test_high_vram_lowers_score(self):
        """Worker with more free VRAM should win (lower score) at equal task count."""
        w_rich = WorkerState(node_id=uuid.uuid4(), url="http://x:1")
        w_poor = WorkerState(node_id=uuid.uuid4(), url="http://x:2")
        w_rich.gpu_vram_free = 16.0
        w_poor.gpu_vram_free = 2.0
        w_rich.active_task_count = 1
        w_poor.active_task_count = 1
        assert w_rich.wlc_score < w_poor.wlc_score

    def test_high_cpu_raises_score(self):
        """Worker burning CPU should lose (higher score) vs idle worker."""
        w_idle = WorkerState(node_id=uuid.uuid4(), url="http://x:1")
        w_busy = WorkerState(node_id=uuid.uuid4(), url="http://x:2")
        w_idle.cpu_usage_percent = 5.0
        w_busy.cpu_usage_percent = 90.0
        w_idle.active_task_count = 1
        w_busy.active_task_count = 1
        assert w_idle.wlc_score < w_busy.wlc_score


# ─────────────────────────────────────────────────────────────
# 3. Heartbeat updates routing
# ─────────────────────────────────────────────────────────────
class TestHeartbeats:

    @pytest.mark.asyncio
    async def test_heartbeat_updates_metrics(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w1:8001")
        hb = make_heartbeat(nid, load=3, cpu=55.0, vram=4.0)
        await lb.update_worker_state(hb)

        w = lb._workers[nid]
        assert w.active_task_count == 3
        assert w.cpu_usage_percent == 55.0
        assert w.gpu_vram_free == 4.0

    @pytest.mark.asyncio
    async def test_heartbeat_from_unknown_worker_is_ignored(self, lb):
        """Should not raise, just log a warning."""
        hb = make_heartbeat(uuid.uuid4())
        await lb.update_worker_state(hb)   # must not throw
        assert len(lb._workers) == 0

    @pytest.mark.asyncio
    async def test_routing_shifts_after_heartbeat(self, lb):
        """
        Register two workers, make worker-A appear loaded via heartbeat,
        then pick — should always pick worker-B.
        """
        id_a, id_b = uuid.uuid4(), uuid.uuid4()
        await lb.register_worker(id_a, "http://a:8001")
        await lb.register_worker(id_b, "http://b:8001")

        # Make A look loaded
        hb_a = make_heartbeat(id_a, load=8, cpu=80.0, vram=1.0)
        hb_b = make_heartbeat(id_b, load=0, cpu=10.0, vram=12.0)
        await lb.update_worker_state(hb_a)
        await lb.update_worker_state(hb_b)

        picked = await lb._pick_worker()
        assert picked is not None
        assert picked.node_id == id_b


# ─────────────────────────────────────────────────────────────
# 4 & 5. Saturated and dead workers are skipped
# ─────────────────────────────────────────────────────────────
class TestWorkerFiltering:

    @pytest.mark.asyncio
    async def test_saturated_worker_not_picked(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w1:8001")
        lb._workers[nid].active_task_count = MAX_TASKS_PER_WORKER
        picked = await lb._pick_worker()
        assert picked is None

    @pytest.mark.asyncio
    async def test_dead_worker_not_picked(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w1:8001")
        await lb.mark_worker_dead(nid)
        picked = await lb._pick_worker()
        assert picked is None

    @pytest.mark.asyncio
    async def test_picks_only_alive_worker_when_mixed(self, lb):
        dead_id  = uuid.uuid4()
        alive_id = uuid.uuid4()
        await lb.register_worker(dead_id,  "http://dead:8001")
        await lb.register_worker(alive_id, "http://alive:8001")
        await lb.mark_worker_dead(dead_id)
        picked = await lb._pick_worker()
        assert picked is not None
        assert picked.node_id == alive_id


# ─────────────────────────────────────────────────────────────
# 6. Stale worker auto-eviction
# ─────────────────────────────────────────────────────────────
class TestStaleEviction:

    @pytest.mark.asyncio
    async def test_stale_worker_evicted_on_pick(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w1:8001")
        # Wind back last_seen to trigger staleness
        lb._workers[nid].last_seen = time.time() - 9999
        picked = await lb._pick_worker()
        assert picked is None
        assert lb._workers[nid].is_alive is False

    @pytest.mark.asyncio
    async def test_fresh_worker_not_evicted(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w1:8001")
        lb._workers[nid].last_seen = time.time()   # just now
        picked = await lb._pick_worker()
        assert picked is not None


# ─────────────────────────────────────────────────────────────
# 7. Queue fills when all workers saturated
# ─────────────────────────────────────────────────────────────
class TestQueueing:

    @pytest.mark.asyncio
    async def test_queue_full_raises(self, lb):
        """
        With no workers registered, every dispatch hits the slow path.
        Once the queue is full, QueueFull should be raised.
        """
        from lb.load_balancer import QUEUE_MAXSIZE
        # Fill queue manually so we don't have to fire QUEUE_MAXSIZE tasks
        for _ in range(QUEUE_MAXSIZE):
            future = asyncio.get_event_loop().create_future()
            lb._queue.put_nowait((make_task(), future))

        with pytest.raises(asyncio.QueueFull):
            await lb.dispatch(make_task())

    @pytest.mark.asyncio
    async def test_optimistic_counter_released_on_failure(self, lb):
        """
        If a worker's HTTP call fails, its task counter must return to
        what it was before the pick — no permanent inflation.
        """
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        before = lb._workers[nid].active_task_count

        with patch.object(lb, "_call_worker", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                await lb.dispatch(make_task())

        # After failure, count should be back to where it started
        assert lb._workers[nid].active_task_count == before


# ─────────────────────────────────────────────────────────────
# 8. Successful end-to-end dispatch (mocked HTTP)
# ─────────────────────────────────────────────────────────────
class TestDispatch:

    def _mock_result(self, task: LB_To_Worker) -> Worker_To_Master:
        return Worker_To_Master(
            task_id            = task.task_id,
            worker_id          = "fake-worker",
            response_text      = "Mocked LLM answer",
            model_used         = "fake",
            provider           = "fake",
            worker_received_at = time.time(),
            inference_start    = time.time(),
            inference_end      = time.time() + 0.1,
            metrics            = {},
            status             = "success",
        )

    @pytest.mark.asyncio
    async def test_dispatch_returns_result(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        task = make_task()

        with patch.object(lb, "_call_worker", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = self._mock_result(task)
            result = await lb.dispatch(task)

        assert result.status == "success"
        assert result.response_text == "Mocked LLM answer"
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_counter_zero_after_dispatch(self, lb):
        """Counter must return to 0 after a successful dispatch."""
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        task = make_task()

        with patch.object(lb, "_call_worker", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = self._mock_result(task)
            await lb.dispatch(task)

        assert lb._workers[nid].active_task_count == 0

    @pytest.mark.asyncio
    async def test_wlc_routes_to_least_loaded(self, lb):
        """
        Two workers: A already has 3 tasks, B has 0.
        Next dispatch must go to B.
        """
        id_a, id_b = uuid.uuid4(), uuid.uuid4()
        await lb.register_worker(id_a, "http://a:8001")
        await lb.register_worker(id_b, "http://b:8001")
        lb._workers[id_a].active_task_count = 3

        chosen_id = None
        original_call = lb._call_worker

        async def spy_call(worker, task):
            nonlocal chosen_id
            chosen_id = worker.node_id
            return self._mock_result(task)

        with patch.object(lb, "_call_worker", side_effect=spy_call):
            await lb.dispatch(make_task())

        assert chosen_id == id_b


# ─────────────────────────────────────────────────────────────
# 9. Retry on transient HTTP failure
# ─────────────────────────────────────────────────────────────
class TestRetry:

    @pytest.mark.asyncio
    async def test_worker_marked_dead_after_all_retries_fail(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")

        with patch.object(lb, "_call_worker", side_effect=RuntimeError("unreachable")):
            with pytest.raises(RuntimeError):
                await lb.dispatch(make_task())

        assert lb._workers[nid].is_alive is False


# ─────────────────────────────────────────────────────────────
# 10. get_worker_stats snapshot
# ─────────────────────────────────────────────────────────────
class TestStats:

    @pytest.mark.asyncio
    async def test_stats_contains_all_workers(self, lb):
        ids = [uuid.uuid4() for _ in range(4)]
        for i, nid in enumerate(ids):
            await lb.register_worker(nid, f"http://w{i}:8001")
        stats = lb.get_worker_stats()
        assert len(stats) == 4

    @pytest.mark.asyncio
    async def test_stats_fields_present(self, lb):
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        stat = lb.get_worker_stats()[0]
        for key in ("node_id", "url", "is_alive", "active_task_count",
                    "cpu_usage_percent", "gpu_vram_free_gb", "weight", "wlc_score"):
            assert key in stat, f"Missing field: {key}"