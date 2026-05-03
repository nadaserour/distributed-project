# ignore it
# tests/test_master.py
#
# Integration tests for master/scheduler.py
#
# Covers:
#   1.  Auth — valid key, missing key, wrong key
#   2.  Admission control / backpressure (429)
#   3.  Successful query end-to-end
#   4.  Result persistence and retrieval via /result
#   5.  Worker registration via /workers/register
#   6.  Heartbeat forwarded to LB via /heartbeat
#   7.  /admin/stats structure
#   8.  Manual mark-dead via /admin/mark-dead
#   9.  503 when all workers are down
#   10. CSV log written on success and failure

import asyncio
import csv
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest # type: ignore
from httpx import AsyncClient, ASGITransport # type: ignore

from common.models import LB_To_Worker, Worker_To_Master
from lb.load_balancer import LoadBalancer
from fault_tolerance.fault_handler import FaultHandler
import master.scheduler as sched


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _user_payload(query: str = "What is a diffusion model?") -> dict:
    return {
        "user_id":      "test-user",
        "query":        query,
        "user_sent_at": time.time(),
        "parameters":   {},
    }


def _good_worker_result(task_id) -> Worker_To_Master:
    now = time.time()
    return Worker_To_Master(
        task_id            = task_id,
        worker_id          = "fake-worker",
        response_text      = "Diffusion models iteratively denoise data.",
        model_used         = "fake",
        provider           = "fake",
        worker_received_at = now,
        inference_start    = now,
        inference_end      = now + 0.1,
        metrics            = {},
        status             = "success",
    )


# ─────────────────────────────────────────────────────────────
# Fixture: isolated master with mocked LB dispatch
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def fresh_master():
    """
    Returns a factory function that yields an AsyncClient
    with lb.dispatch pre-mocked to return a good result.
    Also patches the CSV path so tests don't write to real logs/.
    """
    import tempfile, os

    async def _build(dispatch_side_effect=None):
        fresh_lb = LoadBalancer()
        fresh_fh = FaultHandler(fresh_lb)
        tmp_csv = Path(tempfile.mktemp(suffix=".csv"))

        with (
            patch.object(sched, "lb", fresh_lb),
            patch.object(sched, "fh", fresh_fh),
            patch.object(sched, "LOG_CSV", tmp_csv),
        ):
            sched._ensure_csv_header()   # re-init header for tmp file

            if dispatch_side_effect is not None:
                fresh_lb.dispatch = AsyncMock(side_effect=dispatch_side_effect)
            else:
                async def _auto_dispatch(task: LB_To_Worker):
                    return _good_worker_result(task.task_id)
                fresh_lb.dispatch = AsyncMock(side_effect=_auto_dispatch)

            transport = ASGITransport(app=sched.app)
            async with AsyncClient(
                transport=transport,
                base_url="http://master",
                headers={"x-api-key": "dev-key-1"},
            ) as client:
                yield client, fresh_lb, fresh_fh, tmp_csv

    return _build


# ─────────────────────────────────────────────────────────────
# 1. Auth
# ─────────────────────────────────────────────────────────────
class TestAuth:

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_401(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await AsyncClient(
                transport=ASGITransport(app=sched.app),
                base_url="http://master",
            ).post("/query", json=_user_payload())
            assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_api_key_returns_401(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await AsyncClient(
                transport=ASGITransport(app=sched.app),
                base_url="http://master",
                headers={"x-api-key": "totally-wrong-key"},
            ).post("/query", json=_user_payload())
            assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_passes(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await client.post("/query", json=_user_payload())
            assert r.status_code == 200


# ─────────────────────────────────────────────────────────────
# 2. Backpressure
# ─────────────────────────────────────────────────────────────
class TestBackpressure:

    @pytest.mark.asyncio
    async def test_full_queue_returns_429(self, fresh_master):
        async def _always_queue_full(task):
            raise asyncio.QueueFull()

        async for client, lb, fh, _ in fresh_master(dispatch_side_effect=_always_queue_full):
            r = await client.post("/query", json=_user_payload())
            assert r.status_code == 429
            assert "queue" in r.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_overload_gate_returns_429(self, fresh_master):
        """Simulate lb.queue_depth >= MAX_QUEUE at the Master gate."""
        async for client, lb, fh, _ in fresh_master():
            with patch.object(type(lb), "queue_depth",
                              new_callable=lambda: property(lambda self: 99999)):
                r = await client.post("/query", json=_user_payload())
                assert r.status_code == 429


# ─────────────────────────────────────────────────────────────
# 3. Successful query
# ─────────────────────────────────────────────────────────────
class TestQuerySuccess:

    @pytest.mark.asyncio
    async def test_response_shape(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await client.post("/query", json=_user_payload())
            assert r.status_code == 200
            body = r.json()
            for key in ("request_id", "status", "answer", "total_latency"):
                assert key in body, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_answer_content(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await client.post("/query", json=_user_payload())
            assert "diffusion" in r.json()["answer"].lower()

    @pytest.mark.asyncio
    async def test_total_latency_is_positive(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await client.post("/query", json=_user_payload())
            assert r.json()["total_latency"] > 0

    @pytest.mark.asyncio
    async def test_request_id_is_valid_uuid(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await client.post("/query", json=_user_payload())
            rid = r.json()["request_id"]
            uuid.UUID(rid)   # raises if invalid


# ─────────────────────────────────────────────────────────────
# 4. Result persistence
# ─────────────────────────────────────────────────────────────
class TestResultCache:

    @pytest.mark.asyncio
    async def test_result_retrievable_after_query(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            post_r = await client.post("/query", json=_user_payload())
            rid = post_r.json()["request_id"]
            get_r = await client.get(f"/result/{rid}")
            assert get_r.status_code == 200
            assert get_r.json()["request_id"] == rid

    @pytest.mark.asyncio
    async def test_missing_result_returns_404(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await client.get(f"/result/{uuid.uuid4()}")
            assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_cached_answer_matches_query_response(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            post_r = await client.post("/query", json=_user_payload())
            body = post_r.json()
            get_r = await client.get(f"/result/{body['request_id']}")
            assert get_r.json()["answer"] == body["answer"]


# ─────────────────────────────────────────────────────────────
# 5. Worker registration
# ─────────────────────────────────────────────────────────────
class TestWorkerRegistration:

    @pytest.mark.asyncio
    async def test_register_worker_accepted(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            payload = {"node_id": str(uuid.uuid4()), "url": "http://w1:8001"}
            # Registration endpoint doesn't need API key
            r = await AsyncClient(
                transport=ASGITransport(app=sched.app),
                base_url="http://master",
            ).post("/workers/register", json=payload)
            assert r.status_code == 200
            assert r.json()["status"] == "registered"

    @pytest.mark.asyncio
    async def test_registered_worker_appears_in_stats(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            nid = str(uuid.uuid4())
            await AsyncClient(
                transport=ASGITransport(app=sched.app),
                base_url="http://master",
            ).post("/workers/register", json={"node_id": nid, "url": "http://w1:8001"})
            stats = await client.get("/admin/stats")
            node_ids = [w["node_id"] for w in stats.json()["workers"]]
            assert nid in node_ids


# ─────────────────────────────────────────────────────────────
# 6. Heartbeat forwarding
# ─────────────────────────────────────────────────────────────
class TestHeartbeat:

    @pytest.mark.asyncio
    async def test_heartbeat_updates_lb_state(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            nid = uuid.uuid4()

            # Register first (so LB knows the worker)
            await lb.register_worker(nid, "http://w:8001")

            hb_payload = {
                "node_id":             str(nid),
                "status":              "ready",
                "current_load_count":  7,
                "cpu_usage_percent":   42.0,
                "gpu_vram_free":       6.0,
                "last_seen":           time.time(),
            }
            r = await AsyncClient(
                transport=ASGITransport(app=sched.app),
                base_url="http://master",
            ).post("/heartbeat", json=hb_payload)
            assert r.status_code == 200
            # Verify LB state was updated
            assert lb._workers[nid].active_task_count == 7
            assert lb._workers[nid].cpu_usage_percent == 42.0


# ─────────────────────────────────────────────────────────────
# 7. Admin stats
# ─────────────────────────────────────────────────────────────
class TestAdminStats:

    @pytest.mark.asyncio
    async def test_stats_shape(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await client.get("/admin/stats")
            assert r.status_code == 200
            body = r.json()
            for key in ("queue_depth", "alive_workers", "total_workers",
                        "backpressure_active", "workers", "fault_stats"):
                assert key in body, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_stats_requires_auth(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await AsyncClient(
                transport=ASGITransport(app=sched.app),
                base_url="http://master",
            ).get("/admin/stats")
            assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_fault_stats_fields(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            r = await client.get("/admin/stats")
            fs = r.json()["fault_stats"]
            for key in ("total_worker_deaths", "total_worker_recoveries",
                        "total_tasks_requeued", "currently_in_flight"):
                assert key in fs


# ─────────────────────────────────────────────────────────────
# 8. Manual mark-dead
# ─────────────────────────────────────────────────────────────
class TestMarkDead:

    @pytest.mark.asyncio
    async def test_mark_dead_endpoint(self, fresh_master):
        async for client, lb, fh, _ in fresh_master():
            nid = uuid.uuid4()
            await lb.register_worker(nid, "http://w:8001")
            assert lb._workers[nid].is_alive is True

            r = await client.post(f"/admin/mark-dead/{nid}")
            assert r.status_code == 200
            assert lb._workers[nid].is_alive is False


# ─────────────────────────────────────────────────────────────
# 9. 503 when all workers unavailable
# ─────────────────────────────────────────────────────────────
class TestWorkerUnavailable:

    @pytest.mark.asyncio
    async def test_503_when_dispatch_raises_runtime_error(self, fresh_master):
        async def _always_fail(task):
            raise RuntimeError("all workers dead")

        async for client, lb, fh, _ in fresh_master(dispatch_side_effect=_always_fail):
            r = await client.post("/query", json=_user_payload())
            assert r.status_code == 503


# ─────────────────────────────────────────────────────────────
# 10. CSV logging
# ─────────────────────────────────────────────────────────────
class TestCSVLogging:

    @pytest.mark.asyncio
    async def test_csv_row_written_on_success(self, fresh_master):
        async for client, lb, fh, tmp_csv in fresh_master():
            await client.post("/query", json=_user_payload())
            rows = list(csv.DictReader(tmp_csv.open()))
            assert len(rows) == 1
            assert rows[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_csv_row_written_on_failure(self, fresh_master):
        async def _fail(task):
            raise RuntimeError("boom")

        async for client, lb, fh, tmp_csv in fresh_master(dispatch_side_effect=_fail):
            await client.post("/query", json=_user_payload())
            rows = list(csv.DictReader(tmp_csv.open()))
            assert len(rows) == 1
            assert "error" in rows[0]["status"]

    @pytest.mark.asyncio
    async def test_csv_has_all_timing_columns(self, fresh_master):
        async for client, lb, fh, tmp_csv in fresh_master():
            await client.post("/query", json=_user_payload())
            rows = list(csv.DictReader(tmp_csv.open()))
            row = rows[0]
            for col in ("user_sent_at", "master_received_at", "dispatched_at",
                        "inference_start", "inference_end", "master_responded_at",
                        "total_latency_s", "inference_latency_s"):
                assert col in row and row[col], f"Column missing or empty: {col}"

    @pytest.mark.asyncio
    async def test_multiple_requests_produce_multiple_rows(self, fresh_master):
        async for client, lb, fh, tmp_csv in fresh_master():
            for _ in range(5):
                await client.post("/query", json=_user_payload())
            rows = list(csv.DictReader(tmp_csv.open()))
            assert len(rows) == 5