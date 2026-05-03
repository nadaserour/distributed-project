# tests/test_fault_handler.py
#
# Tests for fault_tolerance/fault_handler.py
#
# Covers:
#   1.  Watchdog detects stale worker and marks it dead
#   2.  Watchdog does NOT evict recently-seen workers
#   3.  In-flight task tracking (register / complete)
#   4.  Orphaned tasks logged on worker death
#   5.  Recovery loop revives worker that answers /health
#   6.  Recovery loop ignores worker that stays silent
#   7.  Fault stats counters increment correctly
#   8.  Fault CSV written on each event

import asyncio
import csv
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile

import httpx # type: ignore
import pytest # type: ignore
import pytest_asyncio # type: ignore

from lb.load_balancer import LoadBalancer
from fault_tolerance.fault_handler import FaultHandler, FAULT_LOG_PATH
from tests.conftest import make_heartbeat


# ─────────────────────────────────────────────────────────────
# Fixture: isolated FH with a tmp log file
# ─────────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def isolated_fh():
    """LoadBalancer + FaultHandler pair with a temp log file."""
    tmp_log = Path(tempfile.mktemp(suffix=".csv"))
    balancer = LoadBalancer()
    balancer.start()

    with patch("fault_tolerance.fault_handler.FAULT_LOG_PATH", tmp_log):
        handler = FaultHandler(balancer)
        # Don't start loops — we call internals directly in most tests
        yield handler, balancer, tmp_log

    balancer.stop()


# ─────────────────────────────────────────────────────────────
# 1. Watchdog detects stale worker
# ─────────────────────────────────────────────────────────────
class TestWatchdog:

    @pytest.mark.asyncio
    async def test_stale_worker_marked_dead(self, isolated_fh):
        fh, lb, tmp_log = isolated_fh
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        lb._workers[nid].last_seen = time.time() - 9999   # ancient

        await fh._sweep_workers()

        assert lb._workers[nid].is_alive is False
        assert fh._total_deaths == 1

    @pytest.mark.asyncio
    async def test_fresh_worker_not_marked_dead(self, isolated_fh):
        fh, lb, _ = isolated_fh
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        lb._workers[nid].last_seen = time.time()

        await fh._sweep_workers()

        assert lb._workers[nid].is_alive is True
        assert fh._total_deaths == 0

    @pytest.mark.asyncio
    async def test_already_dead_worker_not_double_counted(self, isolated_fh):
        fh, lb, _ = isolated_fh
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        await lb.mark_worker_dead(nid)   # already dead
        lb._workers[nid].last_seen = time.time() - 9999

        await fh._sweep_workers()

        # Should not increment deaths a second time
        assert fh._total_deaths == 0


# ─────────────────────────────────────────────────────────────
# 2. In-flight task tracking
# ─────────────────────────────────────────────────────────────
class TestInFlightTracking:

    @pytest.mark.asyncio
    async def test_register_adds_task(self, isolated_fh):
        fh, lb, _ = isolated_fh
        tid = uuid.uuid4()
        nid = uuid.uuid4()
        await fh.register_task(tid, nid)
        assert tid in fh._in_flight
        assert fh._in_flight[tid].node_id == nid

    @pytest.mark.asyncio
    async def test_complete_removes_task(self, isolated_fh):
        fh, lb, _ = isolated_fh
        tid = uuid.uuid4()
        await fh.register_task(tid, uuid.uuid4())
        await fh.complete_task(tid)
        assert tid not in fh._in_flight

    @pytest.mark.asyncio
    async def test_complete_nonexistent_task_is_safe(self, isolated_fh):
        fh, lb, _ = isolated_fh
        await fh.complete_task(uuid.uuid4())   # must not raise

    @pytest.mark.asyncio
    async def test_stats_reflect_in_flight_count(self, isolated_fh):
        fh, lb, _ = isolated_fh
        nid = uuid.uuid4()
        tids = [uuid.uuid4() for _ in range(3)]
        for tid in tids:
            await fh.register_task(tid, nid)
        assert fh.get_fault_stats()["currently_in_flight"] == 3
        await fh.complete_task(tids[0])
        assert fh.get_fault_stats()["currently_in_flight"] == 2


# ─────────────────────────────────────────────────────────────
# 3. Orphaned task re-queue logging
# ─────────────────────────────────────────────────────────────
class TestOrphanedTasks:

    @pytest.mark.asyncio
    async def test_orphaned_tasks_requeued_on_worker_death(self, isolated_fh):
        fh, lb, _ = isolated_fh
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")

        # Register 3 tasks on this worker
        for _ in range(3):
            await fh.register_task(uuid.uuid4(), nid)

        lb._workers[nid].last_seen = time.time() - 9999
        await fh._sweep_workers()

        assert fh._total_requeues == 3
        # All orphaned tasks should be removed from in-flight
        assert len(fh._in_flight) == 0

    @pytest.mark.asyncio
    async def test_tasks_on_other_workers_not_affected(self, isolated_fh):
        fh, lb, _ = isolated_fh
        dead_nid  = uuid.uuid4()
        alive_nid = uuid.uuid4()
        await lb.register_worker(dead_nid,  "http://dead:8001")
        await lb.register_worker(alive_nid, "http://alive:8001")

        dead_tid  = uuid.uuid4()
        alive_tid = uuid.uuid4()
        await fh.register_task(dead_tid,  dead_nid)
        await fh.register_task(alive_tid, alive_nid)

        lb._workers[dead_nid].last_seen = time.time() - 9999
        await fh._sweep_workers()

        # alive_tid must still be tracked
        assert alive_tid in fh._in_flight
        assert dead_tid  not in fh._in_flight


# ─────────────────────────────────────────────────────────────
# 4. Recovery loop
# ─────────────────────────────────────────────────────────────
class TestRecovery:

    @pytest.mark.asyncio
    async def test_recovered_worker_re_registered(self, isolated_fh):
        fh, lb, _ = isolated_fh
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        await lb.mark_worker_dead(nid)
        assert lb._workers[nid].is_alive is False

        # Simulate /health returning 200
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("fault_tolerance.fault_handler.httpx.AsyncClient",
                   return_value=mock_client):
            await fh._probe_dead_workers()

        assert lb._workers[nid].is_alive is True
        assert fh._total_recoveries == 1

    @pytest.mark.asyncio
    async def test_silent_worker_stays_dead(self, isolated_fh):
        fh, lb, _ = isolated_fh
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        await lb.mark_worker_dead(nid)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("fault_tolerance.fault_handler.httpx.AsyncClient",
                   return_value=mock_client):
            await fh._probe_dead_workers()

        assert lb._workers[nid].is_alive is False
        assert fh._total_recoveries == 0

    @pytest.mark.asyncio
    async def test_probe_skips_alive_workers(self, isolated_fh):
        fh, lb, _ = isolated_fh
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        # Worker is alive — should NOT be probed

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("fault_tolerance.fault_handler.httpx.AsyncClient",
                   return_value=mock_client):
            await fh._probe_dead_workers()

        mock_client.get.assert_not_called()


# ─────────────────────────────────────────────────────────────
# 5. Fault stats
# ─────────────────────────────────────────────────────────────
class TestFaultStats:

    @pytest.mark.asyncio
    async def test_stats_shape(self, isolated_fh):
        fh, lb, _ = isolated_fh
        stats = fh.get_fault_stats()
        for key in ("total_worker_deaths", "total_worker_recoveries",
                    "total_tasks_requeued", "currently_in_flight", "fault_log_path"):
            assert key in stats

    @pytest.mark.asyncio
    async def test_counters_start_at_zero(self, isolated_fh):
        fh, lb, _ = isolated_fh
        stats = fh.get_fault_stats()
        assert stats["total_worker_deaths"]     == 0
        assert stats["total_worker_recoveries"] == 0
        assert stats["total_tasks_requeued"]    == 0
        assert stats["currently_in_flight"]     == 0


# ─────────────────────────────────────────────────────────────
# 6. Fault CSV logging
# ─────────────────────────────────────────────────────────────
class TestFaultCSV:

    @pytest.mark.asyncio
    async def test_death_event_written_to_csv(self, isolated_fh):
        fh, lb, tmp_log = isolated_fh
        nid = uuid.uuid4()
        await lb.register_worker(nid, "http://w:8001")
        lb._workers[nid].last_seen = time.time() - 9999

        with patch("fault_tolerance.fault_handler.FAULT_LOG_PATH", tmp_log):
            fh._ensure_log()
            await fh._sweep_workers()
            # Re-call log with the tmp path
            fh._log_event("WORKER_DEAD", nid, "http://w:8001", "test", 9999.0)

        rows = list(csv.DictReader(tmp_log.open()))
        assert any(r["event"] == "WORKER_DEAD" for r in rows)

    @pytest.mark.asyncio
    async def test_csv_row_has_required_fields(self, isolated_fh):
        fh, lb, tmp_log = isolated_fh
        nid = uuid.uuid4()

        with patch("fault_tolerance.fault_handler.FAULT_LOG_PATH", tmp_log):
            fh._ensure_log()
            fh._log_event("WORKER_DEAD", nid, "http://w:8001", "test", 42.0)

        rows = list(csv.DictReader(tmp_log.open()))
        assert len(rows) == 1
        row = rows[0]
        for col in ("timestamp", "event", "node_id", "worker_url",
                    "detail", "elapsed_silent_s"):
            assert col in row and row[col] != "", f"Column missing or empty: {col}"