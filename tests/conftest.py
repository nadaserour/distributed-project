#  another file to ignore, not relevant to current tests
#  tests/conftest.py
#
# Shared pytest fixtures used across all test modules.
# Every fixture that starts a fake worker or the Master app
# sets up and tears down cleanly so tests never bleed into each other.

import asyncio
import time
import uuid
from typing import AsyncGenerator, Generator
from unittest.mock import patch

import pytest # type: ignore
import pytest_asyncio # type: ignore
from httpx import AsyncClient, ASGITransport # type: ignore

# ---------------------------------------------------------------------------
# Make project root importable regardless of working directory
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lb.load_balancer import LoadBalancer, MAX_TASKS_PER_WORKER, QUEUE_MAXSIZE
from fault_tolerance.fault_handler import FaultHandler
from common.models import Worker_Heartbeat, LB_To_Worker


# ---------------------------------------------------------------------------
# Event loop — single loop for the whole session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Fresh LoadBalancer per test
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def lb() -> AsyncGenerator[LoadBalancer, None]:
    balancer = LoadBalancer()
    balancer.start()
    yield balancer
    balancer.stop()
    # drain any pending queue items
    await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Fresh FaultHandler per test
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def fh(lb) -> AsyncGenerator[FaultHandler, None]:
    handler = FaultHandler(lb)
    handler.start()
    yield handler
    handler.stop()


# ---------------------------------------------------------------------------
# In-process fake worker ASGI app (no real ports needed)
# Returns an httpx.AsyncClient pointed at the fake worker.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def fake_worker_client():
    import tests.fake_worker as fw
    fw.reset()
    transport = ASGITransport(app=fw.app)
    async with AsyncClient(transport=transport, base_url="http://fake-worker") as client:
        yield client, fw   # yield both client AND the module so tests can flip knobs


# ---------------------------------------------------------------------------
# Helper: build a valid LB_To_Worker task
# ---------------------------------------------------------------------------
def make_task(query: str = "What is diffusion?") -> LB_To_Worker:
    return LB_To_Worker(
        task_id          = uuid.uuid4(),
        lb_dispatched_at = time.time(),
        instruction      = query,
        parameters       = {},
    )


# ---------------------------------------------------------------------------
# Helper: build a heartbeat for a given node_id
# ---------------------------------------------------------------------------
def make_heartbeat(
    node_id: uuid.UUID,
    load: int   = 0,
    cpu: float  = 10.0,
    vram: float = 8.0,
    status: str = "ready",
) -> Worker_Heartbeat:
    return Worker_Heartbeat(
        node_id             = node_id,
        status              = status,
        current_load_count  = load,
        cpu_usage_percent   = cpu,
        gpu_vram_free       = vram,
        last_seen           = time.time(),
    )


# ---------------------------------------------------------------------------
# Master app client — full FastAPI app with real LB + FH wired in
# Uses ASGI transport so no port is needed.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def master_client():
    # Patch the module-level lb and fh so we control them
    from master import scheduler as sched

    # Replace with fresh instances
    fresh_lb = LoadBalancer()
    fresh_fh = FaultHandler(fresh_lb)

    with patch.object(sched, "lb", fresh_lb), patch.object(sched, "fh", fresh_fh):
        transport = ASGITransport(app=sched.app)
        async with AsyncClient(
            transport=transport,
            base_url="http://master",
            headers={"x-api-key": "dev-key-1"},
        ) as client:
            fresh_lb.start()
            fresh_fh.start()
            yield client, fresh_lb, fresh_fh
            fresh_lb.stop()
            fresh_fh.stop()