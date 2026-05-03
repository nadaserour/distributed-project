#testing wip before rest of project is done, ignore this file for now
# tests/fake_worker.py
#
# A minimal FastAPI app that impersonates a GPU worker.
# Controllable via module-level variables so tests can simulate:
#   - slow inference        (INFERENCE_DELAY)
#   - node failures         (SHOULD_FAIL)
#   - custom responses      (RESPONSE_OVERRIDE)
#
# Run standalone on a port:
#   PORT=8001 python -m tests.fake_worker
# Or import and use the `app` object with httpx.AsyncClient directly.

import asyncio
import time
import uuid as _uuid
from typing import Optional

from fastapi import FastAPI, HTTPException # type: ignore

# ---------------------------------------------------------------------------
# Knobs — tests flip these between requests
# ---------------------------------------------------------------------------
INFERENCE_DELAY:   float         = 0.05   # seconds to sleep per request
SHOULD_FAIL:       bool          = False  # if True, /generate returns 500
FAIL_AFTER:        Optional[int] = None   # fail after N successful requests
RESPONSE_OVERRIDE: Optional[str] = None  # force a specific response_text

_request_count: int = 0   # internal counter

app = FastAPI()
NODE_ID = str(_uuid.uuid4())


def reset():
    """Call between tests to wipe state."""
    global INFERENCE_DELAY, SHOULD_FAIL, FAIL_AFTER, RESPONSE_OVERRIDE, _request_count
    INFERENCE_DELAY   = 0.05
    SHOULD_FAIL       = False
    FAIL_AFTER        = None
    RESPONSE_OVERRIDE = None
    _request_count    = 0


@app.get("/health")
async def health():
    return {"status": "ok", "node_id": NODE_ID}


@app.post("/generate")
async def generate(data: dict):
    global _request_count

    if SHOULD_FAIL:
        raise HTTPException(status_code=500, detail="Simulated worker failure")

    _request_count += 1

    if FAIL_AFTER is not None and _request_count > FAIL_AFTER:
        raise HTTPException(status_code=500, detail="Simulated failure after threshold")

    t_received = time.time()
    await asyncio.sleep(INFERENCE_DELAY)
    t_end = time.time()

    response_text = (
        RESPONSE_OVERRIDE
        if RESPONSE_OVERRIDE is not None
        else f"Fake answer to: {str(data.get('instruction', ''))[:40]}"
    )

    return {
        "task_id":            data.get("task_id", str(_uuid.uuid4())),
        "worker_id":          NODE_ID,
        "response_text":      response_text,
        "model_used":         "fake-model-v1",
        "provider":           "fake",
        "worker_received_at": t_received,
        "inference_start":    t_received,
        "inference_end":      t_end,
        "metrics":            {"tokens": 42},
        "status":             "success",
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import uvicorn  # type: ignore
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("tests.fake_worker:app", host="0.0.0.0", port=port, log_level="warning")