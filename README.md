# Distributed RAG System

A distributed Retrieval-Augmented Generation (RAG) system built with FastAPI. The project routes user questions through a Master node, a dynamic Load Balancer, and one or more GPU worker nodes. Workers retrieve document context from a FAISS index, call a local Ollama model, and return the answer with timing metadata for monitoring and analysis.

The current working runtime is the FastAPI master/worker architecture:

- Master API: `master/scheduler.py`
- Load balancer: `lb/load_balancer.py`
- GPU worker API: `workers/gpu_worker.py`
- Fault monitor: `fault_tolerance/fault_handler.py`
- RAG retriever and index builder: `rag/`
- Streamlit client UI: `client/ui.py`

`main.py` and `client/load_generator.py` appear to be older/demo code paths and do not match the current service architecture as closely as the FastAPI modules above.

## Features

- Master node with API-key authentication, admission control, result caching, CSV request logging, and admin stats.
- Dynamic load balancing using a weighted least-connections strategy based on active tasks, CPU usage, and free GPU VRAM.
- Worker registration and heartbeat reporting.
- Fault tolerance with heartbeat watchdogs, worker death detection, recovery probing, and structured fault event logging.
- RAG pipeline using PDF parsing, sentence-transformer embeddings, and FAISS vector search.
- Ollama-backed LLM inference with a stub fallback if Ollama is unavailable.
- Optional Streamlit chat UI.
- Pytest test coverage for master APIs, load balancing behavior, fault handling, workers, and pipeline behavior.

## Architecture

```text
User / UI
   |
   v
Master Scheduler API
   |
   | validates API key
   | applies backpressure
   | logs request metadata
   v
Load Balancer
   |
   | chooses best alive worker
   | queues requests when saturated
   v
GPU Worker(s)
   |
   | retrieve RAG context from FAISS
   | call Ollama or fallback stub
   v
Master returns final response
```

Supporting components:

- Workers register with the Master at startup.
- Workers send periodic heartbeats to the Master.
- The Master forwards heartbeat state to the Load Balancer.
- The Fault Handler watches for silent workers and probes dead workers for recovery.
- Request timing data is written to `logs/request_log.csv`.
- Fault events are written to `logs/fault_events.csv`.

Architecture diagrams are available in:

- `docs/architecture.png`
- `docs/flowchart.png`

## Project Structure

```text
.
|-- client/
|   |-- ui.py                 # Streamlit chat interface
|   `-- load_generator.py     # Older/demo load generator
|-- common/
|   `-- models.py             # Shared dataclasses for messages and responses
|-- docs/
|   |-- architecture.png
|   `-- flowchart.png
|-- fault_tolerance/
|   `-- fault_handler.py      # Worker watchdog, recovery, fault logging
|-- lb/
|   `-- load_balancer.py      # Weighted least-connections load balancer
|-- llm/
|   `-- inference.py          # Ollama integration and fallback response
|-- logs/
|   |-- request_log.csv
|   `-- fault_events.csv
|-- master/
|   `-- scheduler.py          # Main FastAPI Master API
|-- monitoring/
|   `-- log.py                # Placeholder for monitoring utilities
|-- rag/
|   |-- build_index.py        # Build FAISS index from PDFs
|   |-- build_index.ipynb
|   `-- retriever.py          # Retrieve relevant chunks for a query
|-- tests/
|   `-- test_*.py             # Unit and integration tests
|-- workers/
|   `-- gpu_worker.py         # FastAPI worker service
|-- main.py                   # Older/demo entry point
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10 or newer recommended
- pip
- Optional: Ollama for real LLM responses
- Optional: NVIDIA GPU and `pynvml` for real VRAM reporting

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## RAG Setup

Create a document folder and add PDF files:

```bash
mkdir rag\docs
```

Place your PDFs in:

```text
rag/docs/
```

Build the FAISS index:

```bash
python -m rag.build_index
```

This creates:

```text
rag/index/faiss.index
rag/index/chunks.pkl
```

The worker and retriever expect those files to exist before answering document-grounded questions.

## Ollama Setup

The LLM integration uses Ollama at:

```text
http://localhost:11434
```

The configured model is:

```text
qwen2.5:14b
```

Install Ollama, then pull the model:

```bash
ollama pull qwen2.5:14b
```

If Ollama is not running, the system still responds using a clearly labeled stub fallback from `llm/inference.py`.

## Running the System

Open separate terminals for the Master and each worker.

Start the Master API:

```bash
uvicorn master.scheduler:app --host 0.0.0.0 --port 8000
```

Start one worker:

```bash
uvicorn workers.gpu_worker:app --host 0.0.0.0 --port 8001
```

Start additional workers on different ports:

```bash
uvicorn workers.gpu_worker:app --host 0.0.0.0 --port 8002
uvicorn workers.gpu_worker:app --host 0.0.0.0 --port 8003
```

Each worker generates a deterministic node ID from its port, registers with the Master, and begins sending heartbeats.

## Running the UI

Start the Streamlit client:

```bash
streamlit run client/ui.py
```

The UI sends requests to:

```text
http://localhost:8000/query
```

with the default development API key:

```text
dev-key-1
```

## API Usage

### Submit a Query

```bash
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -H "x-api-key: dev-key-1" ^
  -d "{\"user_id\":\"demo-user\",\"query\":\"What is classifier-free guidance?\",\"user_sent_at\":1710000000,\"parameters\":{}}"
```

Response:

```json
{
  "request_id": "uuid",
  "status": "success",
  "answer": "Generated answer text",
  "total_latency": 1.2345
}
```

### Fetch a Cached Result

```bash
curl http://localhost:8000/result/<request_id> ^
  -H "x-api-key: dev-key-1"
```

### View Admin Stats

```bash
curl http://localhost:8000/admin/stats ^
  -H "x-api-key: dev-key-1"
```

### Mark a Worker Dead

Useful for fault-tolerance testing:

```bash
curl -X POST http://localhost:8000/admin/mark-dead/<node_id> ^
  -H "x-api-key: dev-key-1"
```

## Configuration

The Master reads these environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEYS` | `dev-key-1,dev-key-2,test-key` | Comma-separated valid API keys |
| `LOG_DIR` | `logs` | Directory for request logs |
| `MAX_QUEUE` | `500` | Admission-control queue limit |

Important load-balancer tunables live in `lb/load_balancer.py`:

| Constant | Default | Description |
| --- | --- | --- |
| `MAX_TASKS_PER_WORKER` | `10` | Hard cap before a worker is considered saturated |
| `QUEUE_MAXSIZE` | `500` | Maximum pending queued requests |
| `WORKER_TIMEOUT_SEC` | `300` | Silent-worker timeout used by the Load Balancer |
| `HTTP_REQUEST_TIMEOUT` | `300` | Worker HTTP request timeout |
| `RETRY_ATTEMPTS` | `2` | Retry count for transient worker failures |

Fault-handler settings live in `fault_tolerance/fault_handler.py`:

| Constant | Default | Description |
| --- | --- | --- |
| `WATCHDOG_INTERVAL_SEC` | `10.0` | Worker heartbeat sweep interval |
| `RECOVERY_INTERVAL_SEC` | `30.0` | Dead-worker recovery probe interval |
| `HEARTBEAT_TIMEOUT_SEC` | `25.0` | Silence threshold before declaring a worker dead |
| `HEALTH_CHECK_TIMEOUT` | `5.0` | Timeout for `/health` recovery checks |

## Logs and Outputs

Request logs:

```text
logs/request_log.csv
```

Includes request IDs, user IDs, timing columns, worker IDs, provider/model metadata, latency, and status.

Fault logs:

```text
logs/fault_events.csv
```

Includes worker death events, recovery events, task requeue markers, and silence duration.

