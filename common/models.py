from dataclasses import dataclass
from uuid import UUID
from typing import Dict, Any

@dataclass
class User_Request:
    user_id: str
    query: str
    user_sent_at: float
    parameters: dict[str, any]

@dataclass
class Master_Message_To_LB:
    request_id: UUID
    query: str # Added this back so LB knows the content
    parameters: dict[str, any]
    priority_level: int # Optional

@dataclass
class LB_To_Worker:
    task_id: UUID # Same as request_id
    lb_dispatched_at: float # LB timestamp
    instruction: str
    parameters: dict[str, any]

@dataclass
class Worker_To_Master:
    task_id: UUID
    worker_id: str
    response_text: str
    model_used: str # For your analysis: which model did the worker pick?
    provider: str   # e.g., "Groq", "Local", "Gemini"
    # Timestamps for Master to calculate latencies
    worker_received_at: float 
    inference_start: float
    inference_end: float
    metrics: dict[str, float] # token count, vram, etc.
    status: str

@dataclass
class Worker_Heartbeat:
    node_id: UUID
    status: str
    current_load_count: int
    cpu_usage_percent: float
    gpu_vram_free: float
    last_seen: float

@dataclass
class Final_Response:
    request_id: str
    status: str
    answer: str
    total_latency: float # Calculated by Master (now - User_Request.user_sent_at)